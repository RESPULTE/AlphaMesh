"""
Portfolio Service — manages user holdings and live valuation.

Persistence: Redis (per user_email key). Falls back to session state only.
Supports manual entry, CSV upload, and provides enriched valuation.
"""
from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ui.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

COLS = ["ticker", "name", "quantity", "avg_cost", "currency"]

_EMPTY = pd.DataFrame(columns=COLS)


class PortfolioService:
    """
    All portfolio logic goes here — not in app.py or UI components.
    The UI should call these methods and render what's returned.
    """

    def __init__(self, market_data: MarketDataService):
        self._md = market_data
        self._r  = market_data._r   # shared Redis connection

    # ── Storage key ──────────────────────────────────────────────

    @staticmethod
    def _key(user_email: str) -> str:
        safe = user_email.replace("@", "_").replace(".", "_")
        return f"nx:portfolio:{safe}"

    # ── CRUD ─────────────────────────────────────────────────────

    def load(self, user_email: str) -> pd.DataFrame:
        """Load portfolio from Redis. Returns empty DataFrame if not found."""
        if self._r:
            try:
                raw = self._r.get(self._key(user_email))
                if raw:
                    records = json.loads(raw)
                    if records:
                        return pd.DataFrame(records)[COLS]
            except Exception as exc:
                logger.error(f"[Portfolio] Load failed: {exc}")
        return _EMPTY.copy()

    def save(self, user_email: str, df: pd.DataFrame):
        """Persist portfolio to Redis."""
        if not self._r:
            return
        try:
            records = df.fillna("").to_dict("records")
            self._r.set(self._key(user_email), json.dumps(records))
        except Exception as exc:
            logger.error(f"[Portfolio] Save failed: {exc}")

    def add_position(
        self,
        user_email: str,
        df: pd.DataFrame,
        ticker: str,
        name: str,
        quantity: float,
        avg_cost: float,
        currency: str = "USD",
    ) -> pd.DataFrame:
        """Add or update a position. Returns the updated DataFrame."""
        ticker = ticker.upper()
        row = {"ticker": ticker, "name": name, "quantity": quantity,
               "avg_cost": avg_cost, "currency": currency}
        # Update if exists
        mask = df["ticker"].str.upper() == ticker
        if mask.any():
            df.loc[mask, ["name", "quantity", "avg_cost", "currency"]] = \
                [name, quantity, avg_cost, currency]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self.save(user_email, df)
        return df

    def remove_position(
        self, user_email: str, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Remove a position by ticker."""
        df = df[df["ticker"].str.upper() != ticker.upper()].reset_index(drop=True)
        self.save(user_email, df)
        return df

    # ── Valuation ─────────────────────────────────────────────────

    def valuate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches holdings with live prices, market value, P&L, and weight.
        Returns a new DataFrame; does NOT save.
        """
        if df.empty:
            return df

        df = df.copy()
        df["ticker"]   = df["ticker"].str.upper()
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
        df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce").fillna(0)

        prices = self._md.get_multiple_prices(df["ticker"].tolist())
        df["current_price"]  = df["ticker"].map(prices)
        df["market_value"]   = df["current_price"] * df["quantity"]
        df["cost_basis"]     = df["avg_cost"] * df["quantity"]
        df["unrealized_pnl"] = df["market_value"] - df["cost_basis"]
        df["pnl_pct"] = (
            df["unrealized_pnl"]
            / df["cost_basis"].replace(0, float("nan"))
            * 100
        )
        total = df["market_value"].sum()
        df["weight_pct"] = (df["market_value"] / total * 100) if total > 0 else 0.0

        return df

    def summary(self, valued_df: pd.DataFrame) -> Dict[str, Any]:
        """Returns top-level portfolio summary dict."""
        if valued_df.empty:
            return {"total_value": 0, "total_cost": 0, "total_pnl": 0, "total_pnl_pct": 0, "n_positions": 0}
        total_value = valued_df["market_value"].sum()
        total_cost  = valued_df["cost_basis"].sum()
        total_pnl   = valued_df["unrealized_pnl"].sum()
        pnl_pct     = (total_pnl / total_cost * 100) if total_cost else 0
        return {
            "total_value":   total_value,
            "total_cost":    total_cost,
            "total_pnl":     total_pnl,
            "total_pnl_pct": pnl_pct,
            "n_positions":   len(valued_df),
        }

    # ── Import helpers ────────────────────────────────────────────

    def from_csv(self, file_bytes: bytes) -> pd.DataFrame:
        """
        Parses a CSV upload.
        Accepts columns: ticker/symbol, quantity/shares/units,
        avg_cost/average_cost/price, name/company, currency.
        """
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            logger.error(f"[Portfolio] CSV parse failed: {exc}")
            return _EMPTY.copy()

        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
        rename = {
            "symbol": "ticker", "shares": "quantity", "units": "quantity",
            "average_cost": "avg_cost", "average_cost_per_share": "avg_cost",
            "price": "avg_cost", "purchase_price": "avg_cost",
            "company": "name", "stock": "name",
        }
        df = df.rename(columns=rename)
        for col in COLS:
            if col not in df.columns:
                df[col] = "" if col in ("name", "currency") else 0.0
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["currency"] = df["currency"].fillna("USD")
        return df[COLS].copy()

    def from_broker_placeholder(self) -> str:
        """Returns a message explaining broker API integration."""
        return (
            "Broker API integration (IBKR, Alpaca, etc.) is a backend concern. "
            "Connect your broker via the Settings panel and the portfolio will "
            "sync automatically."
        )

    # ── Context string for agents ─────────────────────────────────

    def to_context_string(self, valued_df: pd.DataFrame) -> str:
        """
        Returns a concise text summary suitable for injection into the agent's
        system context when the user has toggled portfolio into context.
        """
        if valued_df.empty:
            return "PORTFOLIO: Empty — no positions."

        lines = ["PORTFOLIO HOLDINGS (live data):"]
        for _, row in valued_df.iterrows():
            pnl_str = f"{row.get('pnl_pct', 0):+.1f}%" if pd.notna(row.get("pnl_pct")) else "N/A"
            lines.append(
                f"  • {row['ticker']}: {row['quantity']:.2f} shares"
                f" @ ${row.get('current_price', 0):.2f}"
                f" | MV=${row.get('market_value', 0):,.0f}"
                f" | P&L={pnl_str}"
            )
        sm = self.summary(valued_df)
        lines.append(f"\nTotal Market Value: ${sm['total_value']:,.2f}")
        lines.append(f"Total P&L: ${sm['total_pnl']:+,.2f} ({sm['total_pnl_pct']:+.2f}%)")
        return "\n".join(lines)
