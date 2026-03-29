"""
api/services/market_data_service.py

Live market data (quote + intraday chart) backed by yfinance.

Design
──────
The cache is a module-level singleton keyed by ticker so that:
  • FundamentalAnalysisAgent.get_price_data() writes here after its yfinance
    fetch, preventing a duplicate network call when the API endpoint is hit
    immediately after an analysis run.
  • The /api/market/{ticker} endpoint reads from the same cache, falling back
    to a fresh yfinance fetch on a cache miss.

Cache TTLs (configurable via core/config.py):
  MARKET_QUOTE_TTL    — 60 s  (live quote)
  MARKET_INTRADAY_TTL — 300 s (5-min bars, one full trading day)

Thread-safety: The cache dict is mutated only inside asyncio.to_thread()
callbacks which run sequentially within a single asyncio event loop — safe
under uvicorn's single-worker default.  Under Gunicorn multi-worker each
process maintains its own cache (acceptable; market data is public).
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import yfinance as yf

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ── Module-level shared cache ─────────────────────────────────────────────────
# Structure: ticker_key → (written_at_unix, payload)
# Keys: "quote:{TICKER}" and "intraday:{TICKER}"
_CACHE: Dict[str, Tuple[float, object]] = {}


def _cache_get(key: str, ttl: int) -> Optional[object]:
    """Return cached value if within TTL, else None."""
    entry = _CACHE.get(key)
    if entry is None:
        return None
    written_at, value = entry
    if time.time() - written_at > ttl:
        return None
    return value


def _cache_set(key: str, value: object) -> None:
    _CACHE[key] = (time.time(), value)


def write_quote_to_cache(ticker: str, data: dict) -> None:
    """
    Called by FundamentalAnalysisAgent (or any agent) after fetching price data
    so the API market endpoint does not re-fetch the same ticker immediately.
    """
    _cache_set(f"quote:{ticker.upper()}", data)


def write_intraday_to_cache(ticker: str, data: list) -> None:
    """Write intraday bars produced by the agent into the shared cache."""
    _cache_set(f"intraday:{ticker.upper()}", data)


# ── Service class (injected as a FastAPI dependency) ──────────────────────────


class MarketDataService:
    """
    Fetches live market data via yfinance with in-process caching.

    Instantiated once per request via FastAPI Depends() — the underlying
    _CACHE dict is module-level and shared across all instances.
    """

    async def get_quote(self, ticker: str) -> dict:
        """
        Live quote: price, change, % change, market status, company name.
        Returns a safe default dict on any fetch failure.
        """
        ticker = ticker.upper()
        cached = _cache_get(f"quote:{ticker}", settings.MARKET_QUOTE_TTL)
        if cached is not None:
            logger.debug("MarketDataService: quote cache hit for %s", ticker)
            return cached  # type: ignore[return-value]

        data = await asyncio.to_thread(self._fetch_quote_sync, ticker)
        _cache_set(f"quote:{ticker}", data)
        return data

    async def get_intraday(self, ticker: str) -> List[dict]:
        """
        5-minute intraday bars for the current trading day.
        Returns [] on failure so the frontend can render a graceful empty chart.
        """
        ticker = ticker.upper()
        cached = _cache_get(f"intraday:{ticker}", settings.MARKET_INTRADAY_TTL)
        if cached is not None:
            logger.debug("MarketDataService: intraday cache hit for %s", ticker)
            return cached  # type: ignore[return-value]

        data = await asyncio.to_thread(self._fetch_intraday_sync, ticker)
        _cache_set(f"intraday:{ticker}", data)
        return data

    # ── Sync helpers (run inside asyncio.to_thread) ───────────────────────────

    @staticmethod
    def _fetch_quote_sync(ticker: str) -> dict:
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            info = t.info or {}

            current: float = (
                getattr(fi, "last_price", None)
                or info.get("regularMarketPrice", 0)
                or 0
            )
            prev_close: float = (
                getattr(fi, "previous_close", None)
                or info.get("regularMarketPreviousClose", current)
                or current
            )
            change = float(current) - float(prev_close)
            pct = (change / float(prev_close) * 100) if prev_close else 0.0

            market_state = info.get("marketState", "CLOSED")
            status_map = {
                "REGULAR": "LIVE MARKET OPEN",
                "PRE": "PRE-MARKET",
                "POST": "AFTER HOURS",
                "CLOSED": "MARKET CLOSED",
            }
            return {
                "ticker": ticker,
                "companyName": (
                    info.get("longName") or info.get("shortName") or ticker
                ),
                "currentPrice": round(float(current), 2),
                "priceChange": round(float(change), 2),
                "priceChangePercent": round(float(pct), 2),
                "marketStatus": status_map.get(market_state, "MARKET CLOSED"),
            }
        except Exception as exc:
            logger.error(
                "MarketDataService: quote fetch failed for %s: %s", ticker, exc
            )
            return {
                "ticker": ticker,
                "companyName": ticker,
                "currentPrice": 0.0,
                "priceChange": 0.0,
                "priceChangePercent": 0.0,
                "marketStatus": "DATA UNAVAILABLE",
            }

    @staticmethod
    def _fetch_intraday_sync(ticker: str) -> List[dict]:
        try:
            df = yf.Ticker(ticker).history(period="1d", interval="5m")
            if df.empty:
                return []
            df["mid"] = (df["High"] + df["Low"]) / 2
            return [
                {
                    "time": idx.strftime("%H:%M"),
                    "price": round(float(row["mid"]), 2),
                }
                for idx, row in df.iterrows()
            ]
        except Exception as exc:
            logger.error(
                "MarketDataService: intraday fetch failed for %s: %s", ticker, exc
            )
            return []
