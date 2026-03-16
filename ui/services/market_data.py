"""
Market Data Service — yfinance with Redis caching.

All fetched data is cached in Redis with configurable TTLs.
Gracefully degrades to live fetches when Redis is unavailable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

from ui.config import MARKET

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Unified market data interface.

    Usage:
        svc = MarketDataService()
        df  = svc.get_ohlcv("AAPL", period="1y")
        price = svc.get_current_price("AAPL")
    """

    def __init__(self):
        self._r: Optional[Any] = None
        self._try_connect()

    # ── Redis bootstrap ───────────────────────────────────────────

    def _try_connect(self):
        if not _REDIS_AVAILABLE:
            logger.warning("[MarketData] redis-py not installed — caching disabled.")
            return
        try:
            r = _redis_lib.Redis(
                host=MARKET["redis_host"],
                port=MARKET["redis_port"],
                db=MARKET["redis_db"],
                decode_responses=True,
                socket_timeout=1,
                socket_connect_timeout=1,
            )
            r.ping()
            self._r = r
            logger.info("[MarketData] Redis connected ✓")
        except Exception as exc:
            logger.warning(f"[MarketData] Redis unavailable ({exc}) — live fetches only.")

    @property
    def redis_ok(self) -> bool:
        return self._r is not None

    # ── Cache helpers ─────────────────────────────────────────────

    def _get(self, key: str) -> Optional[Any]:
        if not self._r:
            return None
        try:
            raw = self._r.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _set(self, key: str, value: Any, ttl: int):
        if not self._r:
            return
        try:
            self._r.setex(key, ttl, json.dumps(value, default=str))
        except Exception:
            pass

    # ── Current Price ─────────────────────────────────────────────

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Latest market price, cached for cache_ttl_price seconds."""
        key = f"nx:px:{ticker.upper()}"
        cached = self._get(key)
        if cached is not None:
            return float(cached)
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
            if price:
                self._set(key, price, MARKET["cache_ttl_price"])
            return price
        except Exception as exc:
            logger.error(f"[MarketData] Price fetch failed {ticker}: {exc}")
            return None

    def get_multiple_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Batch price fetch — hits cache first, falls back to yf.download."""
        result: Dict[str, Optional[float]] = {}
        missing: List[str] = []

        for t in tickers:
            cached = self._get(f"nx:px:{t.upper()}")
            if cached is not None:
                result[t] = float(cached)
            else:
                missing.append(t)

        if missing:
            try:
                if len(missing) == 1:
                    data = yf.download(missing[0], period="1d", interval="1m", progress=False)
                    if not data.empty:
                        result[missing[0]] = float(data["Close"].iloc[-1])
                        self._set(f"nx:px:{missing[0].upper()}", result[missing[0]], MARKET["cache_ttl_price"])
                else:
                    joined = " ".join(missing)
                    data = yf.download(joined, period="1d", interval="1m", progress=False)
                    for t in missing:
                        try:
                            result[t] = float(data["Close"][t].iloc[-1])
                            self._set(f"nx:px:{t.upper()}", result[t], MARKET["cache_ttl_price"])
                        except Exception:
                            result[t] = None
            except Exception as exc:
                logger.error(f"[MarketData] Batch price failed: {exc}")
                for t in missing:
                    result.setdefault(t, None)

        return result

    # ── OHLCV ─────────────────────────────────────────────────────

    def get_ohlcv(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """OHLCV DataFrame by period string, cached."""
        key = f"nx:ohlcv:{ticker.upper()}:{period}:{interval}"
        cached = self._get(key)
        if cached:
            df = pd.DataFrame(cached)
            df["Date"] = pd.to_datetime(df["Date"])
            return df.set_index("Date")

        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval)
            if df.empty:
                return df
            df.index = df.index.tz_localize(None)
            records = df.reset_index().rename(columns={"Datetime": "Date"}).to_dict("records")
            self._set(key, records, MARKET["cache_ttl_ohlcv"])
            return df
        except Exception as exc:
            logger.error(f"[MarketData] OHLCV failed {ticker}: {exc}")
            return pd.DataFrame()

    def get_ohlcv_range(
        self, ticker: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """OHLCV for explicit date range. Automatically picks interval."""
        delta_days = (end - start).days
        if delta_days <= 30:
            interval = "1d"
        elif delta_days <= 365:
            interval = "1wk"
        else:
            interval = "1mo"

        start_s = start.strftime("%Y-%m-%d")
        end_s   = end.strftime("%Y-%m-%d")
        key = f"nx:ohlcv_range:{ticker.upper()}:{start_s}:{end_s}:{interval}"
        cached = self._get(key)
        if cached:
            df = pd.DataFrame(cached)
            df["Date"] = pd.to_datetime(df["Date"])
            return df.set_index("Date")

        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start_s, end=end_s, interval=interval)
            if df.empty:
                return df
            df.index = df.index.tz_localize(None)
            records = df.reset_index().rename(columns={"Datetime": "Date"}).to_dict("records")
            self._set(key, records, MARKET["cache_ttl_ohlcv"])
            return df
        except Exception as exc:
            logger.error(f"[MarketData] OHLCV range failed {ticker}: {exc}")
            return pd.DataFrame()

    # ── Company Info ──────────────────────────────────────────────

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Company metadata — name, sector, PE, 52-week range, etc."""
        key = f"nx:info:{ticker.upper()}"
        cached = self._get(key)
        if cached:
            return cached
        try:
            info = yf.Ticker(ticker).info
            subset = {
                k: info.get(k)
                for k in (
                    "shortName", "longName", "sector", "industry",
                    "marketCap", "trailingPE", "forwardPE",
                    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                    "dividendYield", "currency",
                    "website", "longBusinessSummary",
                )
            }
            subset["shortName"] = subset["shortName"] or ticker
            self._set(key, subset, MARKET["cache_ttl_info"])
            return subset
        except Exception as exc:
            logger.error(f"[MarketData] Info failed {ticker}: {exc}")
            return {"shortName": ticker, "currency": "USD"}

    # ── Search ────────────────────────────────────────────────────

    def search_ticker(self, query: str) -> List[Dict[str, str]]:
        """Fuzzy ticker search via yfinance Search API."""
        try:
            results = yf.Search(query, max_results=8)
            return [
                {
                    "ticker":   q.get("symbol", ""),
                    "name":     q.get("shortname") or q.get("longname", ""),
                    "exchange": q.get("exchange", ""),
                    "type":     q.get("quoteType", ""),
                }
                for q in results.quotes
                if q.get("symbol")
            ]
        except Exception as exc:
            logger.error(f"[MarketData] Search failed '{query}': {exc}")
            return []

    # ── Cache admin (for UI settings panel) ──────────────────────

    def flush_cache(self, pattern: str = "nx:*"):
        if not self._r:
            return 0
        try:
            keys = self._r.keys(pattern)
            if keys:
                return self._r.delete(*keys)
            return 0
        except Exception:
            return 0

    def cache_stats(self) -> Dict[str, Any]:
        if not self._r:
            return {"available": False}
        try:
            info = self._r.info("memory")
            keys = len(self._r.keys("nx:*"))
            return {
                "available": True,
                "keys": keys,
                "used_memory_human": info.get("used_memory_human", "?"),
            }
        except Exception:
            return {"available": False}
