"""
core/market_data_service.py

Live market data (quote + intraday chart) backed by yfinance.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Awaitable, Dict, List, Optional, Tuple

import aiosqlite
import yfinance as yf

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# -- Module-level shared cache -----------------------------------------------
# Structure: ticker_key ? (written_at_unix, payload)
# Keys: "quote:{TICKER}" and "intraday:{TICKER}"
_CACHE: Dict[str, Tuple[float, object]] = {}
_SQLITE_INITIALIZED = False
_SQLITE_INIT_LOCK = asyncio.Lock()


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


async def _ensure_sqlite_cache() -> None:
    global _SQLITE_INITIALIZED
    if _SQLITE_INITIALIZED:
        return
    async with _SQLITE_INIT_LOCK:
        if _SQLITE_INITIALIZED:
            return
        async with aiosqlite.connect(settings.MARKET_CACHE_DB_PATH) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS market_cache (
                    cache_key  TEXT PRIMARY KEY,
                    payload    TEXT NOT NULL,
                    written_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_cache_written "
                "ON market_cache (written_at DESC)"
            )
            await db.commit()
        _SQLITE_INITIALIZED = True


async def _sqlite_get(key: str, ttl: int) -> Optional[object]:
    try:
        await _ensure_sqlite_cache()
        async with aiosqlite.connect(settings.MARKET_CACHE_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT payload, written_at FROM market_cache WHERE cache_key = ?",
                (key,),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return None
        written_at = float(row["written_at"])
        if time.time() - written_at > ttl:
            return None
        return json.loads(row["payload"])
    except Exception as exc:
        logger.debug("MarketDataService: sqlite cache read failed: %s", exc)
        return None


async def _sqlite_set(key: str, value: object) -> None:
    try:
        await _ensure_sqlite_cache()
        payload = json.dumps(value, default=str)
        async with aiosqlite.connect(settings.MARKET_CACHE_DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO market_cache (cache_key, payload, written_at)
                VALUES (?, ?, ?)
                ON CONFLICT (cache_key) DO UPDATE
                    SET payload = excluded.payload,
                        written_at = excluded.written_at
                """,
                (key, payload, time.time()),
            )
            await db.commit()
    except Exception as exc:
        logger.debug("MarketDataService: sqlite cache write failed: %s", exc)


def _fire_and_forget(coro: Awaitable[None]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(coro)


def write_quote_to_cache(ticker: str, data: dict) -> None:
    """
    Called by FundamentalAnalysisAgent (or any agent) after fetching price data
    so the API market endpoint does not re-fetch the same ticker immediately.
    """
    key = f"quote:{ticker.upper()}"
    _cache_set(key, data)
    _fire_and_forget(_sqlite_set(key, data))


def write_intraday_to_cache(ticker: str, data: list) -> None:
    """Write intraday bars produced by the agent into the shared cache."""
    key = f"intraday:{ticker.upper()}"
    _cache_set(key, data)
    _fire_and_forget(_sqlite_set(key, data))


class MarketDataService:
    """
    Fetches live market data via yfinance with in-process caching.

    Instantiated once per request via FastAPI Depends() � the underlying
    _CACHE dict is module-level and shared across all instances.
    """

    async def get_quote(self, ticker: str) -> dict:
        """
        Live quote: price, change, % change, market status, company name.
        Returns a safe default dict on any fetch failure.
        """
        ticker = ticker.upper()
        return await self._get_with_cache(
            kind="quote",
            ticker=ticker,
            ttl=settings.MARKET_QUOTE_TTL,
            fetcher=self._fetch_quote_sync,
        )

    async def get_intraday(self, ticker: str) -> List[dict]:
        """
        5-minute intraday bars for the current trading day.
        Returns [] on failure so the frontend can render a graceful empty chart.
        """
        ticker = ticker.upper()
        return await self._get_with_cache(
            kind="intraday",
            ticker=ticker,
            ttl=settings.MARKET_INTRADAY_TTL,
            fetcher=self._fetch_intraday_sync,
        )

    async def _get_with_cache(
        self,
        *,
        kind: str,
        ticker: str,
        ttl: int,
        fetcher,
    ):
        key = f"{kind}:{ticker}"
        sqlite_task = asyncio.create_task(_sqlite_get(key, ttl))
        cached = _cache_get(key, ttl)
        if cached is not None:
            logger.debug("MarketDataService: %s memory cache hit for %s", kind, ticker)
            sqlite_task.cancel()
            return cached

        sqlite_value = None
        try:
            sqlite_value = await sqlite_task
        except asyncio.CancelledError:
            sqlite_value = None

        if sqlite_value is not None:
            logger.debug("MarketDataService: %s sqlite cache hit for %s", kind, ticker)
            _cache_set(key, sqlite_value)
            return sqlite_value

        data = await asyncio.to_thread(fetcher, ticker)
        _cache_set(key, data)
        await _sqlite_set(key, data)
        return data

    # -- Sync helpers (run inside asyncio.to_thread) --------------------------

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
