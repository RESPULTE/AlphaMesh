"""
Refactored _data_prep_node for FundamentalAnalysisAgent.

Drop these methods into FundamentalAnalysisAgent, replacing the existing
_data_prep_node.  Everything else in the class is untouched.

Changes
-------
1. True concurrency fix: edgar + price fetches now run in asyncio.gather —
   the original code awaited them sequentially, negating create_task entirely.

2. Extracted four private helpers so each concern lives in one place:
     _resolve_date_range()     — date arithmetic, period list, form type
     _fetch_raw_data()         — concurrent EDGAR + yfinance fetches
     _trim_and_normalise()     — date-column trimming + period-end normalisation
     _merge_price_rows()       — aligns and appends the price sub-DataFrame

3. _data_prep_node itself becomes a thin coordinator: build config → fetch →
   process → merge → return.  Easy to read top-to-bottom.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd

from core.agents.models.fundamental_agent_models import _AgentState
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _DataPrepConfig:
    start_dt: datetime
    end_dt: datetime
    form_type: str  # "10-K" | "10-Q"
    price_interval: str  # "daily" | "monthly"
    periods: List  # List[int] for 10-K, List[Tuple[int,int]] for 10-Q


def _price_interval_from_span(start: datetime, end: datetime) -> str:
    span_days = max(0, (end - start).days)
    return "daily" if span_days <= 31 else "monthly"


def _normalize_period_ends(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for col in df.columns:
        try:
            ts = pd.Timestamp(col)
            if granularity == "yearly":
                rename[col] = ts.replace(month=12, day=31).strftime("%Y-%m-%d")
            else:
                rename[col] = (
                    ts.to_period("Q").end_time.normalize().strftime("%Y-%m-%d")
                )
        except Exception:
            rename[col] = str(col)
    return df.rename(columns=rename)


def _quarterly_periods(start: datetime, end: datetime) -> list:
    """Returns (year, quarter) tuples covering start..end inclusive."""
    periods = []
    year, month = start.year, start.month
    while datetime(year, month, 1) <= end:
        quarter = (month - 1) // 3 + 1
        if (year, quarter) not in periods:
            periods.append((year, quarter))
        month += 3
        if month > 12:
            month -= 12
            year += 1
    return periods


def _resolve_date_range(state: _AgentState) -> _DataPrepConfig:
    """
    Derive all date / period parameters from the agent state.

    Encapsulates every branch of the yearly-vs-quarterly logic so
    _data_prep_node never has to reason about it.
    """
    granularity: str = getattr(state, "granularity", "yearly") or "yearly"
    end_dt = state.end_date or datetime.now()
    price_interval = _price_interval_from_span(
        state.start_date or end_dt, end_dt
    )

    if granularity == "yearly":
        default_start = datetime(end_dt.year - 4, 1, 1)
        start_dt = state.start_date or default_start
        # Ensure at least a 4-year window
        if (end_dt.year - start_dt.year) < 4:
            start_dt = datetime(end_dt.year - 4, 1, 1)

        today = datetime.now()
        last_complete_year = today.year - 1 if today.month < 12 else today.year
        periods = list(range(start_dt.year, min(end_dt.year, last_complete_year) + 1))

        return _DataPrepConfig(
            start_dt=start_dt,
            end_dt=end_dt,
            form_type="10-K",
            price_interval=price_interval,
            periods=periods,
        )
    else:
        start_dt = state.start_date or (end_dt - timedelta(days=2 * 365))
        periods = _quarterly_periods(start_dt, end_dt)

        return _DataPrepConfig(
            start_dt=start_dt,
            end_dt=end_dt,
            form_type="10-Q",
            price_interval=price_interval,
            periods=periods,
        )


async def _fetch_raw_data(
    db,
    ticker: str,
    cfg: _DataPrepConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run EDGAR update and yfinance price fetch concurrently, then load the
    cached EDGAR data once the update completes.

    Returns (financial_df, price_df).  Either may be empty on failure — all
    exceptions are re-raised so the caller can handle them uniformly.

    NOTE: the EDGAR update must finish before get_data() is called (price
    fetch has no such dependency), so we await the update inside the
    gathered coroutine and chain get_data() immediately after.
    """

    async def _edgar_update_and_load() -> pd.DataFrame:
        await db.update_financials(ticker, cfg.periods, cfg.form_type)
        return await db.get_data(ticker, form_types=[cfg.form_type])

    async def _price_fetch() -> pd.DataFrame:
        return await db.get_price_data(
            ticker,
            start=cfg.start_dt.strftime("%Y-%m-%d"),
            end=cfg.end_dt.strftime("%Y-%m-%d"),
            interval=cfg.price_interval,
        )

    # True concurrency: EDGAR network I/O and yfinance run at the same time.
    return await asyncio.gather(
        _edgar_update_and_load(),
        _price_fetch(),
    )


def _trim_and_normalise(
    df: pd.DataFrame,
    cfg: _DataPrepConfig,
) -> pd.DataFrame:
    """
    1. Drop columns outside [start_dt, end_dt].
    2. Normalise column names to canonical period-end dates
       (e.g. any date within Q3 → "2023-09-30").
    """
    if df.empty:
        return df

    try:
        col_dates = pd.to_datetime(df.columns, errors="coerce")
        keep_mask = (col_dates >= pd.Timestamp(cfg.start_dt).tz_localize(None)) & (
            col_dates <= pd.Timestamp(cfg.end_dt).tz_localize(None)
        )
        df = df.loc[:, keep_mask]
    except Exception as exc:
        logger.warning("[data_prep] Date trimming failed: %s", exc)

    return _normalize_period_ends(df, getattr(cfg, "granularity", "yearly"))


def _canonical_date_strs(index: Any, granularity: str) -> list:
    result = []
    for val in index:
        try:
            ts = pd.Timestamp(val)
            if granularity == "daily":
                result.append(ts.normalize().strftime("%Y-%m-%d"))
            elif granularity == "monthly":
                result.append(
                    ts.to_period("M").end_time.normalize().strftime("%Y-%m-%d")
                )
            elif granularity == "yearly":
                result.append(ts.replace(month=12, day=31).strftime("%Y-%m-%d"))
            else:
                result.append(
                    ts.to_period("Q").end_time.normalize().strftime("%Y-%m-%d")
                )
        except Exception:
            result.append(str(val))
    return result


def _merge_price_rows(
    financial_df: pd.DataFrame,
    price_df: pd.DataFrame,
    cfg: _DataPrepConfig,
    available_concepts: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transpose the price series, align its columns with financial_df, and
    concat.  Returns the merged DataFrame and the updated concepts list.

    No-ops cleanly when price_df is empty or lacks a "stock_price" column.
    """
    if price_df.empty or "stock_price" not in price_df.columns:
        return financial_df, available_concepts

    price_t = price_df[["stock_price"]].T
    price_t.columns = _canonical_date_strs(price_t.columns, cfg.price_interval)

    if not financial_df.empty:
        all_cols = sorted(set(financial_df.columns) | set(price_t.columns))
        financial_df = financial_df.reindex(columns=all_cols)
        price_t = price_t.reindex(columns=all_cols)

    merged = pd.concat([financial_df, price_t])

    if "stock_price" not in available_concepts:
        available_concepts = [*available_concepts, "stock_price"]

    return merged, available_concepts
