"""
core/agents/financial_db.py

Manages fetching (via SEC EDGAR), caching (SQLite), and querying of
standardised financial statement data, plus stock price history (yfinance).

Key changes vs original
───────────────────────
1. _fetch_edgar_data_sync: signature fixed (year: int → years: List[int]).
   Fetches all filings for the form type in a thread-safe way via asyncio.to_thread.

2. _process_statement: now resolves labels in priority order:
     standard_concept  (edgartools standardised name, most consistent)
     → concept tag stripped of namespace prefix  (e.g. "us-gaap:Revenues" → "Revenues")
     → raw label  (human-readable but can change between filings)
   This eliminates the primary source of label inconsistency.

3. get_all_concepts: new method that returns every label stored for a ticker.
   Replaces the broken pattern search_label(ticker, []) which always returned
   an empty DataFrame (the keywords guard short-circuits on an empty list).

4. asyncio.get_event_loop() usage replaced with asyncio.to_thread().

5. _fetch_edgar_data_sync robustly handles multi-year fetches for both
   10-K and 10-Q form types.
"""

import asyncio
import re
from typing import Dict, List, Literal, Optional, Set, Tuple, TypeAlias, Union

import aiosqlite
import pandas as pd
from edgar import Company, MultiFinancials, set_identity

from core.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

FormType: TypeAlias = Literal["10-K", "10-Q"]
StatementType: TypeAlias = Literal["income", "balance", "cashflow"]
Period: TypeAlias = Union[int, Tuple[int, int]]

USER_AGENT = "FundamentalAnalysisBot yeapzing@utar.edu.my"
set_identity(USER_AGENT)

DB_PATH = "./data/financial_data.db"
ALL_STATEMENT_TYPES: List[StatementType] = ["income", "balance", "cashflow"]
SUPPORTED_FORM_TYPES: List[FormType] = ["10-K", "10-Q"]

# Namespace prefix regex — strips e.g. "us-gaap:" from XBRL concept tags
_NS_PREFIX_RE = re.compile(r"^[a-zA-Z0-9\-]+:")


# ─────────────────────────────────────────────────────────────────────────────
# FinancialDatabase
# ─────────────────────────────────────────────────────────────────────────────


class FinancialDatabase:
    """
    Manages fetching, storing, and retrieving financial data from EDGAR filings.
    """

    def __init__(self, db_name: str = DB_PATH):
        self.db_name = db_name

    async def initialize(self) -> None:
        """Creates the table and indexes if they do not already exist."""
        async with aiosqlite.connect(self.db_name) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS financials (
                    company        TEXT,
                    period_date    TEXT,
                    form_type      TEXT,
                    statement_type TEXT,
                    label          TEXT,
                    value          REAL,
                    PRIMARY KEY (company, period_date, form_type, statement_type, label)
                );
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_company_form_date "
                "ON financials (company, form_type, period_date);"
            )
            await db.commit()
            logger.info("[DB] Initialized.")

    # ─────────────────────────────────────────────────────────────────────────
    # 1.  CORE DATA FETCHING AND PROCESSING
    # ─────────────────────────────────────────────────────────────────────────

    async def update_financials(
        self, ticker: str, periods: List[Period], form_type: FormType
    ) -> None:
        """Fetches and stores any periods not yet cached in the local DB."""
        uncovered = await self.find_uncovered_periods(ticker, periods, form_type)
        if not uncovered:
            logger.info("[DB] All requested periods for %s already cached.", ticker)
            return
        logger.info("[DB] Uncovered periods to fetch: %s", uncovered)
        await self.fetch_and_store_period(ticker, uncovered, form_type)

    async def fetch_and_store_period(
        self, ticker: str, periods: List[Period], form_type: FormType
    ) -> None:
        """
        Fetches financial data from EDGAR for all required years, processes it
        into long format, filters to exactly the requested periods, and persists
        to SQLite.
        """
        if not periods:
            return

        years = sorted(set(p if isinstance(p, int) else p[0] for p in periods))
        logger.info("[EDGAR] Fetching %s %s for years %s …", ticker, form_type, years)

        try:
            dfs_map = await self._fetch_edgar_data_sync(ticker, years, form_type)
        except Exception as exc:
            logger.error("[EDGAR] Fetch failed for %s: %s", ticker, exc)
            return

        if not dfs_map or not any(not df.empty for df in dfs_map.values()):
            logger.warning(
                "[EDGAR] No financials returned for %s years=%s", ticker, years
            )
            return

        all_rows: List[pd.DataFrame] = []

        for stmt_type, df in dfs_map.items():
            processed = self._process_statement(df, ticker, stmt_type, form_type)
            if processed.empty:
                continue

            processed["period_date_dt"] = pd.to_datetime(processed["period_date"])

            if form_type == "10-Q":
                actual_periods = processed.apply(
                    lambda row: (
                        row["period_date_dt"].year,
                        row["period_date_dt"].quarter,
                    ),
                    axis=1,
                )
                processed = processed[actual_periods.isin(periods)]
            else:  # 10-K
                processed = processed[processed["period_date_dt"].dt.year.isin(years)]

            processed = processed.drop(columns=["period_date_dt"])
            if not processed.empty:
                all_rows.append(processed)

        if not all_rows:
            logger.warning(
                "[DB] No rows remained after period filtering for %s.", ticker
            )
            return

        await self._bulk_insert(pd.concat(all_rows, ignore_index=True))

    async def _fetch_edgar_data_sync(
        self,
        ticker: str,
        years: List[int],  # FIX: was `year: int` — now correctly accepts a list
        form_type: FormType,
    ) -> Dict[StatementType, pd.DataFrame]:
        """
        Runs blocking EDGAR network I/O in a thread via asyncio.to_thread().

        Strategy: fetch all filings for the form type (EDGAR returns them in
        reverse-chronological order). MultiFinancials combines them into aligned
        DataFrames.  Downstream period filtering in fetch_and_store_period
        then trims to only the requested years.
        """

        def _fetch() -> Dict[StatementType, pd.DataFrame]:
            company = Company(ticker)
            # Fetch all filings of the requested form; period filtering happens
            # after we have the data (more robust against EDGAR date edge cases).
            filings = company.get_filings(form=form_type)
            if not filings:
                logger.warning("[EDGAR] No %s filings found for %s", form_type, ticker)
                return {}

            try:
                mf = MultiFinancials.extract(filings)
            except Exception as exc:
                logger.error(
                    "[EDGAR] MultiFinancials.extract failed for %s: %s", ticker, exc
                )
                return {}

            result: Dict[StatementType, pd.DataFrame] = {}

            for stmt_type, method_name in [
                ("income", "income_statement"),
                ("balance", "balance_sheet"),
                ("cashflow", "cashflow_statement"),
            ]:
                try:
                    stmt_obj = getattr(mf, method_name)()
                    df = stmt_obj.to_dataframe()
                    result[stmt_type] = df if df is not None else pd.DataFrame()
                except Exception as exc:
                    logger.warning(
                        "[EDGAR] Could not extract %s for %s: %s",
                        stmt_type,
                        ticker,
                        exc,
                    )
                    result[stmt_type] = pd.DataFrame()

            return result

        return await asyncio.to_thread(_fetch)

    def _process_statement(
        self,
        df: pd.DataFrame,
        ticker: str,
        stmt_type: StatementType,
        form_type: FormType,
    ) -> pd.DataFrame:
        """
        Transforms a raw edgartools DataFrame into long-format rows for DB storage.

        Label resolution priority (most → least consistent across companies/years):
          1. standard_concept column  ← edgartools normalised name (e.g. "Revenues")
          2. concept column stripped  ← XBRL tag sans namespace  (e.g. "us-gaap:Revenues" → "Revenues")
          3. label column             ← human-readable string    (can change between filings)

        The goal is that the DB stores concept names that are consistent across
        companies and fiscal years, eliminating the fuzzy-matching workaround.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # ── Normalise to a flat wide DataFrame with a "label" column ──────────
        # edgartools to_dataframe() can return either:
        #   A) Wide format with concept/label as a column + date columns
        #   B) Wide format with concept/label as the index

        if "label" not in df.columns:
            # Case B: the index holds the concept names — reset it to a column
            df = df.reset_index()
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "label"})

        # ── Resolve the best available label ──────────────────────────────────
        if "standard_concept" in df.columns:
            # Use standardised concept name where available, fall back to raw label
            mask_has_std = df["standard_concept"].notna() & (
                df["standard_concept"] != ""
            )
            df["label"] = df["standard_concept"].where(mask_has_std, other=df["label"])
            df = df.drop(columns=["standard_concept"])

        if "concept" in df.columns:
            # Strip XBRL namespace prefix (e.g. "us-gaap:Revenues" → "Revenues")
            cleaned = (
                df["concept"]
                .astype(str)
                .str.replace(r"^[a-zA-Z0-9\-]+:", "", regex=True)
            )
            # Override label only where the label still looks like a raw XBRL tag
            raw_tag_mask = df["label"].astype(str).str.contains(":", na=False)
            df.loc[raw_tag_mask, "label"] = cleaned[raw_tag_mask]
            df = df.drop(columns=["concept"])

        # ── Melt from wide to long ─────────────────────────────────────────────
        non_date_cols = [
            c for c in df.columns if c in ("label",) or not _looks_like_date(str(c))
        ]
        date_cols = [c for c in df.columns if c not in non_date_cols]

        if not date_cols:
            logger.warning(
                "[DB] No date columns found in %s %s for %s",
                stmt_type,
                form_type,
                ticker,
            )
            return pd.DataFrame()

        melted = df.melt(
            id_vars=["label"],
            value_vars=date_cols,
            var_name="period_date",
            value_name="value",
        )

        # ── Clean labels ──────────────────────────────────────────────────────
        melted["label"] = (
            melted["label"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(",", "_", regex=False)
            .str.replace("/", "|", regex=False)
            .str.replace(r"[()]", "", regex=True)
        )

        # ── Attach metadata ────────────────────────────────────────────────────
        melted["company"] = ticker.upper()
        melted["form_type"] = form_type
        melted["statement_type"] = stmt_type

        # ── Coerce & filter ────────────────────────────────────────────────────
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted.dropna(subset=["value", "label"])
        melted = melted[melted["label"].str.len() > 0]

        melted["period_date"] = pd.to_datetime(
            melted["period_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        melted = melted.dropna(subset=["period_date"])

        return melted[
            ["company", "period_date", "form_type", "statement_type", "label", "value"]
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # 2.  PERIOD COVERAGE
    # ─────────────────────────────────────────────────────────────────────────

    async def get_covered_periods(self, ticker: str) -> Dict[FormType, Set[Period]]:
        """Returns a dict of already-cached periods per form type."""
        ticker = ticker.upper()
        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(
                "SELECT DISTINCT form_type, period_date FROM financials WHERE company = ?",
                (ticker,),
            ) as cursor:
                rows = await cursor.fetchall()

        covered: Dict[FormType, Set[Period]] = {
            ft: set() for ft in SUPPORTED_FORM_TYPES
        }
        for form_type_str, date_str in rows:
            if form_type_str in covered:
                ts = pd.Timestamp(date_str)
                if form_type_str == "10-K":
                    covered[form_type_str].add(ts.year)
                else:
                    covered[form_type_str].add((ts.year, ts.quarter))
        return covered

    async def find_uncovered_periods(
        self, ticker: str, desired_periods: List[Period], form_type: FormType
    ) -> List[Period]:
        """Returns periods in desired_periods that are not yet in the DB."""
        if not desired_periods:
            return []
        covered = (await self.get_covered_periods(ticker)).get(form_type, set())
        return [p for p in desired_periods if p not in covered]

    # ─────────────────────────────────────────────────────────────────────────
    # 3.  DATA RETRIEVAL
    # ─────────────────────────────────────────────────────────────────────────

    async def get_all_concepts(self, ticker: str) -> List[str]:
        """
        Returns every unique label (concept name) stored for the given ticker.

        NOTE: This is the correct replacement for the broken pattern
              search_label(ticker, []) which always returned an empty DataFrame
              because the keywords guard `if not keywords: return pd.DataFrame()`
              short-circuits on an empty list.
        """
        ticker = ticker.upper()
        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(
                "SELECT DISTINCT label FROM financials WHERE company = ? ORDER BY label ASC",
                (ticker,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_labels(self, ticker: str) -> List[str]:
        """Alias for get_all_concepts — kept for backward compatibility."""
        return await self.get_all_concepts(ticker)

    async def get_data(
        self,
        ticker: str,
        statement_types: List[StatementType] = ALL_STATEMENT_TYPES,
        form_types: Optional[List[FormType]] = None,
    ) -> pd.DataFrame:
        """Retrieves and pivots financial data from the DB with optional filtering."""
        ticker = ticker.upper()
        conditions = [
            "company = ?",
            f"statement_type IN ({','.join(['?'] * len(statement_types))})",
        ]
        params: List[Union[str, int]] = [ticker, *statement_types]

        if form_types:
            conditions.append(f"form_type IN ({','.join(['?'] * len(form_types))})")
            params.extend(form_types)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM financials WHERE {where} ORDER BY period_date DESC"

        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                cols = [d[0] for d in cursor.description]

        if not rows:
            return pd.DataFrame()
        return self.pivot_df(pd.DataFrame(rows, columns=cols))

    async def search_label(
        self,
        ticker: str,
        keywords: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Searches for labels matching one or more keywords (SQL LIKE).

        Parameters
        ----------
        ticker   : stock ticker
        keywords : one keyword string or a list; each is matched with LIKE %keyword%
                   Pass a non-empty list — for listing ALL concepts use get_all_concepts().
        start_date, end_date : optional ISO date strings "YYYY-MM-DD"
        """
        ticker = ticker.upper()

        if isinstance(keywords, str):
            keywords = [keywords]

        if not keywords:
            logger.warning(
                "[DB] search_label called with empty keywords for %s — "
                "use get_all_concepts() to list all labels.",
                ticker,
            )
            return pd.DataFrame()

        if start_date:
            start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        if end_date:
            end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        conditions = ["company = ?"]
        sql_params: List[str] = [ticker]

        label_cond = " OR ".join(["label LIKE ?" for _ in keywords])
        conditions.append(f"({label_cond})")
        sql_params.extend(f"%{k}%" for k in keywords)

        if start_date:
            conditions.append("period_date >= ?")
            sql_params.append(start_date)
        if end_date:
            conditions.append("period_date <= ?")
            sql_params.append(end_date)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM financials WHERE {where} ORDER BY period_date DESC"

        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(query, sql_params) as cursor:
                rows = await cursor.fetchall()
                cols = [d[0] for d in cursor.description]

        if not rows:
            return pd.DataFrame()
        return self.pivot_df(pd.DataFrame(rows, columns=cols))

    # ─────────────────────────────────────────────────────────────────────────
    # 4.  STOCK PRICE DATA  (yfinance)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_price_data(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "yearly",
    ) -> pd.DataFrame:
        """
        Fetches OHLCV price data from Yahoo Finance and resamples as requested.

        Parameters
        ----------
        ticker   : stock ticker symbol
        start    : start date "YYYY-MM-DD" (passed to yfinance)
        end      : end date   "YYYY-MM-DD"
        interval : one of "daily", "monthly", "quarterly", "yearly"

        Returns a DataFrame with a 'stock_price' column (midpoint of High and Low).
        """
        import yfinance as yf

        def _fetch() -> pd.DataFrame:
            freq = interval.lower()
            if freq == "daily":
                base = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
            elif freq == "monthly":
                base = yf.Ticker(ticker).history(start=start, end=end, interval="1mo")
            elif freq in ("quarterly", "yearly"):
                base = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
                rule = "QE" if freq == "quarterly" else "YE"
                base = base.resample(rule).agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
            else:
                raise ValueError(
                    f"interval must be 'daily', 'monthly', 'quarterly', or 'yearly', got '{interval}'"
                )

            base = base.dropna(how="all")
            if not base.empty:
                base["stock_price"] = (base["High"] + base["Low"]) / 2
                base = base.rename(
                    columns={
                        "Open": "stock_price_open",
                        "High": "stock_price_high",
                        "Low": "stock_price_low",
                        "Close": "stock_price_close",
                        "Volume": "stock_price_volume",
                    }
                )
            return base

        return await asyncio.to_thread(_fetch)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.  HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def pivot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots a long-format DB result into wide format:
          rows   = label (metric / concept name)
          columns = period_date
        """
        if df.empty:
            return df
        return df.pivot_table(
            index="label", columns="period_date", values="value", aggfunc="first"
        )

    async def _bulk_insert(self, df: pd.DataFrame) -> None:
        """Bulk INSERT OR REPLACE into the financials table."""
        if df.empty:
            return
        ticker = df["company"].iloc[0]
        logger.info("[DB] Saving %d rows for %s …", len(df), ticker)
        records = list(df.itertuples(index=False, name=None))
        async with aiosqlite.connect(self.db_name) as db:
            await db.executemany(
                """INSERT OR REPLACE INTO financials
                   (company, period_date, form_type, statement_type, label, value)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                records,
            )
            await db.commit()
        logger.info("[DB] Save complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _looks_like_date(s: str) -> bool:
    """Heuristic: does this string look like a period-date column header?"""
    return bool(
        s[:4].isdigit()  # starts with 4-digit year
        or (len(s) >= 7 and s[4] in ("-", "/", "Q"))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (python -m core.agents.financial_db)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio as _asyncio

    async def _smoke():
        db = FinancialDatabase()
        await db.initialize()
        await db.update_financials("MSFT", [2022, 2023], "10-K")
        concepts = await db.get_all_concepts("MSFT")
        print(f"MSFT has {len(concepts)} concepts stored.")
        print(concepts[:20])

    _asyncio.run(_smoke())
