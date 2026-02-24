import asyncio
from typing import Dict, List, Literal, Optional, Set, Tuple, TypeAlias, Union

import aiosqlite
import pandas as pd
from edgar import Company, MultiFinancials, set_identity
from core.logger import get_logger

logger = get_logger(__name__)

# --- TYPE DEFINITIONS AND CONFIGURATION ---

# Define clear type aliases for better readability and maintainability
FormType: TypeAlias = Literal["10-K", "10-Q"]
StatementType: TypeAlias = Literal["income", "balance", "cashflow"]
# A Period can be a year (for 10-K) or a (year, quarter) tuple (for 10-Q)
Period: TypeAlias = Union[int, Tuple[int, int]]

USER_AGENT = "FundamentalAnalysisBot yeapzing@utar.edu.my"
set_identity(USER_AGENT)

DB_PATH = "./data/financial_data.db"
ALL_STATEMENT_TYPES: List[StatementType] = ["income", "balance", "cashflow"]
SUPPORTED_FORM_TYPES: List[FormType] = ["10-K", "10-Q"]


class FinancialDatabase:
    """
    Manages fetching, storing, and retrieving financial data from EDGAR filings.
    """

    def __init__(self, db_name: str = DB_PATH):
        self.db_name = db_name

    async def initialize(self):
        """
        Initializes the database, creating the table and indexes if they don't exist.
        The schema now includes the 'form_type' to distinguish annual vs. quarterly data.
        """
        async with aiosqlite.connect(self.db_name) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS financials (
                    company TEXT,
                    period_date TEXT,
                    form_type TEXT,
                    statement_type TEXT,
                    label TEXT,
                    value REAL,
                    PRIMARY KEY (company, period_date, form_type, statement_type, label)
                );
                """
            )
            # Add indexes for common query patterns
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_company_form_date ON financials (company, form_type, period_date);"
            )
            await db.commit()
            logger.info("[DB] Database initialized successfully.")

    # --- 1. CORE DATA FETCHING AND PROCESSING ---

    async def update_financials(
        self, ticker: str, period: List[Period], form_type: FormType
    ) -> None:
        uncoverd_periods = await self.find_uncovered_periods(ticker, period, form_type)
        if len(uncoverd_periods) == 0:
            logger.info(f"All periods for {ticker} are already covered.")
            return

        logger.info(f"Uncovered periods: {uncoverd_periods}")

        await self.fetch_and_store_period(ticker, uncoverd_periods, form_type)

    async def fetch_and_store_period(
        self, ticker: str, periods: List[Period], form_type: FormType
    ) -> None:
        """
        High-level function to fetch, process, and store data for a list of specified periods.

        Args:
            ticker: The stock ticker (e.g., "AAPL").
            periods: A list of periods to fetch, e.g., [2022, 2021] for 10-K or
                     [(2023, 1), (2023, 2)] for 10-Q.
            form_type: The type of form to fetch ('10-K' or '10-Q').
        """
        if not periods:
            logger.warning("No periods specified to fetch.")
            return

        # Extract all unique years required for the API call
        years = sorted(list(set(p if isinstance(p, int) else p[0] for p in periods)))

        logger.info(f"--- [API] Fetching {form_type} data for {ticker} for years {years} ---")

        # 1. Fetch data from EDGAR in a separate thread
        try:
            dfs_map = await self._fetch_edgar_data_sync(ticker, years, form_type)
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return

        if not any(not df.empty for df in dfs_map.values()):
            logger.warning(f"No financials found for {ticker} in years {years}")
            return

        # 2. Process and combine the dataframes
        all_data = []
        for stmt_type, df in dfs_map.items():
            processed = self._process_statement(df, ticker, stmt_type, form_type)
            if not processed.empty:
                # --- START: REGENERATED FILTERING LOGIC ---

                # Convert date column to datetime objects for filtering
                processed["period_date_dt"] = pd.to_datetime(processed["period_date"])

                # If it's a quarterly request, filter for the correct quarters.
                if form_type == "10-Q":
                    # Create a Series of (year, quarter) tuples from the processed data
                    actual_periods = processed.apply(
                        lambda row: (
                            row["period_date_dt"].year,
                            row["period_date_dt"].quarter,
                        ),
                        axis=1,
                    )
                    # Create a boolean mask where the actual period is in our desired list
                    mask = actual_periods.isin(periods)
                    processed = processed[mask]

                # If it's an annual request, filter for the correct years.
                elif form_type == "10-K":
                    mask = processed["period_date_dt"].dt.year.isin(periods)
                    processed = processed[mask]

                # Drop the temporary datetime column
                processed = processed.drop(columns=["period_date_dt"])

                # --- END: REGENERATED FILTERING LOGIC ---

                if not processed.empty:
                    all_data.append(processed)

        if not all_data:
            logger.warning(f"No data found for the specific periods {periods} after filtering.")
            return

        final_df = pd.concat(all_data, ignore_index=True)

        # 3. Bulk insert into the database
        await self._bulk_insert(final_df)

    async def _fetch_edgar_data_sync(
        self,
        ticker: str,
        year: int,
        form_type: FormType,
    ) -> Dict[StatementType, pd.DataFrame]:
        """
        Runs the blocking edgar network requests in a separate thread.
        """

        def _fetch():
            company = Company(ticker)
            filings = company.get_filings(form=form_type, year=year)
            if not filings:
                return {}

            multi_financials = MultiFinancials.extract(filings)
            return {
                "income": multi_financials.income_statement().to_dataframe(),
                "balance": multi_financials.balance_sheet().to_dataframe(),
                "cashflow": multi_financials.cashflow_statement().to_dataframe(),
            }

        # Use type assertion for clarity
        loop = asyncio.get_event_loop()
        result: Dict[StatementType, pd.DataFrame] = await loop.run_in_executor(
            None, _fetch
        )
        return result

    def _process_statement(
        self,
        df: pd.DataFrame,
        ticker: str,
        stmt_type: StatementType,
        form_type: FormType,
    ) -> pd.DataFrame:
        """Transforms a raw dataframe from EDGAR into a long-format for DB storage."""
        if df is None or df.empty:
            return pd.DataFrame()

        # The 'concept' column is redundant if 'label' is the index
        if "concept" in df.columns:
            df = df.drop(columns=["concept"])

        melted = df.melt(id_vars=["label"], var_name="period_date", value_name="value")

        # Clean up label for consistency
        melted["label"] = (
            melted["label"]
            .str.replace(" ", "_")
            .str.replace(",", "_")
            .str.replace("/", "|")
            .str.replace("(", "")
            .str.replace(")", "")
        )

        # Add metadata columns
        melted["company"] = ticker.upper()
        melted["form_type"] = form_type
        melted["statement_type"] = stmt_type

        # Coerce value to numeric and drop rows with invalid data
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted.dropna(subset=["value"])

        # Ensure date is in a consistent string format
        melted["period_date"] = pd.to_datetime(melted["period_date"]).dt.strftime(
            "%Y-%m-%d"
        )

        return melted[
            ["company", "period_date", "form_type", "statement_type", "label", "value"]
        ]

    async def _bulk_insert(self, df: pd.DataFrame):
        """Performs a bulk INSERT OR REPLACE into the database."""
        if df.empty:
            return

        logger.info(f"[DB] Saving {len(df)} rows for {df['company'].iloc[0]}...")
        records = list(df.itertuples(index=False, name=None))

        async with aiosqlite.connect(self.db_name) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO financials (company, period_date, form_type, statement_type, label, value)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            await db.commit()
        logger.info("[DB] Save complete.")

    # --- 2. PERIOD COVERAGE CHECKS ---

    async def get_covered_periods(self, ticker: str) -> Dict[FormType, Set[Period]]:
        """
        Scans the database to find all periods for which data is already stored.

        Returns:
            A dictionary mapping form type to a set of covered periods.
            Example: {'10-K': {2022, 2021}, '10-Q': {(2023, 1), (2023, 2)}}
        """
        ticker = ticker.upper()
        query = (
            "SELECT DISTINCT form_type, period_date FROM financials WHERE company = ?"
        )

        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(query, (ticker,)) as cursor:
                rows = await cursor.fetchall()

        covered: Dict[FormType, Set[Period]] = {
            ft: set() for ft in SUPPORTED_FORM_TYPES
        }
        if not rows:
            return covered

        for form_type_str, date_str in rows:
            form_type = form_type_str  # Assuming it's already of FormType
            if form_type in covered:
                ts = pd.Timestamp(date_str)
                if form_type == "10-K":
                    covered[form_type].add(ts.year)
                elif form_type == "10-Q":
                    covered[form_type].add((ts.year, ts.quarter))

        return covered

    async def find_uncovered_periods(
        self, ticker: str, desired_periods: List[Period], form_type: FormType
    ) -> List[Period]:
        """
        Compares a list of desired periods against the database to find what's missing.
        """
        if not desired_periods:
            return []

        covered_periods = await self.get_covered_periods(ticker)
        covered = covered_periods.get(form_type, set())
        uncovered = [p for p in desired_periods if p not in covered]

        return uncovered

    # --- 3. DATA RETRIEVAL AND QUERYING ---

    async def get_data(
        self,
        ticker: str,
        statement_types: List[StatementType] = ALL_STATEMENT_TYPES,
        form_types: Optional[List[FormType]] = None,
    ) -> pd.DataFrame:
        """
        Retrieves and pivots financial data from the database, with optional form_type filtering.
        """
        ticker = ticker.upper()

        conditions = [
            "company = ?",
            f"statement_type IN ({','.join(['?']*len(statement_types))})",
        ]
        params: List[Union[str, int]] = [ticker] + statement_types

        if form_types:
            conditions.append(f"form_type IN ({','.join(['?']*len(form_types))})")
            params.extend(form_types)

        query = f"""
            SELECT * FROM financials
            WHERE {" AND ".join(conditions)}
            ORDER BY period_date DESC
        """

        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                cols = [desc[0] for desc in cursor.description]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=cols)
        return df.pivot_table(
            index="label", columns=["period_date", "form_type"], values="value"
        )

    async def get_price_data(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        interval: Literal["daily", "monthly", "quarterly", "yearly"] = "daily",
    ) -> pd.DataFrame:
        """
        Get OHLCV price data for a ticker at daily, monthly, quarterly, or yearly frequency.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol, e.g. "AAPL"
        start : str, optional
            Start date in "YYYY-MM-DD" format. If None, Yahoo default is used.
        end : str, optional
            End date in "YYYY-MM-DD" format. If None, Yahoo default is used.
        interval : str
            One of: "daily", "monthly", "quarterly", "yearly"

        Returns
        -------
        pd.DataFrame
            Resampled OHLCV price data.
        """

        import yfinance as yf

        def _data_fetcher():
            # Normalize interval string
            preiod = interval.lower()

            # Yahoo supports daily and monthly directly
            if preiod == "daily":
                base = yf.Ticker(ticker).history(start=start, end=end, interval="1d")

            elif preiod == "monthly":
                base = yf.Ticker(ticker).history(start=start, end=end, interval="1mo")

            # For quarterly and yearly, download daily then resample
            elif preiod in ("quarterly", "yearly"):
                base = yf.Ticker(ticker).history(start=start, end=end, interval="1d")

                rule = "Q" if preiod == "quarterly" else "Y"

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
                    "interval must be: 'daily', 'monthly', 'quarterly', or 'yearly'"
                )
            base = base.dropna(how="all")
            if not base.empty:
                base["stock_price"] = (base["High"] + base["Low"]) / 2
                rename_mapping = {
                    "Open": "stock_price_open",
                    "High": "stock_price_high",
                    "Low": "stock_price_low",
                    "Close": "stock_price_close",
                    "Volume": "stock_price_volume",
                }
                base = base.rename(columns=rename_mapping)

            # Drop empty rows (occurs if date range is too tight)
            return base

        df = await asyncio.to_thread(_data_fetcher)
        return df

    def pivot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pivot_table(index="label", columns="period_date", values="value")

    async def get_labels(self, ticker: str) -> List[str]:
        """
        Efficiently retrieves all unique labels available for a specific ticker
        using a distinct SQL query.
        """
        ticker = ticker.upper()
        query = (
            "SELECT DISTINCT label FROM financials WHERE company = ? ORDER BY label ASC"
        )

        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(query, (ticker,)) as cursor:
                rows = await cursor.fetchall()

        # Flatten list of tuples: [('Label A',), ('Label B',)] -> ['Label A', 'Label B']
        return [row[0] for row in rows]

    async def search_label(
        self,
        ticker: str,
        keywords: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Search for specific labels (rows) across the data using one or multiple keywords,
        with optional date filtering.
        """
        ticker = ticker.upper()

        if start_date:
            start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")

        if end_date:
            end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        # Normalize keywords into a list
        if isinstance(keywords, str):
            keywords = [keywords]

        if not keywords:
            return pd.DataFrame()

        conditions = ["company = ?"]
        params = [ticker]

        label_conditions = " OR ".join(["label LIKE ?" for _ in keywords])
        conditions.append(f"({label_conditions})")
        params.extend([f"%{k}%" for k in keywords])

        # Date filters
        if start_date:
            conditions.append("period_date >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("period_date <= ?")
            params.append(end_date)

        # Final SQL
        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT * FROM financials
            WHERE {where_clause}
            ORDER BY period_date DESC
        """

        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                cols = [description[0] for description in cursor.description]

        if not rows:
            return pd.DataFrame()

        return self.pivot_df(pd.DataFrame(rows, columns=cols))


# --- EXAMPLE USAGE: SMART UPDATE WORKFLOW ---


async def smart_update_workflow():
    """Demonstrates the efficient workflow for updating the database."""
    db = FinancialDatabase()
    await db.initialize()

    TICKER = "MSFT"

    # 1. Define all the periods we want to ensure we have locally.
    desired_periods: List[Period] = [
        (2023, 1),  # Q1 2023 report
        (2023, 2),  # Q2 2023 report
        (2022, 4),  # Q4 2022 report
    ]
    logger.info(f"--- Starting Smart Update for {TICKER} ---")
    logger.info(f"Desired periods: {desired_periods}")

    await db.update_financials(TICKER, desired_periods, "10-Q")

    # 4. Demonstrate retrieving the data with filtering
    logger.info("\n--- Retrieving 2022 & 2021 Annual (10-K) Data Only ---")
    annual_data = await db.get_data(TICKER, form_types=["10-K"])
    logger.info(annual_data.head())

    logger.info("\n--- Retrieving 2023 Quarterly (10-Q) Data Only ---")
    quarterly_data = await db.get_data(TICKER, form_types=["10-Q"])
    logger.info(quarterly_data.head())


if __name__ == "__main__":
    asyncio.run(smart_update_workflow())
