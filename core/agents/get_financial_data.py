import asyncio
from typing import Dict, List, Literal, Optional, Union

import aiosqlite
import pandas as pd
from edgar import Company, MultiFinancials, set_identity

# --- CONFIGURATION ---
USER_AGENT = "FundamentalAnalysisBot yeapzing@utar.edu.my"
set_identity(USER_AGENT)

DB_PATH = "./data/financial_data.db"
ALL_STATEMENT_TYPES = ["income", "balance", "cashflow"]


class FinancialDatabase:
    def __init__(self, db_name: str = DB_PATH):
        self.db_name = db_name

    async def initialize(self):
        """Async initialization of the database schema."""
        async with aiosqlite.connect(self.db_name) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS financials (
                    company TEXT,
                    period_date TEXT,
                    statement_type TEXT,
                    label TEXT,
                    value REAL,
                    PRIMARY KEY (company, period_date, statement_type, label)
                );
            """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_company_date ON financials (company, period_date);"
            )
            await db.commit()

    async def _fetch_edgar_data_sync(
        self,
        ticker: str,
        start_year: int,
        end_year: int,
        form_type: Literal["10-K", "10-Q"] | List[Literal["10-K", "10-Q"]] = "10-K",
    ) -> Dict[str, pd.DataFrame]:
        """
        Runs the blocking edgartools network requests in a separate thread.
        """

        def _fetch():
            company = Company(ticker)
            # Fetch 10-K (Annual) and 10-Q (Quarterly)
            # Adjust 'n' based on how many years of history you need (e.g., 20 filings ≈ 5-10 years)
            filings = company.get_filings(
                form=[form_type], year=list(range(start_year, end_year + 1))
            )

            if not filings:
                return {}

            # Use the efficient MultiFinancials extraction
            multi_financials = MultiFinancials.extract(filings)

            return {
                "income": multi_financials.income_statement().to_dataframe(),
                "balance": multi_financials.balance_sheet().to_dataframe(),
                "cashflow": multi_financials.cashflow_statement().to_dataframe(),
            }

        return await asyncio.to_thread(_fetch)

    def _process_statement(
        self, df: pd.DataFrame, ticker: str, stmt_type: str
    ) -> pd.DataFrame:
        """
        Transforms the pivoted MultiFinancials dataframe into a long-format schema for DB storage.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df.drop(columns=["concept"], inplace=True)

        # The MultiFinancials DF usually has dates as columns and labels as Index.
        # We reset index so 'label' becomes a column.
        melted = df.melt(id_vars=["label"], var_name="period_date", value_name="value")
        melted["label"] = melted["label"].str.replace(" ", "_")

        # Clean up
        melted["company"] = ticker.upper()
        melted["statement_type"] = stmt_type

        # Convert value to numeric and drop invalids
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted.dropna(subset=["value"])

        # Convert period_date to string YYYY-MM-DD ensures consistency
        melted["period_date"] = melted["period_date"].astype(str)

        return melted[["company", "period_date", "statement_type", "label", "value"]]

    async def update_financials(self, ticker: str, start_year: int, end_year: int):
        """
        Fetches data from Edgar and updates the database asynchronously.
        """
        print(f"--- [API] Fetching data for {ticker} ---")

        # 1. Fetch data (Run in thread to avoid blocking)
        try:
            dfs_map = await self._fetch_edgar_data_sync(ticker, start_year, end_year)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return

        if not dfs_map:
            print(f"No financials found for {ticker}")
            return

        # 2. Process DataFrames (CPU bound, but fast enough to keep in main loop or could be threaded)
        all_data = []
        for stmt_type, df in dfs_map.items():
            processed = self._process_statement(df, ticker, stmt_type)
            if not processed.empty:
                all_data.append(processed)

        if not all_data:
            return

        final_df = pd.concat(all_data, ignore_index=True)

        # 3. Bulk Insert into DB
        print(f"[DB] Saving {len(final_df)} rows for {ticker}...")
        records = list(final_df.itertuples(index=False, name=None))

        async with aiosqlite.connect(self.db_name) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO financials (company, period_date, statement_type, label, value)
                VALUES (?, ?, ?, ?, ?)
            """,
                records,
            )
            await db.commit()
        print("[DB] Save complete.")

    async def get_data(
        self,
        ticker: str,
        statement_types: Union[str, List[str]] = ALL_STATEMENT_TYPES,
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieves data from the database.
        """
        ticker = ticker.upper()
        if isinstance(statement_types, str):
            statement_types = [statement_types]

        query = "SELECT * FROM financials WHERE company = ? AND statement_type IN ({})"
        placeholders = ",".join(["?"] * len(statement_types))
        query = query.format(placeholders)

        params = [ticker] + statement_types

        if start_date:
            query += " AND period_date >= ?"
            params.append(start_date)

        query += " ORDER BY period_date DESC"

        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                cols = [description[0] for description in cursor.description]

        if not rows:
            return pd.DataFrame()

        return self.pivot_df(pd.DataFrame(rows, columns=cols))

    def pivot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pivot_table(index="label", columns="period_date", values="value")

    async def search_label(
        self, ticker: str, keywords: Union[str, List[str]]
    ) -> pd.DataFrame:
        """
        Search for specific labels (rows) across the data using one or multiple keywords.
        """
        ticker = ticker.upper()

        # Normalize input to list
        if isinstance(keywords, str):
            keywords = [keywords]

        if not keywords:
            return pd.DataFrame()

        # Dynamically build the OR condition: (label LIKE ? OR label LIKE ?)
        like_conditions = " OR ".join(["label LIKE ?" for _ in keywords])

        query = f"""
            SELECT * FROM financials 
            WHERE company = ? AND ({like_conditions})
            ORDER BY period_date DESC
        """

        # Prepare params: Ticker first, then the keywords wrapped in wildcards
        params = [ticker] + [f"%{k}%" for k in keywords]

        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                cols = [description[0] for description in cursor.description]

        return self.pivot_df(pd.DataFrame(rows, columns=cols))


# --- EXAMPLE USAGE ---
async def main():
    db = FinancialDatabase()
    await db.initialize()

    # Update Data (Fetch from API)
    await db.update_financials("AAPL", 2022, 2024)

    # Retrieve Data
    income_stmt = await db.get_data("AAPL", "income")
    print("\n--- Retrieved Income Statement (Head) ---")
    print(income_stmt.head())


if __name__ == "__main__":
    asyncio.run(main())
