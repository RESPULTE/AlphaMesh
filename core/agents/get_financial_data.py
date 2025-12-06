import sqlite3
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import pandas as pd
from core.agents.concept_resolver import build_concept_resolver
from core.config import settings
from edgar import Company, set_identity

# --- CONFIGURATION ---
USER_AGENT = "FundamentalAnalysisBot yeapzing@utar.edu.my"
ALL_STATEMENT_TYPE = ["income", "balance", "cashflow"]
DEFAULT_YEARS = 5


class FinancialDatabase:
    def __init__(self, db_name: str = "./data/financial_data.db"):
        self.db_name = db_name
        self._cached_existing_years = set()
        self.concept_resolver = {}

        self._init_db()
        set_identity(USER_AGENT)
        settings

    def _get_connection(self):
        return sqlite3.connect(self.db_name)

    def _init_db(self):
        schema_sql = """
        CREATE TABLE IF NOT EXISTS financials (
            company TEXT,
            year INTEGER,
            statement_type TEXT,
            concept TEXT,
            value REAL,
            PRIMARY KEY (company, year, statement_type, concept)
        );
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(schema_sql)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_company_year ON financials (company, year);"
            )
            conn.commit()

    def get_existing_years(self, ticker: str) -> set:
        """Returns a set of years that already exist in the database."""
        # Invalidate cache if needed, but for this agent run we assume single session validity
        if len(self._cached_existing_years) > 0:
            # Note: In a real persistent app, you might want to query this every time
            # or clear cache per ticker. For now, we query DB.
            pass

        query = "SELECT DISTINCT year FROM financials WHERE company = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker,))
            rows = cursor.fetchall()

        return {row[0] for row in rows}

    def _process_dataframe(
        self, df: pd.DataFrame, ticker: str, year: int, stmt_type: str
    ) -> pd.DataFrame:
        """
        Robustly transforms XBRL DataFrame to schema format.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Identify the specific column holding the values
        date_columns = [c for c in df.columns if isinstance(c, str) and c[0].isdigit()]
        if not date_columns:
            date_columns = df.columns.tolist()

        sorted_cols = sorted(date_columns, reverse=True)
        target_col = sorted_cols[0]

        # 2. Handle the Index / Concept Name
        work_df = df.copy()
        if isinstance(work_df.index, pd.RangeIndex) or work_df.index.name is None:
            potential_label_cols = ["Label", "concept", "Abstract", "Item"]
            for col in potential_label_cols:
                if col in work_df.columns:
                    work_df.set_index(col, inplace=True)
                    break

        # 3. Extract and Clean
        try:
            subset = work_df[[target_col]].copy().reset_index()
            concept_col_name = subset.columns[0]
            value_col_name = subset.columns[1]

            subset.rename(
                columns={concept_col_name: "concept", value_col_name: "value"},
                inplace=True,
            )
            subset["concept"] = subset["concept"].str.removeprefix("us-gaap_")
        except KeyError:
            return pd.DataFrame()

        subset = subset.dropna(subset=["value"])
        subset["company"] = ticker
        subset["year"] = year
        subset["statement_type"] = stmt_type
        subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
        subset = subset.dropna(subset=["value"])

        return subset[["company", "year", "statement_type", "concept", "value"]]

    def fetch_new_filings(
        self, ticker: str, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """
        Fetches data from Edgar API but DOES NOT save to DB.
        Returns the data as a Pandas DataFrame.
        """
        ticker = ticker.upper()
        print(f"--- [API] Fetching {ticker} ({start_year} to {end_year}) ---")

        # Check what we already have to avoid re-fetching inside this logic
        # (Though the Agent usually handles the logic of what year to ask for)
        existing_years = self.get_existing_years(ticker)

        try:
            start_dt = datetime.strptime(f"{start_year}-01-01", "%Y-%m-%d")
            end_dt = datetime.strptime(f"{end_year}-12-31", "%Y-%m-%d")
        except ValueError as e:
            print(f"Date format error: {e}")
            return pd.DataFrame()

        try:
            company = Company(ticker)
            all_filings = company.get_filings(form="10-K")
        except Exception as e:
            print(f"Error fetching filings list for {ticker}: {e}")
            return pd.DataFrame()

        if not all_filings:
            return pd.DataFrame()

        # Filter filings
        target_filings = []
        for filing in all_filings:
            f_date = datetime.strptime(str(filing.filing_date), "%Y-%m-%d")
            if start_dt <= f_date <= end_dt:
                target_filings.append(filing)

        if not target_filings:
            print(f"No new filings found between {start_year} and {end_year}.")
            return pd.DataFrame()

        collected_dfs = []

        for filing in target_filings:
            try:
                filing_date = datetime.strptime(str(filing.filing_date), "%Y-%m-%d")
                estimated_year = (
                    filing_date.year - 1 if filing_date.month < 6 else filing_date.year
                )

                if estimated_year in existing_years:
                    continue

                print(f"   -> Processing filing for Fiscal Year {estimated_year}...")
                xbrl = filing.xbrl()
                if not xbrl:
                    continue

                # Process Statements
                stats = xbrl.statements

                # Helper to append to list
                def _append_stmt(stmt, s_type):
                    if stmt:
                        clean = self._process_dataframe(
                            stmt.to_dataframe(), ticker, estimated_year, s_type
                        )
                        if not clean.empty:
                            collected_dfs.append(clean)

                _append_stmt(stats.income_statement(), "income")
                _append_stmt(stats.balance_sheet(), "balance")
                _append_stmt(stats.cashflow_statement(), "cashflow")

            except Exception as e:
                print(f"   -> Failed to process filing: {e}")

        if not collected_dfs:
            return pd.DataFrame()

        # Combine all new data
        final_new_df = pd.concat(collected_dfs, ignore_index=True)

        # Deduplicate
        final_new_df.drop_duplicates(
            subset=["company", "year", "statement_type", "concept"],
            keep="first",
            inplace=True,
        )

        return final_new_df

    def bulk_insert_financials(self, df: pd.DataFrame):
        """
        Performs a single bulk INSERT OR REPLACE into the database.
        """
        if df is None or df.empty:
            return

        print(f"[DB] Bulk saving {len(df)} rows...")

        # Ensure correct columns
        expected_cols = ["company", "year", "statement_type", "concept", "value"]
        save_df = df[expected_cols].copy()

        data_tuples = list(save_df.itertuples(index=False, name=None))

        query = """
            INSERT OR REPLACE INTO financials (company, year, statement_type, concept, value)
            VALUES (?, ?, ?, ?, ?)
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, data_tuples)
                conn.commit()
            print("[DB] Save complete.")
        except Exception as e:
            print(f"[DB] Error during bulk insert: {e}")

    def get_data(
        self,
        ticker: str,
        years: Optional[List[int]] = DEFAULT_YEARS,
        statements: Union[str, List[str]] = ALL_STATEMENT_TYPE,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        query = "SELECT * FROM financials WHERE company = ?"
        params = [ticker]

        if years is None:
            years = [DEFAULT_YEARS]
        if isinstance(years, int):
            years = [years]

        placeholders = ",".join(["?"] * len(years))
        query += f" AND year IN ({placeholders})"
        params.extend(years)

        if statements is None:
            statements = ALL_STATEMENT_TYPE
        if isinstance(statements, str):
            statements = [statements]

        placeholders = ",".join(["?"] * len(statements))
        query += f" AND statement_type IN ({placeholders})"
        params.extend(statements)

        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)

        return df

    def pivot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.pivot_table(
            index=["company", "concept"], columns="year", values="value"
        )

    def resolve_concept(self, ticker: str, concept: str) -> str | None:
        if ticker not in self.concept_resolver:
            all_concepts = self.get_all_concepts_for_company(ticker)
            self.concept_resolver[ticker] = build_concept_resolver(all_concepts)
        return self.concept_resolver[ticker](concept)

    @lru_cache(maxsize=100)
    def get_all_concepts_for_company(self, ticker: str) -> List[str]:
        ticker = ticker.upper()
        query = "SELECT DISTINCT concept FROM financials WHERE company = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker,))
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def get_concept(
        self,
        ticker: str,
        keyword: str | Tuple[str],
        start_year: int | None = None,
        end_year: int | None = None,
        exact: bool = False,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        if isinstance(keyword, Iterable) and not isinstance(keyword, str):
            keywords = list(keyword)
        else:
            keywords = [keyword]

        if exact:
            clause = " OR ".join(["concept = ?" for _ in keywords])
            params = keywords[:]
        else:
            clause = " OR ".join(["concept LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords]

        query = f"SELECT * FROM financials WHERE company = ? AND ({clause})"
        params = [ticker] + params

        if start_year is not None:
            query += " AND year >= ?"
            params.append(start_year)
        if end_year is not None:
            query += " AND year <= ?"
            params.append(end_year)

        query += " ORDER BY year ASC"

        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)

        return self.pivot_data(df)
