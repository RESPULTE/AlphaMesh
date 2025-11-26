import sqlite3
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import pandas as pd
from core.agents.concept_resolver import build_concept_resolver
from edgar import Company, set_identity

# --- CONFIGURATION ---
USER_AGENT = "FinancialResearchBot student@university.edu"
ALL_STATEMENT_TYPE = ["income", "balance", "cashflow"]
DEFAULT_YEARS = 5


class FinancialDatabase:
    def __init__(self, db_name: str = "./data/financial_data.db"):
        self.db_name = db_name
        self._cached_existing_years = set()
        self.concept_resolver = {}

        self._init_db()
        set_identity(USER_AGENT)

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

    @lru_cache(maxsize=100)
    def get_existing_years(self, ticker: str) -> set:
        if len(self._cached_existing_years) > 0:
            return self._cached_existing_years

        query = "SELECT DISTINCT year FROM financials WHERE company = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker,))
            rows = cursor.fetchall()

        self._cached_existing_years.update({row[0] for row in rows})
        return self._cached_existing_years.copy()

    def _process_dataframe(
        self, df: pd.DataFrame, ticker: str, year: int, stmt_type: str
    ) -> pd.DataFrame:
        """
        Robustly transforms XBRL DataFrame to schema format.
        Fixes issues with numeric indices and column mapping.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Identify the specific column holding the values for the filing date.
        # XBRL dataframes usually have dates as columns (e.g., '2023-12-31').
        # We sort to find the most recent date (the current reporting period).
        date_columns = [c for c in df.columns if isinstance(c, str) and c[0].isdigit()]

        if not date_columns:
            # Fallback: if no date columns found, check if all columns are relevant
            # (Sometimes edgar returns columns as objects, we take the first non-metadata one)
            date_columns = df.columns.tolist()

        # Sort dates descending (newest first)
        sorted_cols = sorted(date_columns, reverse=True)
        target_col = sorted_cols[0]  # This is the column with the financial values

        # 2. Handle the Index / Concept Name
        # Create a working copy to avoid SettingWithCopy warnings
        work_df = df.copy()

        # If the index is just numbers (0,1,2...), the concept name is likely inside a column
        # named 'Label', 'item', or similar. We need to move it to the index.
        if isinstance(work_df.index, pd.RangeIndex) or work_df.index.name is None:
            # Common column names used by edgar for the labels
            potential_label_cols = ["Label", "concept", "Abstract", "Item"]
            for col in potential_label_cols:
                if col in work_df.columns:
                    work_df.set_index(col, inplace=True)
                    break

        # 3. Extract only the target value column
        # We now have: Index = Concept Name, Column = Target Date
        try:
            # Slice strictly the target column
            subset = work_df[[target_col]].copy()

            # Reset index to turn the Concept (Index) into a Column
            subset = subset.reset_index()

            # 4. Explicitly Rename Columns
            # The first column is now the Concept (whatever the index name was)
            # The second column is the Value (named after target_col)
            concept_col_name = subset.columns[0]
            value_col_name = subset.columns[1]

            subset.rename(
                columns={concept_col_name: "concept", value_col_name: "value"},
                inplace=True,
            )
            subset["concept"] = subset["concept"].str.removeprefix("us-gaap_")

        except KeyError:
            print(f"   Warning: Could not map columns for {stmt_type}")
            return pd.DataFrame()

        # 5. Clean Data
        # Remove rows where concept is likely just a section header (often value is NaN or empty)
        subset = subset.dropna(subset=["value"])

        # Add Schema Metadata
        subset["company"] = ticker
        subset["year"] = year
        subset["statement_type"] = stmt_type

        # Ensure Value is Numeric
        subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
        subset = subset.dropna(
            subset=["value"]
        )  # Drop lines that couldn't be converted

        # Reorder
        final_df = subset[["company", "year", "statement_type", "concept", "value"]]

        return final_df

    @lru_cache(maxsize=100)
    def get_all_concepts_for_company(self, ticker: str) -> List[str]:
        ticker = ticker.upper()
        query = "SELECT DISTINCT concept FROM financials WHERE company = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker,))
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def update_company_data(self, ticker: str, num_years: int = 5):
        ticker = ticker.upper()
        print(f"--- Processing {ticker} ---")

        existing_years = self.get_existing_years(ticker)

        # Check if we already have enough recent data to satisfy the request.
        # We look back (num_years + 1) to account for fiscal years ending in the previous calendar year.
        current_year = datetime.now().year
        cutoff_year = current_year - (num_years + 1)

        # Count how many years in the DB are recent enough
        recent_years_count = sum(1 for y in existing_years if y > cutoff_year)

        if recent_years_count >= num_years:
            print(
                f"Skipping fetch for {ticker}: Found {recent_years_count} recent years in DB (Requested: {num_years})."
            )
            return

        try:
            company = Company(ticker)
            # Fetch slightly more than needed to ensure we cover fiscal year offsets
            filings = company.get_filings(form="10-K").latest(num_years)
        except Exception as e:
            print(f"Error fetching filings for {ticker}: {e}")
            return

        if not filings:
            print(f"No 10-K filings found for {ticker}")
            return

        processed_count = 0

        for filing in filings:
            # Stop if we have processed enough NEW years or total years
            if processed_count >= num_years:
                break

            try:
                filing_date = datetime.strptime(str(filing.filing_date), "%Y-%m-%d")
                # Estimate Fiscal Year (approximate)
                estimated_year = (
                    filing_date.year - 1 if filing_date.month < 6 else filing_date.year
                )

                if estimated_year in existing_years:
                    # Even inside the loop, we check.
                    # If we hit a year we have, we don't download/process,
                    # but we continue checking older filings just in case there's a gap,
                    # unless we are confident we have the sequence.
                    print(f"Skipping {estimated_year} (Already exists)")

                    # Optimization: If we hit a year that exists, and we know we have
                    # a continuous block of older data, we could technically break here too.
                    # For now, we just skip processing this specific filing.
                    continue

                print(
                    f"Fetching data for {estimated_year} (Filing: {filing.filing_date})..."
                )

                xbrl = filing.xbrl()
                if not xbrl:
                    print(f"No XBRL data found.")
                    continue

                data_frames = []

                # Helper to safely add statements
                def add_statement(stmt_obj, s_type):
                    if stmt_obj:
                        df = stmt_obj.to_dataframe()
                        clean = self._process_dataframe(
                            df, ticker, estimated_year, s_type
                        )
                        if not clean.empty:
                            data_frames.append(clean)

                stats = xbrl.statements
                add_statement(stats.income_statement(), "income")
                add_statement(stats.balance_sheet(), "balance")
                add_statement(stats.cashflow_statement(), "cashflow")

                if data_frames:
                    combined_df = pd.concat(data_frames, ignore_index=True)

                    # Remove duplicates within the same filing (sometimes same tag appears twice)
                    combined_df.drop_duplicates(
                        subset=["company", "year", "statement_type", "concept"],
                        keep="first",
                        inplace=True,
                    )

                    with self._get_connection() as conn:
                        combined_df.to_sql(
                            "financials", conn, if_exists="append", index=False
                        )

                    print(f"  Saved {len(combined_df)} rows.")
                    existing_years.add(estimated_year)
                    processed_count += 1
                else:
                    print(f"No processable data found.")

            except Exception as e:
                print(f"  Failed to process filing: {e}")

        # self._cached_existing_years.update(list(range(current_year - 10, current_year + 1)))

    @lru_cache(maxsize=100)
    def get_data(
        self,
        ticker: str,
        years: Optional[List[int]] = [DEFAULT_YEARS],
        statements: Union[str, List[str]] = ALL_STATEMENT_TYPE,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        query = "SELECT * FROM financials WHERE company = ?"
        params = [ticker]

        # Handle Years (Single Int or List of Ints)
        if years is None:
            years = [DEFAULT_YEARS]

        if isinstance(years, int):
            years = [years]
        placeholders = ",".join(["?"] * len(years))
        query += f" AND year IN ({placeholders})"
        params.extend(years)

        # Handle Statements (Single Str or List of Strs)
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
    def get_concept(
        self,
        ticker: str,
        keyword: str | Tuple[str],
        start_year: int | None = None,
        end_year: int | None = None,
        exact: bool = False,
    ) -> pd.DataFrame:
        """
        Search for specific line items (e.g., 'Revenue', 'Net Income')
        across a range of years (optional). Supports multiple keywords.
        """

        ticker = ticker.upper()

        # --- Normalize keyword(s) ---
        if isinstance(keyword, Iterable) and not isinstance(keyword, str):
            keywords = list(keyword)
        else:
            keywords = [keyword]

        # --- Build where clause ---
        if exact:
            # Exact match (concept = ?)
            clause = " OR ".join(["concept = ?" for _ in keywords])
            params = keywords[:]  # no wildcards
        else:
            # Partial match (concept LIKE %kw%)
            clause = " OR ".join(["concept LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords]

        # --- Build query ---
        query = f"""
            SELECT * FROM financials
            WHERE company = ?
            AND ({clause})
        """

        params = [ticker] + params

        if start_year is not None:
            query += " AND year >= ?"
            params.append(start_year)

        if end_year is not None:
            query += " AND year <= ?"
            params.append(end_year)

        query += " ORDER BY year ASC"

        # --- Execute ---
        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)

        return self.pivot_data(df)

    def save_calculated_metric(self, ticker: str, df: pd.DataFrame):
        """
        Persists calculated metrics into the database.
        Transforms the 'wide' DataFrame (Years as columns) back into the 'long' database format.
        """

        if df is None or df.empty:
            return

        ticker = ticker.upper()
        print(f"[DB] Saving calculated metrics for {ticker}...")

        # 1. Reset index so 'concept' becomes a column we can manipulate
        # Current: Index=Concept, Columns=Years
        work_df = df.copy().reset_index()

        # 2. Melt (Unpivot) the DataFrame
        # Converts columns [2022, 2023, ...] into rows under a 'year' column
        # id_vars='concept' keeps the concept name for every row
        melted_df = work_df.melt(
            id_vars=["company", "concept"], var_name="year", value_name="value"
        )

        # Drop rows with NaN values or invalid years
        melted_df.dropna(subset=["value", "year"], inplace=True)

        # Add Schema Columns
        melted_df["company"] = ticker
        # We tag these as 'calculated' to distinguish from raw XBRL 'income'/'balance' data
        melted_df["statement_type"] = "calculated"

        # 4. Prepare for SQL Insertion
        # Select specific columns in the order expected by the DB schema
        # Schema: company, year, statement_type, concept, value
        final_df = melted_df[["company", "year", "statement_type", "concept", "value"]]

        final_df = final_df[
            ~final_df["concept"].isin(self.get_all_concepts_for_company(ticker))
        ]

        # Convert to list of tuples for bulk insertion
        data_tuples = list(final_df.itertuples(index=False, name=None))

        if not data_tuples:
            print("[DB] No valid data to save after cleaning.")
            return

        # 5. Execute Upsert (INSERT OR REPLACE)
        # This ensures if we recalculate a metric, we update the old value instead of crashing
        upsert_sql = """
            INSERT OR REPLACE INTO financials (company, year, statement_type, concept, value)
            VALUES (?, ?, ?, ?, ?)
            """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(upsert_sql, data_tuples)
                conn.commit()
            print(f"[DB] Successfully saved {len(data_tuples)} calculated data points.")

            # Clear cache so subsequent reads pick up the new data
            self.get_data.cache_clear()
            self.get_concept.cache_clear()

        except Exception as e:
            print(f"[DB] Error saving calculated metrics: {e}")


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    db = FinancialDatabase("financial_data.db")
    company = "MSFT"

    # # Ensure data exists
    # db.update_company_data(company, num_years=5)

    # print("\n--- 2. Filter by Year (2023 Only) ---")
    # # Using the intuitive wrapper
    # print(db.get_fiscal_year(company, 2023))

    # print("\n--- 3. Complex Filter (Cashflow & Balance for 2022 & 2023) ---")
    # # Using the robust get_data method
    # df_complex = db.get_data(
    #     company, years=[2022, 2023], statements=["cashflow", "balance"]
    # )
    # print(db.pivot_data(df_complex))

    # print("\n--- 4. Search for a Concept (e.g., 'Assets') ---")
    print(
        db.get_concept(
            company,
            ["us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax"],
            start_year=2021,
            end_year=2024,
        )
    )

    # import json

    # print(
    #     json.dumps(
    #         db.get_data(company, 2024, ALL_STATEMENT_TYPE)["concept"].tolist(), indent=2
    #     )
    # )
