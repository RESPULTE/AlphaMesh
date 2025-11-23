from collections.abc import Iterable
import sqlite3
import pandas as pd
from edgar import Company, set_identity
from datetime import datetime
from typing import List, Optional, Tuple, Union
from functools import lru_cache

from core.agents.concept_resolver import build_concept_resolver

# --- CONFIGURATION ---
USER_AGENT = "FinancialResearchBot student@university.edu"
ALL_STATEMENT_TYPE = ["income", "balance", "cashflow"]
DEFAULT_YEARS = 5


class FinancialDatabase:
    def __init__(self, db_name: str = "financial_data.db"):
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
        cutoff_year = current_year - (num_years + 2)

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
        return df.pivot_table(index=["concept"], columns="year", values="value")

    def get_fiscal_year(self, ticker: str, year: int) -> pd.DataFrame:
        """Fetch all data (all statements) for a specific year."""
        return self.pivot_data(self.get_data(ticker, years=year))

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
    ) -> pd.DataFrame:
        """
        Search for specific line items (e.g., 'Revenue', 'Net Income')
        across a range of years (optional). Supports multiple keywords.
        """

        ticker = ticker.upper()

        # --- Handle keyword(s) ---
        if isinstance(keyword, Iterable):
            like_clauses = " OR ".join(["concept LIKE ?" for _ in keyword])
            search_terms = [f"%{kw}%" for kw in keyword]
        else:
            like_clauses = "concept LIKE ?"
            search_terms = [f"%{keyword}%"]

        # --- Build query ---
        query = f"""
            SELECT * FROM financials
            WHERE company = ?
            AND ({like_clauses})
        """

        params = [ticker] + search_terms

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

    def save_calculated_metric(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df["statement_type"] = "calculated"
        required = ["company", "year", "statement_type", "concept", "value"]

        if not all(c in df.columns for c in required):
            print(f"   [DB Error] Missing cols: {df.columns}")
            return

        final_df = df[required]
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                for _, row in final_df.iterrows():
                    cursor.execute(
                        "DELETE FROM financials WHERE company=? AND year=? AND statement_type='calculated' AND concept=?",
                        (row["company"], row["year"], row["concept"]),
                    )
                conn.commit()
                final_df.to_sql("financials", conn, if_exists="append", index=False)
                print(
                    f"   [DB] Saved {len(final_df)} rows for '{final_df['concept'].iloc[0]}'."
                )
            except Exception as e:
                print(f"   [DB] Error saving: {e}")


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
