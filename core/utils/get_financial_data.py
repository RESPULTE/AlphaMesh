import sqlite3
import pandas as pd
from edgar import Company, set_identity
from datetime import datetime
from typing import List, Optional, Union

# --- CONFIGURATION ---
USER_AGENT = "FinancialResearchBot student@university.edu"


class FinancialDatabase:
    def __init__(self, db_name: str = "financial_data.db"):
        self.db_name = db_name
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

    def _get_existing_years(self, ticker: str) -> set:
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

    def update_company_data(self, ticker: str, num_years: int = 5):
        ticker = ticker.upper()
        print(f"--- Processing {ticker} ---")

        existing_years = self._get_existing_years(ticker)

        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-K").latest(num_years + 3)
        except Exception as e:
            print(f"Error fetching filings for {ticker}: {e}")
            return

        if not filings:
            print(f"No 10-K filings found for {ticker}")
            return

        processed_count = 0

        for filing in filings:
            if processed_count >= num_years:
                break

            try:
                filing_date = datetime.strptime(str(filing.filing_date), "%Y-%m-%d")
                # Estimate Fiscal Year (approximate)
                estimated_year = (
                    filing_date.year - 1 if filing_date.month < 6 else filing_date.year
                )

                if estimated_year in existing_years:
                    print(f"Skipping {estimated_year} (Already exists)")
                    processed_count += 1
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

    def get_data(
        self,
        ticker: str,
        years: Optional[List[int]] = None,
        statements: Union[str, List[str], None] = None,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        query = "SELECT * FROM financials WHERE company = ?"
        params = [ticker]

        # Handle Years (Single Int or List of Ints)
        if years:
            if isinstance(years, int):
                years = [years]
            placeholders = ",".join(["?"] * len(years))
            query += f" AND year IN ({placeholders})"
            params.extend(years)

        # Handle Statements (Single Str or List of Strs)
        if statements:
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
            index=["statement_type", "concept"], columns="year", values="value"
        )

    def get_income_statement(
        self, ticker: str, year: Optional[int] = None
    ) -> pd.DataFrame:
        """Quickly fetch income statements."""
        return self.pivot_data(self.get_data(ticker, years=year, statements="income"))

    def get_balance_sheet(
        self, ticker: str, year: Optional[int] = None
    ) -> pd.DataFrame:
        """Quickly fetch balance sheets."""
        return self.pivot_data(self.get_data(ticker, years=year, statements="balance"))

    def get_cash_flow(self, ticker: str, year: Optional[int] = None) -> pd.DataFrame:
        """Quickly fetch cash flow statements."""
        return self.pivot_data(self.get_data(ticker, years=year, statements="cashflow"))

    def get_fiscal_year(self, ticker: str, year: int) -> pd.DataFrame:
        """Fetch all data (all statements) for a specific year."""
        return self.pivot_data(self.get_data(ticker, years=year))

    def search_concept(self, ticker: str, keyword: str) -> pd.DataFrame:
        """
        Search for specific line items (e.g., 'Revenue', 'Net Income')
        across all years. Helpful if you don't know the exact XBRL tag.
        """
        ticker = ticker.upper()
        query = """
            SELECT * FROM financials 
            WHERE company = ? AND concept LIKE ?
        """
        # Add wildcards for SQL LIKE
        search_term = f"%{keyword}%"

        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params=(ticker, search_term))

        return self.pivot_data(df)


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    db = FinancialDatabase("financial_data.db")
    company = "NVDA"

    # Ensure data exists
    db.update_company_data(company, num_years=3)

    print("\n--- 1. Filter by Statement (Income Only) ---")
    # Using the intuitive wrapper
    print(db.get_income_statement(company))

    print("\n--- 2. Filter by Year (2023 Only) ---")
    # Using the intuitive wrapper
    print(db.get_fiscal_year(company, 2023))

    print("\n--- 3. Complex Filter (Cashflow & Balance for 2022 & 2023) ---")
    # Using the robust get_data method
    df_complex = db.get_data(
        company, years=[2022, 2023], statements=["cashflow", "balance"]
    )
    print(db.pivot_data(df_complex))

    print("\n--- 4. Search for a Concept (e.g., 'Assets') ---")
    print(db.search_concept(company, "assets"))
