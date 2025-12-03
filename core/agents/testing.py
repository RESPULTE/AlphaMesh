from core.config import settings
from edgar import Company, MultiFinancials
from edgar.xbrl import XBRLS

# --- CONFIGURATION ---


def benchmark(func, *args, **kwargs):
    import time

    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__} took {end - start:.6f} seconds")
    return result


def xbrl_func():

    # Get multiple filings for trend analysis
    company = Company("AAPL")
    filings = company.get_filings(form=["10-K", "10-Q"]).head(7)

    # Create stitched view across multiple filings
    xbrls = XBRLS.from_filings(filings)

    # Access stitched statements
    stitched_statements = xbrls.statements

    # Display multi-period statements with intelligent period selection
    income_trend = stitched_statements.income_statement()
    balance_sheet_trend = stitched_statements.balance_sheet()
    cashflow_trend = stitched_statements.cashflow_statement()

    print("Three-Year Revenue Trend:")
    revenue_trend = income_trend.to_dataframe()
    revenue_row = revenue_trend.loc[revenue_trend["label"] == "Revenue"]
    print(revenue_row)


def multi_func():

    # Get multiple years of 10-K filings
    company = Company("AAPL")
    filings = company.get_filings(form=["10-K", "10-Q"]).head(
        7
    )  # Last 3 annual reports

    # Create multi-period financials
    multi_financials = MultiFinancials.extract(filings)

    # Access statements spanning multiple years
    balance_sheet = multi_financials.balance_sheet()
    income_statement = multi_financials.income_statement()
    cash_flow = multi_financials.cashflow_statement()

    print("Multi-Year Income Statement:")
    print(income_statement)


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    settings
    benchmark(xbrl_func)
    benchmark(multi_func)
