import pandas as pd

"""
FINANCIAL RATIO CALCULATOR
--------------------------
A collection of vectorized functions for fundamental analysis.
Optimized for Pandas DataFrames and NumPy arrays.

Usage:
    df['pe_ratio'] = valuation.price_to_earnings(df['price'], df['eps'])
"""

# Note: We use Union[float, List[float]] in type hints purely so the
# Python code remains valid for batch processing, but the LLM will
# generally prioritize sending single floats.

# ==========================================
# 1. VALUATION TOOLS
# ==========================================


def price_to_earnings(price: float, eps: float) -> float:
    """
    Calculates the Price-to-Earnings (P/E) Ratio.
    Use this when asked to evaluate if a stock is over or undervalued relative to its earnings.
    Formula: Price / EPS.
    """
    return price / eps


def price_to_book(price: float, book_value_per_share: float) -> float:
    """
    Calculates the Price-to-Book (P/B) Ratio.
    Use this to compare market value to book value.
    Formula: Price / Book Value Per Share.
    """
    return price / book_value_per_share


def price_to_sales(price: float, sales_per_share: float) -> float:
    """
    Calculates the Price-to-Sales (P/S) Ratio.
    Useful for valuing companies with no earnings.
    Formula: Price / Sales Per Share.
    """
    return price / sales_per_share


def peg_ratio(pe_ratio: float, earnings_growth_rate: float) -> float:
    """
    Calculates the PEG Ratio (Price/Earnings-to-Growth).
    Use this to determine if a stock is priced reasonably given its growth.
    Formula: P/E Ratio / Earnings Growth Rate (entered as a whole number, e.g., 15 for 15%).
    """
    return pe_ratio / earnings_growth_rate


def ev_to_ebitda(enterprise_value: float, ebitda: float) -> float:
    """
    Calculates the EV/EBITDA ratio.
    Used as a valuation metric that neutralizes capital structure.
    Formula: Enterprise Value / EBITDA.
    """
    return enterprise_value / ebitda


def dividend_yield(dividend_per_share: float, price: float) -> float:
    """
    Calculates the Dividend Yield.
    Formula: Annual Dividend per Share / Price.
    """
    return dividend_per_share / price


# ==========================================
# 2. PROFITABILITY TOOLS
# ==========================================


def gross_margin(gross_profit: float, total_revenue: float) -> float:
    """
    Calculates Gross Margin.
    Represents the percent of total sales revenue that the company retains after incurring direct costs.
    Formula: Gross Profit / Total Revenue.
    """
    return gross_profit / total_revenue


def operating_margin(operating_income: float, total_revenue: float) -> float:
    """
    Calculates Operating Margin.
    Formula: Operating Income / Total Revenue.
    """
    return operating_income / total_revenue


def net_profit_margin(net_income: float, total_revenue: float) -> float:
    """
    Calculates Net Profit Margin.
    Formula: Net Income / Total Revenue.
    """
    return net_income / total_revenue


def return_on_equity(net_income: float, total_equity: float) -> float:
    """
    Calculates Return on Equity (ROE).
    Measures financial performance calculated by dividing net income by shareholders' equity.
    Formula: Net Income / Total Equity.
    """
    return net_income / total_equity


def return_on_assets(net_income: float, total_assets: float) -> float:
    """
    Calculates Return on Assets (ROA).
    Formula: Net Income / Total Assets.
    """
    return net_income / total_assets


# ==========================================
# 3. LIQUIDITY & SOLVENCY TOOLS
# ==========================================


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """
    Calculates the Current Ratio.
    Measures a company's ability to pay short-term obligations or those due within one year.
    Formula: Current Assets / Current Liabilities.
    """
    return current_assets / current_liabilities


def quick_ratio(
    current_assets: float, inventory: float, current_liabilities: float
) -> float:
    """
    Calculates the Quick Ratio (Acid Test).
    Measures liquidity excluding inventory.
    Formula: (Current Assets - Inventory) / Current Liabilities.
    """
    return (current_assets - inventory) / current_liabilities


def debt_to_equity(total_debt: float, total_equity: float) -> float:
    """
    Calculates Debt-to-Equity Ratio.
    Measure of the degree to which a company is financing its operations through debt.
    Formula: Total Debt / Total Equity.
    """
    return total_debt / total_equity


def interest_coverage_ratio(ebit: float, interest_expense: float) -> float:
    """
    Calculates Interest Coverage Ratio.
    Determine how easily a company can pay interest on its outstanding debt.
    Formula: EBIT / Interest Expense.
    """
    return ebit / interest_expense


# ==========================================
# 4. EFFICIENCY TOOLS
# ==========================================


def asset_turnover(total_revenue: float, average_total_assets: float) -> float:
    """
    Calculates Asset Turnover.
    Measures the efficiency of a company's use of its assets in generating sales revenue.
    Formula: Revenue / Average Total Assets.
    """
    return total_revenue / average_total_assets


def inventory_turnover(cogs: float, average_inventory: float) -> float:
    """
    Calculates Inventory Turnover.
    Shows how many times a company has sold and replaced inventory during a given period.
    Formula: COGS / Average Inventory.
    """
    return cogs / average_inventory


# ==========================================
# 5. GROWTH & COMPOSITE TOOLS
# ==========================================


def calculate_cagr(
    beginning_value: float, ending_value: float, periods: float
) -> float:
    """
    Calculates Compound Annual Growth Rate (CAGR).
    Use this to measure the mean annual growth rate of an investment over a specified time period longer than one year.
    Formula: (Ending Value / Beginning Value) ^ (1 / periods) - 1.
    """
    return (ending_value / beginning_value) ** (1 / periods) - 1


def altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    sales: float,
    total_assets: float,
    total_liabilities: float,
) -> float:
    """
    Calculates Altman Z-Score.
    Use this to predict the probability that a firm will go into bankruptcy within two years.
    Formula: 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    """
    A = working_capital / total_assets
    B = retained_earnings / total_assets
    C = ebit / total_assets
    D = market_cap / total_liabilities
    E = sales / total_assets

    return (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)


# ==========================================
# TOOL EXPORT LIST
# ==========================================

# Pass this list to your LangChain Agent
# financial_tools_list = [
#     price_to_earnings,
#     price_to_book,
#     price_to_sales,
#     peg_ratio,
#     ev_to_ebitda,
#     dividend_yield,
#     gross_margin,
#     operating_margin,
#     net_profit_margin,
#     return_on_equity,
#     return_on_assets,
#     current_ratio,
#     quick_ratio,
#     debt_to_equity,
#     interest_coverage_ratio,
#     asset_turnover,
#     inventory_turnover,
#     calculate_cagr,
#     altman_z_score,
# ]

# ==========================================
# USAGE EXAMPLE
# ==========================================

if __name__ == "__main__":
    # Generate dummy data to demonstrate vectorized efficiency
    data = {
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "price": [150.0, 300.0, 2800.0, 3400.0, 900.0],
        "eps": [5.6, 9.0, 110.0, 60.0, 3.0],
        "total_assets": [10000, 20000, 15000, 18000, 8000],
        "total_debt": [4000, 5000, 2000, 6000, 3000],
        "total_equity": [6000, 15000, 13000, 12000, 5000],
        "current_assets": [3000, 8000, 9000, 7000, 2000],
        "current_liabilities": [2000, 4000, 3000, 6000, 2500],
        "inventory": [500, 100, 200, 2000, 1000],
    }

    df = pd.DataFrame(data)

    print("--- Input DataFrame ---")
    print(df[["ticker", "price", "eps"]].head())

    # Calculate multiple ratios at once (Vectorized)
    df["PE_Ratio"] = price_to_earnings(df["price"], df["eps"])
    df["Debt_to_Equity"] = debt_to_equity(df["total_debt"], df["total_equity"])
    df["Quick_Ratio"] = quick_ratio(
        df["current_assets"], df["inventory"], df["current_liabilities"]
    )

    print("\n--- Calculated Ratios ---")
    print(df[["ticker", "PE_Ratio", "Debt_to_Equity", "Quick_Ratio"]])
