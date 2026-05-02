from __future__ import annotations

import pandas as pd

from core.agents.financial_tools import TOOL_REGISTRY


def _valuation_df() -> pd.DataFrame:
    return pd.DataFrame(
        data=[
            [300.0, 200.0, 240.0],  # market_cap
            [360.0, 260.0, 300.0],  # enterprise_value
            [100.0, 80.0, 90.0],  # revenues
            [20.0, 16.0, 18.0],  # EBITDA
            [10.0, 8.0, 9.0],  # net income
            [12.0, 10.0, 11.0],  # free cash flow
        ],
        index=[
            "market_cap",
            "enterprise_value",
            "Revenues",
            "EBITDA",
            "NetIncomeLoss",
            "FreeCashFlow",
        ],
        columns=["2024-12-31", "2022-12-31", "2023-12-31"],
    )


def _default_params() -> dict:
    return {
        "market_cap_metric": "market_cap",
        "enterprise_value_metric": "enterprise_value",
        "revenue_metric": "Revenues",
        "ebitda_metric": "EBITDA",
        "net_income_metric": "NetIncomeLoss",
        "fcf_metric": "FreeCashFlow",
    }


def test_valuation_multiples_snapshot_computes_rows_and_trends() -> None:
    tool = TOOL_REGISTRY["valuation_multiples_snapshot"]
    df = _valuation_df()

    result = tool.execute(df=df, params=tool.parameters_schema(**_default_params()))

    assert result.success is True
    assert result.added_rows is not None
    for row_label in [
        "price_to_sales",
        "price_to_earnings",
        "ev_to_ebitda",
        "price_to_fcf",
        "price_to_sales_pct_change",
        "price_to_earnings_pct_change",
        "ev_to_ebitda_pct_change",
        "price_to_fcf_pct_change",
    ]:
        assert row_label in result.added_rows

    price_to_sales = result.added_rows["price_to_sales"]
    assert price_to_sales["2022-12-31"] == 2.5
    assert round(price_to_sales["2023-12-31"], 4) == 2.6667
    assert price_to_sales["2024-12-31"] == 3.0

    ps_trend = result.added_rows["price_to_sales_pct_change"]
    assert set(ps_trend.keys()) == {"2023-12-31", "2024-12-31"}
    assert ps_trend["2024-12-31"] > 0


def test_valuation_multiples_snapshot_fails_fast_when_required_metrics_missing() -> None:
    tool = TOOL_REGISTRY["valuation_multiples_snapshot"]
    df = _valuation_df().drop(index=["EBITDA"])

    result = tool.execute(df=df, params=tool.parameters_schema(**_default_params()))

    assert result.success is False
    assert result.error is not None
    assert "EBITDA" in result.error


def test_valuation_multiples_snapshot_skips_non_positive_denominators_but_succeeds() -> None:
    tool = TOOL_REGISTRY["valuation_multiples_snapshot"]
    df = _valuation_df()
    df.loc["NetIncomeLoss"] = [10.0, 0.0, -5.0]

    result = tool.execute(df=df, params=tool.parameters_schema(**_default_params()))

    assert result.success is True
    assert result.added_rows is not None

    pe = result.added_rows["price_to_earnings"]
    assert set(pe.keys()) == {"2024-12-31"}
    assert "price_to_sales" in result.added_rows


def test_valuation_multiples_snapshot_fails_when_all_denominators_are_non_positive() -> None:
    tool = TOOL_REGISTRY["valuation_multiples_snapshot"]
    df = _valuation_df()
    df.loc["Revenues"] = [0.0, -1.0, 0.0]
    df.loc["EBITDA"] = [0.0, -1.0, 0.0]
    df.loc["NetIncomeLoss"] = [0.0, -1.0, 0.0]
    df.loc["FreeCashFlow"] = [0.0, -1.0, 0.0]

    result = tool.execute(df=df, params=tool.parameters_schema(**_default_params()))

    assert result.success is False
    assert result.error is not None
    assert "No valuation multiples could be computed" in result.error
