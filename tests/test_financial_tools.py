from __future__ import annotations

import pandas as pd
import pytest

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
    assert "coverage 3/3 periods" in result.summary


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


def test_valuation_multiples_snapshot_summary_reports_partial_coverage() -> None:
    tool = TOOL_REGISTRY["valuation_multiples_snapshot"]
    df = pd.DataFrame(
        data=[
            [300.0, 320.0, 340.0, 360.0],  # market_cap
            [360.0, 380.0, 400.0, 420.0],  # enterprise_value
            [100.0, 0.0, -1.0, 110.0],  # revenues
            [20.0, 21.0, 22.0, 23.0],  # EBITDA
            [10.0, 10.5, 11.0, 11.5],  # net income
            [12.0, 12.5, 13.0, 13.5],  # free cash flow
        ],
        index=[
            "market_cap",
            "enterprise_value",
            "Revenues",
            "EBITDA",
            "NetIncomeLoss",
            "FreeCashFlow",
        ],
        columns=["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"],
    )

    result = tool.execute(df=df, params=tool.parameters_schema(**_default_params()))

    assert result.success is True
    assert result.added_rows is not None
    assert set(result.added_rows["price_to_sales"].keys()) == {
        "2022-12-31",
        "2025-12-31",
    }
    assert "price_to_sales" in result.summary
    assert "coverage 2/4 periods" in result.summary


def test_custom_formula_normalizes_metric_and_dependencies() -> None:
    tool = TOOL_REGISTRY["custom_formula"]
    df = pd.DataFrame(
        data=[[10.0, 12.0], [2.0, 3.0]],
        index=["Revenue", "Cost"],
        columns=["2024-12-31", "2025-12-31"],
    )

    result = tool.execute(
        df=df,
        params=tool.parameters_schema(
            metric_name=" GrossProfit ",
            expression="Revenue - Cost",
            dependencies=[" Revenue ", "Cost  "],
            description="Derived gross profit.",
        ),
    )

    assert result.success is True
    assert result.added_rows is not None
    assert "GrossProfit" in result.added_rows
    assert result.added_rows["GrossProfit"]["2024-12-31"] == 8.0
    assert result.added_rows["GrossProfit"]["2025-12-31"] == 9.0


def test_custom_formula_logs_warning_when_result_series_is_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tool = TOOL_REGISTRY["custom_formula"]
    df = pd.DataFrame(
        data=[[float("nan"), float("nan")]],
        index=["AllNanInput"],
        columns=["2024-12-31", "2025-12-31"],
    )

    with caplog.at_level("WARNING"):
        result = tool.execute(
            df=df,
            params=tool.parameters_schema(
                metric_name="EmptyDerivedRow",
                expression="AllNanInput * 2",
                dependencies=["AllNanInput"],
                description="Should produce all-NaN output.",
            ),
        )

    assert result.success is True
    assert result.added_rows == {"EmptyDerivedRow": {}}
    assert "produced no non-null values" in caplog.text
