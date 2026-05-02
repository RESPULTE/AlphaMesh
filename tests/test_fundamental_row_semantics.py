from __future__ import annotations

import pandas as pd

from core.agents.financial_tools import ToolResult
from core.agents.utils import build_fundamental_row_semantics


def test_build_row_semantics_uses_tool_context_for_percent_rows() -> None:
    df = pd.DataFrame(
        data=[
            [100.0, 120.0],
            [0.20, 0.25],
        ],
        index=["Revenues", "topline_growth"],
        columns=["2023-12-31", "2024-12-31"],
    )
    tool_results = [
        ToolResult(
            tool_name="cagr",
            success=True,
            added_rows={"topline_growth": {"2024-12-31": 0.25}},
        )
    ]

    semantics = build_fundamental_row_semantics(
        financial_data=df,
        tool_results=tool_results,
    )

    assert semantics["topline_growth"].value_kind == "percent"
    assert semantics["topline_growth"].display_unit == "%"


def test_build_row_semantics_marks_pe_invalid_when_eps_non_positive() -> None:
    df = pd.DataFrame(
        data=[
            [25.0, 30.0],
            [-1.2, -0.6],
        ],
        index=["price_to_earnings", "EarningsPerShareBasic"],
        columns=["2023-12-31", "2024-12-31"],
    )

    semantics = build_fundamental_row_semantics(financial_data=df, tool_results=[])

    pe = semantics["price_to_earnings"]
    assert pe.value_kind == "ratio"
    assert pe.invalid is True
    assert "EPS <= 0" in pe.invalid_reason
