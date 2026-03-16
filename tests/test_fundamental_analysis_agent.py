"""
tests/test_fundamental_analysis_agent.py

Comprehensive test suite for the refactored Fundamental Analysis Agent stack.

Coverage map
───────────────────────────────────────────────────────────────────────────────
Layer                   │ Classes / functions under test
────────────────────────┼──────────────────────────────────────────────────────
NumpyFinancialAdapter   │ cagr, npv, dcf_intrinsic_value, all ratio methods
CAGRTool                │ normal, single-point, zero/negative start, wrong metric
DCFTool                 │ normal, with shares, WACC≤terminal error, missing FCF
ProfitabilityTool       │ all margins, missing revenue, partial combinations
DebtSolvencyTool        │ D/E, interest coverage, missing equity
LiquidityTool           │ current+quick ratio, no inventory, missing required
CustomFormulaTool       │ normal, unsafe chars, missing deps, complex expression
ToolRegistry            │ all 6 tools registered, get_tool_descriptions format
FinancialDatabase       │ initialize, _process_statement label resolution,
                        │ get_all_concepts, search_label (warn on empty),
                        │ pivot_df, find_uncovered_periods, _bulk_insert,
                        │ get_price_data (invalid interval)
_data_prep_node         │ happy path, no concepts, price merge, EDGAR failure
_tool_planner_node      │ happy path, LLM failure fallback, empty plan
_tool_executor_node     │ happy path, unknown tool, bad params, empty df,
                        │ sequential row merging, all tools fail
_analyst_node           │ happy path, empty df skip, tool summary included
FundamentalAnalysisOutput│ get_llm_context_str with/without data/tool results
Full pipeline (run())   │ end-to-end mock, CAGR+DCF, raw-data-only query,
                        │ unknown ticker, date range with no data
────────────────────────┴──────────────────────────────────────────────────────

Dependencies
────────────
  pytest, pytest-asyncio, aiosqlite (in-memory), unittest.mock
  No network calls are made — EDGAR and yfinance are fully stubbed.
"""

from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from core.agents.financial_db import FinancialDatabase, _looks_like_date
from core.agents.financial_tools import (
    TOOL_REGISTRY,
    CAGRParams,
    CAGRTool,
    CustomFormulaParams,
    CustomFormulaTool,
    DCFParams,
    DCFTool,
    DebtSolvencyParams,
    DebtSolvencyTool,
    FinanceToolkitAdapter,
    LiquidityParams,
    LiquidityTool,
    NumpyFinancialAdapter,
    ProfitabilityParams,
    ProfitabilityTool,
    ToolResult,
    get_tool_descriptions,
    set_adapter,
)
from core.agents.fundamental_analysis_agent import (
    FundamentalAnalysisAgent,
    FundamentalAnalysisOutput,
    ToolCallSpec,
    ToolPlan,
    _AgentState,
    _format_value,
)
from core.agents.models import BaseAgentInput

# ─────────────────────────────────────────────────────────────────────────────
# Pytest configuration
# ─────────────────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.asyncio


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def adapter():
    """A fresh NumpyFinancialAdapter for each test."""
    return NumpyFinancialAdapter()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    A minimal wide-format DataFrame mirroring what the agent stores:
      rows   = concept labels
      columns = ISO date strings
    """
    dates = ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]
    return pd.DataFrame(
        {
            "2020-12-31": [
                100_000_000,
                20_000_000,
                15_000_000,
                50_000_000,
                30_000_000,
                10_000_000,
                5_000_000,
            ],
            "2021-12-31": [
                120_000_000,
                24_000_000,
                18_000_000,
                60_000_000,
                36_000_000,
                11_000_000,
                5_500_000,
            ],
            "2022-12-31": [
                140_000_000,
                28_000_000,
                21_000_000,
                70_000_000,
                42_000_000,
                12_000_000,
                6_000_000,
            ],
            "2023-12-31": [
                160_000_000,
                32_000_000,
                24_000_000,
                80_000_000,
                48_000_000,
                13_000_000,
                6_500_000,
            ],
        },
        index=[
            "Revenues",
            "GrossProfit",
            "OperatingIncomeLoss",
            "AssetsCurrent",
            "LiabilitiesCurrent",
            "NetIncomeLoss",
            "InventoryNet",
        ],
    )


@pytest.fixture
def sample_df_with_debt(sample_df) -> pd.DataFrame:
    """Adds debt-related rows to the sample DataFrame."""
    df = sample_df.copy()
    debt_row = pd.Series(
        [40_000_000, 38_000_000, 35_000_000, 30_000_000],
        index=df.columns,
        name="LongTermDebt",
    )
    equity_row = pd.Series(
        [60_000_000, 65_000_000, 70_000_000, 75_000_000],
        index=df.columns,
        name="StockholdersEquity",
    )
    interest_row = pd.Series(
        [2_000_000, 1_900_000, 1_800_000, 1_500_000],
        index=df.columns,
        name="InterestExpense",
    )
    return pd.concat(
        [df, debt_row.to_frame().T, equity_row.to_frame().T, interest_row.to_frame().T]
    )


@pytest.fixture
def sample_df_with_fcf(sample_df) -> pd.DataFrame:
    """Adds a FreeCashFlow row derived from the existing data."""
    df = sample_df.copy()
    fcf_row = pd.Series(
        [12_000_000, 15_000_000, 18_000_000, 22_000_000],
        index=df.columns,
        name="FreeCashFlow",
    )
    shares_row = pd.Series(
        [100_000_000, 100_000_000, 100_000_000, 100_000_000],
        index=df.columns,
        name="CommonStockSharesOutstanding",
    )
    return pd.concat([df, fcf_row.to_frame().T, shares_row.to_frame().T])


@pytest.fixture
def tmp_db(tmp_path) -> str:
    """Returns a path to a fresh temporary SQLite database."""
    return str(tmp_path / "test_financials.db")


@pytest.fixture
def base_input() -> BaseAgentInput:
    return BaseAgentInput(
        ticker="AAPL",
        query="Analyse AAPL revenue growth and profitability",
        vector_query="AAPL revenue profitability",
        metrics=[],
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NumpyFinancialAdapter
# ─────────────────────────────────────────────────────────────────────────────


class TestNumpyFinancialAdapter:

    def test_cagr_known_value(self, adapter):
        """100 → 200 over 2 years = 41.42% CAGR."""
        result = adapter.cagr(100, 200, 2)
        assert abs(result - (math.sqrt(2) - 1)) < 1e-9

    def test_cagr_flat_value_is_zero(self, adapter):
        assert abs(adapter.cagr(100, 100, 5)) < 1e-9

    def test_cagr_decline(self, adapter):
        result = adapter.cagr(200, 100, 2)
        assert result < 0

    def test_cagr_zero_start_raises(self, adapter):
        with pytest.raises(ValueError):
            adapter.cagr(0, 100, 3)

    def test_cagr_negative_start_raises(self, adapter):
        with pytest.raises(ValueError):
            adapter.cagr(-50, 100, 3)

    def test_cagr_zero_periods_raises(self, adapter):
        with pytest.raises(ValueError):
            adapter.cagr(100, 200, 0)

    def test_dcf_standard(self, adapter):
        fcfs = [100_000, 110_000, 121_000, 133_100, 146_410]
        result = adapter.dcf_intrinsic_value(
            fcfs, wacc=0.10, terminal_growth_rate=0.025
        )
        assert "enterprise_value" in result
        assert result["enterprise_value"] > 0
        assert result["pv_fcf"] > 0
        assert result["pv_terminal_value"] > 0

    def test_dcf_with_shares(self, adapter):
        fcfs = [1_000_000] * 5
        result = adapter.dcf_intrinsic_value(
            fcfs, wacc=0.10, terminal_growth_rate=0.02, shares_outstanding=100_000
        )
        assert "intrinsic_value_per_share" in result
        assert result["intrinsic_value_per_share"] == pytest.approx(
            result["enterprise_value"] / 100_000
        )

    def test_dcf_wacc_equal_terminal_growth_raises(self, adapter):
        with pytest.raises(ValueError, match="must exceed"):
            adapter.dcf_intrinsic_value([100_000], wacc=0.05, terminal_growth_rate=0.05)

    def test_dcf_wacc_below_terminal_growth_raises(self, adapter):
        with pytest.raises(ValueError):
            adapter.dcf_intrinsic_value([100_000], wacc=0.02, terminal_growth_rate=0.05)

    def test_gross_margin(self, adapter):
        assert adapter.gross_margin(100, 60) == pytest.approx(0.40)

    def test_gross_margin_zero_revenue_returns_nan(self, adapter):
        assert math.isnan(adapter.gross_margin(0, 60))

    def test_operating_margin(self, adapter):
        assert adapter.operating_margin(20, 100) == pytest.approx(0.20)

    def test_net_margin(self, adapter):
        assert adapter.net_margin(10, 100) == pytest.approx(0.10)

    def test_debt_to_equity(self, adapter):
        assert adapter.debt_to_equity(50, 100) == pytest.approx(0.50)

    def test_debt_to_equity_zero_equity_returns_nan(self, adapter):
        assert math.isnan(adapter.debt_to_equity(50, 0))

    def test_interest_coverage(self, adapter):
        # ebit=100, interest=25 → ICR = 4.0
        assert adapter.interest_coverage(100, 25) == pytest.approx(4.0)

    def test_interest_coverage_negative_interest_uses_abs(self, adapter):
        # interest expense reported as negative in some filings
        assert adapter.interest_coverage(100, -25) == pytest.approx(4.0)

    def test_current_ratio(self, adapter):
        assert adapter.current_ratio(200, 100) == pytest.approx(2.0)

    def test_quick_ratio(self, adapter):
        # (200 - 50) / 100 = 1.50
        assert adapter.quick_ratio(200, 50, 100) == pytest.approx(1.50)

    def test_npv_fallback(self, adapter):
        """NPV works even if numpy_financial is not installed (pure-python fallback)."""
        cfs = [-1000, 300, 300, 300, 300, 300]
        result = adapter.npv(0.10, cfs)
        assert isinstance(result, float)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CAGRTool
# ─────────────────────────────────────────────────────────────────────────────


class TestCAGRTool:
    @pytest.fixture(autouse=True)
    def tool(self):
        self.tool = CAGRTool()

    def test_normal_growth(self, sample_df):
        params = CAGRParams(metric="Revenues")
        result = self.tool.execute(sample_df, params)
        assert result.success
        assert result.scalar_value is not None
        assert result.scalar_value > 0
        assert "added_rows" in result.model_fields
        assert "Revenues_CAGR" in result.added_rows

    def test_custom_output_label(self, sample_df):
        params = CAGRParams(metric="Revenues", output_label="rev_cagr")
        result = self.tool.execute(sample_df, params)
        assert result.success
        assert "rev_cagr" in result.added_rows

    def test_cagr_row_broadcast_to_all_columns(self, sample_df):
        params = CAGRParams(metric="Revenues")
        result = self.tool.execute(sample_df, params)
        # The summary row must span all date columns
        cagr_row = result.added_rows["Revenues_CAGR"]
        assert set(cagr_row.keys()) == set(sample_df.columns)

    def test_missing_metric_returns_failure(self, sample_df):
        params = CAGRParams(metric="NonExistentMetric")
        result = self.tool.execute(sample_df, params)
        assert not result.success
        assert "not found" in result.error.lower()

    def test_single_data_point_returns_failure(self):
        df = pd.DataFrame({"2023-12-31": [100]}, index=["Revenues"])
        params = CAGRParams(metric="Revenues")
        result = CAGRTool().execute(df, params)
        assert not result.success
        assert "2 non-null" in result.error

    def test_cagr_with_all_null_row_returns_failure(self):
        df = pd.DataFrame(
            {"2022-12-31": [None], "2023-12-31": [None]},
            index=["Revenues"],
        )
        params = CAGRParams(metric="Revenues")
        result = CAGRTool().execute(df, params)
        assert not result.success

    def test_summary_contains_metric_name(self, sample_df):
        params = CAGRParams(metric="Revenues")
        result = self.tool.execute(sample_df, params)
        assert "Revenues" in result.summary
        assert "CAGR" in result.summary


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DCFTool
# ─────────────────────────────────────────────────────────────────────────────


class TestDCFTool:
    @pytest.fixture(autouse=True)
    def tool(self):
        self.tool = DCFTool()

    def _params(self, fcf_metric="FreeCashFlow", shares_metric=None):
        return DCFParams(
            fcf_metric=fcf_metric,
            wacc=0.10,
            terminal_growth_rate=0.025,
            projection_years=5,
            shares_outstanding_metric=shares_metric,
            wacc_reasoning="Beta ~1.1, risk-free 4.5%, equity premium 5.5%.",
            terminal_growth_reasoning="Mature tech; GDP+0.5% conservatively.",
        )

    def test_normal_enterprise_value(self, sample_df_with_fcf):
        result = self.tool.execute(sample_df_with_fcf, self._params())
        assert result.success
        assert result.series_values["enterprise_value"] > 0
        assert result.reasoning is not None
        assert "10.0%" in result.reasoning or "0.10" in result.reasoning

    def test_with_shares_produces_per_share_value(self, sample_df_with_fcf):
        result = self.tool.execute(
            sample_df_with_fcf,
            self._params(shares_metric="CommonStockSharesOutstanding"),
        )
        assert result.success
        assert "intrinsic_value_per_share" in result.series_values
        assert result.series_values["intrinsic_value_per_share"] > 0

    def test_missing_fcf_metric_returns_failure(self, sample_df):
        result = self.tool.execute(sample_df, self._params(fcf_metric="DoesNotExist"))
        assert not result.success
        assert "not found" in result.error.lower()

    def test_wacc_equals_terminal_growth_returns_failure(self, sample_df_with_fcf):
        params = DCFParams(
            fcf_metric="FreeCashFlow",
            wacc=0.025,
            terminal_growth_rate=0.025,
            wacc_reasoning="equal",
            terminal_growth_reasoning="equal",
        )
        result = self.tool.execute(sample_df_with_fcf, params)
        assert not result.success

    def test_wacc_below_terminal_growth_returns_failure(self, sample_df_with_fcf):
        params = DCFParams(
            fcf_metric="FreeCashFlow",
            wacc=0.01,
            terminal_growth_rate=0.05,
            wacc_reasoning="too low",
            terminal_growth_reasoning="too high",
        )
        result = self.tool.execute(sample_df_with_fcf, params)
        assert not result.success

    def test_single_period_fcf_still_runs(self):
        df = pd.DataFrame({"2023-12-31": [50_000_000]}, index=["FreeCashFlow"])
        params = DCFParams(
            fcf_metric="FreeCashFlow",
            wacc=0.10,
            terminal_growth_rate=0.025,
            wacc_reasoning="test",
            terminal_growth_reasoning="test",
        )
        result = DCFTool().execute(df, params)
        assert result.success

    def test_summary_contains_enterprise_value(self, sample_df_with_fcf):
        result = self.tool.execute(sample_df_with_fcf, self._params())
        assert "Enterprise Value" in result.summary

    def test_reasoning_block_contains_wacc_and_tgr(self, sample_df_with_fcf):
        result = self.tool.execute(sample_df_with_fcf, self._params())
        assert "WACC" in result.reasoning
        assert "terminal growth" in result.reasoning.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ProfitabilityTool
# ─────────────────────────────────────────────────────────────────────────────


class TestProfitabilityTool:
    @pytest.fixture(autouse=True)
    def tool(self):
        self.tool = ProfitabilityTool()

    def _full_params(self):
        return ProfitabilityParams(
            revenue_metric="Revenues",
            cogs_metric="GrossProfit",  # Using GrossProfit as proxy; COGS = Revenue - GrossProfit isn't asserted
            operating_income_metric="OperatingIncomeLoss",
            net_income_metric="NetIncomeLoss",
        )

    def test_all_margins_computed(self, sample_df):
        result = self.tool.execute(sample_df, self._full_params())
        assert result.success
        # All three margins should be added
        assert "operating_margin" in result.added_rows
        assert "net_margin" in result.added_rows

    def test_operating_margin_values_in_range(self, sample_df):
        params = ProfitabilityParams(
            revenue_metric="Revenues",
            operating_income_metric="OperatingIncomeLoss",
        )
        result = self.tool.execute(sample_df, params)
        assert result.success
        for v in result.added_rows["operating_margin"].values():
            assert 0.0 < v < 1.0, f"Operating margin {v} outside (0,1)"

    def test_net_margin_values_in_range(self, sample_df):
        params = ProfitabilityParams(
            revenue_metric="Revenues",
            net_income_metric="NetIncomeLoss",
        )
        result = self.tool.execute(sample_df, params)
        assert result.success
        for v in result.added_rows["net_margin"].values():
            assert 0.0 < v < 1.0

    def test_missing_revenue_returns_failure(self, sample_df):
        params = ProfitabilityParams(revenue_metric="NoSuchMetric")
        result = self.tool.execute(sample_df, params)
        assert not result.success
        assert "NoSuchMetric" in result.error

    def test_partial_params_no_error(self, sample_df):
        """Only revenue + net_income — should succeed without gross margin."""
        params = ProfitabilityParams(
            revenue_metric="Revenues",
            net_income_metric="NetIncomeLoss",
        )
        result = self.tool.execute(sample_df, params)
        assert result.success
        assert "net_margin" in result.added_rows
        assert "gross_margin" not in result.added_rows

    def test_all_optional_metrics_missing_returns_success_with_empty(self, sample_df):
        """Revenue present but no COGS/op_income/net_income → success but no rows."""
        params = ProfitabilityParams(revenue_metric="Revenues")
        result = self.tool.execute(sample_df, params)
        assert result.success
        assert not result.added_rows  # nothing computable


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DebtSolvencyTool
# ─────────────────────────────────────────────────────────────────────────────


class TestDebtSolvencyTool:
    @pytest.fixture(autouse=True)
    def tool(self):
        self.tool = DebtSolvencyTool()

    def _full_params(self):
        return DebtSolvencyParams(
            total_debt_metric="LongTermDebt",
            total_equity_metric="StockholdersEquity",
            ebit_metric="OperatingIncomeLoss",
            interest_expense_metric="InterestExpense",
        )

    def test_dte_ratio_computed(self, sample_df_with_debt):
        result = self.tool.execute(sample_df_with_debt, self._full_params())
        assert result.success
        assert "debt_to_equity" in result.added_rows
        # debt should be shrinking relative to equity over the period
        dte_vals = sorted(result.added_rows["debt_to_equity"].items())
        first_dte = dte_vals[0][1]
        last_dte = dte_vals[-1][1]
        assert last_dte < first_dte  # improving leverage

    def test_interest_coverage_computed(self, sample_df_with_debt):
        result = self.tool.execute(sample_df_with_debt, self._full_params())
        assert result.success
        assert "interest_coverage" in result.added_rows
        for v in result.added_rows["interest_coverage"].values():
            assert v > 1  # EBIT > Interest in our fixture

    def test_missing_equity_skips_dte_gracefully(self, sample_df_with_debt):
        params = DebtSolvencyParams(
            total_debt_metric="LongTermDebt",
            total_equity_metric="NoSuchEquity",
        )
        result = self.tool.execute(sample_df_with_debt, params)
        # Should still succeed (just couldn't compute D/E)
        assert result.success
        assert "debt_to_equity" not in (result.added_rows or {})
        assert (
            "skipped" in result.summary.lower() or "missing" in result.summary.lower()
        )

    def test_no_interest_metrics_skips_coverage(self, sample_df_with_debt):
        params = DebtSolvencyParams(
            total_debt_metric="LongTermDebt",
            total_equity_metric="StockholdersEquity",
        )
        result = self.tool.execute(sample_df_with_debt, params)
        assert result.success
        assert "interest_coverage" not in (result.added_rows or {})


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LiquidityTool
# ─────────────────────────────────────────────────────────────────────────────


class TestLiquidityTool:
    @pytest.fixture(autouse=True)
    def tool(self):
        self.tool = LiquidityTool()

    def test_current_ratio_computed(self, sample_df):
        params = LiquidityParams(
            current_assets_metric="AssetsCurrent",
            current_liabilities_metric="LiabilitiesCurrent",
        )
        result = self.tool.execute(sample_df, params)
        assert result.success
        assert "current_ratio" in result.added_rows
        # Our fixture: AssetsCurrent / LiabilitiesCurrent = 50M/30M ≈ 1.67
        for v in result.added_rows["current_ratio"].values():
            assert v > 1

    def test_quick_ratio_computed_when_inventory_present(self, sample_df):
        params = LiquidityParams(
            current_assets_metric="AssetsCurrent",
            current_liabilities_metric="LiabilitiesCurrent",
            inventory_metric="InventoryNet",
        )
        result = self.tool.execute(sample_df, params)
        assert result.success
        assert "current_ratio" in result.added_rows
        assert "quick_ratio" in result.added_rows
        # Quick ratio should be less than current ratio (inventory excluded)
        for col in result.added_rows["current_ratio"]:
            if col in result.added_rows["quick_ratio"]:
                assert (
                    result.added_rows["quick_ratio"][col]
                    < result.added_rows["current_ratio"][col]
                )

    def test_missing_current_assets_returns_failure(self, sample_df):
        params = LiquidityParams(
            current_assets_metric="NoSuchAssets",
            current_liabilities_metric="LiabilitiesCurrent",
        )
        result = self.tool.execute(sample_df, params)
        assert not result.success
        assert "NoSuchAssets" in result.error

    def test_missing_current_liabilities_returns_failure(self, sample_df):
        params = LiquidityParams(
            current_assets_metric="AssetsCurrent",
            current_liabilities_metric="NoSuchLiabilities",
        )
        result = self.tool.execute(sample_df, params)
        assert not result.success

    def test_no_inventory_no_quick_ratio(self, sample_df):
        params = LiquidityParams(
            current_assets_metric="AssetsCurrent",
            current_liabilities_metric="LiabilitiesCurrent",
        )
        result = self.tool.execute(sample_df, params)
        assert "quick_ratio" not in result.added_rows


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CustomFormulaTool
# ─────────────────────────────────────────────────────────────────────────────


class TestCustomFormulaTool:
    @pytest.fixture(autouse=True)
    def tool(self):
        self.tool = CustomFormulaTool()

    def test_simple_ratio(self, sample_df):
        params = CustomFormulaParams(
            metric_name="net_margin",
            expression="NetIncomeLoss / Revenues",
            dependencies=["NetIncomeLoss", "Revenues"],
            description="Net profit margin",
        )
        result = self.tool.execute(sample_df, params)
        assert result.success
        assert "net_margin" in result.added_rows
        for v in result.added_rows["net_margin"].values():
            assert 0 < v < 1

    def test_subtraction_expression(self, sample_df):
        params = CustomFormulaParams(
            metric_name="net_debt",
            expression="LongTermDebt - AssetsCurrent",
            dependencies=["LongTermDebt", "AssetsCurrent"],
        )
        # LongTermDebt not in sample_df — should fail with missing dep
        result = self.tool.execute(sample_df, params)
        assert not result.success
        assert "LongTermDebt" in result.error

    def test_unsafe_expression_blocked(self, sample_df):
        for dangerous in [
            "__import__('os')",
            "open('/etc/passwd').read()",
            "Revenues; import os",
        ]:
            params = CustomFormulaParams(
                metric_name="evil",
                expression=dangerous,
                dependencies=["Revenues"],
            )
            result = self.tool.execute(sample_df, params)
            assert (
                not result.success
            ), f"Unsafe expression should be blocked: {dangerous}"
            assert "unsafe" in result.error.lower()

    def test_missing_dependency_returns_failure(self, sample_df):
        params = CustomFormulaParams(
            metric_name="mystery",
            expression="EBITDA / Revenues",
            dependencies=["EBITDA", "Revenues"],
        )
        result = self.tool.execute(sample_df, params)
        assert not result.success
        assert "EBITDA" in result.error

    def test_summary_contains_expression(self, sample_df):
        params = CustomFormulaParams(
            metric_name="gross_pct",
            expression="GrossProfit / Revenues",
            dependencies=["GrossProfit", "Revenues"],
        )
        result = self.tool.execute(sample_df, params)
        assert result.success
        assert "gross_pct" in result.summary
        assert "GrossProfit / Revenues" in result.summary


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ToolRegistry & get_tool_descriptions
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistry:

    def test_all_expected_tools_registered(self):
        expected = {
            "cagr",
            "dcf_intrinsic_value",
            "profitability_ratios",
            "debt_solvency",
            "liquidity",
            "custom_formula",
        }
        assert expected == set(TOOL_REGISTRY.keys())

    def test_each_tool_has_execute_method(self):
        for name, tool in TOOL_REGISTRY.items():
            assert callable(getattr(tool, "execute", None)), f"{name} missing execute()"

    def test_each_tool_has_parameters_schema(self):
        for name, tool in TOOL_REGISTRY.items():
            assert (
                tool.parameters_schema is not None
            ), f"{name} missing parameters_schema"

    def test_get_tool_descriptions_contains_all_names(self):
        desc = get_tool_descriptions()
        for tool_name in TOOL_REGISTRY:
            assert f"[{tool_name}]" in desc, f"Missing [{tool_name}] in descriptions"

    def test_get_tool_descriptions_contains_required_marker(self):
        desc = get_tool_descriptions()
        assert "(REQUIRED)" in desc

    def test_adapter_swap(self):
        """Calling set_adapter replaces the global adapter without error."""
        original = NumpyFinancialAdapter()
        new = NumpyFinancialAdapter()  # same type — just verifying no crash
        set_adapter(new)
        set_adapter(original)  # restore

    def test_finance_toolkit_adapter_raises_not_implemented(self):
        """FinanceToolkitAdapter stubs raise NotImplementedError on any call."""
        # Skip if financetoolkit happens to be installed
        try:
            adapter = FinanceToolkitAdapter()
        except ImportError:
            pytest.skip("financetoolkit not installed")
        with pytest.raises(NotImplementedError):
            adapter.cagr(100, 200, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  FinancialDatabase — unit tests (in-memory SQLite, no EDGAR/yfinance)
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialDatabase:

    @pytest.fixture
    def db(self, tmp_db):
        return FinancialDatabase(db_name=tmp_db)

    # ── _looks_like_date helper ───────────────────────────────────────────────

    def test_looks_like_date_valid(self):
        assert _looks_like_date("2023-12-31")
        assert _looks_like_date("2020/01/15")
        assert _looks_like_date("2021Q3")
        assert _looks_like_date("2023")

    def test_looks_like_date_invalid(self):
        assert not _looks_like_date("label")
        assert not _looks_like_date("Revenues")
        assert not _looks_like_date("")

    # ── initialize ────────────────────────────────────────────────────────────

    async def test_initialize_creates_table(self, db):
        await db.initialize()
        import aiosqlite

        async with aiosqlite.connect(db.db_name) as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='financials'"
            ) as cur:
                rows = await cur.fetchall()
        assert len(rows) == 1

    async def test_initialize_idempotent(self, db):
        """Calling initialize twice must not raise or duplicate tables."""
        await db.initialize()
        await db.initialize()

    # ── _process_statement label resolution ───────────────────────────────────

    def test_process_statement_prefers_standard_concept(self, db):
        """standard_concept column should override raw label."""
        raw = pd.DataFrame(
            {
                "label": ["SomeVagueLabel"],
                "standard_concept": ["Revenues"],
                "2023-12-31": [100.0],
            }
        )
        result = db._process_statement(raw, "AAPL", "income", "10-K")
        assert not result.empty
        assert result.iloc[0]["label"] == "Revenues"

    def test_process_statement_strips_xbrl_namespace(self, db):
        """Concept 'us-gaap:NetIncomeLoss' should become 'NetIncomeLoss'."""
        raw = pd.DataFrame(
            {
                "label": ["us-gaap:NetIncomeLoss"],
                "concept": ["us-gaap:NetIncomeLoss"],
                "2023-12-31": [50.0],
            }
        )
        result = db._process_statement(raw, "AAPL", "income", "10-K")
        assert not result.empty
        assert result.iloc[0]["label"] == "NetIncomeLoss"

    def test_process_statement_falls_back_to_raw_label(self, db):
        """No standard_concept or concept column → use raw label."""
        raw = pd.DataFrame(
            {
                "label": ["Total Revenue"],
                "2023-12-31": [200.0],
            }
        )
        result = db._process_statement(raw, "AAPL", "income", "10-K")
        assert not result.empty
        # Spaces → underscores in label cleaning
        assert result.iloc[0]["label"] == "Total_Revenue"

    def test_process_statement_drops_non_numeric_values(self, db):
        raw = pd.DataFrame(
            {
                "label": ["Revenues", "SomeText"],
                "2023-12-31": [100.0, "N/A"],
            }
        )
        result = db._process_statement(raw, "AAPL", "income", "10-K")
        assert not result.empty
        assert len(result) == 1

    def test_process_statement_empty_df_returns_empty(self, db):
        result = db._process_statement(pd.DataFrame(), "AAPL", "income", "10-K")
        assert result.empty

    def test_process_statement_index_as_label(self, db):
        """When label is the index (edgartools sometimes returns this)."""
        raw = pd.DataFrame(
            {"2023-12-31": [100.0]},
            index=pd.Index(["Revenues"], name="label"),
        )
        result = db._process_statement(raw, "AAPL", "income", "10-K")
        assert not result.empty
        assert "Revenues" in result["label"].values

    # ── pivot_df ──────────────────────────────────────────────────────────────

    def test_pivot_df_empty_returns_empty(self, db):
        result = db.pivot_df(pd.DataFrame())
        assert result.empty

    def test_pivot_df_shape(self, db):
        long = pd.DataFrame(
            {
                "label": ["Rev", "Rev", "Income", "Income"],
                "period_date": ["2022-12-31", "2023-12-31", "2022-12-31", "2023-12-31"],
                "value": [100.0, 120.0, 10.0, 12.0],
            }
        )
        wide = db.pivot_df(long)
        assert wide.shape == (2, 2)
        assert set(wide.index) == {"Rev", "Income"}
        assert set(wide.columns) == {"2022-12-31", "2023-12-31"}

    # ── get_all_concepts ──────────────────────────────────────────────────────

    async def test_get_all_concepts_empty_db(self, db):
        await db.initialize()
        concepts = await db.get_all_concepts("AAPL")
        assert concepts == []

    async def test_get_all_concepts_populated(self, db):
        await db.initialize()
        # Insert directly
        import aiosqlite

        async with aiosqlite.connect(db.db_name) as conn:
            await conn.executemany(
                "INSERT INTO financials VALUES (?,?,?,?,?,?)",
                [
                    ("AAPL", "2023-12-31", "10-K", "income", "Revenues", 100.0),
                    ("AAPL", "2023-12-31", "10-K", "income", "NetIncomeLoss", 10.0),
                ],
            )
            await conn.commit()
        concepts = await db.get_all_concepts("AAPL")
        assert "Revenues" in concepts
        assert "NetIncomeLoss" in concepts

    async def test_get_all_concepts_case_insensitive_ticker(self, db):
        await db.initialize()
        import aiosqlite

        async with aiosqlite.connect(db.db_name) as conn:
            await conn.execute(
                "INSERT INTO financials VALUES (?,?,?,?,?,?)",
                ("MSFT", "2023-12-31", "10-K", "income", "Revenues", 200.0),
            )
            await conn.commit()
        result_upper = await db.get_all_concepts("MSFT")
        result_lower = await db.get_all_concepts("msft")
        assert result_upper == result_lower

    # ── search_label ──────────────────────────────────────────────────────────

    async def test_search_label_empty_keywords_returns_empty_df(self, db):
        await db.initialize()
        result = await db.search_label("AAPL", [])
        assert result.empty

    async def test_search_label_single_keyword_match(self, db):
        await db.initialize()
        import aiosqlite

        async with aiosqlite.connect(db.db_name) as conn:
            await conn.executemany(
                "INSERT INTO financials VALUES (?,?,?,?,?,?)",
                [
                    ("AAPL", "2022-12-31", "10-K", "income", "Revenues", 100.0),
                    (
                        "AAPL",
                        "2022-12-31",
                        "10-K",
                        "income",
                        "OperatingIncomeLoss",
                        20.0,
                    ),
                ],
            )
            await conn.commit()
        result = await db.search_label("AAPL", ["Revenues"])
        assert "Revenues" in result.index

    async def test_search_label_date_filter(self, db):
        await db.initialize()
        import aiosqlite

        async with aiosqlite.connect(db.db_name) as conn:
            await conn.executemany(
                "INSERT INTO financials VALUES (?,?,?,?,?,?)",
                [
                    ("AAPL", "2021-12-31", "10-K", "income", "Revenues", 90.0),
                    ("AAPL", "2022-12-31", "10-K", "income", "Revenues", 100.0),
                    ("AAPL", "2023-12-31", "10-K", "income", "Revenues", 110.0),
                ],
            )
            await conn.commit()
        result = await db.search_label(
            "AAPL", ["Revenues"], start_date="2022-01-01", end_date="2022-12-31"
        )
        assert "2022-12-31" in result.columns
        assert "2021-12-31" not in result.columns
        assert "2023-12-31" not in result.columns

    # ── find_uncovered_periods ────────────────────────────────────────────────

    async def test_find_uncovered_all_missing(self, db):
        await db.initialize()
        uncovered = await db.find_uncovered_periods("AAPL", [2020, 2021, 2022], "10-K")
        assert set(uncovered) == {2020, 2021, 2022}

    async def test_find_uncovered_all_present(self, db):
        await db.initialize()
        import aiosqlite

        async with aiosqlite.connect(db.db_name) as conn:
            await conn.executemany(
                "INSERT INTO financials VALUES (?,?,?,?,?,?)",
                [
                    ("AAPL", "2022-12-31", "10-K", "income", "Revenues", 100.0),
                    ("AAPL", "2023-12-31", "10-K", "income", "Revenues", 110.0),
                ],
            )
            await conn.commit()
        uncovered = await db.find_uncovered_periods("AAPL", [2022, 2023], "10-K")
        assert uncovered == []

    async def test_find_uncovered_partial(self, db):
        await db.initialize()
        import aiosqlite

        async with aiosqlite.connect(db.db_name) as conn:
            await conn.execute(
                "INSERT INTO financials VALUES (?,?,?,?,?,?)",
                ("AAPL", "2022-12-31", "10-K", "income", "Revenues", 100.0),
            )
            await conn.commit()
        uncovered = await db.find_uncovered_periods("AAPL", [2021, 2022, 2023], "10-K")
        assert 2021 in uncovered
        assert 2023 in uncovered
        assert 2022 not in uncovered

    async def test_find_uncovered_empty_desired_returns_empty(self, db):
        await db.initialize()
        result = await db.find_uncovered_periods("AAPL", [], "10-K")
        assert result == []

    # ── _bulk_insert ──────────────────────────────────────────────────────────

    async def test_bulk_insert_and_retrieve(self, db):
        await db.initialize()
        rows = pd.DataFrame(
            [
                ("TSLA", "2023-12-31", "10-K", "income", "Revenues", 97_690_000_000.0),
                ("TSLA", "2022-12-31", "10-K", "income", "Revenues", 81_462_000_000.0),
            ],
            columns=[
                "company",
                "period_date",
                "form_type",
                "statement_type",
                "label",
                "value",
            ],
        )
        await db._bulk_insert(rows)
        concepts = await db.get_all_concepts("TSLA")
        assert "Revenues" in concepts

    async def test_bulk_insert_empty_df_is_noop(self, db):
        await db.initialize()
        await db._bulk_insert(pd.DataFrame())  # should not raise

    async def test_bulk_insert_idempotent_replace(self, db):
        await db.initialize()
        row = pd.DataFrame(
            [
                ("AAPL", "2023-12-31", "10-K", "income", "Revenues", 100.0),
            ],
            columns=[
                "company",
                "period_date",
                "form_type",
                "statement_type",
                "label",
                "value",
            ],
        )
        await db._bulk_insert(row)
        # Insert same row with different value → should overwrite (INSERT OR REPLACE)
        row["value"] = 200.0
        await db._bulk_insert(row)
        result = await db.search_label("AAPL", ["Revenues"])
        assert result.loc["Revenues", "2023-12-31"] == 200.0

    # ── get_price_data invalid interval ───────────────────────────────────────

    async def test_get_price_data_invalid_interval_raises(self, db):
        with patch("yfinance.Ticker") as mock_ticker:
            with pytest.raises(ValueError, match="interval must be"):
                await db.get_price_data("AAPL", interval="biweekly")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Agent node unit tests  (nodes isolated with monkeypatching)
# ─────────────────────────────────────────────────────────────────────────────


def _make_agent_with_mock_db(mock_db) -> FundamentalAnalysisAgent:
    """Creates a FundamentalAnalysisAgent whose self.db is replaced with mock_db."""
    with patch.object(FundamentalAnalysisAgent, "__init__", lambda self: None):
        agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
        agent.db = mock_db
        agent._graph = MagicMock()
        return agent


def _make_state(
    sample_df=None, concepts=None, tool_plan=None, tool_results=None
) -> _AgentState:
    state = _AgentState(
        ticker="AAPL",
        query="Analyse AAPL",
        vector_query="AAPL",
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
    )
    if sample_df is not None:
        state.financial_data = sample_df
    if concepts is not None:
        state.available_concepts = concepts
    if tool_plan is not None:
        state.tool_plan = tool_plan
    if tool_results is not None:
        state.tool_results = tool_results
    return state


class TestDataPrepNode:

    async def test_happy_path_populates_df_and_concepts(self, sample_df):
        mock_db = AsyncMock()
        mock_db.update_financials = AsyncMock(return_value=None)
        mock_db.get_all_concepts = AsyncMock(return_value=["Revenues", "NetIncomeLoss"])
        mock_db.search_label = AsyncMock(return_value=sample_df)
        mock_db.get_price_data = AsyncMock(return_value=pd.DataFrame())

        agent = _make_agent_with_mock_db(mock_db)
        state = _make_state()
        result = await agent._data_prep_node(state)

        assert result["financial_data"] is not None
        assert not result["financial_data"].empty
        assert "Revenues" in result["available_concepts"]

    async def test_price_data_merged_into_df(self, sample_df):
        price_df = pd.DataFrame(
            {"stock_price": [150.0, 160.0, 170.0, 180.0]},
            index=pd.to_datetime(
                ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]
            ),
        )
        mock_db = AsyncMock()
        mock_db.update_financials = AsyncMock(return_value=None)
        mock_db.get_all_concepts = AsyncMock(return_value=["Revenues"])
        mock_db.search_label = AsyncMock(return_value=sample_df)
        mock_db.get_price_data = AsyncMock(return_value=price_df)

        agent = _make_agent_with_mock_db(mock_db)
        state = _make_state()
        result = await agent._data_prep_node(state)
        assert "stock_price" in result["financial_data"].index

    async def test_empty_concepts_when_no_data_in_db(self):
        mock_db = AsyncMock()
        mock_db.update_financials = AsyncMock(return_value=None)
        mock_db.get_all_concepts = AsyncMock(return_value=[])
        mock_db.search_label = AsyncMock(return_value=pd.DataFrame())
        mock_db.get_price_data = AsyncMock(return_value=pd.DataFrame())

        agent = _make_agent_with_mock_db(mock_db)
        state = _make_state()
        result = await agent._data_prep_node(state)
        assert result["available_concepts"] == []
        assert result["financial_data"].empty

    async def test_edgar_failure_does_not_crash_node(self):
        """If update_financials raises, data_prep should propagate (not silently swallow)."""
        mock_db = AsyncMock()
        mock_db.update_financials = AsyncMock(side_effect=RuntimeError("EDGAR down"))
        mock_db.get_all_concepts = AsyncMock(return_value=[])
        mock_db.search_label = AsyncMock(return_value=pd.DataFrame())
        mock_db.get_price_data = AsyncMock(return_value=pd.DataFrame())

        agent = _make_agent_with_mock_db(mock_db)
        state = _make_state()
        with pytest.raises(RuntimeError, match="EDGAR down"):
            await agent._data_prep_node(state)


class TestToolPlannerNode:

    def _make_plan(self, calls: list) -> ToolPlan:
        return ToolPlan(
            calls=calls,
            data_summary="Test plan with CAGR tool.",
        )

    async def test_happy_path_returns_tool_plan(self, sample_df):
        mock_plan = self._make_plan(
            [
                ToolCallSpec(
                    tool_name="cagr",
                    parameters={"metric": "Revenues"},
                    reasoning="User asked for revenue growth",
                )
            ]
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_plan
        )

        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=sample_df, concepts=["Revenues", "NetIncomeLoss"])

        with patch("core.agents.fundamental_analysis_agent.service_manager") as mock_sm:
            mock_sm.get_agent.return_value = mock_llm
            result = await agent._tool_planner_node(state)

        assert result["tool_plan"].calls[0].tool_name == "cagr"

    async def test_empty_concepts_returns_empty_plan(self):
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(concepts=[])
        result = await agent._tool_planner_node(state)
        assert result["tool_plan"].calls == []
        assert "No financial data" in result["tool_plan"].data_summary

    async def test_llm_failure_returns_fallback_plan(self, sample_df):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM timeout")
        )
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=sample_df, concepts=["Revenues"])

        with patch("core.agents.fundamental_analysis_agent.service_manager") as mock_sm:
            mock_sm.get_agent.return_value = mock_llm
            result = await agent._tool_planner_node(state)

        assert result["tool_plan"].calls == []
        assert "failed" in result["tool_plan"].data_summary.lower()


class TestToolExecutorNode:

    async def test_happy_path_cagr_tool(self, sample_df):
        plan = ToolPlan(
            calls=[
                ToolCallSpec(
                    tool_name="cagr",
                    parameters={"metric": "Revenues"},
                    reasoning="testing",
                )
            ],
            data_summary="",
        )
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=sample_df, tool_plan=plan)
        result = await agent._tool_executor_node(state)

        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].success
        assert "Revenues_CAGR" in result["financial_data"].index

    async def test_unknown_tool_name_records_failure(self, sample_df):
        plan = ToolPlan(
            calls=[
                ToolCallSpec(
                    tool_name="nonexistent_tool",
                    parameters={},
                    reasoning="testing",
                )
            ],
            data_summary="",
        )
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=sample_df, tool_plan=plan)
        result = await agent._tool_executor_node(state)

        assert result["tool_results"][0].success is False
        assert "not found" in result["tool_results"][0].error.lower()

    async def test_invalid_parameters_records_failure(self, sample_df):
        plan = ToolPlan(
            calls=[
                ToolCallSpec(
                    tool_name="cagr",
                    parameters={"this_field_does_not_exist": 99},
                    reasoning="bad params",
                )
            ],
            data_summary="",
        )
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=sample_df, tool_plan=plan)
        result = await agent._tool_executor_node(state)

        assert result["tool_results"][0].success is False

    async def test_empty_financial_data_records_failure(self):
        plan = ToolPlan(
            calls=[
                ToolCallSpec(
                    tool_name="cagr",
                    parameters={"metric": "Revenues"},
                    reasoning="testing",
                )
            ],
            data_summary="",
        )
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=pd.DataFrame(), tool_plan=plan)
        result = await agent._tool_executor_node(state)
        assert result["tool_results"][0].success is False
        assert "empty" in result["tool_results"][0].error.lower()

    async def test_no_plan_returns_empty_dict(self, sample_df):
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=sample_df, tool_plan=None)
        result = await agent._tool_executor_node(state)
        assert result == {}

    async def test_second_tool_sees_rows_added_by_first(self, sample_df_with_fcf):
        """
        CAGR adds a row 'Revenues_CAGR'; the second tool (custom_formula)
        must see that row in the DataFrame.
        """
        plan = ToolPlan(
            calls=[
                ToolCallSpec(
                    tool_name="cagr",
                    parameters={"metric": "Revenues", "output_label": "rev_cagr"},
                    reasoning="step 1",
                ),
                ToolCallSpec(
                    tool_name="custom_formula",
                    parameters={
                        "metric_name": "rev_times_two",
                        "expression": "rev_cagr * 2",
                        "dependencies": ["rev_cagr"],
                    },
                    reasoning="step 2 uses step 1 output",
                ),
            ],
            data_summary="",
        )
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=sample_df_with_fcf, tool_plan=plan)
        result = await agent._tool_executor_node(state)

        assert result["tool_results"][0].success, "CAGR tool should succeed"
        assert result["tool_results"][
            1
        ].success, "custom_formula should see rev_cagr row"
        assert "rev_times_two" in result["financial_data"].index

    async def test_all_tools_fail_still_returns_original_df(self, sample_df):
        plan = ToolPlan(
            calls=[
                ToolCallSpec(tool_name="nonexistent_1", parameters={}, reasoning=""),
                ToolCallSpec(tool_name="nonexistent_2", parameters={}, reasoning=""),
            ],
            data_summary="",
        )
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=sample_df, tool_plan=plan)
        result = await agent._tool_executor_node(state)

        assert len(result["tool_results"]) == 2
        assert all(not r.success for r in result["tool_results"])
        # DataFrame unchanged
        assert set(result["financial_data"].index) == set(sample_df.index)


class TestAnalystNode:

    def _mock_extract_result(self, text="Analysis text."):
        mock = MagicMock()
        mock.analysis = text
        mock.relationships = []
        mock.parse_success = True
        return mock

    async def test_empty_df_returns_no_data_message(self):
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(sample_df=pd.DataFrame())
        result = await agent._analyst_node(state)
        assert "No financial data" in result["analysis"]
        assert result["relationships_extracted"] is False

    async def test_analyst_calls_extract_with_retry(self, sample_df):
        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(
            sample_df=sample_df,
            tool_results=[
                ToolResult(tool_name="cagr", success=True, summary="Rev CAGR 12%")
            ],
        )
        mock_result = self._mock_extract_result("Strong revenue growth detected.")

        with patch(
            "core.agents.fundamental_analysis_agent.extract_with_retry",
            AsyncMock(return_value=mock_result),
        ) as mock_extract:
            with patch(
                "core.agents.fundamental_analysis_agent.service_manager"
            ) as mock_sm:
                mock_sm.get_agent.return_value = MagicMock()
                with patch(
                    "core.agents.fundamental_analysis_agent.settings"
                ) as mock_settings:
                    mock_settings.EXTRACTION_ENABLED = False
                    result = await agent._analyst_node(state)

        assert result["analysis"] == "Strong revenue growth detected."
        assert result["relationships_extracted"] is True
        mock_extract.assert_called_once()

    async def test_analyst_prompt_contains_tool_results(self, sample_df):
        """Verify the analyst LLM prompt includes tool summaries."""
        captured_prompt = {}

        async def fake_extract(llm, messages):
            captured_prompt["messages"] = messages
            r = MagicMock()
            r.analysis = "Done."
            r.relationships = []
            r.parse_success = False
            return r

        agent = _make_agent_with_mock_db(AsyncMock())
        state = _make_state(
            sample_df=sample_df,
            tool_results=[
                ToolResult(
                    tool_name="cagr", success=True, summary="Revenue CAGR = 17%"
                ),
                ToolResult(
                    tool_name="dcf_intrinsic_value",
                    success=True,
                    summary="Enterprise Value: 3.2T",
                    reasoning="WACC=10%, TGR=2.5%",
                ),
            ],
        )

        with patch(
            "core.agents.fundamental_analysis_agent.extract_with_retry", fake_extract
        ):
            with patch("core.agents.fundamental_analysis_agent.service_manager"):
                with patch("core.agents.fundamental_analysis_agent.settings") as s:
                    s.EXTRACTION_ENABLED = False
                    await agent._analyst_node(state)

        user_content = captured_prompt["messages"][1].content
        assert "Revenue CAGR = 17%" in user_content
        assert "Enterprise Value: 3.2T" in user_content
        assert "WACC=10%" in user_content


# ─────────────────────────────────────────────────────────────────────────────
# 11.  FundamentalAnalysisOutput
# ─────────────────────────────────────────────────────────────────────────────


class TestFundamentalAnalysisOutput:

    def test_get_llm_context_str_with_no_data(self):
        out = FundamentalAnalysisOutput(
            agent_name="fundamentals_agent",
            analysis="No data.",
            financial_data=None,
        )
        ctx = out.get_llm_context_str()
        assert "No financial data" in ctx
        assert "REPORT FROM fundamentals_agent" in ctx

    def test_get_llm_context_str_with_empty_df(self):
        out = FundamentalAnalysisOutput(
            agent_name="fundamentals_agent",
            analysis="No data.",
            financial_data=pd.DataFrame(),
        )
        ctx = out.get_llm_context_str()
        assert "No financial data" in ctx

    def test_get_llm_context_str_with_data(self, sample_df):
        out = FundamentalAnalysisOutput(
            agent_name="fundamentals_agent",
            analysis="Revenue grew 60% over 4 years.",
            financial_data=sample_df,
        )
        ctx = out.get_llm_context_str()
        assert "Rows=Metrics" in ctx
        assert "Revenues" in ctx

    def test_get_llm_context_str_includes_successful_tool_result(self, sample_df):
        out = FundamentalAnalysisOutput(
            agent_name="fundamentals_agent",
            analysis="Done.",
            financial_data=sample_df,
            tool_results=[
                ToolResult(tool_name="cagr", success=True, summary="Rev CAGR 12%")
            ],
        )
        ctx = out.get_llm_context_str()
        assert "TOOL EXECUTION RESULTS" in ctx
        assert "✓" in ctx
        assert "Rev CAGR 12%" in ctx

    def test_get_llm_context_str_includes_failed_tool_error(self, sample_df):
        out = FundamentalAnalysisOutput(
            agent_name="fundamentals_agent",
            analysis="Done.",
            financial_data=sample_df,
            tool_results=[
                ToolResult(
                    tool_name="dcf_intrinsic_value",
                    success=False,
                    error="FCF metric not found.",
                )
            ],
        )
        ctx = out.get_llm_context_str()
        assert "✗" in ctx
        assert "FCF metric not found." in ctx

    def test_get_llm_context_str_includes_reasoning(self, sample_df):
        out = FundamentalAnalysisOutput(
            agent_name="fundamentals_agent",
            analysis="Done.",
            financial_data=sample_df,
            tool_results=[
                ToolResult(
                    tool_name="dcf_intrinsic_value",
                    success=True,
                    summary="EV=2T",
                    reasoning="WACC=10% due to beta=1.2",
                )
            ],
        )
        ctx = out.get_llm_context_str()
        assert "WACC=10%" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# 12.  _format_value helper
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatValue:
    def test_trillion(self):
        assert "Trillion" in _format_value(2.3e12)

    def test_billion(self):
        assert "Billion" in _format_value(1.5e9)

    def test_million(self):
        assert "Million" in _format_value(4.7e6)

    def test_thousand(self):
        assert "Thousand" in _format_value(3_500.0)

    def test_small_value(self):
        result = _format_value(42.7)
        assert "Trillion" not in result
        assert "Billion" not in result

    def test_negative_trillion(self):
        assert "Trillion" in _format_value(-1.1e12)

    def test_negative_billion(self):
        assert "Billion" in _format_value(-2.0e9)


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Full end-to-end agent pipeline (run()) — all I/O mocked
# ─────────────────────────────────────────────────────────────────────────────


def _make_full_mock_agent(sample_df: pd.DataFrame) -> FundamentalAnalysisAgent:
    """
    Builds a FundamentalAnalysisAgent where every external dependency is mocked:
      - FinancialDatabase  (no SQLite, no EDGAR, no yfinance)
      - service_manager.get_agent  (no LLM)
      - extract_with_retry  (no LLM)
    """
    mock_db = AsyncMock(spec=FinancialDatabase)
    mock_db.initialize = AsyncMock(return_value=None)
    mock_db.update_financials = AsyncMock(return_value=None)
    mock_db.get_all_concepts = AsyncMock(return_value=list(sample_df.index))
    mock_db.search_label = AsyncMock(return_value=sample_df)
    mock_db.get_price_data = AsyncMock(return_value=pd.DataFrame())

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    agent.db = mock_db

    # Build the real graph (nodes call self.db which is mocked)
    agent._graph = agent._build_graph()
    return agent


class TestFullAgentPipeline:

    @pytest.fixture
    def full_agent(self, sample_df_with_fcf):
        return _make_full_mock_agent(sample_df_with_fcf), sample_df_with_fcf

    async def test_run_returns_fundamental_analysis_output(self, full_agent):
        agent, sample_df = full_agent
        plan = ToolPlan(
            calls=[
                ToolCallSpec(
                    tool_name="cagr",
                    parameters={"metric": "Revenues"},
                    reasoning="User wants CAGR",
                )
            ],
            data_summary="Will compute CAGR.",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=plan
        )

        extract_mock_result = MagicMock()
        extract_mock_result.analysis = "AAPL shows strong revenue growth."
        extract_mock_result.relationships = []
        extract_mock_result.parse_success = True

        with patch("core.agents.fundamental_analysis_agent.service_manager") as mock_sm:
            mock_sm.get_agent.return_value = mock_llm
            with patch(
                "core.agents.fundamental_analysis_agent.extract_with_retry",
                AsyncMock(return_value=extract_mock_result),
            ):
                with patch("core.agents.fundamental_analysis_agent.settings") as s:
                    s.EXTRACTION_ENABLED = False
                    output = await agent.run(
                        BaseAgentInput(
                            ticker="AAPL",
                            query="Analyse AAPL revenue CAGR",
                            vector_query="AAPL",
                            start_date=datetime(2020, 1, 1),
                            end_date=datetime(2023, 12, 31),
                        )
                    )

        assert isinstance(output, FundamentalAnalysisOutput)
        assert output.financial_data is not None
        assert not output.financial_data.empty
        assert output.analysis == "AAPL shows strong revenue growth."
        assert len(output.tool_results) == 1
        assert output.tool_results[0].tool_name == "cagr"
        assert output.tool_results[0].success

    async def test_run_with_raw_data_only_no_tools(self, full_agent):
        """Planner returning empty calls list → no tools run, raw data returned."""
        agent, _ = full_agent
        empty_plan = ToolPlan(calls=[], data_summary="Raw data only.")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=empty_plan
        )

        extract_mock = MagicMock()
        extract_mock.analysis = "Raw data presented."
        extract_mock.relationships = []
        extract_mock.parse_success = False

        with patch("core.agents.fundamental_analysis_agent.service_manager") as mock_sm:
            mock_sm.get_agent.return_value = mock_llm
            with patch(
                "core.agents.fundamental_analysis_agent.extract_with_retry",
                AsyncMock(return_value=extract_mock),
            ):
                with patch("core.agents.fundamental_analysis_agent.settings") as s:
                    s.EXTRACTION_ENABLED = False
                    output = await agent.run(
                        BaseAgentInput(
                            ticker="AAPL",
                            query="Show me raw financials",
                            vector_query="AAPL",
                            start_date=datetime(2020, 1, 1),
                            end_date=datetime(2023, 12, 31),
                        )
                    )

        assert isinstance(output, FundamentalAnalysisOutput)
        assert output.tool_results == []

    async def test_run_with_no_edgar_data_returns_no_data_analysis(self):
        """When the DB has no data, analyst should return a no-data message without crashing."""
        empty_df = pd.DataFrame()
        mock_db = AsyncMock(spec=FinancialDatabase)
        mock_db.initialize = AsyncMock(return_value=None)
        mock_db.update_financials = AsyncMock(return_value=None)
        mock_db.get_all_concepts = AsyncMock(return_value=[])
        mock_db.search_label = AsyncMock(return_value=empty_df)
        mock_db.get_price_data = AsyncMock(return_value=pd.DataFrame())

        agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
        agent.db = mock_db
        agent._graph = agent._build_graph()

        mock_llm = MagicMock()
        empty_plan = ToolPlan(calls=[], data_summary="No data.")
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=empty_plan
        )

        with patch("core.agents.fundamental_analysis_agent.service_manager") as mock_sm:
            mock_sm.get_agent.return_value = mock_llm
            with patch("core.agents.fundamental_analysis_agent.settings") as s:
                s.EXTRACTION_ENABLED = False
                output = await agent.run(
                    BaseAgentInput(
                        ticker="XXXX",
                        query="Analyse unknown ticker",
                        vector_query="XXXX",
                        start_date=datetime(2020, 1, 1),
                        end_date=datetime(2023, 12, 31),
                    )
                )

        assert isinstance(output, FundamentalAnalysisOutput)
        assert "No financial data" in output.analysis
        assert output.financial_data is None or output.financial_data.empty

    async def test_run_with_dcf_and_cagr_tools(self, sample_df_with_fcf):
        """Multi-tool plan: CAGR first, then DCF."""
        agent = _make_full_mock_agent(sample_df_with_fcf)
        plan = ToolPlan(
            calls=[
                ToolCallSpec(
                    tool_name="cagr",
                    parameters={"metric": "Revenues"},
                    reasoning="revenue growth",
                ),
                ToolCallSpec(
                    tool_name="dcf_intrinsic_value",
                    parameters={
                        "fcf_metric": "FreeCashFlow",
                        "wacc": 0.10,
                        "terminal_growth_rate": 0.025,
                        "projection_years": 5,
                        "shares_outstanding_metric": "CommonStockSharesOutstanding",
                        "wacc_reasoning": "Beta 1.1, RF 4.5%",
                        "terminal_growth_reasoning": "Mature, GDP+0.5%",
                    },
                    reasoning="DCF valuation",
                ),
            ],
            data_summary="CAGR then DCF.",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=plan
        )

        extract_mock = MagicMock()
        extract_mock.analysis = "CAGR strong. DCF suggests undervalued."
        extract_mock.relationships = []
        extract_mock.parse_success = True

        with patch("core.agents.fundamental_analysis_agent.service_manager") as mock_sm:
            mock_sm.get_agent.return_value = mock_llm
            with patch(
                "core.agents.fundamental_analysis_agent.extract_with_retry",
                AsyncMock(return_value=extract_mock),
            ):
                with patch("core.agents.fundamental_analysis_agent.settings") as s:
                    s.EXTRACTION_ENABLED = False
                    output = await agent.run(
                        BaseAgentInput(
                            ticker="AAPL",
                            query="Revenue CAGR and DCF valuation",
                            vector_query="AAPL DCF",
                            start_date=datetime(2020, 1, 1),
                            end_date=datetime(2023, 12, 31),
                        )
                    )

        assert len(output.tool_results) == 2
        assert output.tool_results[0].tool_name == "cagr"
        assert output.tool_results[1].tool_name == "dcf_intrinsic_value"
        assert all(r.success for r in output.tool_results)

    async def test_run_llm_planner_failure_still_produces_raw_output(
        self, sample_df_with_fcf
    ):
        """If the planner LLM fails, the agent should still return raw data + no-tools analysis."""
        agent = _make_full_mock_agent(sample_df_with_fcf)

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM rate limit")
        )

        extract_mock = MagicMock()
        extract_mock.analysis = "Data presented without derived metrics."
        extract_mock.relationships = []
        extract_mock.parse_success = False

        with patch("core.agents.fundamental_analysis_agent.service_manager") as mock_sm:
            mock_sm.get_agent.return_value = mock_llm
            with patch(
                "core.agents.fundamental_analysis_agent.extract_with_retry",
                AsyncMock(return_value=extract_mock),
            ):
                with patch("core.agents.fundamental_analysis_agent.settings") as s:
                    s.EXTRACTION_ENABLED = False
                    output = await agent.run(
                        BaseAgentInput(
                            ticker="AAPL",
                            query="Analyse AAPL",
                            vector_query="AAPL",
                            start_date=datetime(2020, 1, 1),
                            end_date=datetime(2023, 12, 31),
                        )
                    )

        assert isinstance(output, FundamentalAnalysisOutput)
        # Plan failed → no tools ran → tool_results empty
        assert output.tool_results == []
        # But we still have raw data
        assert output.financial_data is not None
        assert not output.financial_data.empty
