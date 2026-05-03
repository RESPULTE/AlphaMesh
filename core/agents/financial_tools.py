"""
core/agents/financial_tools.py

Financial analysis tool registry for the FundamentalAnalysisAgent.

Architecture
------------
  FinancialAnalyticsAdapter (ABC)
      Swappable computation backend. Two implementations provided:
      - NumpyFinancialAdapter  (default, no API key needed)
      - FinanceToolkitAdapter  (stub, swap in when FMP key available)

  FinancialTool (ABC)
      Base class every tool inherits from. Each tool exposes:
      - name          : str
      - description   : str  (shown verbatim to the planner LLM)
      - parameters_schema : Type[BaseModel]
      - execute(df, params) -> ToolResult

  TOOL_REGISTRY : Dict[str, FinancialTool]
      Central lookup used by the agent graph.  Add new tools here.

Tools
-----
  cagr                 — Compound Annual Growth Rate for any metric
  valuation_multiples_snapshot - P/S, P/E, EV/EBITDA, P/FCF + trend rows
  profitability_ratios — Gross / operating / net margin
  debt_solvency        — Debt-to-equity, interest coverage
  liquidity            — Current ratio, quick ratio
  custom_formula       — User / LLM defined metric via pandas-safe expression
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Shared Result Model
# ─────────────────────────────────────────────────────────────────────────────


class ToolResult(BaseModel):
    """Structured output returned by every tool.execute() call."""

    tool_name: str
    success: bool = True
    error: Optional[str] = None

    # Numeric outputs
    scalar_value: Optional[float] = None
    series_values: Optional[Dict[str, float]] = None

    # Rows to merge back into the agent's financial DataFrame.
    # Shape: { row_label -> { date_col -> value } }
    added_rows: Optional[Dict[str, Dict[str, float]]] = None

    # Human-readable justification for tool-specific assumption traceability
    reasoning: Optional[str] = None

    # One-line plain-English summary of results
    summary: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Adapter Layer  (swap backend here without touching any tool logic)
# ─────────────────────────────────────────────────────────────────────────────


class FinancialAnalyticsAdapter(ABC):
    """Abstract backend for all numerical computations."""

    @abstractmethod
    def cagr(self, start_value: float, end_value: float, n_periods: float) -> float: ...

    @abstractmethod
    def gross_margin(self, revenue: float, cogs: float) -> float: ...

    @abstractmethod
    def operating_margin(self, operating_income: float, revenue: float) -> float: ...

    @abstractmethod
    def net_margin(self, net_income: float, revenue: float) -> float: ...

    @abstractmethod
    def debt_to_equity(self, total_debt: float, total_equity: float) -> float: ...

    @abstractmethod
    def interest_coverage(self, ebit: float, interest_expense: float) -> float: ...

    @abstractmethod
    def current_ratio(
        self, current_assets: float, current_liabilities: float
    ) -> float: ...

    @abstractmethod
    def quick_ratio(
        self, current_assets: float, inventory: float, current_liabilities: float
    ) -> float: ...


class NumpyFinancialAdapter(FinancialAnalyticsAdapter):
    """
    Default adapter with pure-Python formulas for core metrics.
    No external API key required.
    """

    def _safe_div(self, numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else float("nan")

    def cagr(self, start_value: float, end_value: float, n_periods: float) -> float:
        if start_value <= 0 or n_periods <= 0:
            raise ValueError("start_value and n_periods must be positive for CAGR")
        return (end_value / start_value) ** (1.0 / n_periods) - 1.0

    def gross_margin(self, revenue: float, cogs: float) -> float:
        return self._safe_div(revenue - cogs, revenue)

    def operating_margin(self, operating_income: float, revenue: float) -> float:
        return self._safe_div(operating_income, revenue)

    def net_margin(self, net_income: float, revenue: float) -> float:
        return self._safe_div(net_income, revenue)

    def debt_to_equity(self, total_debt: float, total_equity: float) -> float:
        return self._safe_div(total_debt, total_equity)

    def interest_coverage(self, ebit: float, interest_expense: float) -> float:
        return self._safe_div(ebit, abs(interest_expense))

    def current_ratio(self, current_assets: float, current_liabilities: float) -> float:
        return self._safe_div(current_assets, current_liabilities)

    def quick_ratio(
        self, current_assets: float, inventory: float, current_liabilities: float
    ) -> float:
        return self._safe_div(current_assets - inventory, current_liabilities)


class FinanceToolkitAdapter(FinancialAnalyticsAdapter):
    """
    Future adapter wrapping the FinanceToolkit library (needs FMP API key).
    All methods raise NotImplementedError until wired in.
    To activate: set_adapter(FinanceToolkitAdapter(api_key="..."))
    """

    def __init__(self, api_key: Optional[str] = None):
        try:
            import financetoolkit  # noqa: F401
        except ImportError:
            raise ImportError(
                "financetoolkit not installed. Run: pip install financetoolkit"
            )
        self._api_key = api_key

    def _not_implemented(self) -> None:
        raise NotImplementedError(
            "FinanceToolkitAdapter is a migration stub. "
            "Implement the method before activating this adapter."
        )

    def cagr(self, *a, **kw) -> float:
        self._not_implemented()  # type: ignore

    def gross_margin(self, *a, **kw) -> float:
        self._not_implemented()  # type: ignore

    def operating_margin(self, *a, **kw) -> float:
        self._not_implemented()  # type: ignore

    def net_margin(self, *a, **kw) -> float:
        self._not_implemented()  # type: ignore

    def debt_to_equity(self, *a, **kw) -> float:
        self._not_implemented()  # type: ignore

    def interest_coverage(self, *a, **kw) -> float:
        self._not_implemented()  # type: ignore

    def current_ratio(self, *a, **kw) -> float:
        self._not_implemented()  # type: ignore

    def quick_ratio(self, *a, **kw) -> float:
        self._not_implemented()  # type: ignore


# Global adapter instance — swap via set_adapter() at startup if needed
_ADAPTER: FinancialAnalyticsAdapter = NumpyFinancialAdapter()


def set_adapter(adapter: FinancialAnalyticsAdapter) -> None:
    """Hot-swap the analytics backend. Call once before the agent starts."""
    global _ADAPTER
    _ADAPTER = adapter


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Tool Base Class
# ─────────────────────────────────────────────────────────────────────────────


class FinancialTool(ABC):
    name: str
    description: str  # shown verbatim in the planner LLM prompt
    parameters_schema: Type[BaseModel]

    @abstractmethod
    def execute(self, df: pd.DataFrame, params: BaseModel) -> ToolResult: ...


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Individual Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────


# ── CAGR ─────────────────────────────────────────────────────────────────────


class CAGRParams(BaseModel):
    metric: str = Field(
        description="Exact row label in the DataFrame to compute CAGR on (e.g. 'Revenues')."
    )
    output_label: Optional[str] = Field(
        default=None,
        description="Label for the new summary row added to the DataFrame (defaults to '<metric>_CAGR').",
    )


class CAGRTool(FinancialTool):
    name = "cagr"
    description = (
        "Computes the Compound Annual Growth Rate (CAGR) of any named metric "
        "over the full date range present in the data. "
        "Adds a constant summary row to the DataFrame for LLM context."
    )
    parameters_schema = CAGRParams

    def execute(self, df: pd.DataFrame, params: CAGRParams) -> ToolResult:  # type: ignore[override]
        try:
            if params.metric not in df.index:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Metric '{params.metric}' not found. Available: {list(df.index[:10])}…",
                )
            row = df.loc[params.metric].dropna()
            if len(row) < 2:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Need ≥2 non-null data points for CAGR. Got {len(row)}.",
                )
            sorted_row = row.sort_index()
            start_val = float(sorted_row.iloc[0])
            end_val = float(sorted_row.iloc[-1])

            try:
                dates = pd.to_datetime(sorted_row.index)
                n_periods = (dates[-1] - dates[0]).days / 365.25
            except Exception:
                n_periods = float(len(sorted_row) - 1)

            if n_periods <= 0:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="Invalid time range for CAGR.",
                )

            cagr_val = _ADAPTER.cagr(start_val, end_val, n_periods)
            output_label = params.output_label or f"{params.metric}_CAGR"
            summary = (
                f"{params.metric} CAGR over {n_periods:.1f} yrs: {cagr_val:+.2%} "
                f"({start_val:,.0f} → {end_val:,.0f})"
            )
            logger.info("[CAGRTool] %s", summary)

            # CAGR is a single summary scalar — place it only in the terminal
            # (most recent) date column so the table doesn't show the same
            # value repeated identically across every period.
            last_col = sorted_row.index[-1]
            added_row = {
                col: cagr_val if col == last_col else float("nan") for col in df.columns
            }

            return ToolResult(
                tool_name=self.name,
                success=True,
                scalar_value=cagr_val,
                added_rows={output_label: added_row},
                summary=summary,
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# -- Valuation Multiples Snapshot -------------------------------------------------


class ValuationMultiplesSnapshotParams(BaseModel):
    market_cap_metric: str = Field(
        description="Exact row label for market capitalization (e.g. 'market_cap')."
    )
    enterprise_value_metric: str = Field(
        description="Exact row label for enterprise value (e.g. 'enterprise_value')."
    )
    revenue_metric: str = Field(
        description="Exact row label for revenue (e.g. 'Revenues')."
    )
    ebitda_metric: str = Field(
        description="Exact row label for EBITDA (e.g. 'EBITDA')."
    )
    net_income_metric: str = Field(
        description="Exact row label for net income (e.g. 'NetIncomeLoss')."
    )
    fcf_metric: str = Field(
        description="Exact row label for free cash flow (e.g. 'FreeCashFlow')."
    )


class ValuationMultiplesSnapshotTool(FinancialTool):
    name = "valuation_multiples_snapshot"
    description = (
        "Computes beginner-friendly valuation multiples across available periods: "
        "P/S, P/E, EV/EBITDA, and P/FCF. "
        "Also adds '<multiple>_pct_change' trend rows to show direction and momentum."
    )
    parameters_schema = ValuationMultiplesSnapshotParams

    def execute(  # type: ignore[override]
        self, df: pd.DataFrame, params: ValuationMultiplesSnapshotParams
    ) -> ToolResult:
        required_metrics = [
            params.market_cap_metric,
            params.enterprise_value_metric,
            params.revenue_metric,
            params.ebitda_metric,
            params.net_income_metric,
            params.fcf_metric,
        ]
        missing = [label for label in required_metrics if label not in df.index]
        if missing:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Required metrics not found: {missing}",
            )

        try:
            sorted_cols = sorted(df.columns, key=lambda c: pd.Timestamp(c))
        except Exception:
            sorted_cols = list(df.columns)

        def _ratio_row(numerator_metric: str, denominator_metric: str) -> Dict[str, float]:
            numerator = pd.to_numeric(df.loc[numerator_metric, sorted_cols], errors="coerce")
            denominator = pd.to_numeric(df.loc[denominator_metric, sorted_cols], errors="coerce")
            output: Dict[str, float] = {}
            for col in sorted_cols:
                n_val = numerator.get(col)
                d_val = denominator.get(col)
                if pd.isna(n_val) or pd.isna(d_val) or float(d_val) <= 0:
                    continue
                output[col] = float(n_val) / float(d_val)
            return output

        base_rows: Dict[str, Dict[str, float]] = {
            "price_to_sales": _ratio_row(params.market_cap_metric, params.revenue_metric),
            "price_to_earnings": _ratio_row(params.market_cap_metric, params.net_income_metric),
            "ev_to_ebitda": _ratio_row(
                params.enterprise_value_metric, params.ebitda_metric
            ),
            "price_to_fcf": _ratio_row(params.market_cap_metric, params.fcf_metric),
        }
        base_rows = {k: v for k, v in base_rows.items() if v}

        if not base_rows:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "No valuation multiples could be computed. Ensure numerator values are "
                    "numeric and denominator metrics are positive."
                ),
            )

        added_rows: Dict[str, Dict[str, float]] = dict(base_rows)
        summary_lines: List[str] = []

        for row_label, row_values in base_rows.items():
            aligned = pd.Series(
                {col: row_values.get(col, np.nan) for col in sorted_cols}, dtype=float
            )
            trend_series = aligned.pct_change(fill_method=None)
            trend_row = {
                col: float(val)
                for col, val in trend_series.items()
                if pd.notna(val) and not np.isinf(val)
            }
            if trend_row:
                added_rows[f"{row_label}_pct_change"] = trend_row

            latest_col = None
            latest_val = None
            for col in reversed(sorted_cols):
                value = row_values.get(col)
                if value is not None and pd.notna(value):
                    latest_col = col
                    latest_val = float(value)
                    break

            latest_trend = trend_row.get(latest_col) if latest_col is not None else None
            trend_text = (
                f"{latest_trend:+.1%}"
                if latest_trend is not None and pd.notna(latest_trend)
                else "n/a"
            )
            if latest_col is not None and latest_val is not None:
                summary_lines.append(
                    f"{row_label} (latest {str(latest_col)[:7]}): {latest_val:.2f}x | trend {trend_text}"
                )

        logger.info(
            "[ValuationMultiplesSnapshotTool] Computed rows: %s",
            list(added_rows.keys()),
        )
        return ToolResult(
            tool_name=self.name,
            success=True,
            added_rows=added_rows,
            summary="\n".join(summary_lines),
        )


class ProfitabilityParams(BaseModel):
    revenue_metric: str = Field(
        description="Exact row label for revenue / net sales (e.g. 'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax')."
    )
    cogs_metric: Optional[str] = Field(
        default=None,
        description="Row label for Cost of Goods Sold (for gross margin).",
    )
    operating_income_metric: Optional[str] = Field(
        default=None,
        description="Row label for operating income / operating profit.",
    )
    net_income_metric: Optional[str] = Field(
        default=None,
        description="Row label for net income / net earnings.",
    )


class ProfitabilityTool(FinancialTool):
    name = "profitability_ratios"
    description = (
        "Computes profitability ratios across all available periods: "
        "gross margin (if COGS available), operating margin, and net profit margin. "
        "Adds one row per computed ratio to the DataFrame."
    )
    parameters_schema = ProfitabilityParams

    def execute(self, df: pd.DataFrame, params: ProfitabilityParams) -> ToolResult:  # type: ignore[override]
        added_rows: Dict[str, Dict[str, float]] = {}
        summaries: List[str] = []

        def _get(metric: Optional[str]) -> Optional[pd.Series]:
            return df.loc[metric] if metric and metric in df.index else None

        rev = _get(params.revenue_metric)
        if rev is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Revenue metric '{params.revenue_metric}' not found.",
            )

        def _ratio_series(numerator_row: pd.Series, fn) -> Dict[str, float]:
            result: Dict[str, float] = {}
            for col in df.columns:
                r = rev.get(col)
                n = numerator_row.get(col)
                if pd.notna(r) and pd.notna(n) and r != 0:
                    result[col] = fn(float(n), float(r))
            return result

        cogs = _get(params.cogs_metric)
        if cogs is not None:
            gm = {
                col: _ADAPTER.gross_margin(float(rev[col]), float(cogs[col]))
                for col in df.columns
                if pd.notna(rev.get(col)) and pd.notna(cogs.get(col))
            }
            if gm:
                added_rows["gross_margin"] = gm
                latest = sorted(gm.items())[-1]
                summaries.append(
                    f"Gross Margin (latest {latest[0][:4]}): {latest[1]:.2%}"
                )

        op_inc = _get(params.operating_income_metric)
        if op_inc is not None:
            om = _ratio_series(op_inc, _ADAPTER.operating_margin)
            if om:
                added_rows["operating_margin"] = om
                latest = sorted(om.items())[-1]
                summaries.append(
                    f"Operating Margin (latest {latest[0][:4]}): {latest[1]:.2%}"
                )

        net_inc = _get(params.net_income_metric)
        if net_inc is not None:
            nm = _ratio_series(net_inc, _ADAPTER.net_margin)
            if nm:
                added_rows["net_margin"] = nm
                latest = sorted(nm.items())[-1]
                summaries.append(
                    f"Net Margin (latest {latest[0][:4]}): {latest[1]:.2%}"
                )

        if not added_rows:
            return ToolResult(
                tool_name=self.name,
                success=True,
                summary="No profitability ratios could be computed — required metrics were missing.",
            )

        logger.info("[ProfitabilityTool] Computed: %s", list(added_rows.keys()))
        return ToolResult(
            tool_name=self.name,
            success=True,
            added_rows=added_rows,
            summary="\n".join(summaries),
        )


# ── Debt & Solvency ───────────────────────────────────────────────────────────


class DebtSolvencyParams(BaseModel):
    total_debt_metric: str = Field(
        description="Row label for total debt (e.g. 'LongTermDebt', 'DebtCurrent', 'LongTermDebtAndCapitalLeaseObligations')."
    )
    total_equity_metric: str = Field(
        description="Row label for total stockholders' equity (e.g. 'StockholdersEquity')."
    )
    ebit_metric: Optional[str] = Field(
        default=None,
        description="Row label for EBIT / operating income. Required for interest coverage ratio.",
    )
    interest_expense_metric: Optional[str] = Field(
        default=None,
        description="Row label for interest expense. Required for interest coverage ratio.",
    )


class DebtSolvencyTool(FinancialTool):
    name = "debt_solvency"
    description = (
        "Computes debt-to-equity ratio across all periods, and optionally the interest coverage "
        "ratio if EBIT and interest expense metrics are available. "
        "Adds computed rows to the DataFrame."
    )
    parameters_schema = DebtSolvencyParams

    def execute(self, df: pd.DataFrame, params: DebtSolvencyParams) -> ToolResult:  # type: ignore[override]
        added_rows: Dict[str, Dict[str, float]] = {}
        summaries: List[str] = []

        def _get(m: Optional[str]) -> Optional[pd.Series]:
            return df.loc[m] if m and m in df.index else None

        debt = _get(params.total_debt_metric)
        equity = _get(params.total_equity_metric)

        if debt is not None and equity is not None:
            dte = {
                col: _ADAPTER.debt_to_equity(float(debt[col]), float(equity[col]))
                for col in df.columns
                if pd.notna(debt.get(col)) and pd.notna(equity.get(col))
            }
            if dte:
                added_rows["debt_to_equity"] = dte
                latest = sorted(dte.items())[-1]
                summaries.append(
                    f"D/E Ratio (latest {latest[0][:4]}): {latest[1]:.2f}x"
                )
        else:
            missing = []
            if debt is None:
                missing.append(params.total_debt_metric)
            if equity is None:
                missing.append(params.total_equity_metric)
            summaries.append(f"D/E skipped — missing: {missing}")

        ebit = _get(params.ebit_metric)
        interest = _get(params.interest_expense_metric)
        if ebit is not None and interest is not None:
            ic = {
                col: _ADAPTER.interest_coverage(float(ebit[col]), float(interest[col]))
                for col in df.columns
                if pd.notna(ebit.get(col)) and pd.notna(interest.get(col))
            }
            if ic:
                added_rows["interest_coverage"] = ic
                latest = sorted(ic.items())[-1]
                summaries.append(
                    f"Interest Coverage (latest {latest[0][:4]}): {latest[1]:.2f}x"
                )

        logger.info("[DebtSolvencyTool] Computed: %s", list(added_rows.keys()))
        if not added_rows:
            return ToolResult(
                tool_name=self.name,
                success=False,  # ← change True → False so the planner sees a real failure
                summary="\n".join(summaries)
                or "No solvency ratios could be computed — check metric names.",
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            added_rows=added_rows if added_rows else None,
            summary="\n".join(summaries) or "No solvency ratios could be computed.",
        )


# ── Liquidity ─────────────────────────────────────────────────────────────────


class LiquidityParams(BaseModel):
    current_assets_metric: str = Field(
        description="Row label for current assets (e.g. 'AssetsCurrent')."
    )
    current_liabilities_metric: str = Field(
        description="Row label for current liabilities (e.g. 'LiabilitiesCurrent')."
    )
    inventory_metric: Optional[str] = Field(
        default=None,
        description="Row label for inventory. Required for quick ratio. If absent, only current ratio is computed.",
    )


class LiquidityTool(FinancialTool):
    name = "liquidity"
    description = (
        "Computes current ratio and (if inventory is available) quick ratio across all periods. "
        "Adds computed rows to the DataFrame."
    )
    parameters_schema = LiquidityParams

    def execute(self, df: pd.DataFrame, params: LiquidityParams) -> ToolResult:  # type: ignore[override]
        added_rows: Dict[str, Dict[str, float]] = {}
        summaries: List[str] = []

        def _get(m: Optional[str]) -> Optional[pd.Series]:
            return df.loc[m] if m and m in df.index else None

        ca = _get(params.current_assets_metric)
        cl = _get(params.current_liabilities_metric)

        if ca is None or cl is None:
            missing = []
            if ca is None:
                missing.append(params.current_assets_metric)
            if cl is None:
                missing.append(params.current_liabilities_metric)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Required metrics not found: {missing}",
            )

        cr = {
            col: _ADAPTER.current_ratio(float(ca[col]), float(cl[col]))
            for col in df.columns
            if pd.notna(ca.get(col)) and pd.notna(cl.get(col))
        }
        if cr:
            added_rows["current_ratio"] = cr
            latest = sorted(cr.items())[-1]
            summaries.append(
                f"Current Ratio (latest {latest[0][:4]}): {latest[1]:.2f}x"
            )

        inv = _get(params.inventory_metric)
        if inv is not None:
            qr = {
                col: _ADAPTER.quick_ratio(
                    float(ca[col]), float(inv[col]), float(cl[col])
                )
                for col in df.columns
                if pd.notna(ca.get(col))
                and pd.notna(inv.get(col))
                and pd.notna(cl.get(col))
            }
            if qr:
                added_rows["quick_ratio"] = qr
                latest = sorted(qr.items())[-1]
                summaries.append(
                    f"Quick Ratio (latest {latest[0][:4]}): {latest[1]:.2f}x"
                )

        logger.info("[LiquidityTool] Computed: %s", list(added_rows.keys()))
        return ToolResult(
            tool_name=self.name,
            success=True,
            added_rows=added_rows,
            summary="\n".join(summaries),
        )


# ── Custom / User-Defined Formula ─────────────────────────────────────────────


class CustomFormulaParams(BaseModel):
    metric_name: str = Field(
        description="Name for the new computed metric row (e.g. 'price_to_fcf', 'net_debt')."
    )
    expression: str = Field(
        description=(
            "pandas.eval()-compatible mathematical expression using existing DataFrame row labels "
            "as variable names. Example: 'NetIncomeLoss / Revenues'. "
            "Spaces in row names must be replaced with underscores in the expression."
        )
    )
    dependencies: List[str] = Field(
        description="List of existing DataFrame row labels referenced in the expression."
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this metric represents.",
    )


class CustomFormulaTool(FinancialTool):
    name = "custom_formula"
    description = (
        "Computes a user-defined or LLM-derived metric using a pandas-safe mathematical expression. "
        "Row labels from the DataFrame are used as variable names. "
        "Use this for any ratio or metric NOT covered by the other tools, "
        "or when the user specifies their own formula."
    )
    parameters_schema = CustomFormulaParams

    # Only allow safe arithmetic characters — block builtins / attribute access
    _SAFE_PATTERN = re.compile(r"^[\w\s\+\-\*/\(\)\.\,]+$")

    def execute(self, df: pd.DataFrame, params: CustomFormulaParams) -> ToolResult:  # type: ignore[override]
        try:
            metric_name = str(params.metric_name or "").strip()
            dependencies = [
                str(dep).strip()
                for dep in (params.dependencies or [])
                if str(dep).strip()
            ]
            if not metric_name:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="metric_name cannot be empty.",
                )
            if not dependencies:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="dependencies must include at least one metric label.",
                )

            if not self._SAFE_PATTERN.match(params.expression):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=(
                        f"Expression '{params.expression}' contains unsafe characters. "
                        "Only alphanumeric identifiers and arithmetic operators are allowed."
                    ),
                )

            missing = [d for d in dependencies if d not in df.index]
            if missing:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Missing dependencies: {missing}. Available: {list(df.index[:10])}…",
                )

            # Transpose so rows become columns for pandas.eval()
            df_t = df.loc[dependencies].T.copy()
            df_t.eval(f"{metric_name} = {params.expression}", inplace=True)
            result_series = df_t[metric_name]
            non_null_result = result_series.dropna()

            if non_null_result.empty:
                logger.warning(
                    "[CustomFormulaTool] Computed '%s' but produced no non-null values. "
                    "dependencies=%s expression=%s",
                    metric_name,
                    dependencies,
                    params.expression,
                )

            added_rows = {metric_name: non_null_result.to_dict()}
            summary = (
                f"Computed '{metric_name}' = {params.expression}. "
                f"{params.description}"
            ).strip()

            logger.info("[CustomFormulaTool] %s", summary)
            return ToolResult(
                tool_name=self.name,
                success=True,
                added_rows=added_rows,
                summary=summary,
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


# ── Period-over-Period Change ─────────────────────────────────────────────────


class PeriodOverPeriodParams(BaseModel):
    metrics: List[str] = Field(
        description=(
            "One or more exact row labels to compute period-over-period changes for "
            "(e.g. ['Revenues', 'NetIncomeLoss', 'gross_margin']). "
            "Accepts both raw EDGAR concepts and previously derived rows."
        )
    )
    absolute_change: bool = Field(
        default=True,
        description=(
            "If True, add a '<metric>_change' row with the raw numeric difference "
            "between each period and the prior one."
        ),
    )
    pct_change: bool = Field(
        default=True,
        description=(
            "If True, add a '<metric>_pct_change' row with the percentage change "
            "(expressed as a decimal, e.g. 0.12 = +12%) between consecutive periods."
        ),
    )


class PeriodOverPeriodTool(FinancialTool):
    name = "period_over_period"
    description = (
        "Computes period-over-period absolute change and/or percentage change for one or "
        "more metrics across all available periods (works for both annual and quarterly data). "
        "Produces '<metric>_change' rows (raw difference) and '<metric>_pct_change' rows "
        "(decimal percentage) which the analyst can reference directly. "
        "Use this whenever the user asks about growth, acceleration, deceleration, "
        "improvement, deterioration, or trend direction for any financial line item."
    )
    parameters_schema = PeriodOverPeriodParams

    def execute(  # type: ignore[override]
        self, df: pd.DataFrame, params: PeriodOverPeriodParams
    ) -> ToolResult:
        if not params.absolute_change and not params.pct_change:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="At least one of absolute_change or pct_change must be True.",
            )

        missing = [m for m in params.metrics if m not in df.index]
        if missing:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Metrics not found in DataFrame: {missing}. "
                f"Available (first 10): {list(df.index[:10])}",
            )

        # Sort columns chronologically so diff() moves forward in time
        try:
            sorted_cols = sorted(df.columns, key=lambda c: pd.Timestamp(c))
        except Exception:
            sorted_cols = list(df.columns)

        added_rows: Dict[str, Dict[str, float]] = {}
        summaries: List[str] = []

        for metric in params.metrics:
            series = df.loc[metric, sorted_cols]
            numeric = pd.to_numeric(series, errors="coerce")

            if params.absolute_change:
                abs_diff = numeric.diff()  # NaN for the first period — intentional
                row_label = f"{metric}_change"
                added_rows[row_label] = {
                    col: val for col, val in abs_diff.items() if pd.notna(val)
                }
                if added_rows[row_label]:
                    latest_col = sorted_cols[-1]
                    latest_val = abs_diff.get(latest_col)
                    if pd.notna(latest_val):
                        direction = "▲" if latest_val >= 0 else "▼"
                        summaries.append(
                            f"{metric} absolute change "
                            f"(latest {str(latest_col)[:7]}): "
                            f"{direction} {latest_val:+,.2f}"
                        )

            if params.pct_change:
                pct = numeric.pct_change()  # NaN for first period — intentional
                row_label = f"{metric}_pct_change"
                added_rows[row_label] = {
                    col: val
                    for col, val in pct.items()
                    if pd.notna(val) and not np.isinf(val)
                }
                if added_rows[row_label]:
                    latest_col = sorted_cols[-1]
                    latest_val = pct.get(latest_col)
                    if pd.notna(latest_val) and not np.isinf(latest_val):
                        direction = "▲" if latest_val >= 0 else "▼"
                        summaries.append(
                            f"{metric} % change "
                            f"(latest {str(latest_col)[:7]}): "
                            f"{direction} {latest_val:+.1%}"
                        )

        # Drop any empty rows (metric had no valid consecutive pairs)
        added_rows = {k: v for k, v in added_rows.items() if v}

        if not added_rows:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "No period-over-period values could be computed. "
                    "The DataFrame may have fewer than 2 periods or all values are NaN."
                ),
            )

        logger.info("[PeriodOverPeriodTool] Computed rows: %s", list(added_rows.keys()))
        return ToolResult(
            tool_name=self.name,
            success=True,
            added_rows=added_rows,
            summary="\n".join(summaries),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, FinancialTool] = {
    CAGRTool.name: CAGRTool(),
    ValuationMultiplesSnapshotTool.name: ValuationMultiplesSnapshotTool(),
    ProfitabilityTool.name: ProfitabilityTool(),
    DebtSolvencyTool.name: DebtSolvencyTool(),
    LiquidityTool.name: LiquidityTool(),
    CustomFormulaTool.name: CustomFormulaTool(),
    PeriodOverPeriodTool.name: PeriodOverPeriodTool(),
}


def get_tool_descriptions() -> str:
    """
    Returns a formatted tool catalogue for injecting into the planner LLM prompt.
    Each entry includes the tool name, description, and all parameter fields.
    """
    lines: List[str] = []
    for tool in TOOL_REGISTRY.values():
        param_lines: List[str] = []
        for field_name, field_info in tool.parameters_schema.model_fields.items():
            default_note = ""
            if field_info.default is not None and field_info.default is not ...:
                default_note = f" [default: {field_info.default}]"
            required_note = " (REQUIRED)" if field_info.is_required() else " (optional)"
            param_lines.append(
                f"    • {field_name}{required_note}{default_note}: {field_info.description}"
            )
        lines.append(
            f"[{tool.name}]\n{tool.description}\nParameters:\n" + "\n".join(param_lines)
        )
    return "\n\n" + "\n\n---\n\n".join(lines) + "\n"
