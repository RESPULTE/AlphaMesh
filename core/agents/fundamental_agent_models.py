import operator
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core.agents.financial_tools import ToolResult
from core.agents.models import BaseAgentInput, BaseAgentOutput

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Tool Call Specification and Iterative Plan Models


class ToolCallSpec(BaseModel):
    """Specification for a single tool call produced by the planner LLM."""

    tool_name: str = Field(
        description=(
            "Exact name of the tool to call from the registry "
            "(e.g. 'cagr', 'dcf_intrinsic_value', 'profitability_ratios', "
            "'custom_formula')."
        )
    )
    parameters: Dict[str, Any] = Field(
        description=(
            "Parameter dict matching the tool's parameters_schema exactly. "
            "All metric fields MUST reference concept names present in the "
            "'Available Concepts' list, OR row labels added in previous iterations."
        )
    )
    reasoning: str = Field(description="One-sentence justification for this tool call.")


class IterativeToolPlan(BaseModel):
    """
    Plan for ONE iteration, produced by the planner LLM.

    All calls in `calls` are executed **in parallel** within this iteration.
    If call B depends on the output of call A, A must be in this iteration
    and B deferred to the next (set needs_more_iterations=True).
    """

    calls: List[ToolCallSpec] = Field(
        description=(
            "Batch of tool calls to execute IN PARALLEL this iteration. "
            "Only include mutually independent calls here — calls whose inputs "
            "depend on outputs of other calls in this batch must wait for the "
            "next iteration. "
            "Return an empty list if the user only wants raw data."
        )
    )
    needs_more_iterations: bool = Field(
        description=(
            "True if another planning + execution iteration is required after "
            "this batch completes (e.g. a derived metric must be computed before "
            "a downstream analysis tool can run). False when this is the final batch."
        )
    )
    iteration_reasoning: str = Field(
        default="",
        description=(
            "Required when needs_more_iterations=True. Explain: what this "
            "iteration computes, what derived metric is being staged, and what "
            "the NEXT iteration will do with the newly created metric."
        ),
    )
    data_summary: str = Field(
        description=(
            "1-2 sentence summary of what data is available and what this "
            "iteration will compute."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Analyst structured output (row selection + written analysis)
# ─────────────────────────────────────────────────────────────────────────────


class RelevantRowsSelection(BaseModel):
    """Structured output from the analyst node."""

    relevant_row_labels: List[str] = Field(
        description=(
            "Exact row label strings (DataFrame index values) to include in the "
            "final financial data table. Include: (a) rows that directly answer "
            "the query, (b) component rows whose values reveal an insight (e.g. "
            "if PE is rising because EPS is falling, include both PE and EPS), "
            "(c) essential context rows used in any calculation. "
            "Exclude rows completely unrelated to the analysis."
        )
    )
    analysis: str = Field(
        description=(
            "Full natural-language financial analysis. Reference all tool results. "
            "Highlight key trends, risks, and insights. Convert large raw numbers "
            "to human-readable form (1.5e9 → '1.5 Billion'). "
            "For DCF: state WACC and terminal growth rate assumptions explicitly "
            "and whether intrinsic value implies the stock is over- or under-valued. "
            "If a derived metric was computed mid-analysis (e.g. FCF computed from "
            "OperatingCF and CapEx), explain how it was derived."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Agent Output
# ─────────────────────────────────────────────────────────────────────────────


class FundamentalAnalysisOutput(BaseAgentOutput):
    """Public output returned by FundamentalAnalysisAgent.run()."""

    agent_name: str = "fundamentals_agent"
    financial_data: Optional[pd.DataFrame] = Field(default=None)
    tool_results: List[ToolResult] = Field(default_factory=list)
    analysis: str = Field(default="")
    entities_enriched: List[str] = Field(default_factory=list)
    subgraph_id: Optional[str] = Field(default=None)
    relationships_extracted: bool = Field(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_llm_context_str(self) -> str:
        if self.financial_data is None or self.financial_data.empty:
            return (
                "### REPORT FROM fundamentals_agent\n"
                "No financial data was found or calculated."
            )
        header = "### REPORT FROM fundamentals_agent (Quantitative Financial Data)\n"
        data_str = self.financial_data.to_string(max_rows=30, float_format="%.4g")

        tool_section = ""
        if self.tool_results:
            summaries = [
                f"  [{r.tool_name}] {'✓' if r.success else '✗'} {r.summary or r.error or ''}"
                for r in self.tool_results
            ]
            reasoning_entries = [
                f"\n  [{r.tool_name} reasoning]\n  {r.reasoning}"
                for r in self.tool_results
                if r.reasoning
            ]
            tool_section = (
                "\n\n### TOOL EXECUTION RESULTS\n"
                + "\n".join(summaries)
                + "".join(reasoning_entries)
            )

        return f"{header} Analysis: {self.analysis}\n\nQuantitative Financial Data (Rows=Metrics, Columns=Dates):\n{data_str}{tool_section}"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Internal Graph State
# ─────────────────────────────────────────────────────────────────────────────


class _AgentState(BaseAgentInput):
    """Internal state carried through LangGraph nodes."""

    # Populated by data_prep
    available_concepts: List[str] = Field(default_factory=list)
    financial_data: Optional[pd.DataFrame] = None

    # Populated / updated by tool_planner each iteration
    tool_plan: Optional[IterativeToolPlan] = None

    # Incremented by tool_executor after each batch
    iteration_count: int = Field(default=0)

    # Accumulated across all iterations
    tool_results: Annotated[List[ToolResult], operator.add] = Field(
        default_factory=list
    )

    # Labels of rows that were computed (not fetched from EDGAR)
    computed_row_labels: List[str] = Field(default_factory=list)

    # Populated by analyst
    analysis: str = Field(default="")
    relationships_extracted: bool = Field(default=False)
    subgraph_id: Optional[str] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)
