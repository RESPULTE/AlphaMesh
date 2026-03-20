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
            "'Available Concepts' list, OR row labels added in previous batches."
        )
    )
    reasoning: str = Field(description="One-sentence justification for this tool call.")


class ToolCallBatch(BaseModel):
    """
    A single parallel-safe batch of tool calls within a multi-step plan.

    All calls in this batch are mutually independent and execute in parallel.
    If call B requires output from call A, they must be in separate batches
    with A's batch appearing first in the list.
    """

    calls: List[ToolCallSpec] = Field(
        description=(
            "Tool calls to execute IN PARALLEL within this batch. "
            "All calls must be mutually independent — no call may depend on "
            "the output of another call in the same batch."
        )
    )
    batch_reasoning: str = Field(
        default="",
        description=(
            "Why these calls are grouped together, and what derived output "
            "this batch produces that the next batch will consume (if any)."
        ),
    )


class IterativeToolPlan(BaseModel):
    """
    A fully pre-planned, ordered sequence of tool execution batches.

    The executor steps through `batches` sequentially without re-invoking
    the LLM planner between steps — each batch runs only after the previous
    batch completes successfully.  The LLM planner is only re-invoked if a
    tool failure occurs mid-execution (fallback recovery path).

    Design principles
    -----------------
    • All dependency chains must be expressed upfront as ordered batches.
      Example (FCF → DCF):
        batch[0]: custom_formula to derive FreeCashFlow
        batch[1]: dcf_intrinsic_value using FreeCashFlow from batch[0]

    • Calls within a batch run in parallel (asyncio.gather).
    • Calls across batches run sequentially (batch N+1 sees batch N's results).

    Replaces the old schema where a single `calls` list + `needs_more_iterations`
    flag drove repeated LLM re-planning between each batch.
    """

    batches: List[ToolCallBatch] = Field(
        description=(
            "Ordered list of execution batches. "
            "Batch 0 executes first; batch N may depend on results from batch N-1. "
            "Return an empty list if the user only wants raw data."
        )
    )
    data_summary: str = Field(
        description=(
            "1-2 sentence summary of what data is available and what the full "
            "plan will compute across all batches."
        )
    )

    # ── Convenience helpers used by the executor ──────────────────────────────

    def batch_count(self) -> int:
        return len(self.batches)

    def get_batch(self, index: int) -> Optional[ToolCallBatch]:
        if 0 <= index < len(self.batches):
            return self.batches[index]
        return None

    def is_empty(self) -> bool:
        return len(self.batches) == 0 or all(len(b.calls) == 0 for b in self.batches)


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

    # Populated by tool_planner; carries the full pre-planned batch list
    tool_plan: Optional[IterativeToolPlan] = None

    # Index into tool_plan.batches — incremented by tool_executor each pass.
    # When current_batch_index == len(tool_plan.batches), execution is done.
    current_batch_index: int = Field(default=0)

    # Incremented by tool_executor after each batch (used for MAX guard only)
    iteration_count: int = Field(default=0)

    # Accumulated across all batches (Annotated with operator.add so LangGraph
    # merges lists rather than replacing them on each state update)
    tool_results: Annotated[List[ToolResult], operator.add] = Field(
        default_factory=list
    )

    # Labels of rows that were computed (not fetched from EDGAR)
    computed_row_labels: List[str] = Field(default_factory=list)

    # Set True when a failure triggers re-planning so the planner prompt
    # can acknowledge the context
    replanning_due_to_failure: bool = Field(default=False)

    # Populated by analyst
    analysis: str = Field(default="")
    relationships_extracted: bool = Field(default=False)
    subgraph_id: Optional[str] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)
