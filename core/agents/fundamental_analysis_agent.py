"""
core/agents/fundamental_analysis_agent.py

Fundamental Analysis Agent — Iterative Tool-Calling LangGraph Architecture
===========================================================================

Graph
─────
START
    │
    ▼
data_prep          — Concurrently: update EDGAR cache, fetch price data,
    │                    load available concepts from DB.
    │                    Output: financial_data (DataFrame), available_concepts (List[str])
    ▼
tool_planner       — LLM produces an IterativeToolPlan:
    │                    • A BATCH of parallel tool calls for *this* iteration.
    │                    • A flag indicating whether more iterations are needed.
    │                    • Explicit reasoning about derived metrics it must
    │                      compute before a downstream tool can run
    │                      (e.g. FCF = OperatingCF − CapEx before DCF).
    ▼
tool_executor      — Runs ALL calls in the current batch **in parallel**
    │                    (asyncio.gather). Merges added_rows into financial_data.
    │                    Persists new time-series rows back to SQLite.
    │                    Increments iteration_count.
    │
    ├─── needs_more_iterations AND iteration_count < MAX_ITERATIONS ──► tool_planner
    │
    └─── done (or limit reached) ──► analyst
    ▼
analyst            — LLM selects the *relevant* rows for the final table
    │                    and writes the analysis.
    │                    Runs memory graph relationship extraction.
    ▼
END

Key improvements
────────────────
• Iterative re-planning loop (max MAX_TOOL_ITERATIONS = 3):
    The LLM detects missing derived metrics (e.g. FCF) and computes
    them via custom_formula BEFORE running DCF, instead of incorrectly
    substituting an available but wrong metric.

• Parallel execution per iteration:
    All calls in a single IterativeToolPlan.calls batch are dispatched
    with asyncio.gather — calls that are mutually independent run at the
    same time. Sequential dependencies are handled across iterations.

• Derived-metric persistence:
    Computed time-series rows (same date columns as financial_data) are
    saved back to SQLite (statement_type='computed') so they survive
    across sessions and are visible to subsequent planner iterations via
    available_concepts.

• LLM-selected relevant-data table:
    The analyst uses a structured output call to pick only the rows that
    are meaningful for the user's query. Component rows that reveal an
    insight (e.g. falling EPS behind a rising PE) are included; unrelated
    asset/liability rows are excluded.
"""

from __future__ import annotations

import asyncio
import operator
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Type

import aiosqlite
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from core.agents.base_agent import AbstractAgent
from core.agents.financial_db import FinancialDatabase
from core.agents.financial_tools import TOOL_REGISTRY, ToolResult, get_tool_descriptions
from core.agents.models import BaseAgentInput, BaseAgentOutput
from core.agents.prompts import (
    _ANALYST_SYSTEM,
    _TOOL_PLANNER_SYSTEM,
    _TOOL_PLANNER_USER,
)
from core.config import settings
from core.logger import get_logger
from core.memory.graph.extraction_prompts import (
    ANALYSIS_ONLY_RELATIONSHIP_PROMPT,
    COMBINED_ANALYSIS_RELATIONSHIP_PROMPT,
)
from core.memory.graph.relationship_extractor import (
    extract_with_retry,
    retry_relationships_only,
)
from core.memory.graph.subgraph_builder import InMemorySubgraphBuilder
from core.memory.stores.subgraph_store import SubgraphStore
from core.services import service_manager

logger = get_logger(__name__)

# ── Iteration ceiling ─────────────────────────────────────────────────────────
MAX_TOOL_ITERATIONS: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Planner LLM Output Models
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Agent
# ─────────────────────────────────────────────────────────────────────────────


class FundamentalAnalysisAgent(AbstractAgent):
    """
    Async LangGraph agent with an iterative tool-calling loop.

    The agent can loop up to MAX_TOOL_ITERATIONS (3) times through
    tool_planner → tool_executor before proceeding to the analyst,
    enabling multi-step derived-metric computation (FCF before DCF, etc.).
    Within each iteration all tool calls execute in parallel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.db = FinancialDatabase()
        self._graph = self._build_graph()

    @staticmethod
    def name() -> str:
        return "fundamentals_agent"

    @staticmethod
    def description() -> str:
        return (
            "Fetches standardised EDGAR financial statements and computes "
            "quantitative metrics (CAGR, DCF, ratios) via an iterative, "
            "parallel tool-calling pipeline. Returns enriched financial data "
            "and a written analysis."
        )

    @staticmethod
    def get_output_schema_class() -> Type[BaseModel]:
        return FundamentalAnalysisOutput

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, input_data: BaseAgentInput) -> FundamentalAnalysisOutput:
        logger.info("[Agent: %s] Started for %s", self.name(), input_data.ticker)
        await self.db.initialize()

        final_state: Dict = await self._graph.ainvoke(
            input_data.model_dump(exclude_none=False),
            config={"recursion_limit": 20},
        )

        return FundamentalAnalysisOutput(
            financial_data=final_state.get("financial_data"),
            analysis=final_state.get("analysis", ""),
            tool_results=final_state.get("tool_results", []),
            entities_enriched=final_state.get("entities_enriched", []),
            subgraph_id=final_state.get("subgraph_id"),
            subgraph_task=final_state.get("subgraph_task"),
            relationships_extracted=final_state.get("relationships_extracted", False),
        )

    # ── Graph wiring ──────────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        workflow = StateGraph(_AgentState)

        workflow.add_node("data_prep", self._data_prep_node)
        workflow.add_node("tool_planner", self._tool_planner_node)
        workflow.add_node("tool_executor", self._tool_executor_node)
        workflow.add_node("analyst", self._analyst_node)

        workflow.add_edge(START, "data_prep")
        workflow.add_edge("data_prep", "tool_planner")
        workflow.add_edge("tool_planner", "tool_executor")

        # ── Iterative loop: executor → planner or analyst ─────────────────────
        workflow.add_conditional_edges(
            "tool_executor",
            self._should_continue,
            {"continue": "tool_planner", "done": "analyst"},
        )

        workflow.add_edge("analyst", END)
        return workflow.compile()

    # ── Routing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _should_continue(state: _AgentState) -> str:
        if state.iteration_count >= MAX_TOOL_ITERATIONS:
            logger.warning(
                "[Router] Max iterations (%d) reached — forcing analyst.",
                MAX_TOOL_ITERATIONS,
            )
            return "done"

        plan = state.tool_plan
        if plan and plan.needs_more_iterations:
            logger.info(
                "[Router] Looping: iteration %d/%d. Reason: %s",
                state.iteration_count,
                MAX_TOOL_ITERATIONS,
                plan.iteration_reasoning or "(no reason given)",
            )
            return "continue"

        # Only retry on failure if the planner actually scheduled tools this round
        # (avoids infinite loop when planner gives up but old failures persist).
        # Crucially, only inspect the CURRENT iteration's results — not the full
        # accumulated history — otherwise a failure from iteration 1 causes
        # endless retries even when all subsequent iterations succeed.
        last_iteration_had_calls = plan is not None and len(plan.calls) > 0
        if last_iteration_had_calls:
            n_calls = len(plan.calls)
            current_batch = (state.tool_results or [])[-n_calls:]
            any_current_failed = any(not r.success for r in current_batch)
            if any_current_failed:
                logger.info(
                    "[Router] Tool failures in current batch — retrying. Iteration %d/%d",
                    state.iteration_count,
                    MAX_TOOL_ITERATIONS,
                )
                return "continue"

        return "done"

    # ── Node: data_prep ───────────────────────────────────────────────────────

    async def _data_prep_node(self, state: _AgentState) -> Dict:
        """
        Concurrently:
        1. Ensures EDGAR data is cached for the requested periods.
        2. Fetches price data from yfinance at matching granularity.
        3. Loads financial data from local DB and normalises period dates.

        Granularity:
        yearly   (default) — 10-K filings, 5-year window.
                            Normalises fiscal-year-end dates to YYYY-12-31
                            to align with yfinance YE prices.
        quarterly           — 10-Q filings, last 8 quarters.
                            Normalises to quarter-end dates.
        """
        ticker: str = state.ticker
        granularity: str = getattr(state, "granularity", "yearly")

        # ── Resolve date window ───────────────────────────────────────────────
        end_dt = state.end_date or datetime.now()
        if granularity == "yearly":
            default_start = datetime(end_dt.year - 4, 1, 1)
            start_dt = state.start_date if state.start_date else default_start
            if (end_dt.year - start_dt.year) < 4:
                start_dt = datetime(end_dt.year - 4, 1, 1)
            form_type = "10-K"
            price_interval = "yearly"
            today = datetime.now()
            last_complete_year = today.year - 1 if today.month < 12 else today.year
            periods = list(
                range(start_dt.year, min(end_dt.year, last_complete_year) + 1)
            )
        else:
            default_start = end_dt - timedelta(days=2 * 365)
            start_dt = state.start_date if state.start_date else default_start
            form_type = "10-Q"
            price_interval = "quarterly"
            periods = _quarterly_periods(start_dt, end_dt)

        logger.info(
            "[Node] data_prep — %s | %s | %s → %s | periods=%s",
            ticker,
            granularity,
            start_dt.date(),
            end_dt.date(),
            periods,
        )

        # ── Concurrent EDGAR update + price fetch ─────────────────────────────
        async def _edgar_update():
            await self.db.update_financials(ticker, periods, form_type)

        async def _price_fetch():
            return await self.db.get_price_data(
                ticker,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=price_interval,
            )

        edgar_task = asyncio.create_task(_edgar_update())
        price_task = asyncio.create_task(_price_fetch())
        await edgar_task  # DB must be populated before querying
        price_df: pd.DataFrame = await price_task

        # ── Load + pivot from DB ──────────────────────────────────────────────
        financial_df = await self.db.get_data(ticker, form_types=[form_type])

        if not financial_df.empty:
            try:
                col_dates = pd.to_datetime(financial_df.columns, errors="coerce")
                keep_mask = (col_dates >= pd.Timestamp(start_dt).tz_localize(None)) & (
                    col_dates <= pd.Timestamp(end_dt).tz_localize(None)
                )
                financial_df = financial_df.loc[:, keep_mask]
            except Exception as exc:
                logger.warning("[data_prep] Date trimming failed: %s", exc)

        # Normalise EDGAR fiscal period dates to canonical period-end
        if not financial_df.empty:
            financial_df = _normalize_period_ends(financial_df, granularity)

        available_concepts = list(financial_df.index) if not financial_df.empty else []

        # ── Merge stock price row ─────────────────────────────────────────────
        if not price_df.empty and "stock_price" in price_df.columns:
            price_t = price_df[["stock_price"]].T
            price_t.columns = _canonical_date_strs(price_t.columns, granularity)

            if not financial_df.empty:
                all_cols = sorted(set(financial_df.columns) | set(price_t.columns))
                financial_df = financial_df.reindex(columns=all_cols)
                price_t = price_t.reindex(columns=all_cols)

            financial_df = pd.concat([financial_df, price_t])
            if "stock_price" not in available_concepts:
                available_concepts.append("stock_price")

        if financial_df.empty:
            logger.warning(
                "[data_prep] No financial data found for %s in the requested range.",
                ticker,
            )
        else:
            thresh = max(1, int(len(financial_df.index) * 0.3))
            financial_df = financial_df.dropna(axis=1, thresh=thresh)

            logger.info(
                "[data_prep] Ready: %d concepts × %d periods for %s",
                len(financial_df.index),
                len(financial_df.columns),
                ticker,
            )

        return {
            "financial_data": financial_df,
            "available_concepts": available_concepts,
        }

    # ── Node: tool_planner ────────────────────────────────────────────────────

    async def _tool_planner_node(self, state: _AgentState) -> Dict:
        """
        LLM produces an IterativeToolPlan for the current iteration.

        The planner is given the current iteration number, all available
        concepts (including any derived from previous iterations), and a
        summary of prior tool results so it knows what's already been done.
        """
        iteration = state.iteration_count + 1  # 1-based for prompt readability
        logger.info(
            "[Node] tool_planner — iteration %d/%d — %s",
            iteration,
            MAX_TOOL_ITERATIONS,
            state.query,
        )

        if not state.available_concepts:
            logger.warning("[tool_planner] No concepts — skipping planning.")
            return {
                "tool_plan": IterativeToolPlan(
                    calls=[],
                    needs_more_iterations=False,
                    data_summary="No financial data available.",
                )
            }

        concepts_display = "\n".join(
            f"  • {c}" for c in sorted(state.available_concepts)[:200]
        )
        if len(state.available_concepts) > 200:
            concepts_display += f"\n  … and {len(state.available_concepts) - 200} more"

        if state.tool_results:
            prior_lines = [
                f"  [{r.tool_name}] {'✓' if r.success else f'✗ {r.error}'} — {r.summary}"
                for r in state.tool_results
            ]
            prior_results_block = "\n".join(prior_lines)
        else:
            prior_results_block = "  (none — first iteration)"

        user_msg = _TOOL_PLANNER_USER.format(
            query=state.query,
            ticker=state.ticker,
            start_date=(
                state.start_date.strftime("%Y-%m-%d") if state.start_date else "N/A"
            ),
            end_date=(state.end_date.strftime("%Y-%m-%d") if state.end_date else "N/A"),
            iteration=iteration,
            max_iterations=MAX_TOOL_ITERATIONS,
            n_concepts=len(state.available_concepts),
            concepts_block=concepts_display,
            prior_results_block=prior_results_block,
            tool_descriptions=get_tool_descriptions(),
        )

        llm = service_manager.get_agent(temperature=0)
        structured_llm = llm.with_structured_output(IterativeToolPlan)

        try:
            tool_plan: IterativeToolPlan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=_TOOL_PLANNER_SYSTEM),
                    HumanMessage(content=user_msg),
                ]
            )
        except Exception as exc:
            logger.error("[tool_planner] LLM call failed: %s", exc)
            tool_plan = IterativeToolPlan(
                calls=[],
                needs_more_iterations=False,
                data_summary=f"Tool planning failed: {exc}. Proceeding with raw data.",
            )

        logger.info(
            "[tool_planner] Iteration %d: %d call(s), needs_more=%s — %s",
            iteration,
            len(tool_plan.calls),
            tool_plan.needs_more_iterations,
            tool_plan.data_summary,
        )
        if tool_plan.iteration_reasoning:
            logger.info(
                "[tool_planner] Multi-step plan: %s", tool_plan.iteration_reasoning
            )
        for i, spec in enumerate(tool_plan.calls, 1):
            logger.info("  %d. %s — %s", i, spec.tool_name, spec.reasoning)

        return {"tool_plan": tool_plan}

    # ── Node: tool_executor ───────────────────────────────────────────────────

    async def _tool_executor_node(self, state: _AgentState) -> Dict:
        """
        Executes ALL calls in the current iteration batch in PARALLEL via
        asyncio.gather. Then:
        1. Merges added_rows back into financial_data.
        2. Persists new time-series rows to SQLite.
        3. Updates available_concepts with newly added row labels.
        4. Increments iteration_count.
        """
        logger.info("[Node] tool_executor — iteration %d", state.iteration_count + 1)

        plan = state.tool_plan
        if not plan or not plan.calls:
            logger.info("[tool_executor] No tools to run.")
            return {"iteration_count": state.iteration_count + 1}

        df: pd.DataFrame = (
            state.financial_data.copy()
            if state.financial_data is not None and not state.financial_data.empty
            else pd.DataFrame()
        )

        # ── Define single-tool runner ─────────────────────────────────────────
        async def _run_one(spec: ToolCallSpec) -> ToolResult:
            tool = TOOL_REGISTRY.get(spec.tool_name)
            if tool is None:
                msg = (
                    f"Tool '{spec.tool_name}' not found in registry. "
                    f"Available: {list(TOOL_REGISTRY.keys())}"
                )
                logger.warning("[tool_executor] %s", msg)
                return ToolResult(tool_name=spec.tool_name, success=False, error=msg)

            try:
                params = tool.parameters_schema(**spec.parameters)
            except Exception as exc:
                msg = f"Invalid parameters for '{spec.tool_name}': {exc}"
                logger.error("[tool_executor] %s", msg)
                return ToolResult(tool_name=spec.tool_name, success=False, error=msg)

            if df.empty:
                return ToolResult(
                    tool_name=spec.tool_name,
                    success=False,
                    error="Cannot execute tool — financial_data is empty.",
                )

            try:
                # Tools are CPU-bound / synchronous; run in thread executor
                loop = asyncio.get_event_loop()
                result: ToolResult = await loop.run_in_executor(
                    None, tool.execute, df, params
                )
            except Exception as exc:
                result = ToolResult(
                    tool_name=spec.tool_name, success=False, error=str(exc)
                )

            return result

        # ── Run all calls in this iteration batch in parallel ─────────────────
        batch_results: List[ToolResult] = await asyncio.gather(
            *[_run_one(spec) for spec in plan.calls]
        )

        # ── Merge added_rows into the working DataFrame ───────────────────────
        # (Must be done sequentially to preserve column alignment.)
        newly_added_labels: List[str] = []
        for result in batch_results:
            if result.success and result.added_rows:
                new_rows = pd.DataFrame.from_dict(result.added_rows, orient="index")
                all_cols = df.columns.union(new_rows.columns)
                df = df.reindex(columns=all_cols)
                new_rows = new_rows.reindex(columns=all_cols)
                # Skip rows that already exist (idempotent merge)
                new_rows = new_rows[~new_rows.index.isin(df.index)]
                if not new_rows.empty:
                    df = pd.concat([df, new_rows])
                newly_added_labels.extend(result.added_rows.keys())
                logger.info(
                    "[tool_executor] Merged rows from %s: %s",
                    result.tool_name,
                    list(result.added_rows.keys()),
                )

        # ── Persist new time-series rows to SQLite ────────────────────────────
        if newly_added_labels and not df.empty:
            await self._persist_computed_rows(
                ticker=state.ticker,
                df=df,
                row_labels=newly_added_labels,
                form_type=(
                    "10-K"
                    if getattr(state, "granularity", "yearly") == "yearly"
                    else "10-Q"
                ),
            )

        # ── Update state ──────────────────────────────────────────────────────
        updated_concepts = list(df.index) if not df.empty else state.available_concepts
        updated_computed = list(
            set(state.computed_row_labels) | set(newly_added_labels)
        )
        accumulated_results = list(state.tool_results) + batch_results

        return {
            "financial_data": df if not df.empty else state.financial_data,
            "tool_results": accumulated_results,
            "iteration_count": state.iteration_count + 1,
            "available_concepts": updated_concepts,
            "computed_row_labels": updated_computed,
        }

    # ── Helper: persist computed rows to SQLite ───────────────────────────────

    async def _persist_computed_rows(
        self,
        ticker: str,
        df: pd.DataFrame,
        row_labels: List[str],
        form_type: str,
    ) -> None:
        """
        Saves newly computed time-series rows back into the `financials` SQLite
        table under statement_type='computed'.

        Only persists rows with ≥ 2 date-valued cells (genuine time series).
        Scalar / single-value rows (e.g. a CAGR percentage) are skipped —
        they have no meaningful time dimension to store.
        """
        records = []
        for label in row_labels:
            if label not in df.index:
                continue
            row = df.loc[label].dropna()
            if len(row) < 2:
                logger.debug("[persist] Skipping scalar/single-period row '%s'.", label)
                continue
            for period_date, value in row.items():
                try:
                    float_val = float(value)
                except (TypeError, ValueError):
                    continue
                records.append(
                    (
                        ticker.upper(),
                        str(period_date),
                        form_type,
                        "computed",
                        label,
                        float_val,
                    )
                )

        if not records:
            return

        try:
            async with aiosqlite.connect(self.db.db_name) as db:
                await db.executemany(
                    """INSERT OR REPLACE INTO financials
                    (company, period_date, form_type, statement_type, label, value)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    records,
                )
                await db.commit()
            logger.info(
                "[persist] Saved %d records for computed rows %s (%s).",
                len(records),
                row_labels,
                ticker,
            )
        except Exception as exc:
            logger.warning("[persist] Failed to save computed rows: %s", exc)

    # ── Node: analyst ─────────────────────────────────────────────────────────

    async def _analyst_node(self, state: _AgentState) -> Dict:
        """
        1. Structured LLM call → selects relevant rows + writes analysis.
        2. Filters financial_data to only the selected rows.
        3. Runs memory graph relationship extraction.
        """
        logger.info("[Node] analyst")

        if state.financial_data is None or state.financial_data.empty:
            return {
                "analysis": "No financial data was retrieved for this query.",
                "relationships_extracted": False,
            }

        # Full DataFrame in human-readable form (all rows, for LLM context)
        full_readable = state.financial_data.map(
            lambda x: _format_value(x) if isinstance(x, (int, float)) else x
        ).to_string(max_rows=60)

        # Tool results summary
        tool_summary_parts = []
        for r in state.tool_results:
            status = "✓" if r.success else f"✗ ERROR: {r.error}"
            tool_summary_parts.append(f"[{r.tool_name}] {status}\n  {r.summary}")
            if r.reasoning:
                tool_summary_parts.append(f"  Assumptions: {r.reasoning}")
        tool_summary = (
            "\n\n".join(tool_summary_parts)
            if tool_summary_parts
            else "No tools were run."
        )

        user_prompt = (
            f"Analyse the following financial data for {state.ticker}.\n\n"
            f"COMPLETE DataFrame (ALL rows — Rows=Metrics, Columns=Dates):\n"
            f"{full_readable}\n\n"
            f"--- Tool Execution Results ---\n{tool_summary}\n\n"
            f"User Question: {state.query}\n\n"
            f"Derived/computed rows added during analysis: "
            f"{state.computed_row_labels or '(none)'}\n\n"
            "TASK:\n"
            "1. Populate `relevant_row_labels` with ONLY the row labels that "
            "   belong in the final table (see system instructions for criteria).\n"
            "2. Populate `analysis` with the full written analysis."
        )

        llm = service_manager.get_agent(temperature=0.7)
        structured_llm = llm.with_structured_output(RelevantRowsSelection)

        try:
            selection: RelevantRowsSelection = await structured_llm.ainvoke(
                [
                    SystemMessage(content=_ANALYST_SYSTEM),
                    HumanMessage(content=user_prompt),
                ]
            )
            analysis_text: str = selection.analysis
            relevant_labels: List[str] = selection.relevant_row_labels
        except Exception as exc:
            logger.error("[analyst] Structured LLM call failed: %s", exc)
            analysis_text = f"Analysis generation failed: {exc}"
            relevant_labels = list(state.financial_data.index)

        # ── Filter DataFrame to relevant rows ─────────────────────────────────
        valid_labels = [
            lbl for lbl in relevant_labels if lbl in state.financial_data.index
        ]
        if valid_labels:
            filtered_df = state.financial_data.loc[valid_labels]
            logger.info(
                "[analyst] Returning %d/%d rows in final table.",
                len(filtered_df),
                len(state.financial_data),
            )
        else:
            logger.warning(
                "[analyst] No valid relevant_row_labels returned — using full DataFrame."
            )
            filtered_df = state.financial_data

        # ── Memory graph relationship extraction ──────────────────────────────
        relationships = []
        relationships_extracted = False
        subgraph_id: Optional[str] = None

        try:
            memory_readable = state.financial_data.map(
                lambda x: _format_value(x) if isinstance(x, (int, float)) else x
            ).to_string(max_rows=30)
            memory_content = (
                f"Financial data for {state.ticker}:\n{memory_readable}\n\n"
                f"Tool results:\n{tool_summary}\n\n"
                f"Analysis:\n{analysis_text}"
            )
            result = await extract_with_retry(
                service_manager.get_agent(temperature=0.7),
                [
                    SystemMessage(content=COMBINED_ANALYSIS_RELATIONSHIP_PROMPT),
                    HumanMessage(content=memory_content),
                ],
            )
            relationships = result.relationships
            relationships_extracted = result.parse_success
        except Exception as exc:
            logger.error("[analyst] Relationship extraction failed: %s", exc)

        if settings.EXTRACTION_ENABLED and state.conversation_id:
            builder = InMemorySubgraphBuilder(
                embedding_func=service_manager.get_embedding_func(),
                fuzzy_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
                semantic_threshold=settings.EXTRACTION_SEMANTIC_THRESHOLD,
            )
            store = service_manager.get_subgraph_store()
            subgraph_id = SubgraphStore.make_key(
                FundamentalAnalysisAgent.name(), state.conversation_id
            )

            async def _build_and_store():
                graph = await builder.build(
                    relationships,
                    source_agent=FundamentalAnalysisAgent.name(),
                )
                await store.save(subgraph_id, graph)

            if relationships_extracted:
                task = asyncio.create_task(_build_and_store())
            else:
                task = asyncio.create_task(
                    retry_relationships_only(
                        service_manager.get_agent(temperature=0.7),
                        analysis_text,
                        FundamentalAnalysisAgent.name(),
                        state.conversation_id,
                        builder,
                        store,
                        subgraph_id,
                        ANALYSIS_ONLY_RELATIONSHIP_PROMPT,
                    )
                )
            if settings.EXTRACTION_IMMEDIATE:
                await task

        return {
            "financial_data": filtered_df,
            "analysis": analysis_text,
            "relationships_extracted": relationships_extracted,
            "subgraph_id": subgraph_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _format_value(x: float) -> str:
    """Converts a raw float to a human-readable denomination string."""
    abs_x = abs(x)
    if abs_x >= 1_000_000_000_000:
        return f"{x / 1_000_000_000_000:.2f} Trillion"
    if abs_x >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f} Billion"
    if abs_x >= 1_000_000:
        return f"{x / 1_000_000:.2f} Million"
    if abs_x >= 1_000:
        return f"{x / 1_000:.2f} Thousand"
    return f"{x:.4g}"


def _quarterly_periods(start: datetime, end: datetime) -> list:
    """Returns (year, quarter) tuples covering start..end inclusive."""
    periods = []
    year, month = start.year, start.month
    while datetime(year, month, 1) <= end:
        quarter = (month - 1) // 3 + 1
        if (year, quarter) not in periods:
            periods.append((year, quarter))
        month += 3
        if month > 12:
            month -= 12
            year += 1
    return periods


def _normalize_period_ends(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Renames DataFrame columns from raw EDGAR fiscal-period dates to canonical
    period-end dates so they align with yfinance resampled prices.

    yearly    : any date in year YYYY  →  YYYY-12-31
    quarterly : any date in quarter Q  →  last day of that quarter
    """
    rename: Dict[str, str] = {}
    for col in df.columns:
        try:
            ts = pd.Timestamp(col)
            if granularity == "yearly":
                rename[col] = ts.replace(month=12, day=31).strftime("%Y-%m-%d")
            else:
                rename[col] = (
                    ts.to_period("Q").end_time.normalize().strftime("%Y-%m-%d")
                )
        except Exception:
            rename[col] = str(col)
    return df.rename(columns=rename)


def _canonical_date_strs(index: Any, granularity: str) -> list:
    """
    Converts a DatetimeIndex (or any iterable of date-like values) to
    canonical period-end date strings matching _normalize_period_ends output.
    """
    result = []
    for val in index:
        try:
            ts = pd.Timestamp(val)
            if granularity == "yearly":
                result.append(ts.replace(month=12, day=31).strftime("%Y-%m-%d"))
            else:
                result.append(
                    ts.to_period("Q").end_time.normalize().strftime("%Y-%m-%d")
                )
        except Exception:
            result.append(str(val))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test  (python -m core.agents.fundamental_analysis_agent)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio as _asyncio

    from core.agents.models import BaseAgentInput

    async def _main():
        agent = FundamentalAnalysisAgent()
        result = await agent.run(
            BaseAgentInput(
                ticker="AAPL",
                query="Perform a DCF valuation for Apple. What is the intrinsic value per share?",
                vector_query="Apple DCF intrinsic value",
            )
        )
        print("=== ANALYSIS ===")
        print(result.analysis)
        print("\n=== RELEVANT FINANCIAL DATA ===")
        if result.financial_data is not None and not result.financial_data.empty:
            print(result.financial_data.to_string())

    _asyncio.run(_main())
