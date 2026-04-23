"""
core/agents/fundamental_analysis_agent.py

Async LangGraph agent — refactored for lower latency.

Changes in this revision
────────────────────────
1. db.initialize() guarded by _initialized flag on FinancialDatabase
   (no-op after the first call; keeps the call in run() for API compatibility
   but eliminates the repeated SQLite round-trip in practice).

2. Iterative planner → single upfront multi-batch plan (Option D hybrid).
   IterativeToolPlan now carries `batches: List[ToolCallBatch]`.  The planner
   LLM emits the FULL ordered dependency chain in one call.  The executor
   steps through batches sequentially without re-invoking the LLM.
   The LLM planner is only called again if a tool failure occurs mid-execution
   (fallback recovery path), at which point `replanning_due_to_failure=True`
   is signalled to the planner for context.

   Old worst-case: 3 LLM planner calls + 3 executor passes = 6+ LLM calls.
   New worst-case: 1 LLM planner call + N executor passes (no LLM between
   passes) + 1 analyst call = 2 LLM calls for the happy path.

3. Relationship extraction removed from the _analyst_node critical path.
   schedule_subgraph_extraction is called with relationships=[] and
   relationships_extracted=False, which routes to the background
   retry_relationships_only path using analysis_text as input.
   The background task runs entirely after the node returns.

Everything else is unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Type

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core.agents.base_agent import AbstractAgent
from core.agents.data_prep import (
    _fetch_raw_data,
    _merge_price_rows,
    _resolve_date_range,
    _trim_and_normalise,
)
from core.agents.financial_db import FinancialDatabase
from core.agents.financial_tools import TOOL_REGISTRY, ToolResult, get_tool_descriptions
from core.agents.models.base_agent_models import BaseAgentInput
from core.agents.models.fundamental_agent_models import (
    FundamentalAnalysisOutput,
    IterativeToolPlan,
    ToolCallBatch,
    ToolCallSpec,
    _AgentState,
)
from core.agents.prompts.fundamental_agent_prompts import (
    FUNDAMENTAL_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
    _ANALYST_SYSTEM,
    _TOOL_PLANNER_SYSTEM,
    _TOOL_PLANNER_USER,
)
from core.agents.sentiment_parser import parse_sentiment_block, strip_sentiment_block
from core.logger import get_logger
from core.memory.graph.graph_queue import make_extraction_task
from core.services import service_manager

logger = get_logger(__name__)

# ── Iteration ceiling (guards the fallback re-planning loop only) ─────────────
MAX_TOOL_ITERATIONS: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────


class FundamentalAnalysisAgent(AbstractAgent):
    """
    Async LangGraph agent with a pre-planned multi-batch tool execution pipeline.

    The planner LLM emits the complete ordered dependency chain in ONE call.
    The executor steps through batches without re-planning between them.
    The planner is only re-invoked on tool failure (fallback recovery).
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
            "quantitative metrics (CAGR, DCF, ratios) via a pre-planned, "
            "parallel tool execution pipeline. Returns enriched financial data "
            "and a written analysis."
        )

    @staticmethod
    def get_output_schema_class() -> Type[BaseModel]:
        return FundamentalAnalysisOutput

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, input_data: BaseAgentInput) -> FundamentalAnalysisOutput:
        logger.info("[Agent: %s] Started for %s", self.name(), input_data.ticker)

        # Fix #1: db.initialize() is now guarded by _initialized flag on
        # FinancialDatabase — this is a no-op after the first call.
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
            sentiment=final_state.get("sentiment"),
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

        # ── Routing: advance batch, re-plan on failure, or proceed to analyst ─
        workflow.add_conditional_edges(
            "tool_executor",
            self._should_continue,
            {
                "next_batch": "tool_executor",  # advance to next batch, no LLM
                "replan": "tool_planner",  # failure fallback — LLM re-plan
                "done": "analyst",
            },
        )

        workflow.add_edge("analyst", END)
        return workflow.compile()

    # ── Routing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _should_continue(state: _AgentState) -> str:
        """
        Three-way routing after each tool_executor pass:

        1. Hard ceiling: force analyst if we've hit MAX_TOOL_ITERATIONS.
           (Prevents runaway loops even in the re-planning fallback path.)

        2. Failure in the current batch AND budget remaining → replan.
           The planner is only re-invoked when something actually went wrong.

        3. More pre-planned batches remain AND no failure → next_batch.
           The executor advances directly without touching the LLM.

        4. No more batches and no failure → done (analyst).
        """
        if state.iteration_count >= MAX_TOOL_ITERATIONS:
            logger.warning(
                "[Router] Max iterations (%d) reached — forcing analyst.",
                MAX_TOOL_ITERATIONS,
            )
            return "done"

        plan = state.tool_plan

        # ── Check for failures in the batch that just ran ─────────────────────
        # We inspect only the slice of tool_results that corresponds to the
        # current batch, not the full accumulated history, to avoid a stale
        # failure from a previous batch triggering a re-plan.
        current_batch_had_calls = (
            plan is not None
            and plan.get_batch(state.current_batch_index - 1) is not None
            and len(plan.get_batch(state.current_batch_index - 1).calls) > 0
        )
        if current_batch_had_calls:
            batch = plan.get_batch(state.current_batch_index - 1)
            n_calls = len(batch.calls)
            current_results = (state.tool_results or [])[-n_calls:]
            if any(not r.success for r in current_results):
                logger.info(
                    "[Router] Failure in batch %d — re-planning. Iteration %d/%d",
                    state.current_batch_index - 1,
                    state.iteration_count,
                    MAX_TOOL_ITERATIONS,
                )
                return "replan"

        # ── Advance to next pre-planned batch if available ────────────────────
        if plan is not None and state.current_batch_index < plan.batch_count():
            logger.info(
                "[Router] Advancing to batch %d/%d (no LLM call).",
                state.current_batch_index,
                plan.batch_count(),
            )
            return "next_batch"

        return "done"

    # ── Node: data_prep ───────────────────────────────────────────────────────

    async def _data_prep_node(self, state: _AgentState) -> Dict:
        """
        Fetch and prepare financial data for the tool planner.

        Step 1  _resolve_date_range  — compute dates, periods, form type
        Step 2  _fetch_raw_data      — EDGAR + yfinance concurrently
        Step 3  _trim_and_normalise  — clip to requested window, normalise col names
        Step 4  _merge_price_rows    — append stock price series
        Step 5  quality gate         — drop sparse columns, return
        """
        ticker: str = state.ticker
        granularity: str = getattr(state, "granularity", "yearly") or "yearly"

        if not ticker:
            logger.warning("[data_prep] No ticker — returning empty state.")
            return {"financial_data": pd.DataFrame(), "available_concepts": []}

        # ── Step 1: dates / periods ───────────────────────────────────────────────
        cfg = _resolve_date_range(state)
        # Attach granularity so _trim_and_normalise can reach it via cfg
        cfg.__dict__["granularity"] = granularity

        logger.info(
            "[data_prep] %s | %s | %s → %s | periods=%s",
            ticker,
            granularity,
            cfg.start_dt,
            cfg.end_dt,
            cfg.periods,
        )

        # ── Step 2: concurrent fetch ──────────────────────────────────────────────
        financial_df, price_df = await _fetch_raw_data(self.db, ticker, cfg)

        # ── Step 3: trim + normalise ──────────────────────────────────────────────
        financial_df = _trim_and_normalise(financial_df, cfg)
        available_concepts = list(financial_df.index) if not financial_df.empty else []

        # ── Step 4: merge price rows ──────────────────────────────────────────────
        financial_df, available_concepts = _merge_price_rows(
            financial_df, price_df, cfg, available_concepts
        )

        # ── Step 5: quality gate ──────────────────────────────────────────────────
        if financial_df.empty:
            logger.warning(
                "[data_prep] No financial data for %s in requested range.", ticker
            )
            return {"financial_data": pd.DataFrame(), "available_concepts": []}

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
        Single LLM call that produces the complete ordered batch plan.

        Called once at the start of the execution pipeline.  Only called
        again if a tool failure triggers the fallback re-planning path
        (replanning_due_to_failure=True on state in that case).
        """
        logger.info(
            "[Node] tool_planner — %s%s",
            state.query,
            " [RE-PLANNING after failure]" if state.replanning_due_to_failure else "",
        )

        if not state.available_concepts:
            logger.warning("[tool_planner] No concepts — skipping planning.")
            return {
                "tool_plan": IterativeToolPlan(
                    batches=[],
                    data_summary="No financial data available.",
                ),
                "current_batch_index": 0,
                "replanning_due_to_failure": False,
            }

        # ── Concepts block ─────────────────────────────────────────────────────
        computed_set = set(state.computed_row_labels or [])
        concept_lines = [
            f"  • {c}{' [computed]' if c in computed_set else ''}"
            for c in sorted(state.available_concepts)[:200]
        ]
        if len(state.available_concepts) > 200:
            concept_lines.append(f"  … and {len(state.available_concepts) - 200} more")
        concepts_block = "\n".join(concept_lines)

        # ── Prior results block ────────────────────────────────────────────────
        if state.tool_results:
            prior_results_block = "\n".join(
                f"  [{r.tool_name}] {'✓' if r.success else f'✗ {r.error}'} — {r.summary}"
                for r in state.tool_results
            )
        else:
            prior_results_block = "  (none)"

        # ── Replanning note for failure context ───────────────────────────────
        replanning_note = ""
        if state.replanning_due_to_failure:
            replanning_note = (
                "NOTE: You are RE-PLANNING because one or more tools failed "
                "in the previous batch (see results above). "
                "Emit only the REMAINING work — do not re-emit calls that "
                "already succeeded."
            )

        user_msg = _TOOL_PLANNER_USER.format(
            query=state.query,
            ticker=state.ticker or "N/A",
            start_date=(
                state.start_date.strftime("%Y-%m-%d") if state.start_date else "N/A"
            ),
            end_date=state.end_date.strftime("%Y-%m-%d") if state.end_date else "N/A",
            replanning_note=replanning_note,
            n_concepts=len(state.available_concepts),
            concepts_block=concepts_block,
            prior_summary=prior_results_block,
            tool_descriptions=get_tool_descriptions(),
            iteration=state.iteration_count + 1,
            max_iterations=MAX_TOOL_ITERATIONS,
        )

        structured_llm = service_manager.get_agent(
            temperature=0
        ).with_structured_output(IterativeToolPlan)

        try:
            tool_plan: IterativeToolPlan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=_TOOL_PLANNER_SYSTEM),
                    HumanMessage(content=user_msg),
                ]
            )
        except Exception as exc:
            logger.error("[tool_planner] LLM call failed: %s", exc)
            return {
                "tool_plan": IterativeToolPlan(
                    batches=[],
                    data_summary=f"Tool planning failed: {exc}. Proceeding with raw data.",
                ),
                "current_batch_index": 0,
                "replanning_due_to_failure": False,
            }

        logger.info(
            "[tool_planner] Plan: %d batch(es) — %s",
            tool_plan.batch_count(),
            tool_plan.data_summary,
        )
        for i, batch in enumerate(tool_plan.batches):
            logger.info(
                "  Batch %d (%d call(s)): %s",
                i,
                len(batch.calls),
                batch.batch_reasoning or "(no reasoning)",
            )
            for j, spec in enumerate(batch.calls, 1):
                logger.info("    %d. %s — %s", j, spec.tool_name, spec.reasoning)

        return {
            "tool_plan": tool_plan,
            "current_batch_index": 0,  # always reset on a new plan
            "replanning_due_to_failure": False,
        }

    # ── Node: tool_executor ───────────────────────────────────────────────────

    async def _tool_executor_node(self, state: _AgentState) -> Dict:
        """
        Execute the current batch (state.current_batch_index) in parallel.

        After execution:
        - Merges added_rows into financial_data.
        - Persists computed time-series rows to SQLite.
        - Advances current_batch_index by 1.
        - Increments iteration_count (for the MAX guard).

        The router then decides: advance to next batch, re-plan, or go to analyst.
        """
        plan = state.tool_plan
        batch_index = state.current_batch_index

        if plan is None or plan.is_empty():
            logger.info("[tool_executor] No plan or empty plan — skipping.")
            return {
                "iteration_count": state.iteration_count + 1,
                "current_batch_index": batch_index + 1,
            }

        batch: ToolCallBatch | None = plan.get_batch(batch_index)
        if batch is None or not batch.calls:
            logger.info(
                "[tool_executor] Batch %d is empty or out of range — skipping.",
                batch_index,
            )
            return {
                "iteration_count": state.iteration_count + 1,
                "current_batch_index": batch_index + 1,
            }

        logger.info(
            "[Node] tool_executor — batch %d/%d (%d call(s))",
            batch_index,
            plan.batch_count() - 1,
            len(batch.calls),
        )

        df = (
            state.financial_data if state.financial_data is not None else pd.DataFrame()
        )

        async def _run_one(spec: ToolCallSpec) -> ToolResult:
            tool = TOOL_REGISTRY.get(spec.tool_name)
            if tool is None:
                return ToolResult(
                    tool_name=spec.tool_name,
                    success=False,
                    error=f"Unknown tool: {spec.tool_name}",
                )
            try:
                return tool.execute(
                    df=df, params=tool.parameters_schema(**spec.parameters)
                )
            except Exception as exc:
                logger.error("[tool_executor] %s failed: %s", spec.tool_name, exc)
                return ToolResult(
                    tool_name=spec.tool_name, success=False, error=str(exc)
                )

        batch_results: List[ToolResult] = await asyncio.gather(
            *[_run_one(s) for s in batch.calls]
        )

        # ── Merge added_rows back into the DataFrame ──────────────────────────
        newly_added_labels: List[str] = []
        for result in batch_results:
            if result.success and result.added_rows:
                new_rows = pd.DataFrame.from_dict(result.added_rows, orient="index")
                all_cols = df.columns.union(new_rows.columns)
                df = df.reindex(columns=all_cols)
                new_rows = new_rows.reindex(columns=all_cols)
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

        updated_concepts = list(df.index) if not df.empty else state.available_concepts
        updated_computed = list(
            set(state.computed_row_labels) | set(newly_added_labels)
        )

        return {
            "financial_data": df,
            "tool_results": list(state.tool_results) + batch_results,
            "iteration_count": state.iteration_count + 1,
            "current_batch_index": batch_index + 1,
            "available_concepts": updated_concepts,
            "computed_row_labels": updated_computed,
        }

    async def _persist_computed_rows(
        self,
        ticker: str,
        df: pd.DataFrame,
        row_labels: List[str],
        form_type: str,
    ) -> None:
        """Saves newly computed time-series rows to SQLite under statement_type='computed'."""
        records = []
        for label in row_labels:
            if label not in df.index:
                continue
            row = df.loc[label].dropna()
            if len(row) < 2:
                logger.debug("[persist] Skipping scalar/single-period row '%s'.", label)
                continue
            for date_col, value in row.items():
                records.append(
                    {
                        "company": ticker.upper(),
                        "period_date": str(date_col),
                        "form_type": form_type,
                        "statement_type": "computed",
                        "label": label,
                        "value": float(value),
                    }
                )
        if records:
            await self.db._bulk_insert(pd.DataFrame(records))
            logger.info(
                "[persist] Saved %d computed rows for %s.", len(records), ticker
            )

    # ── Node: analyst ─────────────────────────────────────────────────────────

    async def _analyst_node(self, state: _AgentState) -> Dict:
        """
        Produce the final written analysis.

        Row selection is fully deterministic — no LLM call is made for it.
        The relevant rows are derived from:
        • IterativeToolPlan.selected_row_labels  (no-tools path)
        • ToolCallSpec.parameters values         (tool input rows)
        • ToolResult.added_rows keys             (tool output rows)

        A single LLM call then writes the analysis against those rows.
        """
        if state.financial_data is None or state.financial_data.empty:
            logger.warning("[analyst] No financial data — returning empty analysis.")
            return {
                "financial_data": pd.DataFrame(),
                "analysis": "No financial data was available for this query.",
                "relationships_extracted": False,
                "subgraph_id": None,
            }

        tool_summary = ""
        if state.tool_results:
            lines = [
                f"[{r.tool_name}] {'✓' if r.success else '✗'} {r.summary or r.error or ''}"
                for r in state.tool_results
            ]
            reasoning_lines = [
                f"[{r.tool_name} reasoning] {r.reasoning}"
                for r in state.tool_results
                if r.reasoning
            ]
            tool_summary = "\n".join(lines + reasoning_lines)

        # ── Deterministic row selection (no LLM) ─────────────────────────────
        filtered_df = self._extract_relevant_rows(state, state.financial_data)
        logger.info(
            "[analyst] Selected %d/%d rows deterministically for query.",
            len(filtered_df),
            len(state.financial_data),
        )

        # ── Write analysis ────────────────────────────────────────────────────
        data_str = filtered_df.to_string(max_rows=30, float_format="%.4g")
        company_context_section = (
            f"\n{state.company_context}\n" if state.company_context else ""
        )
        analysis_prompt = (
            f"Query: {state.query}\n\n"
            f"Ticker: {state.ticker}\n"
            f"{company_context_section}\n"
            f"Financial Data:\n{data_str}\n\n"
            f"Tool Results:\n{tool_summary or 'None'}"
        )
        success = False
        try:
            response = await service_manager.get_agent(temperature=0.7).ainvoke(
                [
                    SystemMessage(content=_ANALYST_SYSTEM),
                    HumanMessage(content=analysis_prompt),
                ]
            )
            raw = response.content if response else ""
            sentiment = parse_sentiment_block(raw)  # ← parse before stripping
            analysis_text = strip_sentiment_block(raw)
            success = True
        except Exception as exc:
            logger.error("[analyst] Analysis LLM call failed: %s", exc)
            analysis_text = "Analysis could not be generated due to an internal error."

        task_id = None
        if state.conversation_id and analysis_text and success:
            turn_id = getattr(state, "turn_id", None) or state.conversation_id
            try:
                task = make_extraction_task(
                    turn_id=turn_id,
                    conversation_id=state.conversation_id,
                    source_agent=self.name(),
                    extraction_text=analysis_text,
                    system_prompt=FUNDAMENTAL_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
                    llm_config={"temperature": 0.7},
                )
                task_id = await service_manager.get_graph_queue_manager().enqueue(task)
            except Exception:
                logger.exception(
                    "_analyst_node: failed to enqueue deferred relationship extraction"
                )

        return {
            "financial_data": filtered_df,
            "analysis": analysis_text,
            "relationships_extracted": False,
            "subgraph_id": task_id,
            "sentiment": sentiment,
        }

    def _extract_relevant_rows(
        self,
        state: _AgentState,
        financial_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Deterministically select the DataFrame rows relevant to this analysis.

        Sources (in priority order):
        1. No-tools path  — selected_row_labels from IterativeToolPlan.
        2. Input rows     — every string / list-of-strings value inside each
                            ToolCallSpec.parameters that matches an index label.
        3. Output rows    — added_rows keys from every successful ToolResult.

        Preserves the original DataFrame row ordering.
        """
        labels: set[str] = set()
        available_index: list[str] = list(financial_data.index)
        available_set: set[str] = set(available_index)

        plan = state.tool_plan
        if plan is not None:
            # No-tools path: planner explicitly nominated the rows to analyse.
            for label in plan.selected_row_labels or []:
                if label in available_set:
                    labels.add(label)

            # Input rows: scan all parameter values in every ToolCallSpec.
            for batch in plan.batches:
                for spec in batch.calls:
                    for v in spec.parameters.values():
                        if isinstance(v, str) and v in available_set:
                            labels.add(v)
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, str) and item in available_set:
                                    labels.add(item)

        # Output rows: every row that a tool wrote back into the DataFrame.
        for result in state.tool_results or []:
            if result.success and result.added_rows:
                for row_label in result.added_rows:
                    if row_label in available_set:
                        labels.add(row_label)

        if not labels:
            logger.warning(
                "[analyst] _extract_relevant_rows: no labels resolved — "
                "falling back to full DataFrame."
            )
            return financial_data

        # Preserve original index ordering.
        ordered = [lbl for lbl in available_index if lbl in labels]
        return financial_data.loc[ordered]
