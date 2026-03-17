"""
core/agents/fundamental_analysis_agent.py

Async LangGraph agent with an iterative tool-calling loop.

Changes in this revision
─────────────────────────
- The inline subgraph build/store block in `_analyst_node` has been replaced
  with a single call to
  `core.memory.graph.subgraph_extraction.schedule_subgraph_extraction`.
  The logic is identical; it now lives in one place shared with
  NewsAnalysisAgent.

- Removed direct imports of:
    InMemorySubgraphBuilder, SubgraphStore, retry_relationships_only,
    ANALYSIS_ONLY_RELATIONSHIP_PROMPT
  These are now fully encapsulated inside subgraph_extraction.py.

Everything else is unchanged.

Key behaviours (unchanged)
──────────────────────────
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
from typing import Any, Dict, List, Type

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core.agents.base_agent import AbstractAgent
from core.agents.financial_db import FinancialDatabase
from core.agents.financial_tools import TOOL_REGISTRY, ToolResult, get_tool_descriptions
from core.agents.fundamental_agent_models import (
    FundamentalAnalysisOutput,
    IterativeToolPlan,
    RelevantRowsSelection,
    _AgentState,
)
from core.agents.fundamental_agent_prompts import (
    _ANALYST_SYSTEM,
    _TOOL_PLANNER_SYSTEM,
    _TOOL_PLANNER_USER,
)
from core.agents.models import BaseAgentInput
from core.logger import get_logger
from core.memory.graph.extraction_prompts import COMBINED_ANALYSIS_RELATIONSHIP_PROMPT
from core.memory.graph.relationship_extractor import extract_with_retry
from core.memory.graph.subgraph_builder import InMemorySubgraphBuilder
from core.services import service_manager

logger = get_logger(__name__)

# ── Iteration ceiling ─────────────────────────────────────────────────────────
MAX_TOOL_ITERATIONS: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Agent
# ─────────────────────────────────────────────────────────────────────────────


class FundamentalAnalysisAgent(AbstractAgent):
    """
    Async LangGraph agent with an iterative tool-calling loop.

    The agent can loop up to MAX_TOOL_ITERATIONS times through
    tool_planner → tool_executor before proceeding to the analyst,
    enabling multi-step derived-metric computation (FCF before DCF, etc.).
    Within each iteration all tool calls execute in parallel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.db = FinancialDatabase()
        self._graph = self._build_graph()
        self._subgraph_builder = InMemorySubgraphBuilder()

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
        ticker = state.ticker or ""
        granularity = state.granularity or "yearly"

        if not ticker:
            logger.warning("[data_prep] No ticker — returning empty state.")
            return {"financial_data": pd.DataFrame(), "available_concepts": []}

        try:
            edgar_svc = service_manager.get_edgar_service()
            if granularity == "quarterly":
                await edgar_svc.ensure_quarterly_cached(ticker)
            else:
                await edgar_svc.ensure_annual_cached(ticker)
        except Exception as exc:
            logger.warning("[data_prep] EDGAR cache failed for %s: %s", ticker, exc)

        try:
            financial_df, available_concepts = await self.db.get_financial_data(
                ticker, granularity=granularity
            )
        except Exception as exc:
            logger.error("[data_prep] DB load failed for %s: %s", ticker, exc)
            return {"financial_data": pd.DataFrame(), "available_concepts": []}

        if financial_df.empty:
            logger.warning("[data_prep] No financial data found for %s", ticker)
            return {"financial_data": financial_df, "available_concepts": []}

        try:
            price_df = await service_manager.get_price_service().get_prices(
                ticker, granularity=granularity
            )
            if price_df is not None and not price_df.empty:
                financial_df = financial_df.join(price_df, how="left")
                available_concepts = list(financial_df.index)
        except Exception as exc:
            logger.warning("[data_prep] Price fetch failed for %s: %s", ticker, exc)

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
        """
        iteration = state.iteration_count + 1
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
                    iteration_reasoning="No concepts to work with.",
                )
            }

        prior_summary = ""
        if state.tool_results:
            prior_lines = [
                f"  [{r.tool_name}] {'✓' if r.success else '✗'} {r.summary or r.error or ''}"
                for r in state.tool_results
            ]
            prior_summary = "Prior tool results:\n" + "\n".join(prior_lines)

        concepts_block = "\n".join(f"  - {c}" for c in state.available_concepts)
        tool_descriptions = get_tool_descriptions()

        user_msg = _TOOL_PLANNER_USER.format(
            iteration=iteration,
            max_iterations=MAX_TOOL_ITERATIONS,
            query=state.query,
            ticker=state.ticker or "N/A",
            concepts_block=concepts_block,
            prior_summary=prior_summary or "None yet.",
            tool_descriptions=tool_descriptions,
        )

        try:
            structured_llm = service_manager.get_agent().with_structured_output(
                IterativeToolPlan
            )
            plan: IterativeToolPlan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=_TOOL_PLANNER_SYSTEM),
                    HumanMessage(content=user_msg),
                ]
            )
            logger.info(
                "[tool_planner] %d calls planned, needs_more=%s",
                len(plan.calls),
                plan.needs_more_iterations,
            )
            return {"tool_plan": plan}
        except Exception as exc:
            logger.error("[tool_planner] LLM call failed: %s", exc)
            return {
                "tool_plan": IterativeToolPlan(
                    calls=[],
                    needs_more_iterations=False,
                    data_summary="Planning failed.",
                    iteration_reasoning=str(exc),
                )
            }

    # ── Node: tool_executor ───────────────────────────────────────────────────

    async def _tool_executor_node(self, state: _AgentState) -> Dict:
        """Execute all tool calls in the current plan in parallel."""
        plan = state.tool_plan
        if not plan or not plan.calls:
            logger.info("[tool_executor] No calls to execute.")
            return {"iteration_count": state.iteration_count + 1}

        async def _run_one(spec) -> ToolResult:
            tool_cls = TOOL_REGISTRY.get(spec.tool_name)
            if tool_cls is None:
                return ToolResult(
                    tool_name=spec.tool_name,
                    success=False,
                    error=f"Unknown tool: {spec.tool_name}",
                )
            try:
                tool = tool_cls()
                result = await tool.execute(
                    params=spec.parameters,
                    financial_data=state.financial_data,
                    db=self.db,
                    ticker=state.ticker or "",
                )
                # Persist computed rows back to state and DB
                if result.output_df is not None and not result.output_df.empty:
                    new_labels = list(result.output_df.index)
                    if state.financial_data is not None:
                        state.financial_data = pd.concat(
                            [state.financial_data, result.output_df]
                        ).loc[lambda df: ~df.index.duplicated(keep="last")]
                    await self.db.persist_computed_rows(
                        state.ticker or "", result.output_df
                    )
                    return ToolResult(
                        tool_name=spec.tool_name,
                        success=True,
                        summary=result.summary,
                        reasoning=spec.reasoning,
                        output_df=result.output_df,
                        new_row_labels=new_labels,
                    )
                return ToolResult(
                    tool_name=spec.tool_name,
                    success=result.success,
                    summary=result.summary,
                    error=result.error,
                    reasoning=spec.reasoning,
                )
            except Exception as exc:
                logger.error("[tool_executor] %s failed: %s", spec.tool_name, exc)
                return ToolResult(
                    tool_name=spec.tool_name,
                    success=False,
                    error=str(exc),
                    reasoning=spec.reasoning,
                )

        results = await asyncio.gather(*[_run_one(s) for s in plan.calls])

        # Merge any new row labels into available_concepts
        new_labels: List[str] = []
        for r in results:
            new_labels.extend(r.new_row_labels or [])
        updated_concepts = list(
            dict.fromkeys(list(state.available_concepts) + new_labels)
        )

        return {
            "tool_results": list(results),
            "iteration_count": state.iteration_count + 1,
            "available_concepts": updated_concepts,
        }

    # ── Node: analyst ─────────────────────────────────────────────────────────

    async def _analyst_node(self, state: _AgentState) -> Dict:
        """
        Produce the final written analysis.

        Steps:
        1. LLM selects which financial data rows are relevant to the query.
        2. LLM writes the analysis, referencing tool results.
        3. Relationship extraction runs; subgraph is scheduled via
           schedule_subgraph_extraction (fire-and-forget unless EXTRACTION_IMMEDIATE).
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

        # ── Step 1: LLM selects relevant rows ────────────────────────────────
        row_labels = list(state.financial_data.index)
        try:
            selector_llm = service_manager.get_agent().with_structured_output(
                RelevantRowsSelection
            )
            selection: RelevantRowsSelection = await selector_llm.ainvoke(
                [
                    SystemMessage(content=_ANALYST_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Query: {state.query}\n\n"
                            f"Available rows:\n"
                            + "\n".join(f"  - {r}" for r in row_labels)
                            + f"\n\nTool results:\n{tool_summary or 'None'}"
                        )
                    ),
                ]
            )
            valid_labels = [
                r for r in (selection.relevant_row_labels or []) if r in row_labels
            ]
            if valid_labels:
                filtered_df = state.financial_data.loc[valid_labels]
                logger.info(
                    "[analyst] Filtered to %d/%d rows for query",
                    len(filtered_df),
                    len(state.financial_data),
                )
            else:
                logger.warning(
                    "[analyst] No valid relevant_row_labels returned — using full DataFrame."
                )
                filtered_df = state.financial_data
        except Exception as exc:
            logger.error("[analyst] Row selection failed: %s", exc)
            filtered_df = state.financial_data

        # ── Step 2: Write analysis ────────────────────────────────────────────
        data_str = filtered_df.to_string(max_rows=30, float_format="%.4g")
        analysis_prompt = (
            f"Query: {state.query}\n\n"
            f"Ticker: {state.ticker}\n\n"
            f"Financial Data:\n{data_str}\n\n"
            f"Tool Results:\n{tool_summary or 'None'}"
        )
        try:
            response = await service_manager.get_agent(temperature=0.7).ainvoke(
                [
                    SystemMessage(content=_ANALYST_SYSTEM),
                    HumanMessage(content=analysis_prompt),
                ]
            )
            analysis_text = response.content if response else ""
        except Exception as exc:
            logger.error("[analyst] Analysis LLM call failed: %s", exc)
            analysis_text = "Analysis could not be generated due to an internal error."

        # ── Step 3: Relationship extraction ──────────────────────────────────
        relationships = []
        relationships_extracted = False

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

        # ── Step 4: Schedule subgraph build/store (fire-and-forget) ──────────
        subgraph_id = await self._subgraph_builder.schedule_subgraph_extraction(
            agent_name=self.name(),
            conversation_id=state.conversation_id or "",
            analysis_text=analysis_text,
            relationships=relationships,
            relationships_extracted=relationships_extracted,
            llm=service_manager.get_agent(temperature=0.7),
        )

        return {
            "financial_data": filtered_df,
            "analysis": analysis_text,
            "relationships_extracted": relationships_extracted,
            "subgraph_id": subgraph_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _format_value(x: Any) -> str:
    """Format a numeric value for display in memory context."""
    if isinstance(x, float):
        if abs(x) >= 1e9:
            return f"{x / 1e9:.2f}B"
        if abs(x) >= 1e6:
            return f"{x / 1e6:.2f}M"
        return f"{x:.4g}"
    return str(x)
