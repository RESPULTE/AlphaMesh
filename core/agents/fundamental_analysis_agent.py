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
from datetime import datetime, timedelta
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
    ToolCallSpec,
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
        Concurrently fetches/caches EDGAR filings and yfinance price data via
        self.db (FinancialDatabase), then loads and normalises the result.

        Granularity:
        yearly   (default) — 10-K filings, 5-year window, prices resampled yearly.
        quarterly           — 10-Q filings, last 8 quarters, prices resampled quarterly.
        """
        ticker: str = state.ticker
        granularity: str = getattr(state, "granularity", "yearly")

        if not ticker:
            logger.warning("[data_prep] No ticker — returning empty state.")
            return {"financial_data": pd.DataFrame(), "available_concepts": []}

        # ── Resolve date window and filing parameters ─────────────────────────
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

        # ── Concurrent EDGAR update + price fetch via self.db ─────────────────
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
            start_date=(
                state.start_date.strftime("%Y-%m-%d") if state.start_date else "N/A"
            ),
            end_date=state.end_date.strftime("%Y-%m-%d") if state.end_date else "N/A",
            max_iterations=MAX_TOOL_ITERATIONS,
            n_concepts=len(state.available_concepts),
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
            *[_run_one(s) for s in plan.calls]
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
            "financial_data": df if not df.empty else state.financial_data,
            "tool_results": list(state.tool_results) + batch_results,
            "iteration_count": state.iteration_count + 1,
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
        """
        Saves newly computed time-series rows back into the financials SQLite
        table under statement_type='computed'.  Scalar/single-period rows are
        skipped as they have no meaningful time dimension to store.
        """
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
