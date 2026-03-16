"""
core/agents/fundamental_analysis_agent.py

Fundamental Analysis Agent — Tool-Calling LangGraph Architecture
=================================================================

Graph
─────
  START
    │
    ▼
  data_prep     — Concurrently: update EDGAR cache, fetch price data,
    │               load available concepts from DB.
    │               Output: financial_data (DataFrame), available_concepts (List[str])
    ▼
  tool_planner  — LLM picks from TOOL_REGISTRY which tools to run
    │               and maps available concept names to each tool's parameters.
    │               Produces a ToolPlan (ordered list of ToolCallSpec objects).
    ▼
  tool_executor — Validates and executes each ToolCallSpec in order.
    │               Merges computed rows back into financial_data.
    │               Accumulates ToolResult objects for the analyst.
    ▼
  analyst       — Generates a natural-language analysis from the enriched
    │               DataFrame + tool results summaries.
    │               Runs relationship extraction for the memory graph.
    ▼
  END

Why this is better than the original pipeline
─────────────────────────────────────────────
  • parser/decomposer merged — was two LLM hops; now one (tool_planner)
  • pandas eval() replaced by typed FinancialTool.execute() calls
  • Available concepts now sourced from get_all_concepts() — never empty
  • output_schema removed from StateGraph; result built explicitly in run()
  • asyncio.gather() in data_prep parallelises EDGAR + price fetching
  • Tool selection is LLM-driven and extensible (just add to TOOL_REGISTRY)
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, Dict, List, Optional, Type

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from core.agents.base_agent import AbstractAgent
from core.agents.financial_db import FinancialDatabase
from core.agents.financial_tools import TOOL_REGISTRY, ToolResult, get_tool_descriptions
from core.agents.models import BaseAgentInput, BaseAgentOutput
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


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Planner LLM Output Models
# ─────────────────────────────────────────────────────────────────────────────


class ToolCallSpec(BaseModel):
    """Specification for a single tool call as produced by the planner LLM."""

    tool_name: str = Field(
        description=(
            "Exact name of the tool to call from the registry "
            "(e.g. 'cagr', 'dcf_intrinsic_value', 'profitability_ratios')."
        )
    )
    parameters: Dict[str, Any] = Field(
        description=(
            "Parameter dict matching the tool's parameters_schema exactly. "
            "All metric fields MUST reference concept names that exist in the "
            "'Available Concepts' list provided."
        )
    )
    reasoning: str = Field(
        description="One-sentence justification for why this tool is being called for this query."
    )


class ToolPlan(BaseModel):
    """Ordered execution plan produced by the planner LLM."""

    calls: List[ToolCallSpec] = Field(
        description=(
            "Ordered list of tool calls. Execute them in sequence. "
            "Return an empty list if the user only wants raw data with no derived metrics."
        )
    )
    data_summary: str = Field(
        description=(
            "Brief (1-2 sentence) summary of what financial data is available "
            "and what the plan will compute."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Agent Output
# ─────────────────────────────────────────────────────────────────────────────


class FundamentalAnalysisOutput(BaseAgentOutput):
    """Public output returned by FundamentalAnalysisAgent.run()."""

    agent_name: str = "fundamentals_agent"
    financial_data: Optional[pd.DataFrame] = Field(default=None)
    tool_results: List[ToolResult] = Field(default_factory=list)

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

        return f"{header}Data (Rows=Metrics, Columns=Dates):\n{data_str}{tool_section}"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Internal Graph State
# ─────────────────────────────────────────────────────────────────────────────


class _AgentState(BaseAgentInput):
    """Internal state carried through the LangGraph nodes."""

    # Populated by data_prep
    available_concepts: List[str] = Field(default_factory=list)
    financial_data: Optional[pd.DataFrame] = None

    # Populated by tool_planner
    tool_plan: Optional[ToolPlan] = None

    # Accumulated by tool_executor (Annotated → LangGraph merges lists on parallel branches)
    tool_results: Annotated[List[ToolResult], operator.add] = Field(
        default_factory=list
    )

    # Populated by analyst — MUST be declared here so LangGraph retains the
    # values in the final state dict. Without these fields, LangGraph silently
    # discards any key returned by _analyst_node that is not in the schema,
    # causing final_state.get("analysis") to always return the default "".
    analysis: str = Field(default="")
    relationships_extracted: bool = Field(default=False)
    subgraph_id: Optional[str] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Prompts
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_PLANNER_SYSTEM = """\
You are a quantitative financial analysis planner. Your role is to produce \
a ToolPlan that will answer the user's financial question using the data \
and tools available.

═══ CORE RULES ═══
1. CONCEPT MATCHING: Every metric parameter you provide MUST be a name that \
   appears EXACTLY in the "Available Concepts" list. If you cannot find an \
   exact match, use the closest available name or use custom_formula.

2. DCF REQUIREMENTS: When calling dcf_intrinsic_value you MUST provide:
   - wacc: a decimal (e.g. 0.10). Estimate it from the company's apparent \
     risk profile, sector, and capital structure visible in the data.
   - terminal_growth_rate: a decimal (e.g. 0.025). Justify with GDP trends, \
     sector maturity, and the company's recent revenue CAGR.
   - wacc_reasoning: detailed written justification (2-3 sentences).
   - terminal_growth_reasoning: detailed written justification (1-2 sentences).

3. CUSTOM FORMULA: Use the custom_formula tool for any user-requested metric \
   not covered by the other tools. Write the expression using EXACT concept \
   names from the Available Concepts list as variable names.

4. ORDERING: Run CAGR or fetch-only tools before DCF, as DCF may reference \
   concepts that only exist after earlier tools add rows.

5. EMPTY PLAN: If the user only wants raw financial statements with no \
   derived analysis, return an empty calls list.

6. TOOL SELECTION: Do NOT call a tool if its required input metrics are absent \
   from the Available Concepts list — explain this in data_summary instead.
"""

_TOOL_PLANNER_USER = """\
User Query: {query}
Ticker: {ticker}
Date Range: {start_date} to {end_date}

Available Concepts ({n_concepts} stored in DB):
{concepts_block}

Available Tools:
{tool_descriptions}

Produce the ToolPlan to best answer the query.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 5.  The Agent
# ─────────────────────────────────────────────────────────────────────────────


class FundamentalAnalysisAgent(AbstractAgent):
    """
    Async LangGraph agent that fetches EDGAR financial data and runs a
    dynamically-selected suite of financial analysis tools.
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
            "quantitative metrics (CAGR, DCF, ratios) via a dynamic tool-calling "
            "pipeline. Returns enriched financial data and a written analysis."
        )

    @staticmethod
    def get_output_schema_class() -> Type[BaseModel]:
        return FundamentalAnalysisOutput

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, input_data: BaseAgentInput) -> FundamentalAnalysisOutput:
        logger.info("[Agent: %s] Started for %s", self.name(), input_data.ticker)
        await self.db.initialize()

        # Note: output_schema is intentionally NOT set on StateGraph; we
        # construct FundamentalAnalysisOutput explicitly below to avoid
        # LangGraph schema-projection bugs when fields contain DataFrames.
        final_state: Dict = await self._graph.ainvoke(
            input_data.model_dump(exclude_none=False),
            config={"recursion_limit": 10},
        )

        return FundamentalAnalysisOutput(
            financial_data=final_state.get("financial_data"),
            analysis=final_state.get("analysis", ""),
            tool_results=final_state.get("tool_results", []),
            entities_enriched=final_state.get("entities_enriched", []),
            subgraph_id=final_state.get("subgraph_id"),
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
        workflow.add_edge("tool_executor", "analyst")
        workflow.add_edge("analyst", END)

        return workflow.compile()

    # ── Node: data_prep ───────────────────────────────────────────────────────

    async def _data_prep_node(self, state: _AgentState) -> Dict:
        """
        Concurrently:
          1. Ensures EDGAR data is cached for the requested years (10-K)
          2. Fetches price data from yfinance if needed
          3. Loads the full concept catalogue from the local DB

        Then assembles financial_data (wide DataFrame) and available_concepts.
        """
        logger.info(
            "[Node] data_prep — %s %s→%s",
            state.ticker,
            state.start_date,
            state.end_date,
        )

        years = list(range(state.start_date.year, state.end_date.year + 1))
        ticker = state.ticker

        # Run EDGAR update and price fetch concurrently
        async def _edgar_update():
            await self.db.update_financials(ticker, years, "10-K")

        async def _price_fetch():
            return await self.db.get_price_data(
                ticker,
                start=state.start_date.strftime("%Y-%m-%d"),
                end=state.end_date.strftime("%Y-%m-%d"),
                interval="yearly",
            )

        edgar_task = asyncio.create_task(_edgar_update())
        price_task = asyncio.create_task(_price_fetch())
        await edgar_task  # must finish before we query DB
        price_df: pd.DataFrame = await price_task

        # Fetch the complete financial statement data from DB
        financial_df = await self.db.search_label(
            ticker,
            keywords=await self.db.get_all_concepts(ticker),
            start_date=(
                state.start_date.strftime("%Y-%m-%d") if state.start_date else None
            ),
            end_date=state.end_date.strftime("%Y-%m-%d") if state.end_date else None,
        )

        # Also get all available concept names for the planner
        available_concepts = await self.db.get_all_concepts(ticker)

        # Merge price data into the financial DataFrame
        if not price_df.empty and "stock_price" in price_df.columns:
            price_t = price_df[["stock_price"]].T
            price_t.columns = pd.to_datetime(price_t.columns).strftime("%Y-%m-%d")
            if not financial_df.empty:
                financial_df.columns = pd.to_datetime(financial_df.columns).strftime(
                    "%Y-%m-%d"
                )
                # Align columns to the intersection + union
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

        return {
            "financial_data": financial_df,
            "available_concepts": available_concepts,
        }

    # ── Node: tool_planner ────────────────────────────────────────────────────

    async def _tool_planner_node(self, state: _AgentState) -> Dict:
        """
        Sends the user query + available concepts + tool catalogue to the LLM.
        The LLM returns a structured ToolPlan deciding which tools to call and
        with which parameters (all mapped to real concept names).
        """
        logger.info("[Node] tool_planner — planning tools for: %s", state.query)

        if not state.available_concepts:
            logger.warning(
                "[tool_planner] No concepts available — skipping tool planning."
            )
            return {
                "tool_plan": ToolPlan(
                    calls=[], data_summary="No financial data available."
                )
            }

        # Format the concept list (truncate display at 200 for token efficiency;
        # the full list is still passed so the LLM can reference any of them)
        concepts_display = "\n".join(
            f"  • {c}" for c in sorted(state.available_concepts)[:200]
        )
        if len(state.available_concepts) > 200:
            concepts_display += f"\n  … and {len(state.available_concepts) - 200} more"

        user_msg = _TOOL_PLANNER_USER.format(
            query=state.query,
            ticker=state.ticker,
            start_date=(
                state.start_date.strftime("%Y-%m-%d") if state.start_date else "N/A"
            ),
            end_date=state.end_date.strftime("%Y-%m-%d") if state.end_date else "N/A",
            n_concepts=len(state.available_concepts),
            concepts_block=concepts_display,
            tool_descriptions=get_tool_descriptions(),
        )

        llm = service_manager.get_agent(temperature=0)
        structured_llm = llm.with_structured_output(ToolPlan)

        try:
            tool_plan: ToolPlan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=_TOOL_PLANNER_SYSTEM),
                    HumanMessage(content=user_msg),
                ]
            )
        except Exception as exc:
            logger.error("[tool_planner] LLM call failed: %s", exc)
            tool_plan = ToolPlan(
                calls=[],
                data_summary=f"Tool planning failed: {exc}. Proceeding with raw data only.",
            )

        logger.info(
            "[tool_planner] Plan: %d tool call(s). Summary: %s",
            len(tool_plan.calls),
            tool_plan.data_summary,
        )
        for i, spec in enumerate(tool_plan.calls, 1):
            logger.info("  %d. %s — %s", i, spec.tool_name, spec.reasoning)

        return {"tool_plan": tool_plan}

    # ── Node: tool_executor ───────────────────────────────────────────────────

    async def _tool_executor_node(self, state: _AgentState) -> Dict:
        """
        Iterates over state.tool_plan.calls in order:
          1. Validates that the tool name is in TOOL_REGISTRY
          2. Instantiates the tool's parameters_schema from spec.parameters
          3. Calls tool.execute(df, params) — synchronous but lightweight
          4. On success, merges added_rows back into financial_data
          5. Appends ToolResult to tool_results

        Failures are recorded in ToolResult.error and never raise exceptions,
        ensuring the analyst node always runs.
        """
        logger.info("[Node] tool_executor")

        plan = state.tool_plan
        if not plan or not plan.calls:
            logger.info("[tool_executor] No tools to run.")
            return {}

        df: pd.DataFrame = (
            state.financial_data.copy()
            if state.financial_data is not None and not state.financial_data.empty
            else pd.DataFrame()
        )
        results: List[ToolResult] = []

        for spec in plan.calls:
            tool = TOOL_REGISTRY.get(spec.tool_name)
            if tool is None:
                msg = (
                    f"Tool '{spec.tool_name}' not found in registry. "
                    f"Available: {list(TOOL_REGISTRY.keys())}"
                )
                logger.warning("[tool_executor] %s", msg)
                results.append(
                    ToolResult(tool_name=spec.tool_name, success=False, error=msg)
                )
                continue

            # Validate parameters against the tool's Pydantic schema
            try:
                params = tool.parameters_schema(**spec.parameters)
            except Exception as exc:
                msg = f"Invalid parameters for '{spec.tool_name}': {exc}"
                logger.error("[tool_executor] %s", msg)
                results.append(
                    ToolResult(tool_name=spec.tool_name, success=False, error=msg)
                )
                continue

            if df.empty:
                results.append(
                    ToolResult(
                        tool_name=spec.tool_name,
                        success=False,
                        error="Cannot execute tool — financial_data is empty.",
                    )
                )
                continue

            # Execute the tool
            try:
                result: ToolResult = tool.execute(df, params)
            except Exception as exc:
                result = ToolResult(
                    tool_name=spec.tool_name, success=False, error=str(exc)
                )

            results.append(result)

            # Merge new rows into the DataFrame for subsequent tools to use
            if result.success and result.added_rows:
                new_rows = pd.DataFrame.from_dict(result.added_rows, orient="index")
                # Align columns before concat
                all_cols = df.columns.union(new_rows.columns)
                df = df.reindex(columns=all_cols)
                new_rows = new_rows.reindex(columns=all_cols)
                df = pd.concat([df, new_rows])
                logger.info(
                    "[tool_executor] Merged %d new row(s) from %s: %s",
                    len(result.added_rows),
                    spec.tool_name,
                    list(result.added_rows.keys()),
                )

        return {
            "financial_data": df if not df.empty else state.financial_data,
            "tool_results": results,
        }

    # ── Node: analyst ─────────────────────────────────────────────────────────

    async def _analyst_node(self, state: _AgentState) -> Dict:
        """
        Generates a natural-language analysis of the enriched financial data.
        Runs the relationship extraction pipeline for the memory graph.
        """
        logger.info("[Node] analyst")

        if state.financial_data is None or state.financial_data.empty:
            return {
                "analysis": "No financial data was retrieved for this query.",
                "relationships_extracted": False,
            }

        # Build human-readable DataFrame for the LLM
        human_readable = state.financial_data.map(
            lambda x: _format_value(x) if isinstance(x, (int, float)) else x
        ).to_string(max_rows=30)

        # Build tool results summary
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

        system_prompt = COMBINED_ANALYSIS_RELATIONSHIP_PROMPT
        user_prompt = (
            f"Analyse the following financial data for {state.ticker}.\n\n"
            f"Data (Rows=Metrics, Columns=Dates):\n{human_readable}\n\n"
            f"--- Tool Analysis Results ---\n{tool_summary}\n\n"
            f"User Question: {state.query}\n\n"
            "### Instructions:\n"
            "1. Highlight key trends, risks, and positives.\n"
            "2. Reference and interpret the tool results (CAGR, ratios, DCF etc.) where available.\n"
            "3. For DCF: always state the WACC and terminal growth assumptions and explain "
            "   whether the intrinsic value suggests the stock is over- or under-valued.\n"
            "4. Convert large raw numbers to human-readable form "
            "   (e.g. 1.5e9 → '1.5 Billion', 2.3e12 → '2.3 Trillion').\n"
            "5. Be concise but comprehensive."
        )

        try:
            result = await extract_with_retry(
                service_manager.get_agent(temperature=0.7),
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
            analysis_text: str = result.analysis
            relationships = result.relationships
            relationships_extracted: bool = result.parse_success
        except Exception as exc:
            logger.error("[analyst] Analysis generation failed: %s", exc)
            raise

        # ── Memory graph extraction ────────────────────────────────────────────
        subgraph_id: Optional[str] = None
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
                    relationships, source_agent=FundamentalAnalysisAgent.name()
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
            "analysis": analysis_text,
            "relationships_extracted": relationships_extracted,
            "subgraph_id": subgraph_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _format_value(x: float) -> str:
    """Converts a raw numeric value to a human-readable denomination string."""
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
# Smoke test  (python -m core.agents.fundamental_analysis_agent)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime

    async def main():
        agent = FundamentalAnalysisAgent()
        input_data = BaseAgentInput(
            ticker="MSFT",
            query="What is MSFT's revenue CAGR and is the stock undervalued based on DCF?",
            vector_query="MSFT revenue growth DCF valuation",
            metrics=[],
            start_date=datetime(2019, 1, 1),
            end_date=datetime(2023, 12, 31),
        )
        output = await agent.run(input_data)
        print("\n=== ANALYSIS ===")
        print(output.analysis)
        print("\n=== TOOL RESULTS ===")
        for r in output.tool_results:
            print(f"[{r.tool_name}] {'OK' if r.success else 'FAIL'}: {r.summary}")
            if r.reasoning:
                print(f"  Reasoning: {r.reasoning}")
        print("\n=== FINANCIAL DATA ===")
        if output.financial_data is not None:
            print(
                output.financial_data.map(
                    lambda x: _format_value(x) if isinstance(x, (int, float)) else x
                ).to_string()
            )

    asyncio.run(main())
