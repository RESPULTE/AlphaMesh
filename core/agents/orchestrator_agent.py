"""
core/agents/orchestrator_agent.py

Changes from previous version
──────────────────────────────
1. run() now calls graph_queue_manager.open_session() before graph.ainvoke()
   and graph_queue_manager.flush_turn() + close_session() after it returns.
   This is the single addition required for the graph queue lifecycle.

2. _synthesize_node() replaces subgraph_svc.schedule() with a direct
   GraphQueueManager.enqueue() call for cross-domain relationships.
   User signal writeback now enqueues GraphTasks for turn-scoped batching.
Everything else is unchanged from the previous version.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph

from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models.base_agent_models import BaseAgentInput, BaseAgentOutput
from core.agents.models.news_agent_models import CitedSource
from core.agents.models.orchestrator_models import (
    FinalResponse,
    OrchestratorPlan,
    OrchestratorState,
    SynthesisResult,
)
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.prompts.orchestrator_agent_prompts import (
    ORCHESTRATOR_PLANNER_SYSTEM_PROMPT,
    SYNTHESISER_SINGLE_AGENT_PROMPT,
    SYNTHESISER_USER_CONTEXT_SECTION,
    build_writeback_system_prompt,
)
from core.agents.utils import (
    _build_clarification_message,
    _build_combined_company_context,
    _extract_last_human_message,
    _last_n_messages,
    _safe_json,
)
from core.config import settings
from core.logger import get_logger
from core.memory.graph.extraction_prompts import DEFERRED_RELATIONSHIP_SYSTEM_PROMPT
from core.memory.graph.graph_queue import make_extraction_task, make_graph_task
from core.memory.user_signal_writeback import (
    build_signal_payload,
    build_user_signal_relationships,
    update_user_signal_cache,
)
from core.services import service_manager

logger = get_logger(__name__)

_RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
_CROSS_RE = re.compile(
    r"<cross_domain_relationships>(.*?)</cross_domain_relationships>",
    re.DOTALL | re.IGNORECASE,
)
_INTEREST_RE = re.compile(
    r"<user_interest_relationships>(.*?)</user_interest_relationships>",
    re.DOTALL | re.IGNORECASE,
)

AVAILABLE_AGENTS: List[type] = [NewsAnalysisAgent, FundamentalAnalysisAgent]
_SYNTHESIS_HISTORY_WINDOW: int = 6


class OrchestratorAgent:
    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)
        self._agents: Dict[str, AbstractAgent] = {
            agent.name(): agent() for agent in AVAILABLE_AGENTS
        }
        self._graph = self._build_graph()

    def name(self) -> str:
        return "Orchestrator Agent"

    @staticmethod
    def get_portfolio(path: str) -> List[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except FileNotFoundError:
            logger.warning("Portfolio file not found at %s", path)
            return []
        except Exception:
            logger.exception("Failed to load portfolio from %s", path)
            return []

    # ── Graph wiring ──────────────────────────────────────────────────────────

    def _router(self, state: OrchestratorState) -> str:
        if state.plan is None:
            return "synthesiser"
        if state.plan.final_answer is not None:
            return "direct_answer"
        if state.plan.target_agents:
            return "validate_and_enrich"
        return "synthesiser"

    def _enrichment_router(self, state: OrchestratorState) -> str:
        if state.plan and state.plan.final_answer is not None:
            return "direct_answer"
        if state.plan and state.plan.target_agents:
            return "execute_agents"
        return "synthesiser"

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState, output_schema=FinalResponse)
        workflow.add_node("planner", self._plan_node)
        workflow.add_node("direct_answer", self._direct_answer_node)
        workflow.add_node("validate_and_enrich", self._validate_and_enrich_node)
        workflow.add_node("execute_agents", self._execute_node)
        workflow.add_node("synthesiser", self._synthesize_node)
        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self._router,
            {
                "direct_answer": "direct_answer",
                "validate_and_enrich": "validate_and_enrich",
                "synthesiser": "synthesiser",
            },
        )
        workflow.add_conditional_edges(
            "validate_and_enrich",
            self._enrichment_router,
            {
                "direct_answer": "direct_answer",
                "execute_agents": "execute_agents",
                "synthesiser": "synthesiser",
            },
        )
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("execute_agents", "synthesiser")
        workflow.add_edge("synthesiser", END)
        return workflow.compile()

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        messages: List[BaseMessage],
        conversation_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> FinalResponse:
        """
        Entry point for one conversation turn.

        Graph queue lifecycle:
          1. open_session() — ensures a ConversationQueue exists for this conversation.
          2. graph.ainvoke() — agents enqueue GraphTasks during execution.
          3. flush_turn()    — signals consumer to process all tasks for this turn.
          4. close_session() is NOT called here — sessions persist across turns
             and are closed by the API layer when the user disconnects.
        """
        user_context_block = "USER CONTEXT: None"
        if user_email:
            try:
                svc = service_manager.get_user_context_service()
                user_context_block = (
                    svc.get_formatted_context(user_email) or "USER CONTEXT: None"
                )
            except Exception:
                logger.exception(
                    "run: failed to read user context from cache for %s", user_email
                )

        turn_id = str(uuid4())
        logger.info("run: turn_id=%s", turn_id)

        # ── Step 1: open session (idempotent — no-op if already open) ─────────
        if conversation_id:
            try:
                await service_manager.get_graph_queue_manager().open_session(
                    conversation_id
                )
            except Exception:
                logger.exception(
                    "run: failed to open graph queue session for '%s'", conversation_id
                )

        initial_state = OrchestratorState(
            messages=messages,
            conversation_id=conversation_id,
            user_email=user_email,
            user_context_block=user_context_block,
            turn_id=turn_id,
        )

        try:
            raw = await self._graph.ainvoke(initial_state)
        except Exception:
            logger.exception("Graph invocation failed")
            return FinalResponse(
                summary="I encountered an internal error. Please try again."
            )

        # ── Step 2: flush turn (fire-and-forget — returns immediately) ────────
        if conversation_id:
            try:
                await service_manager.get_graph_queue_manager().flush_turn(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )
            except Exception:
                logger.exception(
                    "run: failed to flush graph queue turn_id='%s'", turn_id
                )

        return self._coerce_final_response(raw)

    @staticmethod
    def _coerce_final_response(raw: Any) -> FinalResponse:
        if raw is None:
            return FinalResponse(summary="")
        if isinstance(raw, FinalResponse):
            return raw
        if isinstance(raw, dict):
            return FinalResponse(
                summary=raw.get("summary") or "",
                fundamental_data=raw.get("fundamental_data"),
                sources=raw.get("sources") or [],
                agent_analyses=raw.get("agent_analyses") or {},
            )
        return FinalResponse(
            summary=getattr(raw, "summary", "") or "",
            fundamental_data=getattr(raw, "fundamental_data", None),
            sources=getattr(raw, "sources", []) or [],
            agent_analyses=getattr(raw, "agent_analyses", {}) or {},
        )

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_synthesis_prompt(
        self,
        user_context_section: str,
        multi_agent: bool,
        investment_signals: Optional[List] = None,
        learning_signals: Optional[List] = None,
    ) -> ChatPromptTemplate:
        if multi_agent:
            system_prompt = (
                user_context_section
                + "\n\n"
                + build_writeback_system_prompt(investment_signals, learning_signals)
                + "\n\nAgent Findings:\n{context}"
            )
            human_prompt = "Produce cross-domain relationships, user interest relationships (if applicable), and final analysis."
        else:
            system_prompt = (
                user_context_section
                + SYNTHESISER_SINGLE_AGENT_PROMPT
                + "\n\nAgent Findings:\n{context}"
            )
            human_prompt = "Produce the final analysis response."

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", human_prompt),
            ]
        )

    async def _invoke_synthesis_llm(
        self,
        prompt: ChatPromptTemplate,
        state: OrchestratorState,
        context_parts: List[str],
        portfolio_block: str,
        history: List[BaseMessage],
    ) -> str:
        chain = prompt | self._llm
        response = await chain.ainvoke(
            {
                "history": history,
                "context": "\n\n".join(context_parts),
                "user_context": state.user_context_block,
                "portfolio": portfolio_block,
            }
        )
        return response.content if response else ""

    def _parse_synthesis_output(self, raw: str, multi_agent: bool) -> SynthesisResult:
        cross_relationships: List[dict] = []
        interest_edges: List[dict] = []

        if multi_agent:
            cross_match = _CROSS_RE.search(raw)
            if cross_match:
                cross_relationships = _safe_json(cross_match.group(1))
            interest_match = _INTEREST_RE.search(raw)
            if interest_match:
                interest_edges = _safe_json(interest_match.group(1))

        resp_match = _RESPONSE_RE.search(raw)
        analysis_text = resp_match.group(1).strip() if resp_match else raw.strip()

        return SynthesisResult(
            analysis_text=analysis_text,
            cross_relationships=cross_relationships,
            interest_edges=interest_edges,
        )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        available_agents_desc = "\n".join(
            f"  {agent.name()}: {agent.description()}" for agent in AVAILABLE_AGENTS
        )
        system_content = ORCHESTRATOR_PLANNER_SYSTEM_PROMPT.format(
            available_agents_desc=available_agents_desc,
        )
        latest_human = _extract_last_human_message(state.messages)

        try:
            structured_llm = self._llm.with_structured_output(OrchestratorPlan)
            plan: OrchestratorPlan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_content),
                    HumanMessage(content=latest_human),
                ]
            )
            logger.info(
                "_plan_node: agents=%s needs_memory=%s final_answer=%s ticker=%s",
                plan.target_agents,
                plan.needs_memory,
                plan.final_answer is not None,
                plan.ticker,
            )
            return {"plan": plan}
        except Exception:
            logger.exception("_plan_node: LLM call failed — using safe fallback plan")
            return {"plan": OrchestratorPlan()}

    async def _direct_answer_node(self, state: OrchestratorState) -> Dict[str, Any]:
        answer = (state.plan.final_answer or "").strip() if state.plan else ""
        return {"summary": answer, "sources": [], "fundamental_data": None}

    async def _validate_and_enrich_node(
        self, state: OrchestratorState
    ) -> Dict[str, Any]:
        from core.agents.ticker_validation import TickerInfo

        plan = state.plan
        tickers: List[str] = getattr(plan, "tickers", []) if plan else []
        if not tickers:
            return {"company_context_blocks": {}}

        try:
            validator = service_manager.get_ticker_validator()
            results: Dict[str, TickerInfo] = await validator.validate_and_enrich(
                tickers
            )
        except Exception:
            logger.exception("_validate_and_enrich_node: validation failed")
            return {"company_context_blocks": {}}

        needing_confirmation = {
            t: info
            for t, info in results.items()
            if info.needs_confirmation or not info.is_valid
        }
        if needing_confirmation:
            clarification = _build_clarification_message(needing_confirmation)
            updated_plan = plan.model_copy(update={"final_answer": clarification})
            return {"plan": updated_plan, "company_context_blocks": {}}

        company_context_blocks: Dict[str, str] = {}
        for t, info in results.items():
            if info.is_valid and info.is_equity:
                block = info.to_context_block()
                if block:
                    company_context_blocks[t] = block

        ticker_metadata: Dict[str, dict] = {}
        for t, info in results.items():
            if info.is_valid and info.is_equity:
                ticker_metadata[t] = {
                    "long_name": info.long_name,
                    "sector": info.sector,
                    "industry": info.industry,
                    "description": info.description,
                }

        return {
            "company_context_blocks": company_context_blocks,
            "ticker_metadata": ticker_metadata,
        }

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = state.plan
        if not plan or not plan.target_agents:
            return {"agent_outputs": {}}

        valid_names = [n for n in plan.target_agents if n in self._agents]
        primary_ticker = plan.tickers[0] if plan.tickers else None
        combined_context = _build_combined_company_context(
            plan.tickers, state.company_context_blocks
        )

        tasks = []
        for name in valid_names:
            agent_query = plan.per_agent_queries.get(name) or plan.query
            agent_input: BaseAgentInput = plan.model_copy(
                update={
                    "query": agent_query,
                    "ticker": primary_ticker,
                    "conversation_id": state.conversation_id,
                    "company_context": combined_context,
                }
            )
            tasks.append((name, self._agents[name].run(agent_input)))

        if not tasks:
            return {"agent_outputs": {}}

        names, coros = zip(*tasks)
        results = await asyncio.gather(*coros, return_exceptions=True)

        outputs: Dict[str, BaseAgentOutput] = {}
        for name, res in zip(names, results):
            if isinstance(res, Exception):
                logger.error(
                    "_execute_node: agent '%s' failed — %s", name, res, exc_info=res
                )
            else:
                outputs[name] = res

        return {"agent_outputs": outputs}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        context_parts: List[str] = []
        fundamental_df = None
        news_sources: List[CitedSource] = []

        for name, output in state.agent_outputs.items():
            try:
                context_parts.append(output.get_llm_context_str())
            except Exception:
                logger.exception(
                    "_synthesize_node: get_llm_context_str failed for '%s'", name
                )
            if name == "fundamentals_agent":
                fundamental_df = getattr(output, "financial_data", None)
            if name == "news_agent":
                news_sources = getattr(output, "sources", []) or []

        portfolio = self.get_portfolio(settings.PORTFOLIO_JSON_PATH)
        portfolio_block = json.dumps(portfolio, indent=2) if portfolio else "[]"
        multi_agent = len(state.agent_outputs) > 1

        user_context_section = SYNTHESISER_USER_CONTEXT_SECTION.format(
            user_context=state.user_context_block
        )
        history_window = _last_n_messages(state.messages, _SYNTHESIS_HISTORY_WINDOW)

        analysis_text = ""
        synthesis_result = SynthesisResult(analysis_text="")

        if not context_parts and not state.user_context_block and not portfolio_block:
            analysis_text = (
                "I wasn't able to retrieve data for your query at this time. "
                "Please try again or rephrase your question."
            )
        else:
            try:
                investment_signals = (
                    (state.plan.detected_investment_signals or []) if state.plan else []
                )
                learning_signals = (
                    (state.plan.detected_learning_signals or []) if state.plan else []
                )
                prompt = self._build_synthesis_prompt(
                    user_context_section,
                    multi_agent,
                    investment_signals=investment_signals if multi_agent else None,
                    learning_signals=learning_signals if multi_agent else None,
                )
                raw = await self._invoke_synthesis_llm(
                    prompt, state, context_parts, portfolio_block, history_window
                )
                synthesis_result = self._parse_synthesis_output(raw, multi_agent)
                analysis_text = synthesis_result.analysis_text
            except Exception:
                logger.exception("_synthesize_node: LLM synthesis failed")
                analysis_text = (
                    "I encountered an error while synthesising the analysis. "
                    "The raw data was retrieved but could not be summarised."
                )

        # Cross-domain graph write-back (queued via GraphQueueManager.enqueue)
        if multi_agent and state.conversation_id:
            try:
                if synthesis_result.cross_relationships:
                    task = make_graph_task(
                        turn_id=state.turn_id,
                        conversation_id=state.conversation_id,
                        source_agent=self.name(),
                        relationships=synthesis_result.cross_relationships,
                    )
                    await service_manager.get_graph_queue_manager().enqueue(task)
                elif analysis_text:
                    task = make_extraction_task(
                        turn_id=state.turn_id,
                        conversation_id=state.conversation_id,
                        source_agent=self.name(),
                        extraction_text=analysis_text,
                        system_prompt=DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
                        llm_config={"temperature": 0.0},
                    )
                    await service_manager.get_graph_queue_manager().enqueue(
                        task,
                        system_prompt=DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
                    )
            except Exception:
                logger.exception(
                    "_synthesize_node: failed to enqueue cross-domain relationships"
                )
        # User signal write-back (queued via GraphQueueManager.enqueue)
        if state.user_email and state.plan and state.conversation_id:
            user_message = _extract_last_human_message(state.messages)
            payload = build_signal_payload(
                user_email=state.user_email,
                conversation_id=state.conversation_id,
                turn_id=state.turn_id,
                ticker_metadata=state.ticker_metadata,
                user_message=user_message,
                detected_investment_signals=state.plan.detected_investment_signals
                or [],
                detected_learning_signals=state.plan.detected_learning_signals or [],
                interest_edges=synthesis_result.interest_edges or [],
            )
            relationships, cache_entries = await build_user_signal_relationships(
                payload
            )
            if relationships:
                try:
                    task = make_graph_task(
                        turn_id=state.turn_id,
                        conversation_id=state.conversation_id,
                        source_agent=self.name(),
                        relationships=relationships,
                    )
                    await service_manager.get_graph_queue_manager().enqueue(task)
                except Exception:
                    logger.exception(
                        "_synthesize_node: failed to enqueue user signal relationships"
                    )
            if cache_entries:
                update_user_signal_cache(cache_entries, state.user_email)

        per_agent_analyses: Dict[str, str] = {
            name: getattr(output, "analysis", "") or ""
            for name, output in state.agent_outputs.items()
        }

        return {
            "summary": analysis_text,
            "fundamental_data": fundamental_df,
            "sources": news_sources,
            "agent_analyses": per_agent_analyses,
        }





