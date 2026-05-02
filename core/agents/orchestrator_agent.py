"""
core/agents/orchestrator_agent.py

1. run() now calls graph_queue_manager.open_session() before graph.ainvoke()
   and graph_queue_manager.flush_turn() + close_session() after it returns.
   This is the single addition required for the graph queue lifecycle.

2. _synthesize_node() only returns analysis text and no longer enqueues
   deferred relationship extraction; user-interest signal writeback is queued.
Everything else is unchanged from the previous version.
"""

import asyncio
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from core.agents.base_agent import AbstractAgent
from core.agents.analysis_token_stream import AnalysisChunkStreamer, stream_model_text
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models.base_agent_models import BaseAgentInput, BaseAgentOutput
from core.agents.models.orchestrator_models import (
    FinalResponse,
    OrchestratorPlan,
    OrchestratorState,
)
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.prompts.orchestrator_agent_prompts import (
    ORCHESTRATOR_PLANNER_SYSTEM_PROMPT,
    SYNTHESISER_PROMPT,
)
from core.agents.utility.orchestrator_helpers import (
    _collect_latest_agent_memory_summaries,
    _collect_synthesis_inputs,
    _coerce_final_response,
    _extract_response_text,
    _flush_graph_turn,
    _load_portfolio_block,
    _open_graph_session,
    _warm_user_context_cache,
    validate_and_enrich_plan_tickers,
)
from core.agents.utils import (
    _build_combined_company_context,
    _extract_last_human_message,
    build_planner_memory_block,
    build_turn_window_block,
)
from core.config import settings
from core.event_queue import publish_progress, publish_success
from core.logger import get_logger
from core.memory.graph.graph_queue import make_graph_task
from core.memory.graph.models import ALL_MAIN_SECTORS
from core.memory.user_signal_writeback import (
    process_user_signal_writeback,
)
from core.services import service_manager

logger = get_logger(__name__)

AVAILABLE_AGENTS: List[type] = [NewsAnalysisAgent, FundamentalAnalysisAgent]
_SYNTHESIS_TURN_WINDOW: int = 8
_PLANNER_TURN_WINDOW: int = 12
_AGENT_MEMORY_WINDOW: int = 8


class OrchestratorAgent:
    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)
        self._agents: Dict[str, AbstractAgent] = {
            agent.name(): agent() for agent in AVAILABLE_AGENTS
        }
        self._graph = self._build_graph()

    def name(self) -> str:
        return "Orchestrator Agent"

    def _build_runtime_agent_memory_contexts(
        self,
        turns: List[dict],
        *,
        window: int = _AGENT_MEMORY_WINDOW,
    ) -> Dict[str, str]:
        contexts: Dict[str, str] = {}
        for agent_name, agent in self._agents.items():
            builder = getattr(agent, "build_memory_context_from_history", None)
            try:
                if callable(builder):
                    context = builder(turns, window=window)
                else:
                    working_memory = getattr(agent, "_working_memory", None)
                    fallback_builder = getattr(
                        working_memory, "build_context_from_history_summaries", None
                    )
                    if not callable(fallback_builder):
                        continue
                    context = fallback_builder(turns, window=window)
            except Exception:
                logger.exception(
                    "_build_runtime_agent_memory_contexts: context build failed for '%s'",
                    agent_name,
                )
                continue
            text = str(context or "").strip()
            if text:
                contexts[agent_name] = text
        return contexts

    def _router(self, state: OrchestratorState) -> str:
        if state.plan is None:
            return "synthesiser"
        if state.plan.final_answer is not None:
            return "direct_answer"
        if state.plan.target_agents:
            return "execute_agents"
        return "synthesiser"

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState, output_schema=FinalResponse)
        workflow.add_node(
            "prepare_user_interest_context", self._prepare_user_interest_context_node
        )
        workflow.add_node("planner", self._plan_node)
        workflow.add_node("direct_answer", self._direct_answer_node)
        workflow.add_node("execute_agents", self._execute_node)
        workflow.add_node("synthesiser", self._synthesize_node)
        workflow.add_edge(START, "prepare_user_interest_context")
        workflow.add_edge("prepare_user_interest_context", "planner")
        workflow.add_conditional_edges(
            "planner",
            self._router,
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

    def _load_user_context_block(self, user_email: Optional[str]) -> str:
        if not user_email:
            return "USER CONTEXT: None"
        try:
            svc = service_manager.get_user_context_service()
            return svc.get_formatted_context(user_email) or "USER CONTEXT: None"
        except Exception:
            logger.exception(
                "run: failed to read user context from cache for %s", user_email
            )
            return "USER CONTEXT: None"

    def _build_initial_state(
        self,
        *,
        messages: List[BaseMessage],
        conversation_id: Optional[str],
        user_email: Optional[str],
        history_turns: Optional[List[dict]],
        conversation_memory_block: str,
        conversation_memory_hits: Optional[List[dict]],
        turn_id: str,
    ) -> OrchestratorState:
        normalized_history_turns = list(history_turns or [])
        return OrchestratorState(
            messages=messages,
            conversation_id=conversation_id,
            user_email=user_email,
            history_turns=normalized_history_turns,
            agent_memory_summaries=_collect_latest_agent_memory_summaries(
                normalized_history_turns
            ),
            conversation_memory_block=(conversation_memory_block or "(none)"),
            conversation_memory_hits=list(conversation_memory_hits or []),
            user_context_block=self._load_user_context_block(user_email),
            portfolio_block=_load_portfolio_block(user_email),
            turn_id=turn_id,
        )

    async def run(
        self,
        messages: List[BaseMessage],
        conversation_id: Optional[str] = None,
        user_email: Optional[str] = None,
        history_turns: Optional[List[dict]] = None,
        conversation_memory_block: str = "(none)",
        conversation_memory_hits: Optional[List[dict]] = None,
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
        turn_id = str(uuid4())
        logger.info("run: turn_id=%s", turn_id)

        await _open_graph_session(conversation_id)
        await _warm_user_context_cache(user_email)

        initial_state = self._build_initial_state(
            messages=messages,
            conversation_id=conversation_id,
            user_email=user_email,
            history_turns=history_turns,
            conversation_memory_block=conversation_memory_block,
            conversation_memory_hits=conversation_memory_hits,
            turn_id=turn_id,
        )

        try:
            raw = await asyncio.wait_for(
                self._graph.ainvoke(initial_state),
                timeout=settings.ORCHESTRATOR_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "OrchestratorAgent.run() timed out (%.0fs) for conversation '%s'",
                settings.ORCHESTRATOR_TIMEOUT_SECONDS,
                conversation_id,
            )
            return FinalResponse(
                summary="Analysis timed out. Please try a more specific query.",
                turn_id=turn_id,
            )
        except Exception:
            logger.exception("Graph invocation failed")
            return FinalResponse(
                summary="I encountered an internal error. Please try again.",
                turn_id=turn_id,
            )

        await _flush_graph_turn(conversation_id, turn_id)

        return _coerce_final_response(raw)

    def _build_synthesis_system_prompt(
        self,
        user_context: str,
        portfolio: str,
        context_parts: List[str],
    ) -> str:
        rendered = SYNTHESISER_PROMPT.format(
            user_context=user_context, portfolio=portfolio
        )
        context_block = "\n\nAgent Findings:\n" + "\n\n".join(context_parts)
        return rendered + context_block

    async def _prepare_user_interest_context_node(
        self, state: OrchestratorState
    ) -> Dict[str, Any]:
        latest_human = _extract_last_human_message(state.messages).strip()
        try:
            user_context_service = service_manager.get_user_context_service()
            result = await user_context_service.build_targeted_orchestrator_context(
                user_email=state.user_email,
                latest_user_message=latest_human,
                baseline_user_context_block=state.user_context_block,
                portfolio_block=state.portfolio_block,
                llm=self._llm,
            )
            return {
                "user_interest_query_spec": result.query_spec,
                "user_interest_graph_context_block": result.context_block,
                "user_interest_query_debug": result.debug_payload,
            }
        except Exception:
            logger.exception(
                "_prepare_user_interest_context_node: failed while delegating to UserContextService"
            )
            return {
                "user_interest_graph_context_block": "(none)",
                "user_interest_query_debug": {"mode": "error"},
            }

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        available_agents_desc = "\n".join(
            f"  {agent.name()}: {agent.description()}" for agent in AVAILABLE_AGENTS
        )
        system_content = ORCHESTRATOR_PLANNER_SYSTEM_PROMPT.format(
            available_agents_desc=available_agents_desc,
        )
        context_content = (
            "USER CONTEXT (if available):\n"
            f"{state.user_context_block}\n\n"
            "PORTFOLIO HOLDINGS:\n"
            f"{state.portfolio_block}"
        )
        latest_human = _extract_last_human_message(state.messages)
        planner_turn_block = build_turn_window_block(
            state.history_turns, _PLANNER_TURN_WINDOW
        )
        planner_memory_block = build_planner_memory_block(
            self._build_runtime_agent_memory_contexts(state.history_turns)
        )
        planner_conversation_memory_block = state.conversation_memory_block or "(none)"
        user_interest_context_block = state.user_interest_graph_context_block or "(none)"
        canonical_sector_names_block = ", ".join(sorted(ALL_MAIN_SECTORS.keys()))

        try:
            structured_llm = self._llm.with_structured_output(OrchestratorPlan)
            planner_messages: List[BaseMessage] = [
                SystemMessage(content=system_content),
                SystemMessage(content=context_content),
                SystemMessage(
                    content=(
                        "Recent conversation turns (most recent window):\n"
                        f"{planner_turn_block}"
                    )
                ),
                SystemMessage(
                    content=(
                        "Agent-provided memory contexts from prior turn summaries "
                        "(use for continuity during routing and per-agent goal generation):\n"
                        f"{planner_memory_block}"
                    )
                ),
                SystemMessage(
                    content=(
                        "Retrieved private conversation memory chunks "
                        "(use when relevant for continuity; do not treat as external facts):\n"
                        f"{planner_conversation_memory_block}"
                    )
                ),
                SystemMessage(
                    content=(
                        "TARGETED USER-INTEREST GRAPH CONTEXT:\n"
                        f"{user_interest_context_block}"
                    )
                ),
                SystemMessage(
                    content=(
                        "CANONICAL SECTOR NAMES (use exact labels for Sector entities):\n"
                        f"{canonical_sector_names_block}"
                    )
                ),
                HumanMessage(content=latest_human),
            ]
            plan: OrchestratorPlan = await structured_llm.ainvoke(planner_messages)
            publish_progress(
                "orchestrator",
                f"Routing â†’ agents={plan.target_agents or []}, needs_memory={plan.needs_memory}",
            )
            logger.info(
                "_plan_node: agents=%s needs_memory=%s final_answer=%s ticker=%s",
                plan.target_agents,
                plan.needs_memory,
                plan.final_answer is not None,
                plan.ticker,
            )
            state_update: Dict[str, Any] = {"plan": plan}
            state_update.update(await validate_and_enrich_plan_tickers(plan))
            return state_update

        except Exception:
            logger.exception("_plan_node: LLM call failed â€” using safe fallback plan")
            return {"plan": None}

    async def _direct_answer_node(self, state: OrchestratorState) -> Dict[str, Any]:
        answer = (state.plan.final_answer or "").strip() if state.plan else ""
        return {
            "summary": answer,
            "sources": [],
            "fundamental_data": None,
            "fundamentals_visualization": None,
            "fundamentals_raw_display_data": None,
            "fundamentals_task_completed": True,
            "fundamentals_task_completion_reason": "",
            "agent_memory_summaries": {},
            "turn_id": state.turn_id,
        }

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = state.plan
        if not plan or not plan.target_agents:
            return {"agent_outputs": {}}

        valid_names = [n for n in plan.target_agents if n in self._agents]

        publish_progress("orchestrator", f"Dispatching to: {', '.join(valid_names)}")

        primary_ticker = plan.tickers[0] if plan.tickers else None
        combined_context = _build_combined_company_context(
            plan.tickers, state.company_context_blocks
        )

        tasks = []
        agent_memory_contexts = self._build_runtime_agent_memory_contexts(
            state.history_turns
        )
        for name in valid_names:
            agent_goal = (
                (plan.per_agent_goals or {}).get(name)
                or (plan.per_agent_queries or {}).get(name)
                or plan.query
            )
            agent_memory_context = agent_memory_contexts.get(name, "")
            agent_input: BaseAgentInput = plan.model_copy(
                update={
                    "query": "",
                    "goal": agent_goal,
                    "ticker": primary_ticker,
                    "conversation_id": state.conversation_id,
                    "turn_id": state.turn_id,
                    "agent_memory_context": agent_memory_context,
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
                    "_execute_node: agent '%s' failed â€” %s", name, res, exc_info=res
                )
            else:
                outputs[name] = res

        return {"agent_outputs": outputs}

    async def _build_synthesis_text(
        self,
        *,
        state: OrchestratorState,
        context_parts: List[str],
    ) -> str:
        portfolio_block = state.portfolio_block or "[]"
        synthesis_turn_block = build_turn_window_block(
            state.history_turns, _SYNTHESIS_TURN_WINDOW
        )
        latest_human = _extract_last_human_message(state.messages)

        publish_progress("orchestrator", "Synthesising final response...")

        if not context_parts and not state.user_context_block and not portfolio_block:
            return (
                "I wasn't able to retrieve data for your query at this time. "
                "Please try again or rephrase your question."
            )
        streamer: AnalysisChunkStreamer | None = None
        try:
            system_prompt = self._build_synthesis_system_prompt(
                user_context=state.user_context_block,
                portfolio=portfolio_block,
                context_parts=context_parts,
            )
            messages: List[BaseMessage] = [
                SystemMessage(content=system_prompt),
                SystemMessage(
                    content=(
                        "Recent conversation turns (most recent window):\n"
                        f"{synthesis_turn_block}"
                    )
                ),
                SystemMessage(
                    content=(
                        "Retrieved private conversation memory chunks "
                        "(use when relevant for continuity; do not treat as external facts):\n"
                        f"{state.conversation_memory_block or '(none)'}"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Latest user question:\n{latest_human}\n\n"
                        "Produce the final analysis response."
                    )
                ),
            ]
            streamer = AnalysisChunkStreamer(
                source="orchestrator",
                agent="orchestrator",
                node="_build_synthesis_text",
                enabled=settings.ENABLE_ANALYSIS_TOKEN_STREAMING,
            )
            if streamer.enabled:
                streamer.start()
                raw_text = await stream_model_text(
                    llm=self._llm,
                    messages=messages,
                    streamer=streamer,
                )
                final_text = _extract_response_text(raw_text)
                streamer.end(final_text=final_text)
                publish_success("orchestrator", "Synthesis complete.")
                return final_text

            response = await self._llm.ainvoke(messages)
            publish_success("orchestrator", "Synthesis complete.")
            return _extract_response_text(response.text if response else "")
        except Exception:
            if streamer is not None and streamer.enabled:
                streamer.error("Synthesis failed.")
            logger.exception("_synthesize_node: LLM synthesis failed")
            return (
                "I encountered an error while synthesising the analysis. "
                "The raw data was retrieved but could not be summarised."
            )

    async def _write_back_user_signals(self, state: OrchestratorState) -> None:
        if not (state.user_email and state.plan and state.conversation_id):
            return
        investment_signals = state.plan.detected_investment_signals or []
        learning_signals = state.plan.detected_learning_signals or []
        if not investment_signals and not learning_signals:
            return
        user_message = _extract_last_human_message(state.messages)
        writeback_result = await process_user_signal_writeback(
            user_email=state.user_email,
            conversation_id=state.conversation_id,
            turn_id=state.turn_id,
            ticker_metadata=state.ticker_metadata,
            user_message=user_message,
            detected_investment_signals=investment_signals,
            detected_learning_signals=learning_signals,
        )
        if writeback_result.relationships:
            try:
                task = make_graph_task(
                    turn_id=state.turn_id,
                    conversation_id=state.conversation_id,
                    source_agent=self.name(),
                    relationships=writeback_result.relationships,
                )
                await service_manager.get_graph_queue_manager().enqueue(task)
            except Exception:
                logger.exception(
                    "_synthesize_node: failed to enqueue user signal relationships"
                )

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        synthesis_inputs = _collect_synthesis_inputs(state.agent_outputs)
        analysis_text = await self._build_synthesis_text(
            state=state,
            context_parts=synthesis_inputs["context_parts"],
        )
        await self._write_back_user_signals(state)

        return {
            "summary": analysis_text,
            "fundamental_data": synthesis_inputs["fundamental_df"],
            "fundamentals_visualization": synthesis_inputs[
                "fundamentals_visualization"
            ],
            "fundamentals_raw_display_data": synthesis_inputs[
                "fundamentals_raw_display_data"
            ],
            "fundamentals_row_semantics": synthesis_inputs[
                "fundamentals_row_semantics"
            ],
            "fundamentals_task_completed": synthesis_inputs[
                "fundamentals_task_completed"
            ],
            "fundamentals_task_completion_reason": synthesis_inputs[
                "fundamentals_task_completion_reason"
            ],
            "sources": synthesis_inputs["news_sources"],
            "agent_analyses": synthesis_inputs["per_agent_analyses"],
            "agent_memory_summaries": synthesis_inputs["agent_memory_summaries"],
            "tickers": state.plan.tickers if state.plan else [],
            "turn_id": state.turn_id,
        }

