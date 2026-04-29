"""
core/agents/orchestrator_agent.py

1. run() now calls graph_queue_manager.open_session() before graph.ainvoke()
   and graph_queue_manager.flush_turn() + close_session() after it returns.
   This is the single addition required for the graph queue lifecycle.

2. _synthesize_node() only returns analysis text and no longer enqueues
   deferred relationship extraction; only investment-signal writeback is queued.
Everything else is unchanged from the previous version.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from core.agents.base_agent import AbstractAgent
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
from core.agents.ticker_validation import TickerInfo
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
from core.memory.retrieval.models import CitedSource
from core.memory.user_signal_writeback import (
    build_signal_payload,
    build_user_signal_relationships,
    update_user_signal_cache,
)
from core.services import service_manager

logger = get_logger(__name__)

AVAILABLE_AGENTS: List[type] = [NewsAnalysisAgent, FundamentalAnalysisAgent]
_SYNTHESIS_TURN_WINDOW: int = 8
_PLANNER_TURN_WINDOW: int = 12
_AGENT_MEMORY_WINDOW: int = 8


def _build_clarification_message(needs_confirmation: Dict[str, "TickerInfo"]) -> str:
    """Format a user-facing message asking for ticker confirmation."""
    lines = ["Before proceeding, I want to confirm the securities you're asking about:"]
    for ticker, info in needs_confirmation.items():
        if not info.is_valid and info.suggestions:
            suggestions_str = ", ".join(f"**{s}**" for s in info.suggestions[:3])
            lines.append(
                f"• **{ticker}** wasn't recognised as a valid ticker symbol. "
                f"Did you mean one of: {suggestions_str}?"
            )
        elif not info.is_valid:
            lines.append(
                f"• **{ticker}** wasn't recognised as a valid ticker symbol. "
                f"Please double-check the symbol and try again."
            )
        else:
            # Valid but non-equity (ETF, MUTUALFUND, etc.)
            qt = info.quote_type or "unknown type"
            lines.append(
                f"• **{ticker}** appears to be a `{qt}` rather than a common equity. "
                f"Is this correct, or did you mean a different symbol?"
            )
    lines.append("\nPlease reply with the correct ticker symbol(s) and I'll proceed.")
    return "\n".join(lines)


def _sanitize_portfolio_user_email(user_email: str) -> str:
    value = (user_email or "").strip().lower()
    safe = re.sub(r"[^a-z0-9._-]+", "_", value).strip("._-")
    if not safe:
        raise ValueError("Invalid user_email")
    return safe


def _get_user_portfolio_path(base_path: str, user_email: str) -> Path:
    base = Path(base_path)
    safe_user = _sanitize_portfolio_user_email(user_email)
    return base.parent / f"{base.stem}_{safe_user}.json"


def get_portfolio_for_user(base_path: str, user_email: Optional[str]) -> List[dict]:
    if not user_email:
        return []
    try:
        path = _get_user_portfolio_path(base_path, user_email)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        logger.warning("Portfolio file not found for user '%s' at %s", user_email, path)
        return []
    except ValueError:
        logger.warning(
            "Invalid portfolio user_email for path resolution: %s", user_email
        )
        return []
    except Exception:
        logger.exception("Failed to load portfolio for user '%s'", user_email)
        return []


class OrchestratorAgent:
    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)
        self._agents: Dict[str, AbstractAgent] = {
            agent.name(): agent() for agent in AVAILABLE_AGENTS
        }
        self._graph = self._build_graph()

    _get_user_portfolio_path = staticmethod(_get_user_portfolio_path)
    get_portfolio_for_user = staticmethod(get_portfolio_for_user)

    def name(self) -> str:
        return "Orchestrator Agent"

    @staticmethod
    def _collect_latest_agent_memory_summaries(
        turns: List[dict],
    ) -> Dict[str, Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for turn in turns:
            summaries = turn.get("agent_memory_summaries") or {}
            if not isinstance(summaries, dict):
                continue
            for agent_name, payload in summaries.items():
                if isinstance(payload, dict):
                    latest[str(agent_name)] = payload
        return latest

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
            return "validate_and_enrich"
        return "synthesiser"

    def _enrichment_router(self, state: OrchestratorState) -> str:
        if state.plan and state.plan.target_agents:
            return "execute_agents"
        if state.plan and state.plan.final_answer is not None:
            return "direct_answer"
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

    @staticmethod
    def _load_portfolio_block(user_email: Optional[str]) -> str:
        portfolio = get_portfolio_for_user(settings.PORTFOLIO_JSON_PATH, user_email)
        return json.dumps(portfolio, indent=2) if portfolio else "[]"

    @staticmethod
    async def _open_graph_session(conversation_id: Optional[str]) -> None:
        if not conversation_id:
            return
        try:
            await service_manager.get_graph_queue_manager().open_session(
                conversation_id
            )
        except Exception:
            logger.exception(
                "run: failed to open graph queue session for '%s'", conversation_id
            )

    @staticmethod
    async def _flush_graph_turn(conversation_id: Optional[str], turn_id: str) -> None:
        if not conversation_id:
            return
        try:
            await service_manager.get_graph_queue_manager().flush_turn(
                conversation_id=conversation_id,
                turn_id=turn_id,
            )
        except Exception:
            logger.exception("run: failed to flush graph queue turn_id='%s'", turn_id)

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
            agent_memory_summaries=self._collect_latest_agent_memory_summaries(
                normalized_history_turns
            ),
            conversation_memory_block=(conversation_memory_block or "(none)"),
            conversation_memory_hits=list(conversation_memory_hits or []),
            user_context_block=self._load_user_context_block(user_email),
            portfolio_block=self._load_portfolio_block(user_email),
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

        await self._open_graph_session(conversation_id)

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

        await self._flush_graph_turn(conversation_id, turn_id)

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
                fundamentals_visualization=raw.get("fundamentals_visualization"),
                fundamentals_raw_display_data=raw.get("fundamentals_raw_display_data"),
                fundamentals_task_completed=raw.get(
                    "fundamentals_task_completed", True
                ),
                fundamentals_task_completion_reason=raw.get(
                    "fundamentals_task_completion_reason", ""
                ),
                sources=raw.get("sources") or [],
                agent_analyses=raw.get("agent_analyses") or {},
                agent_memory_summaries=raw.get("agent_memory_summaries") or {},
                tickers=raw.get("tickers") or [],
                turn_id=raw.get("turn_id") or "",
            )
        return FinalResponse(
            summary=getattr(raw, "summary", "") or "",
            fundamental_data=getattr(raw, "fundamental_data", None),
            fundamentals_visualization=getattr(raw, "fundamentals_visualization", None),
            fundamentals_raw_display_data=getattr(
                raw, "fundamentals_raw_display_data", None
            ),
            fundamentals_task_completed=getattr(
                raw, "fundamentals_task_completed", True
            ),
            fundamentals_task_completion_reason=getattr(
                raw, "fundamentals_task_completion_reason", ""
            ),
            sources=getattr(raw, "sources", []) or [],
            agent_analyses=getattr(raw, "agent_analyses", {}) or {},
            agent_memory_summaries=getattr(raw, "agent_memory_summaries", {}) or {},
            tickers=getattr(raw, "tickers", []) or [],
            turn_id=getattr(raw, "turn_id", "") or "",
        )

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

    @staticmethod
    def _extract_response_text(raw: str) -> str:
        return raw.strip()

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
            return {"plan": plan}

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

    async def _validate_and_enrich_node(
        self, state: OrchestratorState
    ) -> Dict[str, Any]:
        from core.agents.ticker_validation import TickerInfo

        plan = state.plan
        tickers: List[str] = getattr(plan, "tickers", []) if plan else []
        if not tickers:
            return {"company_context_blocks": {}}

        try:
            publish_progress(
                "orchestrator", f"Validating ticker(s): {', '.join(tickers)}"
            )
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

        confirmed_tickers = [
            t for t, info in results.items() if info.is_valid and info.is_equity
        ]

        # Defensive: covers the case where _plan_node ran on a different async
        # context before the sink was registered, or ticker casing differed.
        if confirmed_tickers:
            from core.event_queue import publish_frontend_event

            publish_frontend_event(
                "orchestrator",
                "ticker_resolved",
                {"ticker": confirmed_tickers[0], "tickers": confirmed_tickers},
            )

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

    @staticmethod
    def _collect_synthesis_inputs(
        agent_outputs: Dict[str, BaseAgentOutput],
    ) -> Dict[str, Any]:
        context_parts: List[str] = []
        fundamental_df = None
        fundamentals_visualization = None
        fundamentals_raw_display_data = None
        fundamentals_task_completed = True
        fundamentals_task_completion_reason = ""
        news_sources: List[CitedSource] = []
        agent_memory_summaries: Dict[str, Dict[str, Any]] = {}
        per_agent_analyses: Dict[str, str] = {}

        for name, output in agent_outputs.items():
            try:
                context_parts.append(output.get_llm_context_str())
            except Exception:
                logger.exception(
                    "_synthesize_node: get_llm_context_str failed for '%s'", name
                )
            if name == "fundamentals_agent":
                fundamental_df = getattr(output, "financial_data", None)
                fundamentals_visualization = getattr(output, "visualization_plan", None)
                fundamentals_raw_display_data = getattr(
                    output, "raw_display_data", None
                )
                fundamentals_task_completed = bool(
                    getattr(output, "task_completed", True)
                )
                fundamentals_task_completion_reason = (
                    getattr(output, "task_completion_reason", "") or ""
                )
            if name == "news_agent":
                news_sources = getattr(output, "sources", []) or []
            memory_summary = getattr(output, "memory_summary", {}) or {}
            if isinstance(memory_summary, dict) and memory_summary:
                agent_memory_summaries[name] = memory_summary
            per_agent_analyses[name] = getattr(output, "analysis", "") or ""

        return {
            "context_parts": context_parts,
            "fundamental_df": fundamental_df,
            "fundamentals_visualization": fundamentals_visualization,
            "fundamentals_raw_display_data": fundamentals_raw_display_data,
            "fundamentals_task_completed": fundamentals_task_completed,
            "fundamentals_task_completion_reason": fundamentals_task_completion_reason,
            "news_sources": news_sources,
            "agent_memory_summaries": agent_memory_summaries,
            "per_agent_analyses": per_agent_analyses,
        }

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
            response = await self._llm.ainvoke(messages)
            publish_success("orchestrator", "Synthesis complete.")
            return self._extract_response_text(response.content if response else "")
        except Exception:
            logger.exception("_synthesize_node: LLM synthesis failed")
            return (
                "I encountered an error while synthesising the analysis. "
                "The raw data was retrieved but could not be summarised."
            )

    async def _write_back_investment_signals(self, state: OrchestratorState) -> None:
        if not (state.user_email and state.plan and state.conversation_id):
            return
        investment_signals = state.plan.detected_investment_signals or []
        if not investment_signals:
            return
        user_message = _extract_last_human_message(state.messages)
        payload = build_signal_payload(
            user_email=state.user_email,
            conversation_id=state.conversation_id,
            turn_id=state.turn_id,
            ticker_metadata=state.ticker_metadata,
            user_message=user_message,
            detected_investment_signals=investment_signals,
            detected_learning_signals=[],
            interest_edges=[],
        )
        relationships, cache_entries = await build_user_signal_relationships(payload)
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

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        synthesis_inputs = self._collect_synthesis_inputs(state.agent_outputs)
        analysis_text = await self._build_synthesis_text(
            state=state,
            context_parts=synthesis_inputs["context_parts"],
        )
        await self._write_back_investment_signals(state)

        return {
            "summary": analysis_text,
            "fundamental_data": synthesis_inputs["fundamental_df"],
            "fundamentals_visualization": synthesis_inputs[
                "fundamentals_visualization"
            ],
            "fundamentals_raw_display_data": synthesis_inputs[
                "fundamentals_raw_display_data"
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
