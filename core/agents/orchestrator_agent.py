"""
core/agents/orchestrator_agent.py

Hardened OrchestratorAgent — refactored for lower latency.

Changes in this revision
─────────────────────────
1.  `load_context` node removed entirely.
    User context is now pre-loaded externally (on session start) and held in
    the UserContextService in-memory cache.  `run()` reads it synchronously
    via `svc.get_formatted_context()` — a pure dict lookup with no I/O — and
    writes the result into initial state before the graph is invoked.

    This eliminates a sequential Neo4j round-trip that previously sat on the
    critical path whenever `needs_memory=True`, blocking agent dispatch for
    the duration of the database read.

    Graph topology simplified from:
      planner → [direct_answer | load_context | execute_agents | synthesiser]
    to:
      planner → [direct_answer | execute_agents | synthesiser]

2.  Planner receives only the latest human message.
    Routing and per-agent query rewriting require only the current user
    intent — not the full conversation transcript.  The ChatPromptTemplate /
    MessagesPlaceholder wrapper in _plan_node is replaced with a direct
    [SystemMessage, HumanMessage] pair, keeping the token load on this
    latency-critical call constant regardless of conversation length.

3.  Synthesiser receives a bounded history window (_SYNTHESIS_HISTORY_WINDOW).
    The synthesis LLM needs enough context to write a coherent, personalised
    response but not the entire transcript.  A fixed window of the most recent
    messages is passed instead of `state.messages`, preventing unbounded token
    growth as conversations grow.

4.  InMemorySubgraphBuilder instantiated without arguments.
    The constructor reads embedding_func and thresholds from service_manager /
    settings internally; passing them as keyword args was a pre-existing bug.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph

from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput, CitedSource
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.orchestrator_models import (
    FinalResponse,
    OrchestratorPlan,
    OrchestratorState,
    SynthesisResult,
)
from core.agents.prompts import (
    ORCHESTRATOR_PLANNER_SYSTEM_PROMPT,
    SYNTHESISER_SINGLE_AGENT_PROMPT,
    SYNTHESISER_USER_CONTEXT_SECTION,
    build_writeback_system_prompt,
)
from core.agents.utils import _safe_create_task
from core.config import settings
from core.logger import get_logger
from core.memory.graph.subgraph_builder import InMemorySubgraphBuilder
from core.memory.user_signal_writeback import (
    DetectedEntity,
    InterestEdge,
    InvestmentSignal,
    LearningSignal,
    UserSignalPayload,
    write_user_signals,
)
from core.services import service_manager

logger = get_logger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────
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

# Number of most-recent messages passed to the synthesis LLM.
# The planner receives only the single latest human message (see _plan_node).
_SYNTHESIS_HISTORY_WINDOW: int = 6


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _extract_last_human_message(messages: List[BaseMessage]) -> str:
    """Return the content of the most recent HumanMessage, or '' if none exists."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
    return ""


def _last_n_messages(messages: List[BaseMessage], n: int) -> List[BaseMessage]:
    """Return the last *n* messages; returns the full list when len ≤ n."""
    return messages[-n:] if len(messages) > n else list(messages)


def _safe_json(text: str) -> List[dict]:
    try:
        result = json.loads(text.strip())
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _build_signal_payload(
    state: OrchestratorState,
    interest_edges: List[dict],
    user_message: str,
) -> UserSignalPayload:
    """
    Convert OrchestratorState signal lists + parsed interest edge dicts into
    the plain UserSignalPayload the memory module expects.
    """
    investment_signals = [
        InvestmentSignal(
            status=s.status,
            target_entities=[
                DetectedEntity(entity_name=e.entity_name, entity_type=e.entity_type)
                for e in s.target_entities
            ],
        )
        for s in (state.plan.detected_investment_signals or [])
    ]
    learning_signals = [
        LearningSignal(
            status=s.status,
            target_entities=[
                DetectedEntity(entity_name=e.entity_name, entity_type=e.entity_type)
                for e in s.target_entities
            ],
        )
        for s in (state.plan.detected_learning_signals or [])
    ]
    edges = [
        InterestEdge(
            entity_name=e.get("entity_name", ""),
            entity_type=e.get("entity_type", ""),
            user_signal_type=e.get("user_signal_type", "investment"),
            target_entity_name=e.get("target_entity_name", ""),
            relationship=e.get("relationship", "RELATED_TO"),
            reason=e.get("reason", ""),
            confidence=e.get("confidence", "low"),
        )
        for e in interest_edges
        if e.get("entity_name") and e.get("target_entity_name")
    ]
    return UserSignalPayload(
        user_email=state.user_email or "",
        conversation_id=state.conversation_id or "",
        user_message=user_message,
        investment_signals=investment_signals,
        learning_signals=learning_signals,
        interest_edges=edges,
    )


# ──────────────────────────────────────────────────────────────────────────────
# OrchestratorAgent
# ──────────────────────────────────────────────────────────────────────────────


class OrchestratorAgent:
    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)
        self._agents: Dict[str, AbstractAgent] = {
            agent.name(): agent() for agent in AVAILABLE_AGENTS
        }
        self._graph = self._build_graph()
        self._subgraph_builder = InMemorySubgraphBuilder()

    # ── Static helpers ────────────────────────────────────────────

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

    # ── Graph wiring ──────────────────────────────────────────────

    def _router(self, state: OrchestratorState) -> str:
        if state.plan is None:
            logger.warning("_router: plan is None — falling back to synthesiser")
            return "synthesiser"
        if state.plan.final_answer is not None:
            return "direct_answer"
        if state.plan.target_agents:
            return "execute_agents"
        return "synthesiser"

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState, output_schema=FinalResponse)

        workflow.add_node("planner", self._plan_node)
        workflow.add_node("direct_answer", self._direct_answer_node)
        workflow.add_node("execute_agents", self._execute_node)
        workflow.add_node("synthesiser", self._synthesize_node)

        workflow.add_edge(START, "planner")
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

    # ── Public entry point ────────────────────────────────────────

    async def run(
        self,
        messages: List[BaseMessage],
        conversation_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> FinalResponse:
        """
        Entry point for one conversation turn.

        User context is read synchronously from the UserContextService
        in-memory cache (O(1), no I/O) before the graph starts.  The cache
        is populated externally on session start; if it is cold the service
        returns "USER CONTEXT: None", which is a safe default.
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

        initial_state = OrchestratorState(
            messages=messages,
            conversation_id=conversation_id,
            user_email=user_email,
            user_context_block=user_context_block,
        )
        try:
            raw = await self._graph.ainvoke(initial_state)
        except Exception:
            logger.exception("Graph invocation failed")
            return FinalResponse(
                summary="I encountered an internal error. Please try again."
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

    # ── Prompt builders ───────────────────────────────────────────

    def _build_synthesis_prompt(
        self,
        user_context_section: str,
        multi_agent: bool,
        investment_signals: Optional[List] = None,
        learning_signals: Optional[List] = None,
    ) -> ChatPromptTemplate:
        """
        Build the synthesis ChatPromptTemplate.

        For multi-agent runs the system prompt is assembled by
        build_writeback_system_prompt() which conditionally injects the
        user-interest-relationships block when signals are present —
        eliminating the need for a second LLM call.
        """
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
        """Invoke the synthesis LLM chain and return raw string output."""
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
        """
        Parse all XML output blocks from a raw synthesis LLM response into a
        typed SynthesisResult.  Never raises — returns empty lists on parse failure.
        """
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

    # ── Nodes ─────────────────────────────────────────────────────

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Route the conversation and rewrite the query per target agent.

        Only the latest human message is sent to the planner LLM — routing
        and query rewriting require only the current user intent, not the full
        transcript.  This keeps the token load on this latency-critical call
        constant regardless of conversation length.
        """
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
                "_plan_node: agents=%s needs_memory=%s per_agent_queries=%s "
                "final_answer=%s start_date=%s end_date=%s ticker=%s "
                "metrics=%s granularity=%s",
                plan.target_agents,
                plan.needs_memory,
                list(plan.per_agent_queries.keys()),
                plan.final_answer is not None,
                plan.start_date,
                plan.end_date,
                plan.ticker,
                plan.metrics,
                plan.granularity,
            )
            return {"plan": plan}
        except Exception:
            logger.exception("_plan_node: LLM call failed — using safe fallback plan")
            return {"plan": OrchestratorPlan()}

    async def _direct_answer_node(self, state: OrchestratorState) -> Dict[str, Any]:
        answer = (state.plan.final_answer or "").strip() if state.plan else ""
        logger.info("_direct_answer_node: %d chars", len(answer))
        return {"summary": answer, "sources": [], "fundamental_data": None}

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Run selected agents in parallel, each receiving a query rewritten
        specifically for its job description.

        Because OrchestratorPlan inherits BaseAgentInput, `plan` already IS a
        valid agent input carrying all shared fields (ticker, dates, metrics,
        granularity).  We simply call model_copy(update={"query": agent_query})
        to produce a per-agent BaseAgentInput with only the query swapped out —
        no dict-filtering or manual field reconstruction needed.
        """
        plan = state.plan
        if not plan or not plan.target_agents:
            logger.warning("_execute_node: no agents to run")
            return {"agent_outputs": {}}

        valid_names = [n for n in plan.target_agents if n in self._agents]

        tasks = []
        for name in valid_names:
            agent_query = plan.per_agent_queries.get(name) or plan.query
            logger.info(
                "_execute_node: dispatching '%s' with query='%.120s'",
                name,
                agent_query,
            )
            agent_input: BaseAgentInput = plan.model_copy(
                update={"query": agent_query, "conversation_id": state.conversation_id}
            )
            tasks.append((name, self._agents[name].run(agent_input)))

        if not tasks:
            logger.warning("_execute_node: no valid agent tasks built")
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
                logger.info("_execute_node: agent '%s' completed", name)

        if not outputs:
            logger.warning("_execute_node: all agents failed")

        return {"agent_outputs": outputs}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Synthesise agent outputs into a final response.

        The synthesis LLM receives a bounded window of the most recent messages
        (`_SYNTHESIS_HISTORY_WINDOW`) rather than the full history, keeping
        token load constant as conversations grow.

        Single LLM call produces up to three output blocks:
          <cross_domain_relationships>  → written to knowledge graph
          <user_interest_relationships> → delegated to memory.user_signal_writeback
          <response>                    → returned to the user
        """
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
        investment_signals = (
            (state.plan.detected_investment_signals or []) if state.plan else []
        )
        learning_signals = (
            (state.plan.detected_learning_signals or []) if state.plan else []
        )

        # Bounded window — prevents unbounded token growth on long conversations.
        history_window = _last_n_messages(state.messages, _SYNTHESIS_HISTORY_WINDOW)

        analysis_text = ""
        synthesis_result = SynthesisResult(analysis_text="")

        if not context_parts and not state.user_context_block and not portfolio_block:
            logger.warning("_synthesize_node: no context from agents, using fallback")
            analysis_text = (
                "I wasn't able to retrieve data for your query at this time. "
                "Please try again or rephrase your question."
            )
        else:
            try:
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

        # ── Cross-domain graph write-back (async, non-blocking) ───
        if (
            multi_agent
            and state.conversation_id
            and synthesis_result.cross_relationships
        ):
            subgraph_id = await self._subgraph_builder.schedule_subgraph_extraction(
                agent_name=self.name(),
                conversation_id=state.conversation_id or "",
                analysis_text=analysis_text,
                relationships=[],
                relationships_extracted=False,
                llm=service_manager.get_agent(temperature=0.7),
            )

        # ── User signal write-back (delegated to memory module) ───
        if state.user_email and state.plan and state.conversation_id:
            user_message = _extract_last_human_message(state.messages)
            payload = _build_signal_payload(
                state, synthesis_result.interest_edges, user_message
            )
            _safe_create_task(write_user_signals(payload))

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
