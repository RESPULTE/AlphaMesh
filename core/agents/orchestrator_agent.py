"""
core/agents/orchestrator_agent.py

Hardened OrchestratorAgent.

Changes in this revision
─────────────────────────
1.  Per-agent query rewriting (Change #1 from design doc).
    OrchestratorPlan now carries `per_agent_queries: Dict[str, str]`.
    The planner LLM populates a tailored retrieval string for every agent
    it selects.  _execute_node reads this dict and builds a separate
    BaseAgentInput per agent — each receiving only its own rewritten query
    instead of the raw user string.  Falls back to `plan.query` for agents
    not present in the dict.

2.  Memory retrieval responsibility moved to NewsAnalysisAgent (Change #3).
    `rewritten_queries` and `memory_task` are removed from OrchestratorPlan
    and OrchestratorState.  The orchestrator no longer creates or passes a
    memory retrieval task — the news agent self-manages this via its own
    `_rewrite_queries_node`.  BaseAgentInput no longer carries `memory_task`
    or `vector_query`.

3.  Single LLM call for synthesis (unchanged from previous revision).
    _run_synthesis_chain parses THREE output blocks from one call:
      <cross_domain_relationships> … </cross_domain_relationships>
      <user_interest_relationships> … </user_interest_relationships>
      <response> … </response>

4.  All graph-write logic for user signals delegated to
    core/memory/user_signal_writeback.write_user_signals() (unchanged).
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from orchestrator_models import (
    FinalResponse,
    OrchestratorPlan,
    OrchestratorState,
    SynthesisResult,
)

from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput, CitedSource
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.prompts import (
    ORCHESTRATOR_PLANNER_SYSTEM_PROMPT,
    SYNTHESISER_SINGLE_AGENT_PROMPT,
    SYNTHESISER_USER_CONTEXT_SECTION,
    build_writeback_system_prompt,
)
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_create_task(coro) -> Optional[asyncio.Task]:
    """Create an asyncio task only when a running loop exists."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        logger.warning("_safe_create_task: no running event loop — task skipped.")
        return None


def _extract_last_human_message(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
    return ""


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

    # ── Static helpers ────────────────────────────────────────────

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
        if state.plan.needs_memory:
            return "load_context"
        if state.plan.target_agents:
            return "execute_agents"
        return "synthesiser"

    def _post_context_router(self, state: OrchestratorState) -> str:
        if state.plan and state.plan.target_agents:
            return "execute_agents"
        return "synthesiser"

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState, output_schema=FinalResponse)

        workflow.add_node("planner", self._plan_node)
        workflow.add_node("load_context", self._load_context_node)
        workflow.add_node("direct_answer", self._direct_answer_node)
        workflow.add_node("execute_agents", self._execute_node)
        workflow.add_node("synthesiser", self._synthesize_node)

        workflow.add_edge(START, "planner")

        workflow.add_conditional_edges(
            "planner",
            self._router,
            {
                "direct_answer": "direct_answer",
                "load_context": "load_context",
                "execute_agents": "execute_agents",
                "synthesiser": "synthesiser",
            },
        )
        workflow.add_conditional_edges(
            "load_context",
            self._post_context_router,
            {
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
        initial_state = OrchestratorState(
            messages=messages,
            conversation_id=conversation_id,
            user_email=user_email,
            user_context=None,
            user_context_block="USER CONTEXT: None",
            user_context_loaded=False,
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
    ) -> str:
        """Invoke the synthesis LLM chain and return raw string output."""
        chain = prompt | self._llm
        response = await chain.ainvoke(
            {
                "history": state.messages,
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
        user_response = resp_match.group(1).strip() if resp_match else raw.strip()

        return SynthesisResult(
            user_response=user_response,
            cross_relationships=cross_relationships,
            interest_edges=interest_edges,
        )

    # ── Nodes ─────────────────────────────────────────────────────

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Ask the LLM planner which agents to activate and rewrite the query per agent."""
        available_agents_desc = "\n".join(
            f"  {agent.name()}: {agent.description()}" for agent in AVAILABLE_AGENTS
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    ORCHESTRATOR_PLANNER_SYSTEM_PROMPT.format(
                        available_agents_desc=available_agents_desc,
                    ),
                ),
                MessagesPlaceholder(variable_name="history"),
            ]
        )
        try:
            structured_llm = self._llm.with_structured_output(OrchestratorPlan)
            chain = prompt | structured_llm
            plan: OrchestratorPlan = await chain.ainvoke({"history": state.messages})
            logger.info(
                "_plan_node: agents=%s needs_memory=%s per_agent_queries=%s final_answer=%s",
                plan.target_agents,
                plan.needs_memory,
                list(plan.per_agent_queries.keys()),
                plan.final_answer is not None,
            )
            return {"plan": plan}
        except Exception:
            logger.exception("_plan_node: LLM call failed — using safe fallback plan")
            return {"plan": OrchestratorPlan()}

    async def _direct_answer_node(self, state: OrchestratorState) -> Dict[str, Any]:
        answer = (state.plan.final_answer or "").strip() if state.plan else ""
        logger.info("_direct_answer_node: %d chars", len(answer))
        return {"summary": answer, "sources": [], "fundamental_data": None}

    async def _load_context_node(self, state: OrchestratorState) -> Dict[str, Any]:
        if not state.user_email:
            logger.warning("_load_context_node: no user_email, skipping")
            return {"user_context_loaded": True}
        try:
            svc = service_manager.get_user_context_service()
            user_context = await svc.load_for_user(state.user_email)
            user_context_block = (
                svc.get_formatted_context(state.user_email) or "USER CONTEXT: None"
            )
            logger.info("_load_context_node: context loaded for %s", state.user_email)
            return {
                "user_context": user_context,
                "user_context_block": user_context_block,
                "user_context_loaded": True,
            }
        except Exception:
            logger.exception("_load_context_node: failed for %s", state.user_email)
            return {"user_context_loaded": True}

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
            # Use the agent-specific rewritten query; fall back to plan.query
            agent_query = plan.per_agent_queries.get(name) or plan.query
            logger.info(
                "_execute_node: dispatching '%s' with query='%.120s'",
                name,
                agent_query,
            )
            # plan IS a BaseAgentInput — copy it, overriding only query and
            # injecting the runtime-only conversation_id (excluded from serialisation).
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

        subgraph_tasks = [
            getattr(out, "subgraph_task", None) for out in outputs.values()
        ]
        await asyncio.gather(
            *[t for t in subgraph_tasks if t is not None], return_exceptions=True
        )

        return {"agent_outputs": outputs}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Synthesise agent outputs into a final response.

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

        user_response = ""
        synthesis_result = SynthesisResult(user_response="")

        # ── LLM synthesis (single call) ───────────────────────────
        if not context_parts and not state.user_context_block and not portfolio_block:
            logger.warning("_synthesize_node: no context from agents, using fallback")
            user_response = (
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
                    prompt, state, context_parts, portfolio_block
                )
                synthesis_result = self._parse_synthesis_output(raw, multi_agent)
                user_response = synthesis_result.user_response
            except Exception:
                logger.exception("_synthesize_node: LLM synthesis failed")
                user_response = (
                    "I encountered an error while synthesising the analysis. "
                    "The raw data was retrieved but could not be summarised."
                )

        # ── Cross-domain graph write-back (async, non-blocking) ───
        if (
            multi_agent
            and state.conversation_id
            and synthesis_result.cross_relationships
        ):
            try:
                builder = InMemorySubgraphBuilder(
                    embedding_func=service_manager.get_embedding_func(),
                    fuzzy_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
                    semantic_threshold=settings.EXTRACTION_SEMANTIC_THRESHOLD,
                )
                cross_graph = await builder.build(
                    synthesis_result.cross_relationships, source_agent="orchestrator"
                )
                if cross_graph.number_of_edges() > 0:
                    _safe_create_task(
                        service_manager.get_ingestor()._upsert_graph_to_neo4j(
                            cross_graph, state.conversation_id
                        )
                    )
            except Exception:
                logger.exception("_synthesize_node: cross-graph write-back failed")

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
            "summary": user_response,
            "fundamental_data": fundamental_df,
            "sources": news_sources,
            "agent_analyses": per_agent_analyses,
        }
