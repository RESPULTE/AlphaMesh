"""
Hardened OrchestratorAgent.

Drop-in replacement for core/agents/orchestrator_agent.py.

Changes vs original
───────────────────
1.  FinalResponse.summary has a default="" so LangGraph's output_schema
    projection never crashes on the final_answer → END path.

2.  _direct_answer_node replaces the bare END route for final_answer.
    It writes summary into state before the graph exits, so the
    output_schema projection always has a valid FinalResponse.

3.  _router guards against state.plan being None.

4.  _plan_node wraps the LLM call in try/except; on failure it falls
    back to a safe OrchestratorPlan that routes straight to the
    synthesiser with empty context rather than crashing the graph.

5.  _execute_node:
    - logs every agent failure instead of silently discarding it
    - guards against plan.target_agents containing unknown agent names
    - adds conversation_id to shared_input (matches newer prompts)

6.  _synthesize_node:
    - handles empty agent_outputs gracefully (all agents failed)
    - wraps LLM call in try/except with a sensible fallback message
    - FIXED signal iteration bug: InvestmentSignalDetection /
      LearningSignalDetection carry a `target_entities` list — the old
      code incorrectly read `signal.entity_name` which doesn't exist at
      the signal level
    - asyncio.create_task is guarded so it never fires outside a
      running event loop

7.  run() robustly unwraps the LangGraph result regardless of whether
    it comes back as a FinalResponse, a plain dict, or None.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Type

import networkx as nx
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput, CitedSource
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.prompts import (
    ORCHESTRATOR_PLANNER_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    SYNTHESISER_SINGLE_AGENT_PROMPT,
    SYNTHESISER_USER_CONTEXT_SECTION,
    SYNTHESISER_WRITEBACK_SYSTEM_PROMPT,
)
from core.config import settings
from core.logger import get_logger
from core.memory.graph.models import (
    RelationshipType,
    UserInvestmentInterestNode,
    UserLearningInterestNode,
)
from core.memory.graph.subgraph_builder import InMemorySubgraphBuilder
from core.memory.retrieval.models import RewrittenQueries
from core.memory.user_context_service import UserContext
from core.services import service_manager

logger = get_logger(__name__)

_RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
_CROSS_RE = re.compile(
    r"<cross_domain_relationships>(.*?)</cross_domain_relationships>",
    re.DOTALL | re.IGNORECASE,
)

AVAILABLE_AGENTS: List[Type[AbstractAgent]] = [
    FundamentalAnalysisAgent,
    NewsAnalysisAgent,
]

_AGENT_NAMES: frozenset = frozenset(a.name() for a in AVAILABLE_AGENTS)


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────


class FinalResponse(BaseModel):
    """Structured output returned to the UI."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # FIX 1: default="" so the output_schema projection never fails when
    # LangGraph exits via the final_answer → END path with no summary written.
    summary: str = ""
    fundamental_data: Optional[pd.DataFrame] = None
    sources: List[CitedSource] = Field(default_factory=list)


class UserInterestEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity_name: str
    entity_type: Literal["Company", "FinancialConcept", "FinancialEvent", "Sector"]


class InvestmentSignalDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target_entities: List[UserInterestEntity] = Field(default_factory=list)
    status: Literal["Bought", "Interested", "Sold", "Avoids"]


class LearningSignalDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target_entities: List[UserInterestEntity] = Field(default_factory=list)
    status: Literal["Interested", "Understood", "Confused", "Not Interested"]


class OrchestratorPlan(BaseAgentInput):
    model_config = ConfigDict(extra="ignore")

    target_agents: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = Field(default=None)
    needs_memory: bool = Field(
        default=False,
        description=(
            "Set True when the user's question requires their personal investment or "
            "learning context to answer well (e.g. questions about their portfolio, "
            "holdings, watchlist, personalised recommendations, or interests)."
        ),
    )
    target_entities: List[str] = Field(default_factory=list)
    rewritten_queries: Optional[RewrittenQueries] = Field(default=None)
    detected_investment_signals: List[InvestmentSignalDetection] = Field(
        default_factory=list
    )
    detected_learning_signals: List[LearningSignalDetection] = Field(
        default_factory=list
    )


class OrchestratorState(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    messages: List[BaseMessage] = Field(default_factory=list)
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: Dict[str, BaseAgentOutput] = Field(default_factory=dict)
    final_response: Optional[FinalResponse] = None

    conversation_id: Optional[str] = None
    user_email: Optional[str] = None
    user_context: Optional[UserContext] = None
    user_context_block: str = ""
    user_context_loaded: bool = False
    memory_task: Optional[Any] = Field(default=None, exclude=True)

    # FIX 2: summary lives in state so the direct-answer node can write it
    # before the graph exits, making output_schema projection safe.
    summary: str = ""
    fundamental_data: Optional[pd.DataFrame] = None
    sources: List[CitedSource] = Field(default_factory=list)


class CrossDomainRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")
    from_name: str
    from_type: Literal["Company", "FinancialEvent", "FinancialConcept", "Sector"]
    relation: RelationshipType
    to_name: str
    to_type: Literal["Company", "FinancialEvent", "FinancialConcept", "Sector"]
    confidence: Literal["high", "low"]
    reason: str
    source_agent_from: Literal["news_agent", "fundamentals_agent"]
    source_agent_to: Literal["news_agent", "fundamentals_agent"]


class MultiSynthesizedResponse(BaseModel):
    cross_domain_relationships: List[CrossDomainRelationship]
    response: str = Field(description="The final user-facing analysis response.")


class SingleSynthesizedResponse(BaseModel):
    response: str = Field(description="The final user-facing analysis response.")


class InterestEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity_name: str
    entity_type: Literal["Company", "FinancialConcept", "FinancialEvent", "Sector"]
    user_signal_type: Literal["investment", "learning"]
    target_entity_name: str
    relationship: Literal["THREATENS", "SUPPORTS", "CLARIFIES", "CONFUSES_FURTHER"]
    reason: str
    confidence: Literal["high", "low"]


class UserInterestMapping(BaseModel):
    model_config = ConfigDict(extra="ignore")
    investment_threats: List[InterestEdge] = Field(default_factory=list)
    learning_helpers: List[InterestEdge] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_create_task(coro) -> Optional[asyncio.Task]:
    """Create an asyncio task only when a running loop exists."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        logger.warning("No running event loop — background task skipped.")
        return None


def _extract_last_human_message(messages: List[BaseMessage]) -> str:
    """Return the content of the most recent HumanMessage, or '' if none."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            return msg.content or ""
    return messages[-1].content if messages else ""


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
        """
        Three-way split after the planner:

        1. final_answer set          ? direct_answer
        2. needs_memory = True       ? load_context
        3. agents needed, no memory  ? execute_agents
        4. no agents, no memory      ? synthesiser
        """
        if state.plan is None:
            logger.warning("_router: plan is None � falling back to synthesiser")
            return "synthesiser"

        if state.plan.final_answer is not None:
            return "direct_answer"

        if state.plan.needs_memory:
            return "load_context"

        if state.plan.target_agents:
            return "execute_agents"

        return "synthesiser"

    def _post_context_router(self, state: OrchestratorState) -> str:
        """Called after load_context. Decides whether agents are needed."""
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
        """
        Run the orchestrator graph and always return a FinalResponse.
        Handles all three shapes LangGraph can return:
          • FinalResponse instance  (output_schema projection succeeded)
          • dict                    (raw state dict)
          • None                    (should no longer happen with direct_answer node,
                                     but guarded defensively)
        """
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
                summary="I encountered an internal error. Please try again.",
            )

        # FIX 1 + 7: normalise all possible return shapes
        return self._coerce_final_response(raw)

    @staticmethod
    def _coerce_final_response(raw: Any) -> FinalResponse:
        """Convert whatever LangGraph returned into a safe FinalResponse."""
        if raw is None:
            return FinalResponse(summary="")

        if isinstance(raw, FinalResponse):
            return raw

        if isinstance(raw, dict):
            return FinalResponse(
                summary=raw.get("summary") or "",
                fundamental_data=raw.get("fundamental_data"),
                sources=raw.get("sources") or [],
            )

        # Last resort: attribute-based extraction
        return FinalResponse(
            summary=getattr(raw, "summary", "") or "",
            fundamental_data=getattr(raw, "fundamental_data", None),
            sources=getattr(raw, "sources", []) or [],
        )

    def _build_synthesis_prompt(
        self, user_context_section: str, multi_agent: bool
    ) -> ChatPromptTemplate:
        if multi_agent:
            system_prompt = (
                user_context_section
                + SYNTHESISER_WRITEBACK_SYSTEM_PROMPT
                + "\n\nAgent Findings:\n{context}"
            )
            human_prompt = "Produce cross-domain relationships and final analysis."
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

    async def _run_synthesis_chain(
        self,
        prompt: ChatPromptTemplate,
        state: OrchestratorState,
        context_parts: List[str],
        portfolio_block: str,
    ) -> str:
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

    # ── Nodes ─────────────────────────────────────────────────────

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Ask the LLM planner which agents to activate.
        FIX 4: Wrapped in try/except — any LLM failure produces a safe
        fallback plan instead of crashing the graph.
        """
        now = datetime.now()
        available_agents_desc = ", ".join(
            f"{a.name()}: {a.description()}" for a in AVAILABLE_AGENTS
        )
        system_prompt = ORCHESTRATOR_PLANNER_SYSTEM_PROMPT.format(
            available_agents_desc=available_agents_desc,
            query_rewrite_system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
        )

        try:
            planner_llm = self._llm.with_structured_output(OrchestratorPlan)
            plan: OrchestratorPlan = await planner_llm.ainvoke(
                [SystemMessage(content=system_prompt)] + state.messages
            )
        except Exception:
            logger.exception("_plan_node: LLM planner failed — using fallback plan")
            # Safe fallback: route to synthesiser with no agents, no rewrite
            last_query = _extract_last_human_message(state.messages)
            plan = OrchestratorPlan(
                query=last_query,
                vector_query=last_query,
                ticker=None,
                start_date=now - timedelta(days=365),
                end_date=now,
                target_agents=[],
                final_answer=None,
            )

        # Default date guards
        if plan.end_date is None:
            plan.end_date = now
        if plan.start_date is None:
            plan.start_date = plan.end_date - timedelta(days=365)

        # Filter target_agents to only known agents (prevents KeyError downstream)
        unknown = [a for a in plan.target_agents if a not in _AGENT_NAMES]
        if unknown:
            logger.warning("_plan_node: unknown agents in plan %s — removed", unknown)
            plan.target_agents = [a for a in plan.target_agents if a in _AGENT_NAMES]

        memory_task = None
        if (
            plan.target_agents
            and plan.rewritten_queries
            and plan.rewritten_queries.active_domains
        ):
            try:
                memory_task = _safe_create_task(
                    service_manager.get_retriever().comprehensive_retrieve(
                        plan.rewritten_queries
                    )
                )
            except Exception:
                logger.exception("_plan_node: memory retrieval task creation failed")

        return {"plan": plan, "memory_task": memory_task}

    async def _direct_answer_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        FIX 2: Handles the final_answer path by writing summary into state
        before the graph exits. Previously routed bare to END which left
        summary unpopulated and caused the output_schema projection to crash.
        """
        answer = (state.plan.final_answer or "").strip() if state.plan else ""
        logger.info(
            "_direct_answer_node: returning direct answer (%d chars)", len(answer)
        )
        return {"summary": answer, "sources": [], "fundamental_data": None}

    async def _load_context_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Load user context from UserContextService (cache-first, cold-start hits Neo4j).
        Only reached when the planner set needs_memory=True.
        On failure, continues with empty context so the rest of the graph still runs.
        """
        if not state.user_email:
            logger.warning("_load_context_node: no user_email in state, skipping")
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
            logger.exception(
                "_load_context_node: failed to load context for %s", state.user_email
            )
            return {"user_context_loaded": True}

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Run selected agents in parallel.
        FIX 5: Logs individual agent failures; guards against unknown agents.
        """
        plan = state.plan
        if not plan or not plan.target_agents:
            logger.warning("_execute_node: no agents to run")
            return {"agent_outputs": {}}

        # Build shared input — use model_dump but exclude memory_task (not serialisable)
        try:
            shared_input = BaseAgentInput(
                **{
                    k: v
                    for k, v in plan.model_dump().items()
                    if k in BaseAgentInput.model_fields
                },
                memory_task=state.memory_task,
                conversation_id=state.conversation_id,
            )
        except Exception:
            logger.exception("_execute_node: failed to build BaseAgentInput")
            return {"agent_outputs": {}}

        valid_names = [n for n in plan.target_agents if n in self._agents]
        tasks = [self._agents[name].run(shared_input) for name in valid_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs: Dict[str, BaseAgentOutput] = {}
        for name, res in zip(valid_names, results):
            if isinstance(res, Exception):
                logger.error(
                    "_execute_node: agent '%s' failed — %s", name, res, exc_info=res
                )
            else:
                outputs[name] = res
                logger.info("_execute_node: agent '%s' completed successfully", name)

        if not outputs:
            logger.warning(
                "_execute_node: all agents failed — synthesiser will have empty context"
            )

        subgraph_tasks = [
            getattr(out, "subgraph_task", None) for out in outputs.values()
        ]
        await asyncio.gather(
            *[t for t in subgraph_tasks if t is not None],
            return_exceptions=True,
        )

        return {"agent_outputs": outputs}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Synthesise agent outputs into a final response.

        FIX 5: Handles empty agent_outputs gracefully.
        FIX 5: Wraps LLM call in try/except with a sensible fallback.
        FIX 6: Corrects signal entity iteration (was reading signal.entity_name
               which doesn't exist — entities live in signal.target_entities).
        FIX 7: asyncio.create_task guarded via _safe_create_task.
        """
        context_parts: List[str] = []
        fundamental_df = None
        news_sources: List[CitedSource] = []
        all_enriched_entities = []

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
            all_enriched_entities.extend(getattr(output, "entities_enriched", []))

        portfolio = self.get_portfolio(settings.PORTFOLIO_JSON_PATH)
        portfolio_block = json.dumps(portfolio, indent=2) if portfolio else "[]"
        multi_agent = len(state.agent_outputs) > 1
        user_context_section = SYNTHESISER_USER_CONTEXT_SECTION.format(
            user_context=state.user_context_block
        )

        user_response = ""
        cross_relationships: List[dict] = []

        # ── LLM synthesis ─────────────────────────────────────────
        if not context_parts and not state.user_context_block and not portfolio_block:
            # All agents failed or no agents ran — produce a safe fallback
            logger.warning(
                "_synthesize_node: no context from agents, skipping LLM synthesis"
            )
            user_response = (
                "I wasn't able to retrieve data for your query at this time. "
                "Please try again or rephrase your question."
            )
        else:
            try:
                prompt = self._build_synthesis_prompt(user_context_section, multi_agent)
                raw = await self._run_synthesis_chain(
                    prompt,
                    state,
                    context_parts,
                    portfolio_block,
                )

                if multi_agent:
                    match = _CROSS_RE.search(raw)
                    if match:
                        try:
                            cross_relationships = json.loads(match.group(1).strip())
                        except json.JSONDecodeError:
                            cross_relationships = []

                resp_match = _RESPONSE_RE.search(raw)
                user_response = (
                    resp_match.group(1).strip() if resp_match else raw.strip()
                )

            except Exception:
                logger.exception("_synthesize_node: LLM synthesis failed")
                user_response = (
                    "I encountered an error while synthesising the analysis. "
                    "The raw data was retrieved but could not be summarised."
                )

        # ── Graph write-back (async, non-blocking) ────────────────
        if multi_agent and state.conversation_id:
            try:
                builder = InMemorySubgraphBuilder(
                    embedding_func=service_manager.get_embedding_func(),
                    fuzzy_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
                    semantic_threshold=settings.EXTRACTION_SEMANTIC_THRESHOLD,
                )
                cross_graph = nx.DiGraph()
                if cross_relationships:
                    cross_graph = await builder.build(
                        cross_relationships, source_agent="orchestrator"
                    )

                if cross_graph.number_of_edges() > 0:
                    _safe_create_task(
                        service_manager.get_ingestor()._upsert_graph_to_neo4j(
                            cross_graph, state.conversation_id
                        )
                    )
            except Exception:
                logger.exception("_synthesize_node: graph write-back setup failed")

        # ── User interest signal write-back ───────────────────────
        if state.user_email and state.plan:
            await self._write_user_signals(state, context_parts)

        return {
            "summary": user_response,
            "fundamental_data": fundamental_df,
            "sources": news_sources,
        }

    async def _write_user_signals(
        self,
        state: OrchestratorState,
        context_parts: List[str],
    ) -> None:
        """
        Persist investment and learning interest signals to the user graph.

        FIX 6 (signal iteration): InvestmentSignalDetection has
        `target_entities: List[UserInterestEntity]` not `entity_name` /
        `entity_type` at the top level. We iterate target_entities correctly.
        """
        user_message = _extract_last_human_message(state.messages)
        ingestor = service_manager.get_ingestor()
        user_context_service = service_manager.get_user_context_service()
        entity_cache: Dict[str, Any] = {}
        wrote_any = False

        # ── Investment signals ────────────────────────────────────
        for signal in state.plan.detected_investment_signals or []:
            for entity in signal.target_entities or []:  # ← FIX: was signal.entity_name
                try:
                    resolved_id = await ingestor.resolve_entity_id(
                        entity.entity_name,
                        entity.entity_type,
                        entity_cache=entity_cache,
                    )
                    if not resolved_id:
                        continue
                    node = UserInvestmentInterestNode(
                        id="",
                        user_email=state.user_email,
                        status=signal.status,
                        reason=user_message,
                        confidence="high",
                        updated_at=datetime.now(timezone.utc),
                        target_entity_ids=[resolved_id],
                    )
                    user_context_service.schedule_upsert_fire_and_forget(
                        node, state.user_email
                    )
                    wrote_any = True
                except Exception:
                    logger.exception(
                        "_write_user_signals: investment signal failed for '%s'",
                        entity.entity_name,
                    )

        # ── Learning signals ──────────────────────────────────────
        for signal in state.plan.detected_learning_signals or []:
            for entity in signal.target_entities or []:  # ← FIX: was signal.entity_name
                try:
                    resolved_id = await ingestor.resolve_entity_id(
                        entity.entity_name,
                        entity.entity_type,
                        entity_cache=entity_cache,
                    )
                    if not resolved_id:
                        continue
                    node = UserLearningInterestNode(
                        id="",
                        user_email=state.user_email,
                        status=signal.status,
                        reason=user_message,
                        updated_at=datetime.now(timezone.utc),
                        target_entity_ids=[resolved_id],
                    )
                    user_context_service.schedule_upsert_fire_and_forget(
                        node, state.user_email
                    )
                    wrote_any = True
                except Exception:
                    logger.exception(
                        "_write_user_signals: learning signal failed for '%s'",
                        entity.entity_name,
                    )

        # ── User interest subgraph ────────────────────────────────
        detected_investment = state.plan.detected_investment_signals or []
        detected_learning = state.plan.detected_learning_signals or []

        if not (detected_investment or detected_learning) or not state.conversation_id:
            return

        try:
            user_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a financial relationship extractor. Given the user context and "
                        "agent analyses, extract only relationships that connect entities to the "
                        "user's investment or learning signals. Return only the structured fields "
                        "required by the schema.",
                    ),
                    (
                        "human",
                        "USER CONTEXT:\n{user_context}\n\n"
                        "DETECTED INVESTMENT SIGNALS:\n{investment_signals}\n\n"
                        "DETECTED LEARNING SIGNALS:\n{learning_signals}\n\n"
                        "AGENT FINDINGS:\n{context}\n",
                    ),
                ]
            )
            structured_llm = self._llm.with_structured_output(UserInterestMapping)
            mapping: UserInterestMapping = await (user_prompt | structured_llm).ainvoke(
                {
                    "user_context": state.user_context_block,
                    "investment_signals": detected_investment,
                    "learning_signals": detected_learning,
                    "context": "\n\n".join(context_parts),
                }
            )
        except Exception:
            logger.exception("_write_user_signals: UserInterestMapping LLM call failed")
            return

        edges = (mapping.investment_threats or []) + (mapping.learning_helpers or [])
        if not edges:
            return

        # Build a type-lookup from all signal target_entities
        target_type_lookup: Dict[str, str] = {}
        for signal in detected_investment + detected_learning:
            for entity in signal.target_entities or []:
                if entity.entity_name and entity.entity_type:
                    target_type_lookup[entity.entity_name.lower()] = entity.entity_type

        rels = []
        for edge in edges:
            target_type = target_type_lookup.get(edge.target_entity_name.lower())
            if not target_type:
                continue
            rels.append(
                {
                    "from_name": edge.entity_name,
                    "from_type": edge.entity_type,
                    "relation": edge.relationship,
                    "to_name": edge.target_entity_name,
                    "to_type": target_type,
                    "confidence": edge.confidence,
                    "reason": edge.reason,
                    "extra_props": {"derived_for_user_email": state.user_email},
                }
            )

        if not rels:
            return

        try:
            builder = InMemorySubgraphBuilder(
                embedding_func=service_manager.get_embedding_func(),
                fuzzy_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
                semantic_threshold=settings.EXTRACTION_SEMANTIC_THRESHOLD,
            )
            interest_graph = await builder.build(rels, source_agent="orchestrator")
            _safe_create_task(
                service_manager.get_ingestor()._upsert_graph_to_neo4j(
                    interest_graph, state.conversation_id
                )
            )
        except Exception:
            logger.exception(
                "_write_user_signals: interest subgraph build/upsert failed"
            )
            return

        # Update last_analysis_summary on user interest targets
        try:
            user_graph = service_manager.get_neo4j_adapter()
            entity_cache2: Dict[str, Any] = {}

            def _first_sentence(text: str) -> str:
                text = (text or "").strip()
                sentence = text.split(".")[0].strip()
                return f"{sentence}." if sentence else ""

            for edge in edges:
                target_type = target_type_lookup.get(edge.target_entity_name.lower())
                if not target_type:
                    continue
                try:
                    target_id = await ingestor.resolve_entity_id(
                        edge.target_entity_name,
                        target_type,
                        entity_cache=entity_cache2,
                    )
                    if not target_id:
                        continue
                    summary = _first_sentence(edge.reason)
                    if summary:
                        await user_graph.update_targets_last_analysis_summary(
                            state.user_email, target_id, summary
                        )
                except Exception:
                    logger.exception(
                        "_write_user_signals: summary update failed for '%s'",
                        edge.target_entity_name,
                    )
        except Exception:
            logger.exception(
                "_write_user_signals: last_analysis_summary update loop failed"
            )
