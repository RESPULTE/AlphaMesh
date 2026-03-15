import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Type

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

# --- Import Sub-Agents & Their Outputs ---
from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput, CitedSource
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    SYNTHESISER_WRITEBACK_SYSTEM_PROMPT,
)
from core.logger import get_logger
from core.memory.graph.models import (
    RelationshipType,
    UserInvestmentInterestNode,
    UserLearningInterestNode,
)
from core.memory.retrieval.models import RewrittenQueries
from core.memory.user_context_service import UserContext

# --- Import Core Services ---
from core.services import service_manager

logger = get_logger(__name__)

AVAILABLE_AGENTS: List[Type[AbstractAgent]] = [
    FundamentalAnalysisAgent,
    NewsAnalysisAgent,
]


class FinalResponse(BaseModel):
    """The structured output returned to the UI for professional rendering."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str
    fundamental_data: Optional[pd.DataFrame] = None
    sources: List[CitedSource] = Field(default_factory=list)


class UserInterestEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entity_name: str
    entity_type: Literal["Company", "FinancialConcept", "FinancialEvent", "Sector"]


class InvestmentSignalDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_entities: List[UserInterestEntity] = Field(
        default_factory=list,
        description="List of entities the user expressed an investment stance towards.",
    )
    status: Literal["Bought", "Interested", "Sold", "Avoids"]


class LearningSignalDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_entities: List[UserInterestEntity] = Field(
        default_factory=list,
        description="List of entities the user expressed a learning interest in.",
    )
    status: Literal["Interested", "Understood", "Confused", "Not Interested"]


class OrchestratorPlan(BaseAgentInput):
    model_config = ConfigDict(extra="ignore")

    target_agents: List[str] = Field(
        default_factory=list, description="Agents to activate."
    )
    final_answer: Optional[str] = Field(
        default=None, description="Direct answer if no agents needed (e.g. greetings)."
    )
    target_entities: List[str] = Field(
        default_factory=list,
        description="Entities explicitly mentioned in the query for disambiguation.",
    )

    rewritten_queries: Optional[RewrittenQueries] = Field(default=None)
    detected_investment_signals: List[InvestmentSignalDetection] = Field(
        default_factory=list
    )
    detected_learning_signals: List[LearningSignalDetection] = Field(
        default_factory=list
    )


class OrchestratorState(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Changed from query: str to messages: List[BaseMessage]
    messages: List[BaseMessage] = Field(default_factory=list)
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: Dict[str, BaseAgentOutput] = Field(default_factory=dict)
    final_response: Optional[FinalResponse] = None

    # NEW: write-back payload populated by _synthesize_node
    conversation_id: Optional[str] = None  # passed in from caller
    user_email: Optional[str] = None
    user_context: Optional[UserContext] = None
    user_context_block: str = ""
    memory_task: Optional[Any] = Field(default=None, exclude=True)


class RelationshipEntry(BaseModel):
    """Typed relationship for synthesizer writeback."""

    model_config = ConfigDict(extra="ignore")

    from_name: str
    from_type: Literal["Company", "FinancialEvent", "FinancialConcept", "Sector"]
    relation: RelationshipType  # enforced enum — was Dict key "relation": str
    to_name: str
    to_type: Literal["Company", "FinancialEvent", "FinancialConcept", "Sector"]
    confidence: Literal["high", "low"]


class SynthesizedResponse(BaseModel):
    relationships: List[RelationshipEntry]
    response: str = Field(description="The final user-facing analysis response.")


class OrchestratorAgent:
    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)
        self._agents: Dict[str, AbstractAgent] = {
            agent.name(): agent() for agent in AVAILABLE_AGENTS
        }
        self._graph = self._build_graph()

    def _router(self, state: OrchestratorState) -> str:
        if state.plan.final_answer is not None:
            return "END"
        return (
            "Execute_Selected_Agents"
            if state.plan.target_agents
            else "portfolio_analyst"
        )

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState, output_schema=FinalResponse)
        workflow.add_node("planner", self._plan_node)
        workflow.add_node("Execute_Selected_Agents", self._execute_node)
        workflow.add_node("portfolio_analyst", self._synthesize_node)

        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self._router,
            {
                "END": END,
                "Execute_Selected_Agents": "Execute_Selected_Agents",
                "portfolio_analyst": "portfolio_analyst",
            },
        )
        workflow.add_edge("Execute_Selected_Agents", "portfolio_analyst")
        workflow.add_edge("portfolio_analyst", END)
        return workflow.compile()

    async def run(
        self,
        messages: List[BaseMessage],
        conversation_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> FinalResponse:
        """Entry point accepting a list of LangChain messages."""
        user_context = None
        user_context_block = "USER CONTEXT: None"
        if user_email:
            user_context = (
                await service_manager.get_user_context_service().load_for_user(
                    user_email
                )
            )

            user_context_block = (
                service_manager.get_user_context_service().get_formatted_context(
                    user_email, limit=15
                )
                if user_email
                else "USER CONTEXT: None"
            )
        initial_state = OrchestratorState(
            messages=messages,
            conversation_id=conversation_id,
            user_email=user_email,
            user_context=user_context,
            user_context_block=user_context_block,
        )
        final_state = await self._graph.ainvoke(initial_state)

        return FinalResponse(**final_state)

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        now = datetime.now()
        available_agents_desc = ", ".join(
            [f"{a.name()}: {a.description()}" for a in AVAILABLE_AGENTS]
        )

        system_prompt = (
            "You are a Financial Orchestrator. Review the chat history and decide which agents to call.\n"
            f"AVAILABLE AGENTS: {available_agents_desc}\n"
            f"USER CONTEXT:\n{state.user_context_block}\n"
            "Use user context to influence query rewrites and agent selection.\n"
            "Only populate detected_* signals when the user explicitly states intent or confusion.\n"
            "For investment signals, require explicit stance verbs (buy, bought, sell, sold, avoid, avoids, interested).\n"
            "For learning signals, require explicit learning intent or confusion.\n"
            "If the user's latest message is a greeting or doesn't require data, provide 'final_answer'.\n"
            "If the latest message refers to a company mentioned earlier (e.g., 'its revenue'), "
            "ensure you extract the correct ticker from history.\n\n"
            f"{QUERY_REWRITE_SYSTEM_PROMPT}"
        )

        planner_llm = self._llm.with_structured_output(OrchestratorPlan)

        # We pass the full message history to the planner
        plan: OrchestratorPlan = await planner_llm.ainvoke(
            [SystemMessage(content=system_prompt)] + state.messages
        )

        # Default date logic
        if plan.end_date is None:
            plan.end_date = now
        if plan.start_date is None:
            plan.start_date = plan.end_date - timedelta(days=365)

        memory_task = None
        if (
            plan.target_agents
            and plan.rewritten_queries
            and plan.rewritten_queries.active_domains
        ):
            memory_task = asyncio.create_task(
                service_manager.get_retriever().comprehensive_retrieve(
                    plan.rewritten_queries
                )
            )

        return {
            "plan": plan,
            "memory_task": memory_task,
        }

    async def _query_user_graph_context(
        self, query: str, conversation_id: Optional[str]
    ) -> List[dict]:
        """
        Future iteration: retrieve user-specific graph context ranked by recency.
        Queries Neo4j for entities and relationships related to the user's
        conversation history, ordered by `ingested_at` descending.

        Returns: List of entity/relationship dicts (empty list until implemented)
        """
        return []

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = state.plan
        shared_input = BaseAgentInput(
            **plan.model_dump(), memory_task=state.memory_task
        )

        tasks = [
            self._agents[name].run(shared_input)
            for name in plan.target_agents
            if name in self._agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = {
            name: res
            for name, res in zip(plan.target_agents, results)
            if not isinstance(res, Exception)
        }
        return {"agent_outputs": outputs}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        MODIFIED: Emits CoT <relationships> block before the user-facing response.
        The relationships block serves as explicit reasoning grounding the analysis.
        Write-back is fired asynchronously after this node completes.
        """
        context_parts = []
        fundamental_df = None
        news_sources = []
        all_enriched_entities = []

        for name, output in state.agent_outputs.items():
            context_parts.append(output.get_llm_context_str())
            if name == "fundamentals_agent":
                fundamental_df = getattr(output, "financial_data", None)
            if name == "news_agent":
                news_sources = getattr(output, "sources", [])
            # Collect enriched entities from ALL agents
            all_enriched_entities.extend(getattr(output, "entities_enriched", []))

        # Use the CoT synthesiser prompt instead of the old system prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYNTHESISER_WRITEBACK_SYSTEM_PROMPT
                    + "\n\nAgent Findings:\n{context}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Produce the relationship reasoning and final analysis."),
            ]
        )

        chain = prompt | self._llm.with_structured_output(SynthesizedResponse)
        response_data = await chain.ainvoke(
            {
                "history": state.messages,
                "context": "\n\n".join(context_parts),
                "user_context": state.user_context_block,
            }
        )

        relationships = (
            [r.model_dump() for r in response_data.relationships]
            if response_data
            else []
        )
        user_response = response_data.response if response_data else ""

        # --- Fire write-back asynchronously (non-blocking) ---
        # This runs after the node returns — user never waits for it
        if state.conversation_id:
            asyncio.create_task(
                service_manager.get_ingestor().run_conversation_writeback(
                    relationships=relationships,
                    enriched_entities=all_enriched_entities,
                    user_email=None,  # pass from caller if multi-tenant needed
                    conversation_id=state.conversation_id,
                )
            )

        if state.user_email and state.plan:
            user_message = ""
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                    user_message = msg.content
                    break
            if not user_message and state.messages:
                user_message = state.messages[-1].content

            ingestor = service_manager.get_ingestor()
            user_context_service = service_manager.get_user_context_service()
            entity_cache = {}
            wrote_any = False

            for signal in state.plan.detected_investment_signals or []:
                resolved_id = await ingestor.resolve_entity_id(
                    signal.entity_name,
                    signal.entity_type,
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

            for signal in state.plan.detected_learning_signals or []:
                resolved_id = await ingestor.resolve_entity_id(
                    signal.entity_name,
                    signal.entity_type,
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

            if wrote_any:
                user_context_service.invalidate(state.user_email)

        return {
            "summary": user_response,
            "fundamental_data": fundamental_df,
            "sources": news_sources,
        }
