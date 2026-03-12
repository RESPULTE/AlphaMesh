import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

# --- Import Sub-Agents & Their Outputs ---
from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput, CitedSource
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.logger import get_logger
from core.memory.conversation_writeback import run_conversation_writeback
from core.memory.prompts import SYNTHESISER_WRITEBACK_SYSTEM_PROMPT

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


class OrchestratorPlan(BaseAgentInput):
    model_config = ConfigDict(extra="ignore")

    target_agents: List[str] = Field(description="Agents to activate.")
    request_requires_agents: bool = Field(
        description="True if query needs agent tools."
    )
    final_answer: Optional[str] = Field(
        default=None, description="Direct answer if no agents needed (e.g. greetings)."
    )


class OrchestratorState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Changed from query: str to messages: List[BaseMessage]
    messages: List[BaseMessage] = Field(default_factory=list)
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: Dict[str, BaseAgentOutput] = Field(default_factory=dict)
    final_response: Optional[FinalResponse] = None
    # NEW: write-back payload populated by _synthesize_node
    writeback_relationships: List[dict] = Field(default_factory=list)
    writeback_entities: List[Any] = Field(default_factory=list)
    conversation_id: Optional[str] = None  # passed in from caller
    graph_context: List[dict] = Field(default_factory=list)


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
            if state.plan.request_requires_agents
            else "portfolio_analyst"
        )

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)
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
        self, messages: List[BaseMessage], conversation_id: Optional[str] = None
    ) -> FinalResponse:
        """Entry point accepting a list of LangChain messages."""
        initial_state = OrchestratorState(
            messages=messages,
            conversation_id=conversation_id,
        )
        final_state = await self._graph.ainvoke(initial_state)

        if final_state.get("final_response"):
            return final_state["final_response"]

        return FinalResponse(
            summary=final_state["plan"].final_answer
            or "I couldn't process that request."
        )

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        now = datetime.now()
        available_agents_desc = ", ".join(
            [f"{a.name()}: {a.description()}" for a in AVAILABLE_AGENTS]
        )
        graph_context = await self._query_user_graph_context(
            query=state.messages[-1].content if state.messages else "",
            conversation_id=state.conversation_id,
        )

        system_prompt = (
            "You are a Financial Orchestrator. Review the chat history and decide which agents to call.\n"
            f"AVAILABLE AGENTS: {available_agents_desc}\n"
            "If the user's latest message is a greeting or doesn't require data, provide 'final_answer'.\n"
            "If the latest message refers to a company mentioned earlier (e.g., 'its revenue'), "
            "ensure you extract the correct ticker from history."
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

        return {"plan": plan, "graph_context": graph_context}

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
        shared_input = BaseAgentInput(**plan.model_dump())

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

        chain = prompt | self._llm
        response = await chain.ainvoke(
            {"history": state.messages, "context": "\n\n".join(context_parts)}
        )
        raw = response.content.strip()

        # --- Parse <relationships> block (fault-tolerant) ---
        relationships = []
        rel_match = re.search(r"<relationships>(.*?)</relationships>", raw, re.DOTALL)
        if rel_match:
            try:
                relationships = json.loads(rel_match.group(1).strip())
                if not isinstance(relationships, list):
                    relationships = []
            except json.JSONDecodeError:
                relationships = []  # malformed JSON must NOT break user response

        # --- Parse <response> block ---
        resp_match = re.search(r"<response>(.*?)</response>", raw, re.DOTALL)
        user_response = resp_match.group(1).strip() if resp_match else raw

        # --- Fire write-back asynchronously (non-blocking) ---
        # This runs after the node returns — user never waits for it
        if state.conversation_id:
            asyncio.create_task(
                run_conversation_writeback(
                    relationships=relationships,
                    enriched_entities=all_enriched_entities,
                    user_email=None,  # pass from caller if multi-tenant needed
                    conversation_id=state.conversation_id,
                )
            )

        return {
            "final_response": FinalResponse(
                summary=user_response,
                fundamental_data=fundamental_df,
                sources=news_sources,
            ),
            "writeback_relationships": relationships,
            "writeback_entities": all_enriched_entities,
        }
