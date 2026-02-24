import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# --- Import Sub-Agents & Their Outputs ---
from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput
from core.agents.news_analysis_agent import CitedSource, NewsAnalysisAgent

# --- Import Core Services ---
from core.services import service_manager

from core.logger import get_logger

logger = get_logger(__name__)

AVAILABLE_AGENTS: List[Type[AbstractAgent]] = [
    FundamentalAnalysisAgent,
    NewsAnalysisAgent,
]


class FinalResponse(BaseModel):
    """The structured output returned to the UI for professional rendering."""

    summary: str
    fundamental_data: Optional[pd.DataFrame] = None
    sources: List[CitedSource] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class OrchestratorPlan(BaseAgentInput):
    target_agents: List[str] = Field(description="Agents to activate.")
    request_requires_agents: bool = Field(
        description="True if query needs agent tools."
    )
    final_answer: Optional[str] = Field(
        default=None, description="Direct answer if no agents needed (e.g. greetings)."
    )


class OrchestratorState(BaseModel):
    # Changed from query: str to messages: List[BaseMessage]
    messages: List[BaseMessage] = Field(default_factory=list)
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: Dict[str, BaseAgentOutput] = Field(default_factory=dict)
    final_response: Optional[FinalResponse] = None


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

    async def run(self, messages: List[BaseMessage]) -> FinalResponse:
        """Entry point accepting a list of LangChain messages."""
        initial_state = OrchestratorState(messages=messages)
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

        return {"plan": plan}

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
        context_parts = []
        fundamental_df = None
        news_sources = []

        for name, output in state.agent_outputs.items():
            context_parts.append(output.get_llm_context_str())
            if name == "fundamentals_agent":
                fundamental_df = getattr(output, "financial_data", None)
            if name == "news_agent":
                news_sources = getattr(output, "sources", [])

        # Using MessagesPlaceholder to inject the history into the synthesis
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Senior Financial Analyst. Write a cohesive narrative summary based on findings.\n"
                    "**IMPORTANT**: Use numeric in-text citations like [1], [2].\n"
                    "Findings:\n{context}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Produce the final analysis based on our conversation."),
            ]
        )

        chain = prompt | self._llm
        response = await chain.ainvoke(
            {"history": state.messages, "context": "\n".join(context_parts)}
        )

        return {
            "final_response": FinalResponse(
                summary=response.content,
                fundamental_data=fundamental_df,
                sources=news_sources,
            )
        }
