import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# --- Import Sub-Agents & Their Outputs ---
from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput
from core.agents.news_analysis_agent import CitedSource, NewsAnalysisAgent

# --- Import Core Services ---
from core.services import service_manager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Orchestrator")

AVAILABLE_AGENTS: List[Type[AbstractAgent]] = [
    FundamentalAnalysisAgent,
    NewsAnalysisAgent,
]


class FinalResponse(BaseModel):
    """The structured output returned to the UI."""

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
        default=None, description="Direct answer if no agents needed."
    )


class OrchestratorState(BaseModel):
    query: str
    plan: Optional[OrchestratorPlan] = None
    # Store raw output objects to preserve DataFrames and Source lists
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
        if state.plan.request_requires_agents:
            return "Execute_Selected_Agents"
        return "portfolio_analyst"

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

    async def run(self, query: str) -> FinalResponse:
        logger.info(f"🎼 [Orchestrator] Started: {query}")
        initial_state = OrchestratorState(query=query)
        final_state = await self._graph.ainvoke(initial_state)

        if final_state.get("final_response"):
            return final_state["final_response"]

        # Fallback for direct LLM answers (simple greetings)
        return FinalResponse(
            summary=final_state["plan"].final_answer or "No response generated."
        )

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        logger.info("🧠 [Planner] Planning route...")
        now = datetime.now()
        available_agents_desc = ", ".join(
            [f"{a.name()}: {a.description()}" for a in AVAILABLE_AGENTS]
        )

        system_prompt = (
            "You are a Financial Orchestrator. Decide which agents to call.\n"
            f"AVAILABLE AGENTS: {available_agents_desc}\n"
            "If the query is a greeting or trivial, provide a 'final_answer' and set request_requires_agents=False."
        )

        planner_llm = self._llm.with_structured_output(OrchestratorPlan)
        plan: OrchestratorPlan = await planner_llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=state.query)]
        )

        # Date normalization
        if plan.end_date is None:
            plan.end_date = now
        if plan.start_date is None:
            plan.start_date = plan.end_date - timedelta(days=365)

        return {"plan": plan}

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        logger.info("⚙️  [Executor] Running agents...")
        plan = state.plan
        shared_input = BaseAgentInput(**plan.model_dump())

        tasks = [
            self._agents[name].run(shared_input)
            for name in plan.target_agents
            if name in self._agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = {}
        for name, res in zip(plan.target_agents, results):
            if not isinstance(res, Exception):
                outputs[name] = res
        return {"agent_outputs": outputs}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        logger.info("✍️  [Synthesizer] Compiling report...")

        # 1. Prepare context for the LLM
        context_parts = []
        fundamental_df = None
        news_sources = []

        for name, output in state.agent_outputs.items():
            context_parts.append(output.get_llm_context_str())

            # Extract structured data for the FinalResponse
            if name == "fundamentals_agent":
                fundamental_df = getattr(output, "financial_data", None)
            if name == "news_agent":
                news_sources = getattr(output, "sources", [])

        prompt = ChatPromptTemplate.from_template(
            "You are a Senior Financial Analyst. Synthesize the findings below into a narrative report.\n"
            "User Question: {query}\n\n"
            "Findings:\n{context}\n\n"
            "Instructions:\n"
            "1. Focus on a narrative summary. DO NOT create markdown tables (they will be added separately).\n"
            "2. Preserve ALL in-text citations like [1] or [2].\n"
            "3. End with a 2-sentence 'Bottom Line' summary."
        )

        chain = prompt | self._llm
        response = await chain.ainvoke(
            {"query": state.query, "context": "\n".join(context_parts)}
        )

        final_resp = FinalResponse(
            summary=response.content,
            fundamental_data=fundamental_df,
            sources=news_sources,
        )

        return {"final_response": final_resp}
