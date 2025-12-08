# orchestrator_agent.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

# --- Import Sub-Agents & Their Outputs ---
from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput
from core.agents.news_analysis_agent import NewsAnalysisAgent

# --- Import Core Services ---
from core.services import service_manager
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Orchestrator")


class OrchestratorPlan(BaseAgentInput):
    """
    Extends the BaseInput to include routing logic.
    """

    target_agents: List[Literal["fundamentals_agent", "news_agent"]] = Field(
        description="Which agents to activate. Select based on the user's needs."
    )


class OrchestratorState(BaseModel):
    """
    Internal state of the Orchestrator Graph.
    """

    query: str
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: dict[str, str] = Field(default_factory=dict)
    final_answer: Optional[str] = None


class OrchestratorAgent:
    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)

        self._agents: Dict[str, AbstractAgent] = {
            "fundamentals_agent": FundamentalAnalysisAgent(),
            "news_agent": NewsAnalysisAgent(),
        }

        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)

        workflow.add_node("planner", self._plan_node)
        workflow.add_node("executor", self._execute_node)
        # RENAMED: from _synthesize_node to _analyst_node
        workflow.add_node("analyst", self._synthesize_node)

        workflow.add_edge(START, "planner")

        workflow.add_conditional_edges(
            "planner",
            lambda state: (
                "executor" if state.plan and state.plan.target_agents else "analyst"
            ),
        )

        workflow.add_edge("executor", "analyst")
        workflow.add_edge("analyst", END)

        return workflow.compile()

    async def run(self, query: str) -> str:
        """Main entry point."""
        logger.info(f"🎼 [Orchestrator] Started. Query: {query}")
        initial_state = OrchestratorState(query=query)
        final_state = await self._graph.ainvoke(initial_state)
        return final_state.get("final_answer", "Error: No final answer generated.")

    # --- Node Implementations ---

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Uses LLM to decide which agents to route to.
        """
        logger.info("🧠 [Planner] Extracting shared parameters and routing...")
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        one_year_ago_str = (now - timedelta(days=365)).strftime("%Y-%m-%d")

        system_prompt = (
            "You are a Financial Orchestrator. Your job is to:\n"
            "1. Extract core parameters (Ticker, Dates, Metrics) from the query into a standardized format.\n"
            "2. Select the specific agents required to answer the question.\n\n"
            "### DATE GUIDELINES (CRITICAL):\n"
            f"- **Today's Date:** {today_str}\n"
            f"- **'Recent' Definition:** If the user asks for 'recent' or 'latest' data without a specific year, "
            f"set 'start_date' to {one_year_ago_str} and 'end_date' to {today_str}.\n"
            "- **Safety:** Do NOT set 'end_date' into the future.\n\n"
            "### AVAILABLE AGENTS:\n"
            "- 'fundamentals_agent': For financial ratios, balance sheets, margins, FCF, valuation.\n"
            "- 'news_agent': For qualitative analysis, recent events, sentiment, reasons for price moves.\n"
        )
        planner_llm = self._llm.with_structured_output(OrchestratorPlan)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.query),
        ]
        plan: OrchestratorPlan = await planner_llm.ainvoke(messages)

        if plan.end_date is None or plan.end_date > now:
            plan.end_date = now
        if plan.start_date is None or plan.start_date > plan.end_date:
            plan.start_date = plan.end_date - timedelta(days=365)

        logger.info(f"   -> Plan Target Agents: {plan.target_agents}")
        return {"plan": plan}

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Parallel execution of agents. Now returns a list of BaseAgentOutput objects.
        """
        logger.info("⚙️  [Executor] Running agents in parallel...")
        plan = state.plan
        shared_input_data = BaseAgentInput(**plan.model_dump())

        tasks = [
            self._agents[agent_name].run(shared_input_data)
            for agent_name in plan.target_agents
            if agent_name in self._agents
        ]

        if not tasks:
            return {"agent_outputs": []}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_outputs = {}
        for name, res in zip(plan.target_agents, results):
            if isinstance(res, Exception):
                logger.error(f"   ❌ Agent failed: {res}")
                # Optionally create an error output object
            elif isinstance(res, BaseAgentOutput):
                logger.info(f"   -> Received output from {res.agent_name}")
                agent_outputs[name] = res.get_llm_context_str()
                logger.info(
                    f"   -> Agent output for {res.agent_name} : \n\n {agent_outputs[name]}... \n\n\n"
                )
            else:
                logger.error(
                    f"   ❌ Unexpected output type from agent {name}: {type(res)}"
                )

        return {"agent_outputs": agent_outputs}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Aggregates the JSON outputs from the agents into a final answer.
        """
        logger.info("✍️  [Synthesizer] Compiling final report...")

        if not state.agent_outputs:
            return {"final_answer": "No agents were executed or all failed."}

        # Format context
        reports = []
        for name, data in state.agent_outputs.items():
            reports.append(f"### REPORT FROM {name.upper()}:\n{data}\n")

        combined_context = "\n".join(reports)

        prompt = ChatPromptTemplate.from_template(
            "Hello! Your role is to act as a helpful Senior Financial Analyst who is great at communicating. "
            "Your task is to review the findings from your fellow agents and present them in a clear, cohesive summary for the user. "
            "Please start with a friendly and approachable tone.\n\n"
            "**User's Original Question:** {query}\n\n"
            "**Collected Agent Findings:**\n{context}\n\n"
            "**Your Instructions:**\n"
            "1.  **Summarize and Synthesize:** Do not just copy the findings. Your main goal is to slightly shorten and summarize the information from the 'Collected Agent Findings'. Weave them together into a single, easy-to-understand response.\n"
            "2.  **Improve Cohesion:** Ensure your writing flows naturally. Seamlessly integrate any quantitative data with the qualitative news and analysis.\n"
            "3.  **Strictly No New Information:** You must only use the information provided in the context above. Do not add any external knowledge or make up new facts. If specific information wasn't found, it's important to state that it's missing.\n"
            "4.  **Final Summary:** At the very end of your response, please add a short and brief summary (just a few sentences) that recaps the most critical points of what has been found.\n"
            "5.  **Formatting:** Use professional Markdown for your final response."
        )

        chain = prompt | self._llm
        response = await chain.ainvoke(
            {"query": state.query, "context": combined_context}
        )

        return {"final_answer": response.content}


# --- Execution Block for Testing ---
if __name__ == "__main__":

    async def main():
        orchestrator = OrchestratorAgent()
        query = "Why did NVDA stock rise so much recently? Check the news and also look at its net profit margin."
        final_response = await orchestrator.run(query)
        print("\n" + "=" * 50 + "\nFINAL OUTPUT:\n" + "=" * 50)
        print(final_response)

    asyncio.run(main())
