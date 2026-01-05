# orchestrator_agent.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# --- Import Sub-Agents & Their Outputs ---
from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput
from core.agents.news_analysis_agent import NewsAnalysisAgent

# --- Import Core Services ---
from core.services import service_manager

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Orchestrator")

AVAILABLE_AGENTS: List[Type[AbstractAgent]] = [
    FundamentalAnalysisAgent,
    NewsAnalysisAgent,
]


class OrchestratorPlan(BaseAgentInput):
    """
    Extends the BaseInput to include routing logic.
    """

    target_agents: List[str] = Field(
        description="Which agents to activate. Select based on the user's needs."
    )

    request_requires_agents: bool = Field(
        description="True if the query requires agent calls, False if it can be answered directly by the orchestrator (e.g., simple greetings)."
    )

    final_answer: Optional[str] = Field(
        default=None,
        description="If the query can be answered directly by the orchestrator, provide the answer here.",
    )


class OrchestratorState(BaseModel):
    """
    Internal state of the Orchestrator Graph.
    """

    query: str
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: dict[str, str] = Field(default_factory=dict)
    final_answer: Optional[str] = None

    request_requires_agents: bool = True


class OrchestratorAgent:
    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)

        self._agents: Dict[str, AbstractAgent] = {
            agent.name(): agent() for agent in AVAILABLE_AGENTS
        }

        self._graph = self._build_graph()

    def _router(self, state: OrchestratorState) -> str:
        if state.plan.final_answer != None:
            return "END"
        elif state.plan.request_requires_agents:
            return "Execute_Selected_Agents"

        if len(state.plan.target_agents) == 1:
            return "portfolio_analyst"

        return "END"

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)

        workflow.add_node("planner", self._plan_node)
        workflow.add_node("Execute_Selected_Agents", self._execute_node)
        # RENAMED: from _synthesize_node to _analyst_node
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
        available_agents_desc = ", ".join(
            [f"{a.name()}: {a.description()}" for a in AVAILABLE_AGENTS]
        )

        system_prompt = (
            "You are a Financial Orchestrator. Your job is to:\n"
            "1. Extract core parameters (Ticker, Dates, Metrics) from the query into a standardized format.\n"
            "2. Select the specific agents required to answer the question.\n\n"
            "3. determine whether the user's query requires the calling of other agents.\n\n"
            "if the query is trivial and can be resolved without calling other agents, please directly generate the answer in response to the user query.\n\n"
            "### DATE GUIDELINES (CRITICAL):\n"
            f"- **Today's Date:** {today_str}\n"
            f"- **'Recent' Definition:** If the user asks for 'recent' or 'latest' data without a specific year, "
            f"set 'start_date' to {one_year_ago_str} and 'end_date' to {today_str}.\n"
            "- **Safety:** Do NOT set 'end_date' into the future.\n\n"
            "### AVAILABLE AGENTS:\n"
            f"{available_agents_desc}"
        )
        planner_llm = self._llm.with_structured_output(OrchestratorPlan)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.query),
        ]
        plan: OrchestratorPlan = await planner_llm.ainvoke(messages)

        if not plan.request_requires_agents:
            logger.info("   -> Planner resolved query directly. Final Answer provided.")
            return {"plan": plan}

        if plan.end_date is None or plan.end_date > now:
            plan.end_date = now
        if plan.start_date is None or plan.start_date > plan.end_date:
            plan.start_date = plan.end_date - timedelta(days=365)

        logger.info(f"   -> Plan Target Agents={plan.target_agents}")

        logger.info(
            f"   -> Plan Ticker={plan.ticker}, Start={plan.start_date.strftime('%Y-%m-%d')}, End={plan.end_date.strftime('%Y-%m-%d')}"
        )

        logger.info(f"   -> Plan Query={plan.query}, Vector Query={plan.vector_query}")

        logger.info(f"   -> Plan Metrics={plan.metrics}")

        return {"plan": plan, "request_requires_agents": True}

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Parallel execution of agents. Now returns a list of BaseAgentOutput objects.
        """
        logger.info("⚙️  [Execute_Selected_Agents] Running agents in parallel...")
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
            "1. **Summarize and Synthesize (Citation-Safe):** Slightly shorten and summarize the information from the "
            "'Collected Agent Findings'. You may rephrase sentences, but **you must preserve all in-text citations exactly as they appear** "
            "(e.g. [1], [2], (Reuters, 2024)). If a sentence contains a citation, the rewritten sentence must still contain that citation.\n"
            "2. **No Citation Loss or Modification:** Do **not** remove, merge, renumber, invent, or alter citations in any way. "
            "Do not introduce new citations. Citations must only come from the provided context.\n"
            "4.  **Improve Cohesion:** Ensure your writing flows naturally. Seamlessly integrate any quantitative data with the qualitative news and analysis.\n"
            "5.  **Strictly No New Information:** You must only use the information provided in the context above. Do not add any external knowledge or make up new facts. If specific information wasn't found, it's important to state that it's missing.\n"
            "6.  **Final Summary:** At the very end of your response, please add a short and brief summary (just a few sentences) that recaps the most critical points of what has been found.\n"
            "7.  **Formatting:** Use professional Markdown for your final response."
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
        query = (
            "Why did NVDA stock rise so much recently? did any events happened lately?"
        )
        final_response = await orchestrator.run(query)
        print("\n" + "=" * 50 + "\nFINAL OUTPUT:\n" + "=" * 50)
        print(final_response)

    asyncio.run(main())

    # agent = OrchestratorAgent()
    # png_bytes = agent._graph.get_graph().draw_mermaid_png()

    # with open("orchestrator.png", "wb") as f:
    #     f.write(png_bytes)

    # print("Saved graph as graph.png")
