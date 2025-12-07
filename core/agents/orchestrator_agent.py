# orchestrator_agent.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

# --- Import Sub-Agents ---
from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.news_analysis_agent import NewsAnalysisAgent

# --- Import Core Services ---
from core.services import service_manager
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Orchestrator")


# ==========================================
# 1. SHARED INPUT SCHEMA
# ==========================================


class BaseAgentInput(BaseModel):
    """
    The unified input schema shared by the Orchestrator and all Sub-Agents.
    """

    query: str = Field(description="The original user query for context.")
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL).")
    metrics: List[str] = Field(
        default_factory=list,
        description="List of financial metrics to analyze (if applicable).",
    )
    start_date: Optional[datetime] = Field(
        default=None, description="Start date (format: YYYY-MM-DD)."
    )
    end_date: Optional[datetime] = Field(
        default=None, description="End date (format: YYYY-MM-DD)."
    )

    @field_validator("start_date", "end_date", mode="before")
    def parse_dates(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                # Fallback for ISO or other formats LLM might spit out
                return datetime.fromisoformat(v)
        return v


# ==========================================
# 2. ORCHESTRATOR SPECIFIC MODELS
# ==========================================


class OrchestratorPlan(BaseAgentInput):
    """
    Extends the BaseInput to include routing logic.
    The LLM fills this out to define WHAT data to pass and WHO to call.
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
    agent_results: Dict[str, str] = Field(default_factory=dict)
    final_answer: Optional[str] = None


# ==========================================
# 3. THE ORCHESTRATOR AGENT
# ==========================================


class OrchestratorAgent:
    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)

        # Initialize Sub-Agents
        # We Map Name -> Instance
        self._agents: Dict[str, AbstractAgent] = {
            "fundamentals_agent": FundamentalAnalysisAgent(),
            "news_agent": NewsAnalysisAgent(),
        }

        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)

        workflow.add_node("planner", self._plan_node)
        workflow.add_node("executor", self._execute_node)
        workflow.add_node("synthesizer", self._synthesize_node)

        workflow.add_edge(START, "planner")

        # Conditional logic: If plan has agents, execute. Else, skip to end (or handle error).
        workflow.add_conditional_edges(
            "planner",
            lambda state: (
                "executor" if state.plan and state.plan.target_agents else "synthesizer"
            ),
            {"executor": "executor", "synthesizer": "synthesizer"},
        )

        workflow.add_edge("executor", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    async def run(self, query: str) -> str:
        """Main entry point."""
        logger.info(f"🎼 [Orchestrator] Started. Query: {query}")

        initial_state = OrchestratorState(query=query)
        result = await self._graph.ainvoke(initial_state)

        return result["final_answer"]

    # --- Node Implementations ---

    async def _plan_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Uses LLM Structured Output to populate the BaseAgentInput
        and decide which agents to route to.
        Includes Date Context and Safety Defaults.
        """
        logger.info("🧠 [Planner] Extracting shared parameters and routing...")

        # 1. Calculate Dynamic Date Context
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        one_year_ago_str = (now - timedelta(days=365)).strftime("%Y-%m-%d")

        # 2. Construct Prompt with Context
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

        # We force the LLM to return the OrchestratorPlan schema
        planner_llm = self._llm.with_structured_output(OrchestratorPlan)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.query),
        ]

        # 3. Invoke LLM
        plan: OrchestratorPlan = await planner_llm.ainvoke(messages)

        # 4. Programmatic Safety Guards (Post-Processing)
        # If LLM returns None for dates, we enforce the "Recent" rule or "Year-to-Date" rule here.
        if plan.end_date is None:
            plan.end_date = now

        # Cap future dates to today
        if plan.end_date > now:
            plan.end_date = now

        if plan.start_date is None:
            # Default to 1 year lookback if start date is missing
            plan.start_date = plan.end_date - timedelta(days=365)

        # Ensure start_date is not after end_date
        if plan.start_date > plan.end_date:
            plan.start_date = plan.end_date - timedelta(days=365)

        logger.info(f"   -> Plan Target Agents: {plan.target_agents}")
        logger.info(
            f"   -> Date Range: {plan.start_date.date()} to {plan.end_date.date()}"
        )

        return {"plan": plan}

    async def _execute_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Parallel execution.
        Constructs ONE BaseAgentInput object and passes it to all selected agents.
        """
        logger.info("⚙️  [Executor] Running agents in parallel...")

        plan = state.plan

        # 1. Create the Shared Input Object
        # We strip out the 'target_agents' field to get a pure BaseAgentInput
        shared_input_data = BaseAgentInput(
            query=plan.query,
            ticker=plan.ticker,
            metrics=plan.metrics,
            start_date=plan.start_date,
            end_date=plan.end_date,
        )

        tasks = []
        active_agent_names = []

        # 2. Dispatch to Agents
        for agent_name in plan.target_agents:
            agent_instance = self._agents.get(agent_name)

            if agent_instance:
                logger.info(f"   -> Calling {agent_name}...")

                # IMPORTANT: We pass the shared_input_data directly.
                # The agent.run() method must be typed to accept BaseAgentInput.
                tasks.append(agent_instance.run(shared_input_data))
                active_agent_names.append(agent_name)
            else:
                logger.error(f"   ❌ Agent {agent_name} not found in registry.")

        # 3. Await All
        if not tasks:
            return {"agent_results": {}}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Process Results
        agent_results = {}
        for name, res in zip(active_agent_names, results):
            if isinstance(res, Exception):
                error_msg = f"Error: {str(res)}"
                logger.error(f"   ❌ {name} failed: {error_msg}")
                agent_results[name] = error_msg
            else:
                # We assume the agents return Pydantic models.
                # We dump them to JSON strings for the synthesizer.
                if isinstance(res, BaseModel):
                    agent_results[name] = res.model_dump_json()
                else:
                    agent_results[name] = str(res)

        return {"agent_results": agent_results}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Aggregates the JSON outputs from the agents into a final answer.
        """
        logger.info("✍️  [Synthesizer] Compiling final report...")

        if not state.agent_results:
            return {"final_answer": "No agents were executed or all failed."}

        # Format context
        reports = []
        for name, data in state.agent_results.items():
            reports.append(f"### REPORT FROM {name.upper()}:\n{data}\n")

        combined_context = "\n".join(reports)

        prompt = ChatPromptTemplate.from_template(
            "You are a Senior Financial Analyst. Synthesize the following reports to answer the user's question.\n\n"
            "**User Query:** {query}\n\n"
            "**Agent Data:**\n{context}\n\n"
            "**Instructions:**\n"
            "1. Integrate the quantitative data and qualitative news seamlessly.\n"
            "2. If data is missing, state it clearly.\n"
            "3. Provide a professional Markdown response."
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

        # Example Query
        query = "Why did NVDA stock drop recently? Check the news and also look at its net profit margin."

        print(f"\nQUERY: {query}\n" + "=" * 50)
        final_response = await orchestrator.run(query)
        print("\n" + "=" * 50 + "\nFINAL OUTPUT:\n" + "=" * 50)
        print(final_response)

    asyncio.run(main())
