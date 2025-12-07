# orchestrator_agent.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

# --- Import Sub-Agents & Their Outputs ---
from core.agents.base_agent import AbstractAgent
from core.agents.fundamental_analysis_agent import (
    FundamentalAnalysisAgent,
    FundamentalAnalysisOutput,
)
from core.agents.news_analysis_agent import NewsAnalysisAgent, NewsAnalysisOutput

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
    # REFACTORED: This now holds the raw Pydantic model outputs from the agents.
    agent_outputs: List[Any] = Field(default_factory=list)
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

        # Conditional logic: If plan has agents, execute. Else, skip to synthesizer.
        workflow.add_conditional_edges(
            "planner",
            lambda state: (
                "executor" if state.plan and state.plan.target_agents else "synthesizer"
            ),
        )

        workflow.add_edge("executor", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    async def run(self, query: str) -> str:
        """Main entry point."""
        logger.info(f"🎼 [Orchestrator] Started. Query: {query}")

        initial_state = OrchestratorState(query=query)
        # The final result is a state object, we extract the final_answer field
        final_state = await self._graph.ainvoke(initial_state)

        return final_state.get("final_answer", "Error: No final answer generated.")

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
        if plan.end_date is None:
            plan.end_date = now

        if plan.end_date > now:
            plan.end_date = now

        if plan.start_date is None:
            plan.start_date = plan.end_date - timedelta(days=365)

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
        REFACTORED: Now collects raw Pydantic model outputs.
        """
        logger.info("⚙️  [Executor] Running agents in parallel...")

        plan = state.plan

        # 1. Create the Shared Input Object
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
                tasks.append(agent_instance.run(shared_input_data))
                active_agent_names.append(agent_name)
            else:
                logger.error(f"   ❌ Agent {agent_name} not found in registry.")

        # 3. Await All
        if not tasks:
            return {"agent_outputs": []}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Process Results and collect raw outputs
        agent_outputs = []
        for name, res in zip(active_agent_names, results):
            if isinstance(res, Exception):
                error_msg = f"Agent '{name}' failed with error: {str(res)}"
                logger.error(f"   ❌ {error_msg}")
                agent_outputs.append(error_msg)  # Append error string for context
            elif isinstance(res, BaseModel):
                logger.info(
                    f"   -> Received output from {name} of type {type(res).__name__}"
                )
                agent_outputs.append(res)  # Append the raw Pydantic model
            else:
                # Fallback for unexpected return types
                agent_outputs.append(str(res))

        return {"agent_outputs": agent_outputs}

    async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Aggregates the raw data from agent outputs into a final, reasoned answer.
        REFACTORED: This is now the primary reasoning and analysis step.
        """
        logger.info("✍️  [Synthesizer] Compiling final report from raw data...")

        if not state.agent_outputs:
            return {"final_answer": "No agents were executed or all failed."}

        # Format context from the raw Pydantic models
        reports = []
        for i, output in enumerate(state.agent_outputs):
            report_str = f"--- Data from Report {i+1} ---\n"
            if isinstance(output, FundamentalAnalysisOutput):
                report_str += "Source Agent: fundamentals_agent\n"
                if (
                    output.financial_data is not None
                    and not output.financial_data.empty
                ):
                    report_str += "Type: Financial DataFrame\n"
                    report_str += (
                        "Data:\n" + output.financial_data.to_string(max_rows=20) + "\n"
                    )
                else:
                    report_str += "Data: No financial data was returned.\n"

            elif isinstance(output, NewsAnalysisOutput):
                report_str += "Source Agent: news_agent\n"
                if output.sources:
                    report_str += "Type: List of News Articles\n"
                    # Format sources for clarity
                    source_context = "\n".join(
                        [
                            f'  - Title: {s.title}\n    URL: {s.url}\n    Content Snippet: "{s.page_content[:250]}..."'
                            for s in output.sources
                        ]
                    )
                    report_str += "Data:\n" + source_context + "\n"
                else:
                    report_str += "Data: No news articles were found.\n"

            elif isinstance(output, str):
                # Handle error strings from the executor
                report_str += (
                    f"Source Agent: Execution Error\nError Details: {output}\n"
                )
            else:
                # Generic fallback
                report_str += f"Source Agent: Unknown\nData: {str(output)}\n"

            reports.append(report_str)

        combined_context = "\n".join(reports)

        prompt = ChatPromptTemplate.from_template(
            "You are a Senior Financial Analyst. Your task is to synthesize the following raw data reports from different specialized agents to provide a comprehensive answer to the user's question.\n\n"
            "**User's Original Query:** {query}\n\n"
            "**Raw Agent Data:**\n{context}\n\n"
            "**Your Instructions:**\n"
            "1. **Synthesize, Don't Summarize:** Do not just list the data. Integrate the quantitative data (financials) and qualitative data (news) into a seamless, coherent narrative.\n"
            "2. **Address the Core Question:** Directly answer the user's query using the provided data as evidence.\n"
            "3. **Identify Key Insights:** Highlight trends, correlations, and important figures. For example, if the user asks why a stock dropped, connect news events with financial data if possible.\n"
            "4. **Acknowledge Missing Data:** If the data is insufficient to answer the question, state it clearly.\n"
            "5. **Professional Formatting:** Present your final answer in well-structured Markdown. Use headings, lists, and bold text to improve readability."
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
