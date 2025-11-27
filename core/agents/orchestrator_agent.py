from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from core.agents.fundamental_analysis_agent import (
    build_graph as build_fundamentals_graph,
)
from core.agents.news_analysis_agent import (
    create_graph_workflow,
    create_retriever_tool,
    query_and_ingest_stock_news,
)

# --- Service & Agent Imports ---
from core.services import service_manager
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# 1. Configuration & Enum (The Source of Truth)
# -------------------------------------------------------------------


class AgentType(str, Enum):
    """
    Defines the available agents.
    """

    NEWS = "news_agent"
    FUNDAMENTALS = "fundamentals_agent"


# Metadata describing what each agent is good at.
# The Supervisor uses this to decide how to break down the query.
AGENT_METADATA = {
    AgentType.NEWS: (
        "Focuses on qualitative data: news, market sentiment, "
        "reasons for price volatility, and macro events."
    ),
    AgentType.FUNDAMENTALS: (
        "Focuses on quantitative data: financial statements, balance sheets, "
        "revenue numbers, margins, and growth ratios."
    ),
}

# -------------------------------------------------------------------
# 2. State Definitions
# -------------------------------------------------------------------


def merge_dictionaries(a: Dict, b: Dict) -> Dict:
    """Reducer to allow parallel nodes to update dictionaries simultaneously."""
    return {**a, **b}


class OrchestratorState(BaseModel):
    """
    The master state of the orchestration layer.
    """

    # --- Inputs ---
    raw_input: str = Field(..., description="Original user input")

    # --- Internal Processing ---
    extracted_ticker: Optional[str] = None

    # [NEW] Maps AgentName -> Specific Query for that agent
    # e.g. {"news_agent": "Why did it drop?", "fundamentals_agent": "What is the P/E?"}
    agent_instructions: Dict[str, str] = Field(default_factory=dict)

    # List of agents selected for execution (used for routing)
    target_agents: List[str] = Field(default_factory=list)

    # --- Outputs ---
    # Parallel storage for results
    agent_outputs: Annotated[Dict[str, str], merge_dictionaries] = Field(
        default_factory=dict
    )

    # --- Final ---
    final_output: Optional[str] = None


# -------------------------------------------------------------------
# 3. Structured Output Models (Decomposition Logic)
# -------------------------------------------------------------------


class AgentTask(BaseModel):
    """
    Represents a specific sub-task assigned to a specific agent.
    """

    assigned_agent: AgentType = Field(
        description="The specialist agent to handle this part."
    )
    specific_instruction: str = Field(
        description="The rewritten, specific sub-query for this agent only. "
        "Do not include parts of the question relevant to other agents."
    )


class RouteDecision(BaseModel):
    """
    The Supervisor's output: Validation + Decomposition + Ticker Extraction.
    """

    is_valid: bool = Field(description="False if input is nonsense.")
    clarification_message: Optional[str] = Field(description="Message if invalid.")

    ticker: Optional[str] = Field(description="The primary ticker symbol involved.")

    # [NEW] List of sub-tasks instead of a global list of agents
    tasks: List[AgentTask] = Field(
        description="Break down the user query into specific tasks for the relevant agents."
    )


# -------------------------------------------------------------------
# 4. Prompt Generation
# -------------------------------------------------------------------


def build_router_system_prompt() -> str:
    """
    Constructs the Supervisor's instructions dynamically from AGENT_METADATA.
    """
    agent_descriptions = "\n".join(
        [f"- **{agent.value}**: {desc}" for agent, desc in AGENT_METADATA.items()]
    )

    return (
        "You are a Senior Financial Orchestrator. Your goal is to decompose complex user queries "
        "into specific sub-tasks for specialized agents.\n\n"
        f"**Available Specialists:**\n{agent_descriptions}\n\n"
        "**Instructions:**\n"
        "1. Analyze the user's input.\n"
        "2. If the input covers multiple topics (e.g., 'News' AND 'Fundamentals'), break it down.\n"
        "3. Assign each part to the correct specialist.\n"
        "4. **Rewrite** the sub-query for that specific agent so it is self-contained and clear.\n"
        "5. Extract the primary Ticker symbol if present.\n\n"
        "**Example:**\n"
        "User: 'Why did NVDA drop and is it profitable?'\n"
        "Output: \n"
        "  - Task 1 (News): 'What are the reasons for the recent price drop of NVDA?'\n"
        "  - Task 2 (Fundamentals): 'Analyze the profitability ratios and net income of NVDA.'"
    )


# -------------------------------------------------------------------
# 5. Agent Registry & Worker Nodes
# -------------------------------------------------------------------


class AgentRegistry:
    _fundamentals_graph = None

    @classmethod
    def get_agent_graph(cls, agent_type: AgentType, ticker: str = "SPY"):
        """Factory method to get the correct graph."""
        if agent_type == AgentType.NEWS:
            llm = service_manager.get_agent()
            query_and_ingest_stock_news(ticker)
            tool = create_retriever_tool(ticker)
            return create_graph_workflow(llm, tool)

        elif agent_type == AgentType.FUNDAMENTALS:
            if cls._fundamentals_graph is None:
                cls._fundamentals_graph = build_fundamentals_graph()
            return cls._fundamentals_graph

        raise ValueError(f"Unknown Agent: {agent_type}")


def generic_worker_node(
    state: OrchestratorState, agent_type: AgentType
) -> Dict[str, Any]:
    """
    Standard worker:
    1. Looks up ITS specific instruction from state.
    2. Runs the agent.
    3. Saves output.
    """
    ticker = state.extracted_ticker or "SPY"

    # [CRITICAL] Retrieve only the instruction meant for this agent
    specific_query = state.agent_instructions.get(agent_type.value)

    if not specific_query:
        return {"agent_outputs": {agent_type.value: "Error: No instruction found."}}

    print(f"--- [Worker: {agent_type.value}] Executing: '{specific_query}' ---")

    try:
        agent = AgentRegistry.get_agent_graph(agent_type, ticker)
        response = agent.invoke({"messages": [HumanMessage(content=specific_query)]})
        output_text = response["messages"][-1].content
    except Exception as e:
        output_text = f"Error executing {agent_type.value}: {str(e)}"

    return {"agent_outputs": {agent_type.value: output_text}}


# Specific wrappers for LangGraph Node Names
def news_worker_node(state: OrchestratorState) -> Dict[str, Any]:
    return generic_worker_node(state, AgentType.NEWS)


def fundamentals_worker_node(state: OrchestratorState) -> Dict[str, Any]:
    return generic_worker_node(state, AgentType.FUNDAMENTALS)


# -------------------------------------------------------------------
# 6. Supervisor & Aggregator Nodes
# -------------------------------------------------------------------


def supervisor_node(state: OrchestratorState) -> Dict[str, Any]:
    print(f"\n--- [Orchestrator] Decomposing: '{state.raw_input}' ---")

    llm = service_manager.get_agent(temperature=0)
    system_prompt = build_router_system_prompt()

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    router = prompt | llm.with_structured_output(RouteDecision)
    decision: RouteDecision = router.invoke({"input": state.raw_input})

    # Validation Check
    if not decision.is_valid or not decision.tasks:
        return {
            "target_agents": [],
            "final_output": decision.clarification_message
            or "Could not process request.",
        }

    # Transform the List[Task] into the format needed for the State
    # 1. Which agents to trigger?
    target_agents = [task.assigned_agent.value for task in decision.tasks]

    # 2. What specific instruction does each agent get?
    instructions = {
        task.assigned_agent.value: task.specific_instruction for task in decision.tasks
    }

    print(f"--- [Orchestrator] Plan: {instructions} ---")

    return {
        "target_agents": target_agents,
        "agent_instructions": instructions,
        "extracted_ticker": decision.ticker,
        "agent_outputs": {},  # Reset outputs
    }


def aggregator_node(state: OrchestratorState) -> Dict[str, Any]:
    print("--- [Orchestrator] Aggregating Responses ---")

    if not state.agent_outputs:
        return {"final_output": "Error: No data received from agents."}

    # Synthesize
    formatted_data = "\n\n".join(
        [
            f"## Report from {agent_name.upper()}\nQuery: {state.agent_instructions.get(agent_name)}\nResponse: {content}"
            for agent_name, content in state.agent_outputs.items()
        ]
    )

    prompt_text = (
        "You are a Financial Research Lead. You have received reports from your sub-agents.\n"
        "Synthesize these partial reports into a single, cohesive answer to the user's original question.\n\n"
        "**Original User Question:** {original_input}\n\n"
        "**Sub-Agent Reports:**\n{agent_data}\n\n"
        "**Requirements:**\n"
        "- Do not explicitly mention 'The News Agent said X'. Just state the facts.\n"
        "- Connect the qualitative (News) and quantitative (Fundamentals) insights if possible.\n"
        "- Be professional and concise."
    )

    llm = service_manager.get_agent(temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt_text)

    response = (prompt | llm).invoke(
        {"agent_data": formatted_data, "original_input": state.raw_input}
    )

    return {"final_output": response.content}


# -------------------------------------------------------------------
# 7. Graph Construction
# -------------------------------------------------------------------


def build_orchestrator_graph():
    workflow = StateGraph(OrchestratorState)

    # 1. Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node(AgentType.NEWS.value, news_worker_node)
    workflow.add_node(AgentType.FUNDAMENTALS.value, fundamentals_worker_node)
    workflow.add_node("aggregator", aggregator_node)

    # 2. Entry Point
    workflow.add_edge(START, "supervisor")

    # 3. Dynamic Routing
    def route_logic(state: OrchestratorState):
        if not state.target_agents:
            return END
        return state.target_agents

    workflow.add_conditional_edges(
        "supervisor",
        route_logic,
        {
            AgentType.NEWS.value: AgentType.NEWS.value,
            AgentType.FUNDAMENTALS.value: AgentType.FUNDAMENTALS.value,
            END: END,
        },
    )

    # 4. Fan-In to Aggregator
    workflow.add_edge(AgentType.NEWS.value, "aggregator")
    workflow.add_edge(AgentType.FUNDAMENTALS.value, "aggregator")
    workflow.add_edge("aggregator", END)

    return workflow.compile()


# -------------------------------------------------------------------
# 8. Execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    app = build_orchestrator_graph()

    def run_demo(q):
        print(f"\n{'='*50}\nUSER QUERY: {q}\n{'='*50}")
        res = app.invoke({"raw_input": q})
        print(f"\n>> FINAL ANSWER:\n{res.get('final_output')}\n")

    # Example: Complex query requiring split
    run_demo("Why did NVDA drop recently and what is their current net profit margin?")
