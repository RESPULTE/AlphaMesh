import datetime
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
from pydantic import BaseModel, Field

# LangChain / LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# --- MOCK IMPORTS (Replace these with your actual project imports) ---
# In a real scenario, these would come from your local files.
from core.services import service_manager
from core.agents import fundamental_metric_helper
from core.agents.get_financial_data import FinancialDatabase


# -------------------------------------------------------------------

# --- 1. State Definition ---


class AgentState(BaseModel):
    """
    The state object passed between nodes in the LangGraph.
    """

    messages: List[BaseMessage]
    user_query: str
    ticker: Optional[str] = None
    period_start: Optional[int] = None
    period_end: Optional[int] = None
    concepts: Optional[List[str]] = None
    financial_data_json: Optional[str] = (
        None  # Store DF as JSON string to pass between nodes
    )
    error: Optional[str] = None


# --- 2. Helper Classes & Tools ---


class ScopeParser(BaseModel):
    """Structured output for the parser node."""

    ticker: str = Field(description="The stock ticker symbol.")
    start_year: int = Field(description="The start year of the analysis.")
    end_year: int = Field(description="The end year of the analysis.")
    concepts: List[str] = Field(
        description="List of relevant financial concept names for the query."
    )


# Global variable to hold the DataFrame temporarily for tools to access
# This avoids passing the massive DataFrame into the LLM Context Window.
CURRENT_CONTEXT_DATA: Optional[pd.DataFrame] = None


@tool
def calculate_pe_ratio_tool():
    """Calculates the Price-to-Earnings (P/E) Ratio based on loaded data."""
    if CURRENT_CONTEXT_DATA is None:
        return "Error: No financial data loaded."
    try:
        # In reality, you call: fundamental_metric_helper.calculate_pe_ratio(CURRENT_CONTEXT_DATA)
        result = fundamental_metric_helper.calculate_pe_ratio(CURRENT_CONTEXT_DATA)
        return f"The P/E Ratio is {result}"
    except Exception as e:
        return f"Error calculating PE: {str(e)}"


@tool
def calculate_cagr_tool(metric_name: str = "revenue"):
    """Calculates the Compound Annual Growth Rate (CAGR) for a specific metric (default: revenue)."""
    if CURRENT_CONTEXT_DATA is None:
        return "Error: No financial data loaded."
    try:
        result = fundamental_metric_helper.calculate_cagr(
            CURRENT_CONTEXT_DATA, metric_name
        )
        return f"The CAGR for {metric_name} is {result:.2%}"
    except Exception as e:
        return f"Error calculating CAGR: {str(e)}"


@tool
def calculate_debt_to_equity_tool():
    """Calculates the Debt-to-Equity ratio."""
    if CURRENT_CONTEXT_DATA is None:
        return "Error: No financial data loaded."
    try:
        result = fundamental_metric_helper.calculate_debt_to_equity(
            CURRENT_CONTEXT_DATA
        )
        return f"The Debt-to-Equity Ratio is {result}"
    except Exception as e:
        return f"Error calculating Debt/Equity: {str(e)}"


# List of available tools
analysis_tools = [
    calculate_pe_ratio_tool,
    calculate_cagr_tool,
    calculate_debt_to_equity_tool,
]

# --- 3. Nodes ---


def parse_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Parse User Input.
    Extracts Company Ticker and Period. Defaults to last 5 years if unspecified.
    """
    print(f"--- [Node] Parsing Input: {state.user_query} ---")

    llm = service_manager.get_agent()

    current_year = datetime.datetime.now().year
    default_start = current_year - 5

    parser_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"The current year is {current_year}."
                f"Extract the Company Ticker, Year Range and the relevant financial metric from the user query. "
                f"If the company is named (e.g. 'Apple'), convert it to Ticker (e.g. 'AAPL'). "
                f"If no period is specified, default to {default_start} to {current_year}. "
                "return JSON",
            ),
            ("human", "{query}"),
        ]
    )

    # Use structured output for reliability
    structured_llm = llm.with_structured_output(ScopeParser)
    chain = parser_prompt | structured_llm

    try:
        result = chain.invoke({"query": state.user_query})
        return {
            "ticker": result.ticker,
            "period_start": result.start_year,
            "period_end": result.end_year,
            "concepts": result.concepts,
        }
    except Exception as e:
        return {"error": f"Failed to parse input: {str(e)}"}


def data_manager_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Check DB & Fetch Data.
    Ensures data is available locally, then loads it into the global context.
    """
    if state.error:
        return {}  # Skip if previous error

    ticker = state.ticker
    start = state.period_start
    end = state.period_end
    concepts = state.concepts

    print(f"--- [Node] Checking Data for {ticker} ({start}-{end}) ---")

    db = FinancialDatabase()

    try:

        # 2. Validation: If empty or missing years, fetch from API
        # (Simplified logic: if None, fetch)
        if not set(range(start, end + 1)).issubset(db.get_existing_years(ticker)):
            print("   > Data missing locally. Fetching from external source...")
            db.update_company_data(ticker, 2025 - min(start, end) + 1)

        df = db.search_concept(ticker, concepts, start_year=start, end_year=end)

        if df is None or df.empty:
            return {"error": f"Could not retrieve data for {ticker}."}

        # 3. Set Global Context for Tools
        # We do not put the DF in the state 'messages' to save tokens.
        # We put it in a global variable that Tools can access.
        global CURRENT_CONTEXT_DATA
        CURRENT_CONTEXT_DATA = df

        # We store a lightweight reference or summary in state if needed
        return {"financial_data_json": str(df.columns.tolist())}

    except Exception as e:
        return {"error": f"Database Error: {str(e)}"}


def analysis_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Agent Reasoning.
    Decides which tool to call based on the user query and available tools.
    """
    if state.error:
        return {"messages": [AIMessage(content=state.error)]}

    print("--- [Node] Analyst Agent Reasoning ---")

    llm = service_manager.get_agent()
    llm_with_tools = llm.bind_tools(analysis_tools)

    # Construct context for the agent
    sys_msg = SystemMessage(
        content=f"You are a Fundamental Analysis Agent. "
        f"You are analyzing {state.ticker} from {state.period_start} to {state.period_end}. "
        f"Financial data is already loaded in the system context. "
        f"Select the appropriate tool to answer the user's question. "
        f"If the calculation is done, summarize the results."
    )

    # Get the history (messages)
    # If this is the first pass, add the system message and user query
    messages = state.messages
    if not messages:
        messages = [sys_msg, HumanMessage(content=state.user_query)]
    else:
        # Ensure system message is present if strictly needed,
        # usually LangGraph handles history, but we prepend context here.
        messages = [sys_msg] + messages

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Conditional Edge: Checks if the Agent made a tool call or a final response.
    """
    messages = state.messages
    last_message = messages[-1]

    if state.error:
        return "__end__"

    if last_message.tool_calls:
        return "tools"
    return "__end__"


# --- 4. Graph Construction ---


def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("parser", parse_node)
    workflow.add_node("data_manager", data_manager_node)
    workflow.add_node("analyst", analysis_agent_node)
    workflow.add_node("tools", ToolNode(analysis_tools))

    # Define Edges
    workflow.set_entry_point("parser")

    # Parser -> Data Manager
    workflow.add_edge("parser", "data_manager")

    # Data Manager -> Analyst
    workflow.add_edge("data_manager", "analyst")

    # Analyst -> Conditional (Tools or End)
    workflow.add_conditional_edges(
        "analyst",
        should_continue,
    )

    # Tools -> Analyst (Loop back with result)
    workflow.add_edge("tools", "analyst")

    return workflow.compile()


# --- 5. Execution Entry Point ---


def run_fundamental_analysis(prompt: str):
    app = build_graph()

    print(f"Initializing Agent with prompt: '{prompt}'")

    initial_state = AgentState(messages=[], user_query=prompt)

    # Run the graph
    final_state = None
    for output in app.stream(initial_state):
        for key, value in output.items():
            # Visualize the flow
            print(f"Finished Node: {key}")

            print(f"State Update: {value}\n")
            pass

    # Extract Final Response
    # The final state is usually accessible via invoke, but with stream we grab the last yield
    # Ideally, we re-invoke to get the final state object simply:
    final_state_dict = app.invoke(initial_state)

    last_msg = final_state_dict["messages"][-1]
    print("\n" + "=" * 30)
    print("FINAL ANALYSIS REPORT")
    print("=" * 30)
    print(last_msg.content)


if __name__ == "__main__":
    # Example usage
    user_prompt = "Analyze the pe ratio for the company apple over the last 3 years."
    run_fundamental_analysis(user_prompt)
