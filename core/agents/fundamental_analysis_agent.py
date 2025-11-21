import inspect
import json
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from datetime import datetime

from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)

# Project imports
from core.services import ServiceManager
from core.agents import fundamental_metric_helper
from core.agents.get_financial_data import FinancialDatabase

DEFAULT_PERIOD_YEARS = 5


# --- 1. Shared Context for Data Management ---
class AnalysisContext:
    """
    Singleton-like class to hold the financial data (DataFrame)
    so it can be shared between the Fetch tool and Calculation tools
    without passing raw data through the LLM context window.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.ticker: str = ""

    def set_data(self, df: pd.DataFrame, ticker: str):
        self.df = df
        self.ticker = ticker

    def get_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError(
                "Data has not been fetched yet. Please call 'get_financial_columns' first."
            )
        return self.df


# Initialize global context
context = AnalysisContext()


# --- 2. Initial Parsing (Scope Definition) ---
def parse_user_scope(user_prompt: str, agent) -> Tuple[str, Tuple[int, int]]:
    """
    Extracts just the Company and Period to define the scope of the DB query.
    The specific *columns* are left for the agent to decide later.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Extract only the Company Ticker and the Year Period from the user prompt.
        If no period is specified, use the last 5 years.
        
        User prompt: {user_prompt}
        
        Respond in JSON format: {{'ticker': '...', 'period': '...'}}
        Example Period formats: '2020-2024' or '5' (for last 5 years).
        """
    )
    chain = prompt | agent
    result: AIMessage = chain.invoke({"user_prompt": user_prompt})

    try:
        # Clean up potential markdown formatting from LLM
        content = result.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)

        company = parsed.get("ticker")
        period = parsed.get("period")

        if not period:
            end_year = datetime.now().year
            start_year = end_year - DEFAULT_PERIOD_YEARS + 1
        else:
            if isinstance(period, str) and "-" in period:
                start_year, end_year = map(int, period.split("-"))
            elif isinstance(period, (list, tuple)):
                start_year, end_year = int(period[0]), int(period[1])
            else:
                # Handle single number as "last X years" or specific year
                val = int(period)
                if val < 100:  # Assume it's a duration (e.g., 5 years)
                    end_year = datetime.now().year
                    start_year = end_year - val + 1
                else:
                    start_year = end_year = val

        return company, (start_year, end_year)

    except Exception as e:
        # Fallback defaults if parsing fails
        print(f"Parsing warning: {e}. Using defaults.")
        return "AAPL", (2020, 2024)


# --- 3. Tool Definitions ---


def create_calculation_tools() -> List[Tool]:
    """
    Dynamically scans fundamental_metric_helper and creates tools that accept COLUMN NAMES.
    The tool implementation looks up the actual data from the global 'context'.
    """
    tools = []
    functions = inspect.getmembers(fundamental_metric_helper, inspect.isfunction)

    for name, func in functions:
        if name.startswith("_"):
            continue

        sig = inspect.signature(func)
        params = sig.parameters

        # We create a wrapper that takes strings (column names) instead of Series
        def make_wrapper(original_func, param_names):
            def wrapper(**kwargs):
                # 1. Get Data from Context
                try:
                    df = context.get_data()
                except ValueError as e:
                    return str(e)

                # 2. Prepare arguments
                func_args = {}
                for p in param_names:
                    if p == "df":
                        func_args["df"] = df
                        continue

                    # The LLM provides a string (column name) or a raw value
                    arg_value = kwargs.get(p)

                    if isinstance(arg_value, str) and arg_value in df.columns:
                        # Pass the actual Series data
                        func_args[p] = df[arg_value]
                    else:
                        # Pass the raw value (e.g., risk_free_rate=0.02)
                        func_args[p] = arg_value

                # 3. Execute
                try:
                    result = original_func(**func_args)
                    if isinstance(result, pd.Series):
                        return result.to_dict()  # Return as dict for LLM readability
                    return result
                except Exception as e:
                    return f"Calculation Error in {original_func.__name__}: {e}"

            return wrapper

        # Define tool arguments based on function signature (excluding 'df')
        # We want the LLM to know it should pass column names strings.
        wrapper_func = make_wrapper(func, list(params.keys()))

        param_desc = ", ".join([f"{p} (column name)" for p in params if p != "df"])
        desc = (
            f"Calculates {name}. "
            f"Arguments required: {param_desc}. "
            "Pass the NAME of the column (string) containing the data, not the data itself."
        )

        tools.append(Tool(name=name, func=wrapper_func, description=desc))

    return tools


# --- 4. Main Agent Workflow ---


def run_analysis(user_prompt: str) -> Dict[str, Any]:
    service_manager = ServiceManager()
    llm = service_manager.get_llm()

    # 1. Parse Scope (Ticker/Year)
    company, (start_year, end_year) = parse_user_scope(user_prompt, llm)

    # 2. Prepare Tools
    # Tool to fetch data
    db = FinancialDatabase()

    fetch_tool = Tool(
        name="get_financial_columns",
        func=lambda columns: db.search_concept(company, start_year, end_year, columns),
        description="Fetches financial data. Input should be a list of column names required for the analysis (e.g. ['Revenue', 'EPS']). Call this BEFORE any calculation.",
    )

    # Tools to calculate metrics
    calc_tools = create_calculation_tools()
    all_tools = [fetch_tool] + calc_tools

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(all_tools)

    # 3. Agent Loop (ReAct Pattern)
    print(f"\n--- Starting Analysis for {company} ({start_year}-{end_year}) ---")

    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                f"You are a financial analysis assistant. You are analyzing {company} from {start_year} to {end_year}. "
                "You have a two-step process:\n"
                "1. IDENTIFY which data columns are needed for the user's request and use 'get_financial_columns' to load them.\n"
                "2. Once data is loaded, select the appropriate calculation tool to compute the result.\n"
                "Always verify data is loaded before calculating."
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    final_result = None

    # Allow up to 5 turns (Reason -> Fetch -> Reason -> Calculate -> Result)
    for i in range(5):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # Agent is done and providing the final answer
            final_result = ai_msg.content
            break

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(
                f"Step {i+1}: Agent calling tool '{tool_name}' with args: {tool_args}"
            )

            # Find and Execute Tool
            selected_tool = next((t for t in all_tools if t.name == tool_name), None)

            if selected_tool:
                try:
                    # Special handling for the fetch tool to ensure list parsing
                    if tool_name == "get_financial_columns" and "columns" in tool_args:
                        if isinstance(tool_args["columns"], str):
                            # Sometimes LLM sends "['A','B']" as string
                            tool_args["columns"] = eval(tool_args["columns"])

                    tool_output = selected_tool.func(**tool_args)
                except Exception as e:
                    tool_output = f"Error executing {tool_name}: {e}"
            else:
                tool_output = f"Tool {tool_name} not found."

            print(f"   > Tool Output: {str(tool_output)[:100]}...")  # Truncate log

            # Append tool result to history
            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            )

    return {
        "company": company,
        "period": (start_year, end_year),
        "result": final_result,
    }


if __name__ == "__main__":
    # Example 1: Simple Metric
    query = "Calculate the PE Ratio for Apple"
    # Example 2: Metric requiring multiple columns
    # query = "What is the Debt to Equity ratio for Tesla?"

    try:
        output = run_analysis(query)
        print("\n--- Final Analysis Result ---")
        print(output["result"])
    except Exception as e:
        print(f"Execution failed: {e}")
