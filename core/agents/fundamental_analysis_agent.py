import inspect
import datetime
from typing import Dict, Any
import pandas as pd

# LangChain / LangGraph Imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from core.services import service_manager
from core.agents import fundamental_metric_helper
from core.agents.get_financial_data import FinancialDatabase

# --- 1. Context Management ---


class AnalysisContext:
    """
    A Singleton-like class to hold the financial DataFrame in memory
    during the agent's execution lifecycle.

    This prevents passing massive JSON/DataFrames through the LLM context window.
    """

    _data: Dict[str, pd.DataFrame] = {}
    _active_ticker: str = None

    @classmethod
    def set_data(cls, ticker: str, df: pd.DataFrame):
        cls._data[ticker] = df
        cls._active_ticker = ticker

    @classmethod
    def get_active_data(cls) -> pd.DataFrame:
        if not cls._active_ticker or cls._active_ticker not in cls._data:
            raise ValueError(
                "No financial data loaded. Please call 'fetch_financial_data' first."
            )
        return cls._data[cls._active_ticker]

    @classmethod
    def clear(cls):
        cls._data = {}
        cls._active_ticker = None


# --- 2. Tool Definitions ---


@tool
def fetch_financial_data(
    ticker: str, start_year: int = None, end_year: int = None
) -> str:
    """
    Fetches financial data for a specific company and period from the local database.
    If data is missing, it attempts to download it.

    Args:
        ticker: The stock symbol (e.g., 'AAPL', 'TSLA').
        start_year: The start year (integer). Defaults to 5 years ago if not provided.
        end_year: The end year (integer). Defaults to current year if not provided.
    """
    current_year = datetime.datetime.now().year

    # Default logic for Period
    if end_year is None:
        end_year = current_year
    if start_year is None:
        start_year = end_year - 5

    print(f"🛠️ Tool Triggered: Fetching data for {ticker} ({start_year}-{end_year})...")

    try:
        db = FinancialDatabase()

        # Check if data exists locally (logic assumed to be in FinancialDatabase)
        # If not, query it. This method is assumed to return a Pandas DataFrame.
        # We assume the method signature matches this logic.
        df = db.get_data(ticker, list(range(start_year, end_year + 1)))

        if df is None or df.empty:
            return f"Error: No data found for {ticker} between {start_year} and {end_year}."

        # Store in Context for calculation tools to access
        AnalysisContext.set_data(ticker, df)

        columns_preview = ", ".join(df.columns[:5])
        return (
            f"Successfully loaded data for {ticker} ({start_year}-{end_year}). "
            f"Rows: {len(df)}. Available columns include: {columns_preview}..."
        )

    except Exception as e:
        return f"Database Error: {str(e)}"


def _create_dynamic_calculation_tools():
    """
    Dynamically inspects `fundamental_metric_helper` and creates LangChain tools
    for every calculation function found there.
    """
    tools = []

    # Inspect the helper module
    functions_list = [
        o
        for o in inspect.getmembers(fundamental_metric_helper)
        if inspect.isfunction(o[1])
    ]

    for name, func in functions_list:
        # We create a wrapper tool for each function
        # We assume the helper functions take a DataFrame as the first argument 'df'
        # or that we can inject the context data.

        def make_tool_func(f, func_name):
            @tool(func_name)
            def dynamic_tool(**kwargs) -> str:
                """
                Executes a specific financial metric calculation.
                Requires 'fetch_financial_data' to be called first.
                """
                try:
                    # Retrieve data from context
                    df = AnalysisContext.get_active_data()

                    # Execute the helper function, injecting the dataframe
                    # We assume the helper functions signature is like: func(df, **kwargs)
                    result = f(df, **kwargs)

                    return f"Result for {func_name}: {result}"
                except ValueError as ve:
                    return str(
                        ve
                    )  # Return the error message to the agent so it knows to fetch data
                except Exception as e:
                    return f"Calculation Error: {str(e)}"

            return dynamic_tool

        # Create the tool with the specific name and docstring from the helper
        wrapped_tool = make_tool_func(func, name)
        wrapped_tool.description = (
            func.__doc__ or f"Calculates {name}."
        ) + " Requires data to be loaded first."
        tools.append(wrapped_tool)

    return tools


# --- 3. Agent Construction ---


class FundamentalAnalysisAgent:
    def __init__(self):
        # 1. Load Preconfigured LLM
        self.llm = service_manager.get_llm()

        # 2. Prepare Tools
        self.fetch_tool = fetch_financial_data
        self.calc_tools = _create_dynamic_calculation_tools()
        self.tools = [self.fetch_tool] + self.calc_tools

        # 3. Define System Prompt
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = f"""
        You are an expert Fundamental Analysis Agent. Today is {current_date}.
        
        Your workflow is strictly:
        1. **Parse** the user's request to identify the Company (Ticker) and Period.
           - If the user implies "last 5 years", calculate the years based on today's date.
        2. **Fetch** the data using `fetch_financial_data`. 
           - YOU MUST DO THIS BEFORE ANY CALCULATION.
        3. **Select** the appropriate calculation tool based on the user's metric request (e.g., 'calculate_pe_ratio', 'calculate_cagr').
           - The data is stored in a shared context, you just need to call the calculation tool.
        4. **Answer** the user's question with the numeric result and a brief insight.
        
        If the tool returns an error saying "No financial data loaded", you must call `fetch_financial_data` immediately.
        """

        # 4. Create the Graph (ReAct Agent)
        # We use the prebuilt create_react_agent which handles the loop: LLM -> Tool -> LLM
        self.graph = create_agent(
            self.llm, self.tools, system_prompt=self.system_prompt
        )

    def analyze(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point for the agent.
        """
        # Clear previous context to ensure clean state
        AnalysisContext.clear()

        print(f"\n--- Agent Starting: {user_input} ---")

        inputs = {"messages": [HumanMessage(content=user_input)]}

        try:
            # Stream the steps to see internal reasoning (optional, for debugging)
            final_state = None
            for chunk in self.graph.stream(inputs, stream_mode="values"):
                final_state = chunk

            # Extract final message
            last_message = final_state["messages"][-1]
            return {
                "status": "success",
                "response": last_message.content,
                "history": final_state["messages"],
            }

        except Exception as e:
            return {
                "status": "error",
                "response": f"An error occurred during analysis: {str(e)}",
            }


# --- 4. Execution Example ---

if __name__ == "__main__":
    # Instantiate the agent
    agent = FundamentalAnalysisAgent()

    # Example Query 1
    query = "What is the CAGR for Revenue for Nvidia over the last 5 years?"
    result = agent.analyze(query)

    print("\n>>> FINAL RESPONSE:")
    print(result["response"])

    # Example Query 2 (Specific years)
    # query_2 = "Calculate the Debt-to-Equity ratio for Apple for 2020 to 2022."
    # result_2 = agent.analyze(query_2)
    # print("\n>>> FINAL RESPONSE 2:")
    # print(result_2["response"])
