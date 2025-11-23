import datetime
import pandas as pd
import operator
from typing import Any, List, Dict, Optional, Annotated, Set
from pydantic import BaseModel, ConfigDict, Field

# LangChain / LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --- LOCAL IMPORTS (Assumed existing) ---
from core.services import service_manager
from core.agents.concept_resolver import COMMON_FINANCIAL_CONCEPTS

# Initialize Resolver globally


# -------------------------------------------------------------------
# 1. STATE DEFINITIONS
# -------------------------------------------------------------------


class InputState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]


class OutputState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]


class AgentState(InputState, OutputState):
    # --- Scope ---
    ticker: Optional[str] = None
    period_start: Optional[int] = None
    period_end: Optional[int] = None

    metric_to_process: Set[str] = Field(default_factory=list)
    composite_metrics: Dict[str, Set[str]] = Field(default_factory=dict)

    # --- Data ---
    financial_data: Optional[pd.DataFrame] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# -------------------------------------------------------------------
# 2. WORKER NODES
# -------------------------------------------------------------------


class ScopeParser(BaseModel):
    ticker: str
    start_year: int
    end_year: int
    metrics: List[str]


def parser_node(state: InputState) -> AgentState:
    print(f"\n--- [Node] Parser ---")
    llm = service_manager.get_agent()
    current_year = datetime.datetime.now().year

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Current year: {current_year}. Extract Ticker, Year Range, and LIST of Metrics. Default to last 5 years. Return JSON.",
            ),
            ("human", "{query}"),
        ]
    )

    # Safely handle input messages
    query = state.messages[-1].content if state.messages else ""
    res = (prompt | llm.with_structured_output(ScopeParser)).invoke({"query": query})

    print(
        f"[Parser] Extracted - Ticker: {res.ticker}, Years: {res.start_year}-{res.end_year}, Metrics: {res.metrics}"
    )

    db = service_manager.get_financial_database()
    db.update_company_data(
        res.ticker.upper(), num_years=min(res.start_year, res.end_year, 5)
    )

    metrics_to_process = set()
    composite_metrics = dict()
    for metric in res.metrics:
        resolved_metric = db.resolve_concept(res.ticker.upper(), metric)
        print(f"[Fetcher] Resolving metric '{metric}' -> '{resolved_metric}'")

        if resolved_metric is None:
            print(f"[Fetcher] Could not resolve concept for '{metric}'")
            composite_metrics[metric] = set()
            continue
        metrics_to_process.add(resolved_metric)

    return {
        "ticker": res.ticker.upper(),
        "period_start": res.start_year,
        "period_end": res.end_year,
        "metric_to_process": metrics_to_process,
        "composite_metrics": composite_metrics,
        "financial_data": pd.DataFrame(),
    }


def fetch_data_node(state: AgentState) -> AgentState:
    """
    Fetches data. If the tag isn't in state yet, it resolves it here first.
    """
    print(f"\n--- [Node] Fetcher ---")
    print(f"Fetching data for ticker: {state.metric_to_process}")

    db = service_manager.get_financial_database()
    new_data = db.get_concept(
        state.ticker,
        tuple(state.metric_to_process),
        state.period_start,
        state.period_end,
    )

    if new_data is None or new_data.empty:
        print(
            f"[Fetcher] No data found for the company'{state.ticker}' (metrics to process: '{state.metric_to_process}')"
        )

    print(f"[Fetcher] Fetched data for {len(new_data)} metrics.")
    print(new_data)

    return {"financial_data": state.financial_data.combine_first(new_data)}


def decomposer_node(state: AgentState) -> Dict[str, Any]:
    # Extract the list of metrics needing decomposition
    targets = state.composite_metrics.keys()
    if not targets:
        return {}

    llm = service_manager.get_agent(temperature=0)
    print(f"--- [Decomposer] Batch processing: {targets} ---")

    # Define Schema for Batch Processing
    class Recipe(BaseModel):
        metric: str
        ingredients: Set[str]

    class BatchResponse(BaseModel):
        breakdowns: List[Recipe]

    # Single API Call
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial decomposition assistant. "
                "Break each metric into its constituent components using only these concepts:\n"
                f"{COMMON_FINANCIAL_CONCEPTS}\n"
                "Return a JSON list of objects with 'metric' and 'ingredients'. Obey instructions exactly.",
            ),
            ("human", "Metrics to decompose: {metrics}"),
        ]
    )

    res = (prompt | llm.with_structured_output(BatchResponse)).invoke(
        {"metrics": ", ".join(targets)}
    )

    updated_composite_metrics = state.composite_metrics.copy()
    updated_metrics_to_process = state.metric_to_process.copy()
    db = service_manager.get_financial_database()
    for item in res.breakdowns:
        if not item.ingredients:
            continue

        print(f"   > {item.metric} -> {item.ingredients}")

        ingredients = set()
        for i in item.ingredients:
            resolved_ingredient = db.resolve_concept(state.ticker, i)
            if resolved_ingredient is None:
                print(
                    f"[Decomposer] Could not resolve concept for ingredient '{i}' of composite metric '{item.metric}'"
                )
                continue
            ingredients.add(resolved_ingredient)
            print(f"[Decomposer] Resolved ingredient '{i}' to '{resolved_ingredient}'")

        updated_composite_metrics[item.metric] = ingredients
        updated_metrics_to_process.update(ingredients)

    return {
        "composite_metrics": updated_composite_metrics,
        "metric_to_process": updated_metrics_to_process,
    }


def calculator_node(state: AgentState) -> AgentState:
    print(f"\n--- [Node] Calculator (Row-Based) ---")

    # 1. Load Data
    df = state.financial_data.copy()
    targets = list(state.composite_metrics.keys())

    # Basic validation
    if df is None or df.empty:
        print("[Calculator] No financial data available.")
        return {}

    if not targets:
        print("[Calculator] No composite metrics requested.")
        return {}

    llm = service_manager.get_agent()

    # 2. Define the Schema
    #    We instruct the agent that it is working with ROWS, not columns.
    class FinancialFormula(BaseModel):
        metric_name: str = Field(
            description="The name of the new metric to be created (e.g., 'NetProfitMargin')"
        )
        expression: str = Field(
            description=(
                "A valid Python/Pandas expression to calculate this metric using Row-based operations. "
                "The DataFrame 'df' has Metrics as the Index and Years as Columns. "
                "You MUST use `df.loc['metric_name']` to access data. "
                "Example: `df.loc['us-gaap_NetIncome'] / df.loc['us-gaap_Revenues']`"
            )
        )
        reasoning: str = Field(
            description="Why this formula is mathematically correct for the financial concept."
        )

    class CalculationPlan(BaseModel):
        calculations: List[FinancialFormula]

    # 3. Prepare Context for the Agent
    #    We provide the exact list of available row indices (the us-gaap tags)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Financial Data Engineer. You are working with a Time-Series DataFrame `df`.\n"
                    "**Structure:**\n"
                    "- Index (Rows): Financial Concepts (e.g., 'us-gaap_Assets')\n"
                    "- Columns: Years (e.g., '2022', '2023')\n\n"
                    "**Your Goal:** Generate Python expressions to calculate new financial metrics based on the user's request.\n"
                    "**Rules:**\n"
                    "1. You MUST access data using `df.loc['Exact_Index_Name']`.\n"
                    "2. Operations should be vectorized (e.g., Row A / Row B).\n"
                    "3. Do not use column access like `df['Name']`."
                ),
            ),
            (
                "human",
                (
                    f"Available Row Indices (Ingredients): {df.index.tolist()}\n\n"
                    f"Requested Metrics to Calculate: {targets}\n\n"
                    "Provide the Pandas formulas to compute these metrics as new rows."
                ),
            ),
        ]
    )

    # 4. Invoke the Agent
    try:
        print(f"[Calculator] Planning formulas for: {targets}")
        chain = prompt | llm.with_structured_output(CalculationPlan)
        result = chain.invoke({})
    except Exception as e:
        print(f"[Calculator] Planning failed: {e}")
        return {}

    # 5. Execute the Formulas safely
    calculated_count = 0

    # Prepare the execution scope
    # We make 'df' available, along with pandas for advanced logic if they need it
    local_scope = {"df": df}
    global_scope = {"pd": pd}

    for calc in result.calculations:
        print(f"   > Generating {calc.metric_name}...")
        print(f"     Expression: {calc.expression}")

        try:
            # Evaluate the string expression generated by the LLM
            # expected result is a pd.Series (Year 2022, Year 2023...)
            computed_series = eval(calc.expression, global_scope, local_scope)

            # Validate result shape (it should match the number of columns/years)
            if isinstance(computed_series, (pd.Series, pd.DataFrame)):
                # Add the new row to the DataFrame
                # We use .loc to insert the new metric as a new row
                df.loc[calc.metric_name] = computed_series
                calculated_count += 1
            else:
                print(
                    f"     [Warning] Result was not a Series/DataFrame. Got {type(computed_series)}."
                )

        except KeyError as e:
            print(f"     [Error] Agent used a tag that doesn't exist in the index: {e}")
        except Exception as e:
            print(f"     [Error] Calculation syntax error: {e}")

    print(f"[Calculator] Finished. Added {calculated_count} new rows.")

    db = service_manager.get_financial_database()
    db.save_calculated_metric(state.ticker, df)

    # Return updated state
    return {"financial_data": df}


def analyst_node(state: AgentState) -> OutputState:
    print(f"--- [Node] Analyst ---")
    llm = service_manager.get_agent(temperature=1.0)
    data_str = (
        state.financial_data.to_string()
        if state.financial_data is not None
        else "No Data"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a financial analyst. Answer based on the data."),
            ("human", f"Query: {state.messages[-1].content}\n\nData:\n{data_str}"),
        ]
    )

    return {"messages": [(prompt | llm).invoke({})]}


# -------------------------------------------------------------------
# 4. GRAPH CONSTRUCTION
# -------------------------------------------------------------------


def build_graph():
    workflow = StateGraph(
        AgentState, input_schema=InputState, output_schema=OutputState
    )

    # Add Nodes
    workflow.add_node("parser", parser_node)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("analyst", analyst_node)

    # Entry Point
    workflow.add_edge(START, "parser")
    workflow.add_edge("parser", "decomposer")
    workflow.add_edge("decomposer", "fetch_data")
    workflow.add_conditional_edges(
        "fetch_data",
        lambda state: "calculator" if state.composite_metrics else "analyst",
        {"calculator": "calculator", "analyst": "analyst"},
    )
    workflow.add_edge("calculator", "analyst")
    workflow.add_edge("analyst", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()
    user_input = {
        "messages": [
            HumanMessage(
                content="Analyze revenue, earnings and free cash flow for MSFT for last 3 years."
            )
        ]
    }

    print("Starting Agent...")
    final_output = app.invoke(user_input)

    print("\n" + "=" * 40)
    print(final_output["messages"][-1].content)

    # from PIL import Image as PILImage
    # import io

    # png_data = app.get_graph().draw_mermaid_png()

    # # Load into PIL
    # img = PILImage.open(io.BytesIO(png_data))

    # # Open in new window
    # img.show()
