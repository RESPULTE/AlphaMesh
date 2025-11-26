import datetime
import operator
import re
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd

# --- LOCAL IMPORTS (Assumed existing) ---
from core.services import service_manager

# LangChain / LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

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

    metric_to_process: Annotated[list[str], operator.add] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)

    # --- Data ---
    financial_data: pd.DataFrame = Field(default_factory=pd.DataFrame)

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

    metrics_to_process = []
    formulas = []
    for metric in res.metrics:
        resolved_metric = db.resolve_concept(res.ticker.upper(), metric)
        print(f"[Fetcher] Resolving metric '{metric}' -> '{resolved_metric}'")

        if resolved_metric is None:
            print(f"[Fetcher] Could not resolve concept for '{metric}'")
            formulas.append(metric)
            continue
        metrics_to_process.append(resolved_metric)

    return {
        "ticker": res.ticker.upper(),
        "period_start": res.start_year,
        "period_end": res.end_year,
        "metric_to_process": metrics_to_process,
        "formulas": formulas,
    }


# ! update this to have metric_to_process if needed
# ? assumes that the orchestrator gives out a structured response here
class BatchResponse(BaseModel):
    formulas: List[str]


def decomposer_node(state: AgentState) -> Dict[str, Any]:
    # Extract the list of metrics needing decomposition
    if len(state.formulas) == 0:
        return {}

    print(f"--- [Decomposer] Batch processing: {state.formulas} ---")

    # Single API Call
    db = service_manager.get_financial_database()

    # ! update this to have metric_to_process if needed
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial decomposition assistant. "
                "For each requested metric, provide ONLY an explicit mathematical formula using exclusively the concepts listed below:\n"
                f"{db.get_all_concepts_for_company(state.ticker)}\n"
                "Rules:\n"
                "1. The formula must strictly follow this style:\n"
                "   metric = (concept_1) + (concept_2) - (concept_3) * (concept_4) / (concept_5)\n"
                "2. Use brackets '()' around each constituent concept.\n"
                "3. Use only +, -, *, / operators as appropriate.\n"
                "4. Do NOT include explanations, text, or extra metadata.\n"
                "5. Output exactly one formula per metric, nothing else.",
            ),
            ("human", "Metrics to decompose: {metrics}"),
        ]
    )
    llm = service_manager.get_agent(temperature=0)

    res: BatchResponse = (prompt | llm.with_structured_output(BatchResponse)).invoke(
        {"metrics": ", ".join(state.formulas)}
    )

    updated_metrics_to_process = state.metric_to_process.copy()
    updated_formulas = []

    for formula in res.formulas:
        updated_metrics_to_process.extend(re.findall(r"\((.*?)\)", formula))

        lhs, rhs = formula.split("=")
        new_lhs = lhs.strip().replace(" ", "_")
        new_rhs = rhs.replace("(", "").replace(")", "")
        updated_formulas.append(f"{new_lhs} = {new_rhs}")
        print(f"{new_lhs} = {new_rhs}")

    return {
        "formulas": updated_formulas,
        "metric_to_process": updated_metrics_to_process,
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
        exact=True,
    )

    if new_data is None or new_data.empty:
        print(
            f"[Fetcher] No data found for the company'{state.ticker}' (metrics to process: '{state.metric_to_process}')"
        )

    print(f"[Fetcher] Fetched data for {len(new_data)} metrics.")
    print(new_data)

    return {
        "financial_data": (
            state.financial_data.combine_first(new_data)
            if not state.financial_data.empty
            else new_data
        ),
        "metric_to_process": [],
    }


def calculator_node(state: AgentState) -> AgentState:
    db = service_manager.get_financial_database()

    work_df = state.financial_data.T
    work_df.columns = [concept for _, concept in work_df.columns]
    for f in state.formulas:
        try:
            # lhs, rhs = f.split("=")
            work_df.eval(f, inplace=True)
            # state.financial_data.loc[(state.ticker, lhs)] = retval
        except Exception as e:
            print(f"could not process the formula: {f}\n, due to {e}")

    work_df.columns = pd.MultiIndex.from_tuples(
        [(state.ticker, concept) for concept in work_df.columns],
        names=state.financial_data.index.names,
    )
    state.financial_data = work_df.T

    db.save_calculated_metric(state.ticker, state.financial_data)

    return {"financial_data": state.financial_data, "formulas": []}


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
        lambda state: "calculator" if state.formulas else "analyst",
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
                content="Analyze revenue, earnings and free cash flow, and debt to asset ratio for META for last 3 years."
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
