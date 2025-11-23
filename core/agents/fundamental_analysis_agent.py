import datetime
import pandas as pd
import operator
from typing import List, Dict, Any, Optional, Literal, Annotated
from pydantic import BaseModel, Field

# LangChain / LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START

# --- LOCAL IMPORTS (Assumed existing) ---
from core.services import service_manager
from core.agents.concept_resolver import ConceptResolver
from core.agents.get_financial_data import FinancialDatabase

# Initialize Resolver globally
RESOLVER = ConceptResolver()


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

    # --- Execution Stack (LIFO) ---
    task_stack: List[str] = Field(default_factory=list)

    # --- Knowledge Graph ---
    resolved_tags: Dict[str, str] = Field(default_factory=dict)
    formulas: Dict[str, List[str]] = Field(default_factory=dict)

    # --- Data ---
    financial_data: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True


# -------------------------------------------------------------------
# 2. WORKER NODES
# -------------------------------------------------------------------


class ScopeParser(BaseModel):
    ticker: str
    start_year: int
    end_year: int
    metrics: List[str]


def parser_node(state: AgentState) -> Dict[str, Any]:
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

    return {
        "ticker": res.ticker.upper(),
        "period_start": res.start_year,
        "period_end": res.end_year,
        "task_stack": res.metrics,
        "financial_data": pd.DataFrame(),
    }


def fetch_data_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetches data. If the tag isn't in state yet, it resolves it here first.
    """
    current_task = state.task_stack[0]

    # 1. Determine Tag (Check cache or resolve fresh)
    tag = state.resolved_tags.get(current_task)
    if not tag:
        tag = RESOLVER.resolve(current_task)

    print(f"--- [Node] Fetcher: '{current_task}' -> {tag} ---")

    # 2. Fetch
    updates = {"task_stack": state.task_stack[1:]}  # Default: pop task

    if tag:
        # Save the resolution for future reference
        updates["resolved_tags"] = {**state.resolved_tags, current_task: tag}

        db = FinancialDatabase()
        db.update_company_data(state.ticker, num_years=5)
        RESOLVER.update_company_concepts(db.get_all_concepts_for_company(state.ticker))

        new_data = db.search_concept(
            state.ticker, [tag], state.period_start, state.period_end
        )

        if not new_data.empty:
            current_df = state.financial_data
            updated_df = (
                new_data
                if (current_df is None or current_df.empty)
                else current_df.combine_first(new_data)
            )
            updates["financial_data"] = updated_df

    return updates


def decomposer_node(state: AgentState) -> Dict[str, Any]:
    current_task = state.task_stack[0]
    print(f"--- [Node] Decomposer: '{current_task}' ---")
    llm = service_manager.get_agent()

    class Ingredients(BaseModel):
        names: List[str]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Break down the metric into 2 standard financial concepts found in 10-K. Return JSON.",
            ),
            ("human", f"Metric: {current_task}"),
        ]
    )

    res = (prompt | llm.with_structured_output(Ingredients)).invoke({})

    # Push ingredients to front of stack
    new_stack = res.names + [current_task] + state.task_stack[1:]

    return {
        "task_stack": new_stack,
        "formulas": {**state.formulas, current_task: res.names},
    }


def calculator_node(state: AgentState) -> Dict[str, Any]:
    target = state.task_stack[0]
    ingredients = state.formulas.get(target, [])
    print(f"--- [Node] Calculator: '{target}' ---")

    df = state.financial_data
    cols = []

    # Identify columns for ingredients
    for ing in ingredients:
        # Check by name OR by tag
        tag = state.resolved_tags.get(ing)
        if ing in df.columns:
            cols.append(ing)
        elif tag and tag in df.columns:
            cols.append(tag)

    if len(cols) < 2:
        # Missing ingredients (logic error or fetch fail), pop to avoid infinite loop
        return {"task_stack": state.task_stack[1:]}

    # Calculate
    def safe_div(row):
        n, d = row.get(cols[0], 0), row.get(cols[1], 0)
        return n / d if d else 0.0

    result = df.apply(safe_div, axis=1).to_frame(name=target)

    return {
        "financial_data": df.combine_first(result),
        "task_stack": state.task_stack[1:],
    }


def analyst_node(state: AgentState) -> Dict[str, Any]:
    print(f"--- [Node] Analyst ---")
    llm = service_manager.get_agent()
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
# 3. CONDITIONAL ROUTING LOGIC
# -------------------------------------------------------------------


def decide_next_step(
    state: AgentState,
) -> Literal["analyst", "calculator", "fetch_data", "decomposer"]:
    """
    Determines the next node based on the state of the stack and data availability.
    This replaces the 'Monitor' node.
    """
    stack = state.task_stack

    # 1. Empty Stack -> Finished
    if not stack:
        return "analyst"

    current = stack[0]

    # 2. Optimization: If data already exists, skip processing?
    # We can handle this by routing to fetcher/calculator and letting them pop immediately,
    # OR check here. checking here is cleaner for the graph flow.
    df = state.financial_data
    tag = state.resolved_tags.get(current)
    if df is not None and not df.empty:
        if current in df.columns or (tag and tag in df.columns):
            # If data exists, we need to pop it.
            # Since edges can't update state, we must route to a node that will pop it.
            # We route to 'fetch_data' which handles "already existing" logic gracefully (by popping).
            return "fetch_data"

    # 3. Formula Logic
    if current in state.formulas:
        return "calculator"

    # 4. Resolution Logic
    # We perform a lookahead check.
    # If it's already a known tag OR resolves successfully, go to Fetcher.
    if current in state.resolved_tags or RESOLVER.resolve(current) is not None:
        return "fetch_data"

    # 5. Fallback -> Decompose
    return "decomposer"


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

    # Connect Every Worker to the Router (Conditional Edge)
    # The parser prepares the initial stack, then we decide where to go.
    # Every processing node (fetch, decompose, calc) loops back via this decision.

    route_config = {
        "analyst": "analyst",
        "calculator": "calculator",
        "fetch_data": "fetch_data",
        "decomposer": "decomposer",
    }

    workflow.add_conditional_edges("parser", decide_next_step, route_config)
    workflow.add_conditional_edges("fetch_data", decide_next_step, route_config)
    workflow.add_conditional_edges("decomposer", decide_next_step, route_config)
    workflow.add_conditional_edges("calculator", decide_next_step, route_config)

    workflow.add_edge("analyst", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()
    user_input = {
        "messages": [
            HumanMessage(
                content="Analyze revenue, earnings and profit for MSFT for last 3 years."
            )
        ]
    }

    print("Starting Agent...")
    final_output = app.invoke(user_input)

    print("\n" + "=" * 40)
    print(final_output["messages"][-1].content)
