import datetime
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# LangChain / LangGraph imports
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- LOCAL IMPORTS ---
from core.services import service_manager
from core.agents.concept_resolver import ConceptResolver
from core.agents.get_financial_data import FinancialDatabase

# Initialize Resolver globally
RESOLVER = ConceptResolver()


# -------------------------------------------------------------------
# 2. STATE DEFINITION
# -------------------------------------------------------------------


class AgentState(BaseModel):
    messages: List[BaseMessage]
    user_query: str
    ticker: Optional[str] = None
    period_start: Optional[int] = None
    period_end: Optional[int] = None

    # --- UNIFIED QUEUE MANAGEMENT ---
    # Single queue for everything (top-level metrics AND ingredients)
    queue: List[str] = Field(default_factory=list)

    # The item currently being processed from the queue
    current_concept: Optional[str] = None

    # --- RESOLUTION STATE ---
    # Stores successful resolutions: {"Concept Name": "XBRL_Tag"}
    resolved_tags: Dict[str, str] = Field(default_factory=dict)

    # Stores calculation logic: {"PE Ratio": ["Price", "EPS"]}
    # Used to know if a concept needs calculation instead of fetching
    formulas: Dict[str, List[str]] = Field(default_factory=dict)

    # --- DATA STATUS ---
    data_status: str = "Empty"
    financial_context: Optional[pd.DataFrame] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True


# -------------------------------------------------------------------
# 3. CALCULATION LOGIC
# -------------------------------------------------------------------


def calculate_ratio_logic(
    current_context: Optional[pd.DataFrame],
    numerator_tag: str,
    denominator_tag: str,
) -> pd.Series:
    """
    Calculates a ratio based on the provided DataFrame context.
    """
    if current_context is None or current_context.empty:
        return "Error: No data context available for calculation.", current_context

    df = current_context.copy()

    def safe_division(row):
        n, d = row[numerator_tag], row[denominator_tag]
        if d == 0 or pd.isna(d) or pd.isna(n):
            return 0.0
        return n / d

    return df.apply(safe_division, axis=0)


# -------------------------------------------------------------------
# 4. NODES
# -------------------------------------------------------------------


class ScopeParser(BaseModel):
    ticker: str
    start_year: int
    end_year: int
    target_metrics: List[str]


def parser_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n--- [Node] Parser: '{state.user_query}' ---")
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

    try:
        res = (prompt | llm.with_structured_output(ScopeParser)).invoke(
            {"query": state.user_query}
        )
        # Initialize the queue with the requested metrics
        return {
            "ticker": res.ticker.upper(),
            "period_start": res.start_year,
            "period_end": res.end_year,
            "queue": res.target_metrics,
            "current_concept": None,
        }
    except Exception as e:
        return {"messages": [BaseMessage(content=f"Error parsing: {e}")]}


def scheduler_node(state: AgentState) -> Dict[str, Any]:
    """
    Pops the next item from the FIFO queue.
    """
    queue = state.queue
    if not queue:
        print("--- [Node] Scheduler: Queue Empty. Done. ---")
        return {"current_concept": None}

    next_concept = queue[0]
    remaining = queue[1:]

    print(
        f"--- [Node] Scheduler: Popped '{next_concept}'. Queue size: {len(remaining)} ---"
    )

    return {"current_concept": next_concept, "queue": remaining}


def resolver_node(state: AgentState) -> Dict[str, Any]:
    """
    Unified resolver. Tries to find a standard XBRL tag.
    If found, we mark it for Fetching.
    If not found, we mark it for Decomposition.
    """
    concept = state.current_concept
    print(f"--- [Node] Resolver: Checking '{concept}' ---")

    # Try to resolve
    tag = RESOLVER.resolve(concept)

    if tag:
        print(f"   > Resolved '{concept}' -> '{tag}'")
        # Store the mapping.
        # Note: We store the Original Name -> Resolved Tag
        return {"resolved_tags": {**state.resolved_tags, concept: tag}}
    else:
        print(f"   > Could not resolve '{concept}' directly.")
        return {}  # No state update, routing will send to decomposer


def decomposer_node(state: AgentState) -> Dict[str, Any]:
    """
    Breaks down a complex metric, adds ingredients to the front of the queue,
    and saves the formula for later calculation.
    """
    concept = state.current_concept
    print(f"--- [Node] Decomposer: Breaking down '{concept}' ---")
    llm = service_manager.get_agent()

    class Ingredients(BaseModel):
        names: List[str] = Field(description="List of 2 raw financial concepts needed.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Break down the requested metric into standard 10-K items (e.g. 'PE Ratio' -> ['Price', 'Earnings Per Share']). Return JSON.",
            ),
            ("human", f"Metric: {concept}"),
        ]
    )

    res = (prompt | llm.with_structured_output(Ingredients)).invoke({})
    ingredients = res.names
    print(f"   > Ingredients: {ingredients}")

    # UPDATE THE QUEUE:
    # 1. We need to process ingredients first.
    # 2. Then we need to come back to 'concept' to calculate it.
    # New Queue = [Ingredient1, Ingredient2, Original_Concept, ...Old_Queue]

    # IMPORTANT: We verify we aren't creating an infinite loop.
    # If the 'concept' is already in formulas, we shouldn't be here, but as a safeguard:
    new_queue_front = ingredients + [concept]
    combined_queue = new_queue_front + state.queue

    # Store the formula: Concept -> [Ing1, Ing2]
    new_formulas = {**state.formulas, concept: ingredients}

    return {"queue": combined_queue, "formulas": new_formulas}


def fetch_data_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetches data for the current_concept using its resolved tag.
    """
    concept = state.current_concept
    tag = state.resolved_tags.get(concept)
    print(f"--- [Node] Fetch Data: '{concept}' (Tag: {tag}) ---")

    if not tag:
        return {"data_status": "Error: No tag found"}

    db = FinancialDatabase()
    # Ensure company data is present
    db.update_company_data(state.ticker, num_years=5)

    # Fetch specific tag
    df = db.search_concept(state.ticker, [tag], state.period_start, state.period_end)

    current_context = state.financial_context
    if not df.empty:
        if current_context is not None and not current_context.empty:
            combined = pd.concat([current_context, df])
            combined = combined.drop_duplicates()
            return {"financial_context": combined, "data_status": "Data Loaded"}
        else:
            return {"financial_context": df, "data_status": "Data Loaded"}

    return {"data_status": "No Data Found"}


def calculator_node(state: AgentState) -> Dict[str, Any]:
    """
    Executed when a concept is in 'formulas' and its ingredients are ready.
    """
    target = state.current_concept
    ingredients = state.formulas.get(target, [])
    print(f"--- [Node] Calculator: Calculating '{target}' using {ingredients} ---")

    # Look up the resolved tags for the ingredients
    # The ingredients (e.g., "Price") should now be in resolved_tags because they were processed earlier in queue
    resolved_ingredient_tags = []
    for ing in ingredients:
        tag = state.resolved_tags.get(ing)
        if not tag:
            print(f"   > Error: Ingredient '{ing}' was not resolved/fetched.")
            return {
                "messages": [
                    BaseMessage(
                        content=f"Failed to calculate {target}: missing data for {ing}"
                    )
                ]
            }
        resolved_ingredient_tags.append(tag)

    if len(resolved_ingredient_tags) < 2:
        return {
            "messages": [
                BaseMessage(
                    content=f"Calculator requires 2 ingredients, found {len(resolved_ingredient_tags)}"
                )
            ]
        }

    calculated_data = calculate_ratio_logic(
        state.financial_context,
        resolved_ingredient_tags[0],  # Numerator
        resolved_ingredient_tags[1],  # Denominator
    )

    calculated_data = calculated_data.to_frame().T
    calculated_data.index = ["P/E Ratio"]

    updated_context = pd.concat([state.financial_context, calculated_data])

    # We treat the calculated metric as "Resolved" now, so we don't try to decompose it again if it appears
    # (though strictly it shouldn't appear again based on queue logic)

    return {
        "messages": [BaseMessage(content="calculated successfully", type="ai")],
        "financial_context": updated_context,
    }


def analyst_node(state: AgentState) -> Dict[str, Any]:
    print(f"--- [Node] Analyst ---")
    llm = service_manager.get_agent()
    context_df = state.financial_context

    data_str = (
        context_df.to_string()
        if context_df is not None and not context_df.empty
        else "No Data"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial analyst. Answer the user's query using the data below.",
            ),
            ("human", f"Query: {state.user_query}\n\nData:\n{data_str}"),
        ]
    )

    res = (prompt | llm).invoke({})
    return {"messages": [res]}


# -------------------------------------------------------------------
# 5. ROUTING LOGIC
# -------------------------------------------------------------------


def route_scheduler(state: AgentState) -> Literal["resolver", "calculator", "analyst"]:
    """
    Decides what to do with the item popped from the queue.
    """
    concept = state.current_concept

    # 1. If queue was empty, concept is None -> Go to Analyst
    if concept is None:
        return "analyst"

    # 2. Check if this concept is a known 'formula' (meaning it was decomposed earlier)
    # If it is, that means we put it back in the queue to wait for ingredients.
    # Now it's back, so we calculate.
    if concept in state.formulas:
        return "calculator"

    # 3. Otherwise, it's a raw concept (either user input or a decomposed ingredient)
    # We need to resolve it.
    return "resolver"


def route_resolver(state: AgentState) -> Literal["fetch_data", "decomposer"]:
    """
    After resolver runs, did we find a tag?
    """
    concept = state.current_concept

    # Check if the current concept exists in the resolved map
    if concept in state.resolved_tags:
        return "fetch_data"

    # If not found, we must decompose it
    return "decomposer"


# -------------------------------------------------------------------
# 6. GRAPH CONSTRUCTION
# -------------------------------------------------------------------


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("parser", parser_node)
    workflow.add_node("scheduler", scheduler_node)

    # Single resolver node
    workflow.add_node("resolver", resolver_node)

    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("analyst", analyst_node)

    # Entry
    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "scheduler")

    # Scheduler Routing
    workflow.add_conditional_edges(
        "scheduler",
        route_scheduler,
        {"resolver": "resolver", "calculator": "calculator", "analyst": "analyst"},
    )

    # Resolver Routing
    workflow.add_conditional_edges(
        "resolver",
        route_resolver,
        {"fetch_data": "fetch_data", "decomposer": "decomposer"},
    )

    # Decomposer sends back to scheduler (updated queue)
    workflow.add_edge("decomposer", "scheduler")

    # Fetch Data sends back to scheduler (get next item)
    workflow.add_edge("fetch_data", "scheduler")

    # Calculator sends back to scheduler (get next item)
    workflow.add_edge("calculator", "scheduler")

    workflow.add_edge("analyst", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()
    prompt = "Calculate pe ratio for CRWD for last 3 years."
    print(f"Starting: '{prompt}'")

    initial = AgentState(messages=[], user_query=prompt)
    out = app.invoke(initial)

    print("\n" + "=" * 40)
    print("FINAL REPORT")
    print("=" * 40)
    print(out["messages"][-1].content)
