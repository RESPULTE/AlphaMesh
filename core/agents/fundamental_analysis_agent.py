import datetime
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# LangChain / LangGraph imports
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- LOCAL IMPORTS ---
from core.services import service_manager
from core.agents.get_financial_data import FinancialDatabase
from core.agents.concept_resolver import ConceptResolver

# Initialize Resolver globally
RESOLVER = ConceptResolver()

# -------------------------------------------------------------------
# 1. DATABASE EXTENSION
# -------------------------------------------------------------------


class ExtendedFinancialDatabase(FinancialDatabase):
    """Extends DB to handle saving calculated metrics safely."""

    def save_calculated_metric(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df["statement_type"] = "calculated"
        required = ["company", "year", "statement_type", "concept", "value"]

        if not all(c in df.columns for c in required):
            print(f"   [DB Error] Missing cols: {df.columns}")
            return

        final_df = df[required]
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                for _, row in final_df.iterrows():
                    # Delete existing calculation to allow overwrite
                    cursor.execute(
                        "DELETE FROM financials WHERE company=? AND year=? AND statement_type='calculated' AND concept=?",
                        (row["company"], row["year"], row["concept"]),
                    )
                conn.commit()
                final_df.to_sql("financials", conn, if_exists="append", index=False)
                print(
                    f"   [DB] Saved {len(final_df)} rows for '{final_df['concept'].iloc[0]}'."
                )
            except Exception as e:
                print(f"   [DB] Error saving: {e}")


# -------------------------------------------------------------------
# 2. STATE DEFINITION
# -------------------------------------------------------------------


class AgentState(BaseModel):
    messages: List[BaseMessage]
    user_query: str
    ticker: Optional[str] = None
    period_start: Optional[int] = None
    period_end: Optional[int] = None

    # --- QUEUE MANAGEMENT ---
    metrics_queue: List[str] = Field(default_factory=list)
    current_metric: Optional[str] = None

    # --- EPHEMERAL FLAGS ---
    direct_resolved_tag: Optional[str] = None
    is_complex_calculation: bool = False
    ingredient_names: List[str] = Field(default_factory=list)
    resolved_ingredients: Dict[str, str] = Field(default_factory=dict)

    # --- GLOBAL DATA STATUS ---
    data_status: str = "Empty"


GLOBAL_DATAFRAME_CONTEXT: Optional[pd.DataFrame] = None

# -------------------------------------------------------------------
# 3. TOOLS
# -------------------------------------------------------------------


@tool
def calculate_and_store_ratio(
    ticker: str, metric_name: str, numerator_tag: str, denominator_tag: str
):
    """
    Robust ratio calculator:
      - Reads GLOBAL_DATAFRAME_CONTEXT (which must be normalized to MultiIndex
        ['company','concept','year'] with a 'value' column).
      - Extracts series for numerator_tag and denominator_tag for the requested ticker,
        aligns years, computes ratio (numerator / denominator).
      - Saves results to DB and appends calculated rows to GLOBAL_DATAFRAME_CONTEXT.
    Returns a short status string for logging.
    """
    global GLOBAL_DATAFRAME_CONTEXT
    if GLOBAL_DATAFRAME_CONTEXT is None or GLOBAL_DATAFRAME_CONTEXT.empty:
        return "Error: No data context."

    # Ensure the expected index and column exist
    df = GLOBAL_DATAFRAME_CONTEXT

    # Accept either the normalized MultiIndex or a "flat" table and normalize on the fly
    def normalize_context(d: pd.DataFrame) -> pd.DataFrame:
        # If there are columns 'company','concept','year','value', convert to multiindex
        cols = set(d.columns)
        if {"company", "concept", "year", "value"}.issubset(cols):
            tmp = d.copy()
            tmp["year"] = tmp["year"].astype(int)
            tmp = tmp.set_index(["company", "concept", "year"])
            # keep single 'value' column
            return tmp[["value"]]
        # If already has multiindex with names that include 'company'/'concept'/'year'
        if isinstance(d.index, pd.MultiIndex):
            names = list(d.index.names)
            if all(x in names for x in ("company", "concept", "year")):
                # ensure value column exists
                if "value" in d.columns:
                    return d[["value"]]
        # otherwise try to coerce
        raise ValueError("GLOBAL_DATAFRAME_CONTEXT not in expected format")

    try:
        norm = normalize_context(df)
    except Exception as e:
        return f"Context normalization error: {e}"

    # helper to get a Series indexed by year for (ticker, tag)
    def get_series_for(tag: str) -> Optional[pd.Series]:
        try:
            sel = norm.loc[(ticker, tag)]
            # sel is a DataFrame with index = year, column 'value'
            if isinstance(sel, pd.DataFrame):
                s = sel["value"]
            elif isinstance(sel, pd.Series):
                # case when selecting yields Series (unlikely) -> try to coerce
                s = sel
            else:
                return None
            # ensure index are ints (years)
            s.index = s.index.astype(int)
            s = s.sort_index()
            return s
        except KeyError:
            return None

    num_s = get_series_for(numerator_tag)
    den_s = get_series_for(denominator_tag)

    if num_s is None:
        return f"Missing data for numerator tag '{numerator_tag}'"
    if den_s is None:
        return f"Missing data for denominator tag '{denominator_tag}'"

    # Align on years (outer join would produce NaNs; we only compute where both present)
    joined = pd.concat([num_s, den_s], axis=1, keys=["num", "den"])
    joined = joined.dropna(subset=["num", "den"])
    if joined.empty:
        return "No overlapping years with data to compute ratio."

    try:
        ratio = joined["num"] / joined["den"]
    except Exception as e:
        return f"Math error while computing ratio: {e}"

    # Prepare rows to save
    db_rows = []
    for year, val in ratio.items():
        # year is index (int)
        if pd.notna(val):
            db_rows.append(
                {
                    "company": ticker,
                    "year": int(year),
                    "concept": metric_name,
                    "value": float(val),
                }
            )

    if not db_rows:
        return "No valid numeric ratio values to save."

    # Save to DB
    ExtendedFinancialDatabase().save_calculated_metric(pd.DataFrame(db_rows))

    # Append into the global context so analyst sees it
    try:
        calc_df = pd.DataFrame(db_rows)  # columns company, year, concept, value
        calc_df = calc_df.set_index(["company", "concept", "year"])[["value"]]
        # Avoid duplicate index rows; prefer calculated overwrite
        if GLOBAL_DATAFRAME_CONTEXT is None or GLOBAL_DATAFRAME_CONTEXT.empty:
            GLOBAL_DATAFRAME_CONTEXT = calc_df
        else:
            # combine, with calculated rows overwriting existing same (company,concept,year)
            combined = pd.concat(
                [
                    GLOBAL_DATAFRAME_CONTEXT[
                        ~GLOBAL_DATAFRAME_CONTEXT.index.isin(calc_df.index)
                    ],
                    calc_df,
                ]
            )
            GLOBAL_DATAFRAME_CONTEXT = combined.sort_index()
    except Exception as e:
        # Non-fatal: DB saved but global append failed
        return f"Saved to DB but failed to append to context: {e}"

    return f"Calculated '{metric_name}' for {len(db_rows)} year(s) and saved to DB/context."


# -------------------------------------------------------------------
# 4. NODES
# -------------------------------------------------------------------


class ScopeParser(BaseModel):
    ticker: str
    start_year: int
    end_year: int
    target_metrics: List[str] = Field(
        description="List of financial metrics requested."
    )


def parser_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n--- [Node] Parser: '{state.user_query}' ---")
    llm = service_manager.get_agent()
    current_year = datetime.datetime.now().year

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Current year: {current_year}. Extract Ticker, Year Range, and LIST of Metrics. "
                "Default to last 5 years. Return JSON.",
            ),
            ("human", "{query}"),
        ]
    )

    try:
        res = (prompt | llm.with_structured_output(ScopeParser)).invoke(
            {"query": state.user_query}
        )
        return {
            "ticker": res.ticker.upper(),
            "period_start": res.start_year,
            "period_end": res.end_year,
            "metrics_queue": res.target_metrics,
            "current_metric": None,
        }
    except Exception as e:
        return {"messages": [BaseMessage(content=f"Error parsing: {e}")]}


def scheduler_node(state: AgentState) -> Dict[str, Any]:
    queue = state.metrics_queue
    if not queue:
        print("--- [Node] Scheduler: Queue Empty. Done. ---")
        return {"current_metric": None}

    next_metric = queue[0]
    remaining = queue[1:]
    print(
        f"--- [Node] Scheduler: Starting '{next_metric}' (Remaining: {len(remaining)}) ---"
    )

    return {
        "current_metric": next_metric,
        "metrics_queue": remaining,
        "direct_resolved_tag": None,
        "is_complex_calculation": False,
        "ingredient_names": [],
        "resolved_ingredients": {},
    }


def resolve_target_node(state: AgentState) -> Dict[str, Any]:
    metric = state.current_metric
    print(f"--- [Node] Resolve Target: '{metric}' ---")
    tag = RESOLVER.resolve(metric)

    if tag:
        print(f"   > Direct match found: {tag}")
        return {"direct_resolved_tag": tag, "is_complex_calculation": False}
    else:
        print(f"   > No direct match. Needs decomposition.")
        return {"direct_resolved_tag": None, "is_complex_calculation": True}


def decomposer_node(state: AgentState) -> Dict[str, Any]:
    metric = state.current_metric
    print(f"--- [Node] Decomposer: Breaking down '{metric}' ---")
    llm = service_manager.get_agent()

    class Ingredients(BaseModel):
        names: List[str] = Field(description="List of 2-3 raw financial concepts.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Break down the requested metric into standard 10-K items (e.g. 'PE Ratio' -> ['Price', 'EPS']). Return JSON.",
            ),
            ("human", f"Metric: {metric}"),
        ]
    )

    res = (prompt | llm.with_structured_output(Ingredients)).invoke({})
    print(f"   > Ingredients: {res.names}")
    return {"ingredient_names": res.names}


def resolve_ingredients_node(state: AgentState) -> Dict[str, Any]:
    print(f"--- [Node] Resolve Ingredients ---")
    ingredients = state.ingredient_names
    resolved_map = {}

    for name in ingredients:
        tag = RESOLVER.resolve(name)
        if tag:
            resolved_map[name] = tag
        else:
            print(f"   > Warning: Could not resolve ingredient '{name}'")

    return {"resolved_ingredients": resolved_map}


def fetch_data_node(state: AgentState) -> Dict[str, Any]:
    print(f"--- [Node] Fetch Data ---")
    db = ExtendedFinancialDatabase()
    db.update_company_data(state.ticker, num_years=5)

    tags_to_fetch = state.direct_resolved_tag
    if state.resolved_ingredients:
        tags_to_fetch = list(state.resolved_ingredients.values())

    if not tags_to_fetch:
        return {"data_status": "Error: No tags to fetch"}

    df = db.search_concept(
        state.ticker, tags_to_fetch, state.period_start, state.period_end
    )

    global GLOBAL_DATAFRAME_CONTEXT
    if not df.empty:
        if GLOBAL_DATAFRAME_CONTEXT is not None:
            GLOBAL_DATAFRAME_CONTEXT = pd.concat([GLOBAL_DATAFRAME_CONTEXT, df])
            GLOBAL_DATAFRAME_CONTEXT = GLOBAL_DATAFRAME_CONTEXT[
                ~GLOBAL_DATAFRAME_CONTEXT.index.duplicated(keep="first")
            ]
        else:
            GLOBAL_DATAFRAME_CONTEXT = df
        return {"data_status": "Data Loaded"}

    return {"data_status": "No Data Found"}


def calculator_setup_node(state: AgentState) -> Dict[str, Any]:
    """
    Prepare and call the actual calculator function. We pick numerator/denominator
    heuristically from resolved_ingredients (order preserved). If there are >2
    ingredients, we pick the first two. If only two ingredients are present the
    first is numerator, second denominator.
    This function *calls* calculate_and_store_ratio directly and returns a message.
    """
    print(f"--- [Node] Calculator Setup ---")
    # Defensive checks
    metric = state.current_metric
    resolved_map = state.resolved_ingredients or {}

    if not resolved_map:
        return {
            "messages": [
                BaseMessage(content="Calculator: no resolved ingredients available.")
            ]
        }
    # order-preserving extraction (dict may be ordered)
    tags = list(resolved_map.values())
    if len(tags) < 2:
        return {
            "messages": [
                BaseMessage(
                    content="Calculator: need at least 2 resolved tags to form a ratio."
                )
            ]
        }
    numerator_tag = tags[0]
    denominator_tag = tags[1]

    # call calculator tool
    try:
        result = calculate_and_store_ratio(
            state.ticker, metric, numerator_tag, denominator_tag
        )
    except Exception as e:
        result = f"Calculator raised exception: {e}"

    return {"messages": str(result)}


def analyst_node(state: AgentState) -> Dict[str, Any]:
    print(f"--- [Node] Analyst ---")
    llm = service_manager.get_agent()
    global GLOBAL_DATAFRAME_CONTEXT

    data_str = (
        GLOBAL_DATAFRAME_CONTEXT.to_string()
        if GLOBAL_DATAFRAME_CONTEXT is not None
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


def route_scheduler(state: AgentState) -> Literal["resolve_target", "analyst"]:
    if state.current_metric:
        return "resolve_target"
    return "analyst"


def route_target_resolution(state: AgentState) -> Literal["fetch_data", "decomposer"]:
    if state.direct_resolved_tag:
        return "fetch_data"
    return "decomposer"


def route_after_fetch(state: AgentState) -> Literal["scheduler", "calculator_setup"]:
    if state.is_complex_calculation:
        return "calculator_setup"
    return "scheduler"


# -------------------------------------------------------------------
# 6. GRAPH CONSTRUCTION
# -------------------------------------------------------------------


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("parser", parser_node)
    workflow.add_node("scheduler", scheduler_node)
    workflow.add_node("resolve_target", resolve_target_node)
    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("resolve_ingredients", resolve_ingredients_node)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("calculator_setup", calculator_setup_node)
    workflow.add_node("tools", ToolNode([calculate_and_store_ratio]))
    workflow.add_node("analyst", analyst_node)

    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "scheduler")

    workflow.add_conditional_edges(
        "scheduler",
        route_scheduler,
        {"resolve_target": "resolve_target", "analyst": "analyst"},
    )
    workflow.add_conditional_edges(
        "resolve_target",
        route_target_resolution,
        {"fetch_data": "fetch_data", "decomposer": "decomposer"},
    )

    workflow.add_edge("decomposer", "resolve_ingredients")
    workflow.add_edge("resolve_ingredients", "fetch_data")

    workflow.add_conditional_edges(
        "fetch_data",
        route_after_fetch,
        {"scheduler": "scheduler", "calculator_setup": "calculator_setup"},
    )

    workflow.add_edge("calculator_setup", "tools")
    workflow.add_edge("tools", "scheduler")
    workflow.add_edge("analyst", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()
    prompt = "Calculate P/E ratio and revenue for CRWD for last 3 years."
    print(f"Starting: '{prompt}'")
    initial = AgentState(messages=[], user_query=prompt)
    out = app.invoke(initial)
    print("\n" + "=" * 40)
    print("FINAL REPORT")
    print("=" * 40)
    print(out["messages"][-1].content)
