import datetime
import operator
from typing import Annotated, Any, List, Optional, Type

import pandas as pd
from core.agents.base_agent import AbstractAgent, AgentOutput
from core.services import service_manager
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

# --- 1. Internal Structured Output Models (The "Solution 2" Fix) ---


class CalculatedMetric(BaseModel):
    """
    Represents a single metric to be calculated.
    Using this strict schema prevents parsing errors (e.g., splitting strings by '=').
    """

    target_metric_name: str = Field(
        description="The name of the new metric to calculate (e.g., 'net_profit_margin'). Use snake_case."
    )
    pandas_eval_expression: str = Field(
        description="The mathematical formula compatible with pandas.eval(). Example: 'price / revenue' (Right Hand Side of the formula only)"
    )
    dependencies: List[str] = Field(
        description="A list of the base financial concepts required for this formula. Example: ['net_income', 'revenue']"
    )


class DecompositionPlan(BaseModel):
    """The structured output expected from the Decomposer LLM."""

    calculations: List[CalculatedMetric]


class _AgentState(BaseModel):
    """Internal state for the agent workflow."""

    messages: Annotated[List[BaseMessage], operator.add]

    # Inputs
    ticker: str
    metrics: List[str]
    start_year: int
    end_year: int

    # Processing
    metrics_to_fetch: Annotated[List[str], operator.add] = Field(default_factory=list)
    calculations_to_run: List[CalculatedMetric] = Field(default_factory=list)
    financial_data: pd.DataFrame = Field(default_factory=pd.DataFrame)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- 2. Public Input Schema (Used by Orchestrator) ---


class FundamentalAnalysisInput(BaseModel):
    """Input schema for the Fundamental Analysis Agent."""

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL).")
    metrics: List[str] = Field(description="List of financial metrics to analyze.")
    start_year: Optional[int] = Field(default=None, description="Start year (integer).")
    end_year: Optional[int] = Field(default=None, description="End year (integer).")
    raw_input: str = Field(description="The original user query.")


# --- 3. The Agent Class ---


class FundamentalAnalysisAgent(AbstractAgent):
    """
    Refactored Agent using Structured Outputs for internal logic.
    """

    def __init__(self):
        super().__init__()
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "fundamentals_agent"

    @property
    def description(self) -> str:
        return (
            "Focuses on quantitative data: financial statements, margins, and ratios. "
            "Returns strict financial data analysis."
        )

    @classmethod
    def get_input_schema_class(cls) -> Type[BaseModel]:
        return FundamentalAnalysisInput

    def run(self, input_data: FundamentalAnalysisInput) -> AgentOutput:
        print(f"--- [Agent: {self.name}] Started for {input_data.ticker} ---")

        # Defaults
        current_year = datetime.datetime.now().year
        s_year = input_data.start_year if input_data.start_year else (current_year - 5)
        e_year = input_data.end_year if input_data.end_year else current_year

        initial_state = {
            "messages": [HumanMessage(content=input_data.raw_input)],
            "ticker": input_data.ticker.upper(),
            "metrics": input_data.metrics,
            "start_year": s_year,
            "end_year": e_year,
            "metrics_to_fetch": [],
            "calculations_to_run": [],
            "financial_data": pd.DataFrame(),
        }

        final_state = self._graph.invoke(initial_state)
        output_content = final_state["messages"][-1].content

        return AgentOutput(agent_name=self.name, output=output_content)

    def _build_graph(self):
        workflow = StateGraph(_AgentState)

        workflow.add_node("parser", self._parser_node)
        workflow.add_node("decomposer", self._decomposer_node)
        workflow.add_node("fetch_data", self._fetch_data_node)
        workflow.add_node("calculator", self._calculator_node)
        workflow.add_node("analyst", self._analyst_node)

        workflow.add_edge(START, "parser")

        # Conditional: If parser found unknown metrics, go to decomposer. Else fetch data.
        workflow.add_conditional_edges(
            "parser",
            lambda state: "decomposer" if state.calculations_to_run else "fetch_data",
            {"decomposer": "decomposer", "fetch_data": "fetch_data"},
        )

        workflow.add_edge("decomposer", "fetch_data")
        workflow.add_edge("fetch_data", "calculator")
        workflow.add_edge("calculator", "analyst")
        workflow.add_edge("analyst", END)

        return workflow.compile()

    # --- Node Implementations ---

    def _parser_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Separates metrics into 'fetchable' (exist in DB) and 'complex' (need formulas).
        """
        print(f"--- [Node] Parser ---")
        db = service_manager.get_financial_database()

        to_fetch = []
        unknown_metrics = []  # These will become dummy calculations initially

        for metric in state.metrics:
            resolved = db.resolve_concept(state.ticker, metric)
            if resolved:
                to_fetch.append(resolved)
            else:
                # We flag this as a 'calculation' needed, but we don't know the formula yet.
                # We pass the name to the decomposer.
                unknown_metrics.append(metric)

        # We temporarily store unknown metrics in 'calculations_to_run' as placeholders
        # The decomposer will replace these placeholders with real formulas.
        placeholders = [
            CalculatedMetric(
                target_metric_name=m, pandas_eval_expression="", dependencies=[]
            )
            for m in unknown_metrics
        ]

        return {"metrics_to_fetch": to_fetch, "calculations_to_run": placeholders}

    def _decomposer_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Uses LLM with Structured Output to break down complex metrics into formulas.
        """
        # Identify which metrics need formulas (those with empty expressions)
        targets = [
            c.target_metric_name
            for c in state.calculations_to_run
            if not c.pandas_eval_expression
        ]

        if not targets:
            return {}

        print(f"--- [Node] Decomposer: Deriving formulas for {targets} ---")
        db = service_manager.get_financial_database()
        available_concepts = db.get_all_concepts_for_company(state.ticker)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial quant. Decompose the requested metrics into mathematical formulas.",
                ),
                (
                    "human",
                    f"Available Database Concepts: {available_concepts}\n\n"
                    f"Metrics to decompose: {targets}\n\n"
                    "Provide a calculation plan using ONLY the available concepts.",
                ),
            ]
        )

        # Enforce the strict schema via tool calling/structured output
        llm = service_manager.get_agent(temperature=0)
        structured_llm = llm.with_structured_output(DecompositionPlan)

        chain = prompt | structured_llm
        result: DecompositionPlan = chain.invoke({})

        # Extract new dependencies to fetch
        new_dependencies = []
        for calc in result.calculations:
            new_dependencies.extend(calc.dependencies)

        print(f"[Decomposer] Plan: {result.calculations}")

        return {
            "calculations_to_run": result.calculations,  # Replaces the placeholders
            "metrics_to_fetch": new_dependencies,  # Add these to the fetch list
        }

    def _fetch_data_node(self, state: _AgentState) -> dict[str, Any]:
        print(f"--- [Node] Fetcher ---")
        # Unique metrics only
        metrics = list(set(state.metrics_to_fetch))

        if not metrics:
            return {}

        db = service_manager.get_financial_database()
        db.update_company_data(
            state.ticker, num_years=state.end_year - state.start_year + 1
        )

        df = db.get_concept(
            state.ticker, metrics, state.start_year, state.end_year, exact=True
        )

        # Merge with existing if any
        current_df = state.financial_data
        combined_df = current_df.combine_first(df) if not current_df.empty else df

        return {"financial_data": combined_df}

    def _calculator_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Executes the formulas using pandas eval.
        """
        print(f"--- [Node] Calculator ---")
        df = state.financial_data.copy()

        if df.empty or not state.calculations_to_run:
            return {}

        # 1. Flatten MultiIndex for eval (Ticker, Metric) -> Metric
        # We assume the dataframe columns are MultiIndex (Ticker, Metric)
        # We simplify to just 'Metric' for the evaluation context
        df_eval = df.copy()
        try:
            df_eval.index = [c[1] for c in df.index]  # Keep only metric name
        except IndexError:
            pass  # Handle cases where it might not be MultiIndex

        # 2. Transpose so years are rows, metrics are columns (standard for eval)
        df_eval = df_eval.T

        for calc in state.calculations_to_run:
            try:
                print(
                    f"[Calculator] {calc.target_metric_name} = {calc.pandas_eval_expression}"
                )
                # Execute formula
                df_eval.eval(
                    f"{calc.target_metric_name} = {calc.pandas_eval_expression}",
                    inplace=True,
                )
            except Exception as e:
                print(f"[Calculator] Error calculating {calc.target_metric_name}: {e}")

        # 3. Transpose back and restore structure
        # We only care about the NEW calculated columns
        # Filter df_eval for columns that match our calculation targets
        calculated_cols = [
            c.target_metric_name
            for c in state.calculations_to_run
            if c.target_metric_name in df_eval.columns
        ]

        if not calculated_cols:
            return {}

        result_df = df_eval[calculated_cols].T

        # Restore MultiIndex (Ticker, Metric)
        result_df.index = pd.MultiIndex.from_tuples(
            [(state.ticker, col) for col in result_df.index],
            names=["company", "concept"],
        )

        final_df = state.financial_data.combine_first(result_df)

        print(final_df)
        db = service_manager.get_financial_database()
        db.save_calculated_metric(state.ticker, result_df)

        return {"financial_data": final_df}

    def _analyst_node(self, state: _AgentState) -> dict[str, Any]:
        print(f"--- [Node] Analyst ---")
        llm = service_manager.get_agent(temperature=0.7)

        data_str = (
            state.financial_data.to_string()
            if not state.financial_data.empty
            else "No data."
        )

        msg = (
            f"Analyze the following financial data for {state.ticker}.\n"
            f"Data:\n{data_str}\n\n"
            f"User Question: {state.messages[0].content}\n"
            "Highlight key trends, risks, and positives."
        )

        response = llm.invoke(msg)
        return {"messages": [response]}
