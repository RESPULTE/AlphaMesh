import asyncio
import difflib
import operator
from datetime import datetime
from typing import Annotated, List, Optional, Type

import pandas as pd
from core.agents.base_agent import AbstractAgent, AgentOutput
from core.services import service_manager  # Assuming this still handles LLM retrieval

# Import the new database class
from get_financial_data import FinancialDatabase
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- 1. Internal Structured Output Models ---


class CalculatedMetric(BaseModel):
    """
    Represents a single metric to be calculated.
    """

    target_metric_name: str = Field(
        description="The name of the new metric to calculate, use camel case _ instead of spaces."
    )
    pandas_eval_expression: str = Field(
        description="The mathematical formula compatible with pandas.eval(). Example: 'A / B' (Right Hand Side only)"
    )
    dependencies: List[str] = Field(
        description="A list of the base financial concepts required for this formula."
    )


class DecompositionPlan(BaseModel):
    """The structured output expected from the Decomposer LLM."""

    calculations: List[CalculatedMetric]


class FundamentalAnalysisOutput(BaseModel):
    """The structured output expected from the Fundamental Analysis Agent."""

    detailed_analysis: str
    data: pd.DataFrame = Field(default_factory=pd.DataFrame)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class _AgentState(BaseModel):
    """Internal state for the agent workflow."""

    # Inputs
    query: str
    ticker: str
    metrics: List[str]
    start_date: datetime
    end_date: datetime

    # Processing
    metrics_to_fetch: Annotated[List[str], operator.add] = Field(default_factory=list)
    calculations_to_run: List[CalculatedMetric] = Field(default_factory=list)

    # Financial Data: Index=Label, Columns=Date
    financial_data: pd.DataFrame = Field(default_factory=pd.DataFrame)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    output: Optional[str] = ""


# --- 2. Public Input Schema ---


class FundamentalAnalysisInput(BaseModel):
    """Input schema for the Fundamental Analysis Agent."""

    query: str = Field(description="The original user query for context.")

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL).")
    metrics: List[str] = Field(description="List of financial metrics to analyze.")

    start_date: Optional[datetime] = Field(
        default=None, description="Start date (str, format: YYYY-MM-DD)."
    )
    end_date: Optional[datetime] = Field(
        default=None, description="End date (str, format: YYYY-MM-DD)."
    )

    @field_validator("start_date", "end_date", mode="before")
    def parse_dates(cls, v):
        if isinstance(v, datetime) or v is None:
            return v
        return datetime.strptime(v, "%Y-%m-%d")


# --- 3. The Agent Class ---


class FundamentalAnalysisAgent(AbstractAgent):
    """
    Refactored Async Agent using Structured Outputs and new FinancialDatabase.
    """

    def __init__(self):
        super().__init__()
        self._graph = self._build_graph()
        self.db = FinancialDatabase()  # Instantiate the DB class

    @property
    def name(self) -> str:
        return "fundamentals_agent"

    @property
    def description(self) -> str:
        return (
            "Focuses on quantitative data: financial statements, margins, and ratios. "
            "Returns strict financial data analysis asynchronously."
        )

    @classmethod
    def get_input_schema_class(cls) -> Type[BaseModel]:
        return FundamentalAnalysisInput

    @classmethod
    def get_output_schema_class(cls) -> Type[BaseModel]:
        return FundamentalAnalysisOutput

    async def run(self, input_data: FundamentalAnalysisInput) -> AgentOutput:
        """Async entry point for the agent."""
        print(f"--- [Agent: {self.name}] Started for {input_data.ticker} ---")

        # Ensure DB is initialized
        await self.db.initialize()

        retval: FundamentalAnalysisOutput = await self._graph.ainvoke(input_data)

        return AgentOutput(agent_name=self.name, output=retval.detailed_analysis)

    def _build_graph(self):
        workflow = StateGraph(
            _AgentState,
            output_schema=self.get_output_schema_class(),
        )

        workflow.add_node("parser", self._parser_node)
        workflow.add_node("decomposer", self._decomposer_node)
        workflow.add_node("fetch_data", self._fetch_data_node)
        workflow.add_node("calculator", self._calculator_node)
        workflow.add_node("analyst", self._analyst_node)

        workflow.add_edge(START, "parser")

        # Conditional: If parser found unknown metrics, go to decomposer. Else fetch data directly.
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

    async def _parser_node(self, state: _AgentState) -> _AgentState:
        """
        Ensures data exists, then checks which metrics are raw DB columns vs calculated.
        """
        print(f"--- [Node] Parser ---")

        # 1. We must ensure data exists in the DB to perform label lookups
        # In a real app, you might want to check if data exists before updating to save time,
        # but here we update to ensure freshness/availability.
        sd = state.start_date
        ed = state.end_date

        await self.db.update_financials(state.ticker, sd.year, ed.year)

        to_fetch = []
        unknown_metrics = []

        search_df = await self.db.search_label(state.ticker, state.metrics)
        if search_df.empty:
            placeholders = [
                CalculatedMetric(
                    target_metric_name=m, pandas_eval_expression="", dependencies=[]
                )
                for m in state.metrics
            ]

            return {
                "calculations_to_run": placeholders,
                "metrics": state.metrics,
                "metrics_to_fetch": to_fetch,
                "start_date": sd,
                "end_date": ed,
                "query": state.query,
                "ticker": state.ticker,
            }

        available_labels = set(search_df.index)

        for metric in state.metrics:
            close_matches = difflib.get_close_matches(
                metric, available_labels, n=1, cutoff=0.8
            )

            if close_matches:
                found_label = close_matches[0]
                print(f"Resolving '{metric}' to fuzzy match '{found_label}'")
                to_fetch.append(found_label)
            else:
                unknown_metrics.append(metric.replace(" ", "_"))

        placeholders = [
            CalculatedMetric(
                target_metric_name=m, pandas_eval_expression="", dependencies=[]
            )
            for m in unknown_metrics
        ]

        return {
            "calculations_to_run": placeholders,
            "metrics": state.metrics,
            "metrics_to_fetch": to_fetch,
            "start_date": sd,
            "end_date": ed,
            "query": state.query,
            "ticker": state.ticker,
        }

    async def _decomposer_node(self, state: _AgentState) -> _AgentState:
        """
        Uses LLM to decompose complex metrics into formulas based on available data.
        """
        targets = [
            c.target_metric_name
            for c in state.calculations_to_run
            if not c.pandas_eval_expression
        ]

        if not targets:
            return {}

        print(f"--- [Node] Decomposer: Deriving formulas for {targets} ---")

        # Retrieve all available labels to give the LLM context
        available_concepts = await self.db.get_labels(state.ticker)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial quant. Decompose the requested metrics into mathematical formulas.\n"
                    "1. Use precise labels from the 'Available Concepts' list.\n"
                    "2. DO NOT use any labels other than the provided ones.\n"
                    "3. CRITICAL: In the 'pandas_eval_expression' AND 'target_metric_name', replace all spaces in the variable names with underscores '_'.\n"
                    "   Example: If available concept is 'Gross Profit', use 'Gross_Profit' in the formula.\n",
                ),
                (
                    "human",
                    f"Available Concepts: {available_concepts}\n\n"
                    f"Metrics to decompose: {targets}\n\n"
                    "Provide a calculation plan using ONLY the available concepts.",
                ),
            ]
        )

        llm = service_manager.get_agent(temperature=0)
        structured_llm = llm.with_structured_output(DecompositionPlan)

        chain = prompt | structured_llm
        result: DecompositionPlan = await chain.ainvoke({})

        # Extract new dependencies to fetch
        new_dependencies = []
        for calc in result.calculations:
            new_dependencies.extend(calc.dependencies)

        print(f"[Decomposer] Plan: {result.calculations}")

        return {
            "calculations_to_run": result.calculations,
            "metrics_to_fetch": new_dependencies,
        }

    async def _fetch_data_node(self, state: _AgentState) -> _AgentState:
        """
        Retrieves the actual data values from the database.
        """
        print(f"--- [Node] Fetcher ---")
        # Unique metrics only
        # We fetch everything requested directly or as a dependency
        metrics_to_query = list(set(state.metrics_to_fetch))

        if not metrics_to_query:
            return {}

        price = pd.DataFrame()
        for i, m in enumerate(metrics_to_query):
            if "price" in m.lower():
                price = await self.db.get_price_data(
                    state.ticker, state.start_date, state.end_date, "yearly"
                )
                del metrics_to_query[i]

        # Filter strictly for the requested labels if possible, or just keep all.
        # Since get_data gets entire statements usually, we might have more than we need, which is fine.
        # However, to be precise:
        filtered_df = await self.db.search_label(
            state.ticker, metrics_to_query, state.start_date, state.end_date
        )

        # Merge with existing if any
        current_df = state.financial_data

        # Simple merge strategy: if current_df is empty, take new one.
        if current_df.empty:
            combined_df = filtered_df
        else:
            # combine_first aligns on index (label) and columns (date)
            combined_df = current_df.combine_first(filtered_df)

        if not price.empty:
            combined_df = pd.concat([combined_df, price.T], axis=1)

        return {"financial_data": combined_df}

    async def _calculator_node(self, state: _AgentState) -> _AgentState:
        """
        Executes the formulas using pandas eval.
        """
        print(f"--- [Node] Calculator ---")
        df = state.financial_data.copy()

        if df.empty or not state.calculations_to_run:
            return {}

        # The DataFrame is: Index=Label, Columns=Dates.
        # pandas.eval works best on Columns=Variables.
        # So we Transpose -> Calculate -> Transpose Back.

        df_eval = df.T  # Now Index=Dates, Columns=Labels

        for calc in state.calculations_to_run:
            try:
                expr = calc.pandas_eval_expression
                target = calc.target_metric_name
                print(f"[Calculator] {target} = {expr}")

                # Check if dependencies exist in columns
                missing_deps = [
                    d for d in calc.dependencies if d not in df_eval.columns
                ]
                if missing_deps:
                    print(
                        f"[Calculator] Warning: Missing dependencies for {target}: {missing_deps}"
                    )
                    # Attempt fuzzy lookup or skip?
                    # For safety, we skip if exact columns missing to prevent crash
                    continue

                df_eval.eval(f"{target} = {expr}", inplace=True)
            except Exception as e:
                print(f"[Calculator] Error calculating {calc.target_metric_name}: {e}")

        # Extract only the newly calculated columns + original
        # Transpose back: Index=Label (including new metrics), Columns=Dates
        final_df = df_eval.T

        # Optional: Save calculated metrics back to DB?
        # The new DB schema is (company, period_date, statement_type, label, value).
        # We could save these as statement_type='calculated'.
        # For this refactor, we stick to keeping it in state.

        return {"financial_data": final_df}

    async def _analyst_node(self, state: _AgentState) -> FundamentalAnalysisOutput:
        print(f"--- [Node] Analyst ---")
        llm = service_manager.get_agent(temperature=0.7)

        # Format dataframe for readability
        data_str = (
            state.financial_data.to_string()
            if not state.financial_data.empty
            else "No data found."
        )

        msg = (
            f"Analyze the following financial data for {state.ticker}.\n"
            f"Data (Rows=Metrics, Cols=Dates):\n{data_str}\n\n"
            f"User Question: {state.query}\n"
            "Highlight key trends, risks, and positives."
        )

        # Use ainvoke for async LLM call
        response = await llm.ainvoke(msg)
        return FundamentalAnalysisOutput(
            detailed_analysis=response.content, data=state.financial_data
        )


if __name__ == "__main__":
    # Note: Requires service_manager and FinancialDatabase mock/implementation to run.
    # The following is for demonstration of the flow structure.
    agent = FundamentalAnalysisAgent()

    # Example Usage:
    async def main():
        # Scenario 1: Basic metrics
        print("--- Running Scenario 1: Basic Metrics ---")
        input_basic = FundamentalAnalysisInput(
            ticker="AAPL",
            metrics=["market cap"],
            start_date="2020-01-01",
            end_date="2022-01-01",
            query="What were Apple's revenues and net income from 2020 to 2022?",
        )
        output_basic = await agent.run(input_basic)
        print(f"Scenario 1 Output: {output_basic.output}\n")

        # Scenario 2: Complex metric (Net Profit Margin)
        # print("--- Running Scenario 2: Complex Metric (Net Profit Margin) ---")
        # input_complex = FundamentalAnalysisInput(
        #     ticker="MSFT",
        #     metrics=["net_profit_margin", "free cash flow"],
        #     start_year=2020,
        #     end_year=2022,
        #     query="Analyse MSFT's net profit margin and free cash flow from 2020 to 2022.",
        # )
        # output_complex = await agent.run(input_complex)
        # print(f"Scenario 2 Output: {output_complex.output}\n")

    asyncio.run(main())  # Uncomment to test with actual setup
