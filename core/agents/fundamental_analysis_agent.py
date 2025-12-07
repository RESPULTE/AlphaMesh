import asyncio
import difflib
import operator
from typing import Annotated, List, Optional, Type

import pandas as pd
from core.agents.base_agent import AbstractAgent
from core.agents.models import BaseAgentInput
from core.services import service_manager  # Assuming this still handles LLM retrieval

# Import the new database class
from get_financial_data import FinancialDatabase
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

# --- 1. Internal Structured Output Models ---
# ! only support 10-k as of now


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

    detailed_analysis: Optional[str] = None
    financial_data: pd.DataFrame = Field(default_factory=pd.DataFrame)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class _AgentState(BaseAgentInput, FundamentalAnalysisOutput):
    """Internal state for the agent workflow."""

    # Processing
    metrics_to_fetch: Annotated[List[str], operator.add] = Field(default_factory=list)
    calculations_to_run: List[CalculatedMetric] = Field(default_factory=list)


# --- 2. Public Input Schema ---


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
    def get_output_schema_class(cls) -> Type[BaseModel]:
        return FundamentalAnalysisOutput

    async def run(self, input_data: BaseAgentInput) -> FundamentalAnalysisOutput:
        """Async entry point for the agent."""
        print(f"--- [Agent: {self.name}] Started for {input_data.ticker} ---")

        # Ensure DB is initialized
        await self.db.initialize()

        retval = await self._graph.ainvoke(input_data)

        return FundamentalAnalysisOutput(
            detailed_analysis=retval["detailed_analysis"],
            financial_data=retval["financial_data"],
        )

    def _build_graph(self):
        workflow = StateGraph(_AgentState)

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
        ed = state.end_date.replace(month=12, day=31)

        # * ONLY SUPPORTING 10K for now, in the future, will inher user request for which form to query
        await self.db.update_financials(
            state.ticker, list(range(sd.year, ed.year + 1)), "10-K"
        )

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
                    "   Example: If available concept is 'Gross Profit', use 'Gross_Profit' in the formula.\n"
                    "4. If a formula requires the market price of the stock, use 'stock_price' as the name for that component.\n",
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

        # Separate price metrics from financial metrics to fetch them differently
        price_metrics = []
        financial_metrics = []
        for m in metrics_to_query:
            if "price" in m.lower():
                price_metrics.append(m)
            else:
                financial_metrics.append(m)

        if price_metrics:
            # Fetch yearly price data (Returns Index=Date, Columns=Metrics)
            price = await self.db.get_price_data(
                state.ticker, state.start_date, state.end_date, "yearly"
            )
            # ! hard coded
            price = price.loc[:, ["stock_price"]]

        # Filter strictly for the requested labels if possible
        filtered_df = await self.db.search_label(
            state.ticker, financial_metrics, state.start_date, state.end_date
        )

        # Merge with existing state data if any
        current_df = state.financial_data

        if current_df.empty:
            combined_df = filtered_df
        else:
            # combine_first aligns on index (label) and columns (date)
            combined_df = current_df.combine_first(filtered_df)

        # Process Price Data Alignment
        if not price.empty:
            # If we have existing financial data, we must align the price dates (index)
            # to the financial data columns (dates) by matching the Year.
            if not combined_df.empty:
                actual_date = combined_df.columns.copy()
                financial_years = [int(d.split("-")[0]) for d in actual_date]

                # Create a new index for price data based on matching years
                new_index = []
                for price_date in price.index:
                    # If we find a matching year in financials, use the financial date
                    if price_date.year in financial_years:
                        new_index.append(
                            actual_date[financial_years.index(price_date.year)]
                        )
                    else:
                        # Otherwise keep the original price date
                        new_index.append(price_date)

                price.index = new_index

            # Transpose Price: Now Index=Metrics, Columns=Dates
            price_t = price.T

            # Concatenate along axis=0 to add the price rows to the financial rows
            # This relies on the columns (Dates) now matching.
            combined_df = pd.concat([combined_df, price_t], axis=0)

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

        # Format dataframe for readability
        data_str = (
            state.financial_data.to_string()
            if not state.financial_data.empty
            else "No data found."
        )

        msg = (
            f"Analyze the following financial data for {state.ticker}.\n"
            f"Data (Rows=Metrics, Cols=Dates):\n{data_str}\n\n"
            f"User Question: {state.query}\n\n"
            "### Analysis Instructions:\n"
            "1. Highlight key trends, risks, and positives.\n"
            "2. **Number Formatting:** When citing numbers from the data, you MUST automatically convert "
            "scientific notation (e.g., 1.5e9) or large raw integers into human-readable denominations "
            "(Million, Billion, Trillion). For example, convert '1.5e9' to '1.5 Billion'."
        )

        # Use ainvoke for async LLM call
        response = await service_manager.get_agent(temperature=0.7).ainvoke(msg)

        return {"detailed_analysis": response.content}


if __name__ == "__main__":
    # Note: Requires service_manager and FinancialDatabase mock/implementation to run.
    # The following is for demonstration of the flow structure.
    agent = FundamentalAnalysisAgent()

    # Example Usage:
    async def main():
        # Scenario 1: Basic metrics
        print("--- Running Scenario 1: Basic Metrics ---")
        # input_basic = FundamentalAnalysisInput(
        #     ticker="AAPL",
        #     metrics=["total market capitalization"],
        #     start_date="2020-01-01",
        #     end_date="2022-01-01",
        #     query="What were Apple's market cap from 2020 to 2022?",
        # )
        # output_basic = await agent.run(input_basic)
        # print(
        #     f"Scenario 1 Output: {output_basic.detailed_analysis}\n\n\n{output_basic.financial_data}"
        # )

        # Scenario 2: Complex metric (Net Profit Margin)
        print("--- Running Scenario 2: Complex Metric (Net Profit Margin) ---")
        input_complex = BaseAgentInput(
            ticker="MSFT",
            metrics=["net_profit_margin", "free cash flow"],
            start_date="2020-01-01",
            end_date="2022-12-31",
            query="Analyse MSFT's net profit margin and free cash flow from 2020 to 2022.",
        )
        output_complex = await agent.run(input_complex)
        print(
            f"Scenario 1 Output: {output_complex.detailed_analysis}\n\n\n{output_complex.financial_data}"
        )

    asyncio.run(main())  # Uncomment to test with actual setup
