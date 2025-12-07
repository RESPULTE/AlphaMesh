import asyncio
import difflib
import operator
from typing import Annotated, List, Optional, Type

import pandas as pd
from core.agents.base_agent import AbstractAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput
from core.services import service_manager
from get_financial_data import FinancialDatabase
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

# --- 1. Internal Structured Output Models ---


class CalculatedMetric(BaseModel):
    target_metric_name: str = Field(
        description="The name of the new metric to calculate, use camel case _ instead of spaces."
    )
    pandas_eval_expression: str = Field(
        description="The mathematical formula compatible with pandas.eval(). Example: 'A / B'"
    )
    dependencies: List[str] = Field(
        description="A list of the base financial concepts required for this formula."
    )


class DecompositionPlan(BaseModel):
    calculations: List[CalculatedMetric]


class FundamentalAnalysisOutput(BaseAgentOutput):
    """Data container for the Fundamental Analysis Agent."""

    agent_name: str = "fundamentals_agent"
    financial_data: Optional[pd.DataFrame] = Field(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_llm_context_str(self) -> str:
        """Formats the DataFrame into a readable string for the analyst LLM."""
        if self.financial_data is None or self.financial_data.empty:
            return "### REPORT FROM fundamentals_agent\nNo financial data was found or calculated."

        header = "### REPORT FROM fundamentals_agent (Quantitative Financial Data)\n"
        # Using to_string() is effective for LLM consumption
        data_str = self.financial_data.to_string(max_rows=20, float_format="%.2f")
        return f"{header}Data (Rows=Metrics, Columns=Dates):\n{data_str}"


class _AgentState(BaseAgentInput):
    metrics_to_fetch: Annotated[List[str], operator.add] = Field(default_factory=list)
    calculations_to_run: List[CalculatedMetric] = Field(default_factory=list)
    financial_data: Optional[pd.DataFrame] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- 2. The Agent Class ---


class FundamentalAnalysisAgent(AbstractAgent):
    """
    Refactored Async Agent focused on fetching and calculating financial data.
    """

    def __init__(self):
        super().__init__()
        self._graph = self._build_graph()
        self.db = FinancialDatabase()

    @property
    def name(self) -> str:
        return "fundamentals_agent"

    @property
    def description(self) -> str:
        return "Fetches and calculates quantitative financial data (ratios, statements). Returns raw data."

    @classmethod
    def get_output_schema_class(cls) -> Type[BaseModel]:
        return FundamentalAnalysisOutput

    async def run(self, input_data: BaseAgentInput) -> FundamentalAnalysisOutput:
        """Async entry point for the agent."""
        print(f"--- [Agent: {self.name}] Started for {input_data.ticker} ---")
        await self.db.initialize()

        final_state = await self._graph.ainvoke(input_data.model_dump())

        return FundamentalAnalysisOutput(
            financial_data=final_state.get("financial_data"),
        )

    def _build_graph(self):
        workflow = StateGraph(_AgentState)

        workflow.add_node("parser", self._parser_node)
        workflow.add_node("decomposer", self._decomposer_node)
        workflow.add_node("fetch_data", self._fetch_data_node)
        workflow.add_node("calculator", self._calculator_node)

        workflow.add_edge(START, "parser")

        # REWRITTEN: Use direct attribute access in conditional logic
        workflow.add_conditional_edges(
            "parser",
            lambda state: "decomposer" if state.calculations_to_run else "fetch_data",
        )

        workflow.add_edge("decomposer", "fetch_data")
        workflow.add_edge("fetch_data", "calculator")
        workflow.add_edge("calculator", END)

        return workflow.compile()

    # --- Node Implementations ---

    async def _parser_node(self, state: _AgentState) -> dict:
        """
        Checks which metrics are raw DB columns vs. which need to be calculated.
        """
        # REWRITTEN: Use state.attribute access
        print(f"--- [Node] Parser ---")

        await self.db.update_financials(
            state.ticker,
            list(range(state.start_date.year, state.end_date.year + 1)),
            "10-K",
        )

        to_fetch = []
        unknown_metrics = []

        available_labels_df = await self.db.search_label(state.ticker, [])
        if available_labels_df.empty:
            unknown_metrics = state.metrics
        else:
            available_labels = set(available_labels_df.index)
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
            "metrics_to_fetch": to_fetch,
        }

    async def _decomposer_node(self, state: _AgentState) -> dict:
        """
        Uses LLM to decompose complex metrics into formulas based on available data.
        """
        # REWRITTEN: Use state.attribute access
        targets = [
            c.target_metric_name
            for c in state.calculations_to_run
            if not c.pandas_eval_expression
        ]

        if not targets:
            return {}

        print(f"--- [Node] Decomposer: Deriving formulas for {targets} ---")

        available_concepts = await self.db.get_labels(state.ticker)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial quant. Decompose the requested metrics into mathematical formulas.\n"
                    "1. Use precise labels from the 'Available Concepts' list.\n"
                    "2. DO NOT use any labels other than the provided ones.\n"
                    "3. CRITICAL: In 'pandas_eval_expression' and 'target_metric_name', replace spaces with underscores '_'.\n"
                    "4. If a formula requires stock market price, use 'stock_price'.\n",
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
        result: DecompositionPlan = await (prompt | structured_llm).ainvoke({})

        new_dependencies = [
            dep for calc in result.calculations for dep in calc.dependencies
        ]

        print(f"[Decomposer] Plan: {result.calculations}")

        return {
            "calculations_to_run": result.calculations,
            "metrics_to_fetch": new_dependencies,
        }

    async def _fetch_data_node(self, state: _AgentState) -> dict:
        """
        Retrieves the actual data values from the database.
        """
        # REWRITTEN: Use state.attribute access
        print(f"--- [Node] Fetcher ---")
        metrics_to_query = list(set(state.metrics_to_fetch))

        if not metrics_to_query:
            return {"financial_data": pd.DataFrame()}

        financial_df = await self.db.search_label(
            state.ticker, metrics_to_query, state.start_date, state.end_date
        )

        if any("price" in m.lower() for m in metrics_to_query):
            price_df = await self.db.get_price_data(
                state.ticker, state.start_date, state.end_date, "yearly"
            )
            if not price_df.empty and "stock_price" in price_df.columns:
                price_t = price_df[["stock_price"]].T
                price_t.columns = [d.strftime("%Y-%m-%d") for d in price_t.columns]

                # Align column types if necessary before combining
                if not financial_df.empty:
                    financial_df.columns = pd.to_datetime(
                        financial_df.columns
                    ).strftime("%Y-%m-%d")
                    financial_df = pd.concat([financial_df, price_t]).loc[
                        ~financial_df.index.duplicated(keep="first")
                    ]

        return {"financial_data": financial_df}

    async def _calculator_node(self, state: _AgentState) -> dict:
        """
        Executes the formulas from the decomposer using pandas eval.
        """
        # REWRITTEN: Use state.attribute access
        print(f"--- [Node] Calculator ---")
        df = state.financial_data
        calculations = state.calculations_to_run

        if df is None or df.empty or not calculations:
            return {}

        df_eval = df.T

        for calc in calculations:
            try:
                expr = calc.pandas_eval_expression
                target = calc.target_metric_name
                print(f"[Calculator] {target} = {expr}")

                missing_deps = [
                    d for d in calc.dependencies if d not in df_eval.columns
                ]
                if missing_deps:
                    print(
                        f"[Calculator] Warning: Missing dependencies for {target}: {missing_deps}"
                    )
                    continue

                df_eval.eval(f"{target} = {expr}", inplace=True)
            except Exception as e:
                print(f"[Calculator] Error calculating {calc.target_metric_name}: {e}")

        return {"financial_data": df_eval.T}


if __name__ == "__main__":

    async def main():
        agent = FundamentalAnalysisAgent()

        print("--- Running Scenario: Complex Metric (Net Profit Margin) ---")
        input_complex = BaseAgentInput(
            ticker="MSFT",
            metrics=["net_profit_margin", "free cash flow"],
            start_date="2020-01-01",
            end_date="2022-12-31",
            query="Get MSFT's net profit margin and free cash flow from 2020 to 2022.",
        )
        output_complex = await agent.run(input_complex)
        print("\n--- RAW AGENT OUTPUT ---\n")
        print(output_complex.financial_data)

    asyncio.run(main())
