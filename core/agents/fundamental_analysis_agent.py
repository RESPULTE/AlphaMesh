import asyncio
import difflib
import operator
from typing import Annotated, List, Optional, Type

import pandas as pd
from core.agents.base_agent import AbstractAgent
from core.agents.get_financial_data import FinancialDatabase
from core.agents.models import BaseAgentInput, BaseAgentOutput
from core.services import service_manager
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from core.logger import get_logger

logger = get_logger(__name__)

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

    @staticmethod
    def name() -> str:
        return "fundamentals_agent"

    @staticmethod
    def description() -> str:
        return "Fetches and calculates quantitative financial data (ratios, statements). Returns raw data."

    @staticmethod
    def get_output_schema_class() -> Type[BaseModel]:
        return FundamentalAnalysisOutput

    async def run(self, input_data: BaseAgentInput) -> FundamentalAnalysisOutput:
        """Async entry point for the agent."""
        logger.info(f"--- [Agent: {self.name}] Started for {input_data.ticker} ---")
        await self.db.initialize()

        final_state = await self._graph.ainvoke(input_data.model_dump())

        return FundamentalAnalysisOutput(
            financial_data=final_state.get("financial_data"),
            analysis=final_state.get("analysis"),
        )

    def _build_graph(self):
        workflow = StateGraph(_AgentState, output_schema=FundamentalAnalysisOutput)

        workflow.add_node("parser", self._parser_node)
        workflow.add_node("decomposer", self._decomposer_node)
        workflow.add_node("fetch_data", self._fetch_data_node)
        workflow.add_node("calculator", self._calculator_node)
        workflow.add_node("analyst", self._generate_analysis)

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

    async def _parser_node(self, state: _AgentState) -> dict:
        """
        Checks which metrics are raw DB columns vs. which need to be calculated.
        """
        # REWRITTEN: Use state.attribute access
        logger.info(f"--- [Node] Parser ---")

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
                    logger.info(f"Resolving '{metric}' to fuzzy match '{found_label}'")
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
        Handles multi-level decomposition (e.g., Price -> FCF -> [OCF - CapEx]).
        """
        targets = [
            c.target_metric_name
            for c in state.calculations_to_run
            if not c.pandas_eval_expression
        ]

        if not targets:
            return {}

        logger.info(f"--- [Node] Decomposer: Deriving formulas for {targets} ---")

        available_concepts = await self.db.get_labels(state.ticker)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert financial quant and data engineer. Decompose the requested metrics into mathematical formulas executable in pandas.\n\n"
                    "RULES:\n"
                    "1. **Check 'Available Concepts':** If a requested metric relies on data NOT in the list (e.g., 'Free Cash Flow'), you must create an **Intermediate Calculation** first.\n"
                    "2. **CRITICAL - UNIT CONSISTENCY (Per Share vs. Total):**\n"
                    "   - The variable 'stock_price' represents the price of ONE share.\n"
                    "   - You generally cannot divide 'stock_price' by a total company metric (like 'Total Revenue' or 'Total Free Cash Flow').\n"
                    "   - If calculating a valuation ratio (e.g., 'Price to Free Cash Flow'), you MUST:\n"
                    "       a) Identify the total metric (e.g., Free Cash Flow).\n"
                    "       b) Identify 'Shares Outstanding' (or 'Weighted Average Shares') from the available concepts.\n"
                    "       c) Calculate the **Per Share** metric (e.g., `fcf_per_share = free_cash_flow / shares_outstanding`).\n"
                    "       d) Calculate the final ratio using the per-share metric (e.g., `price_to_fcf = stock_price / fcf_per_share`).\n"
                    "3. **Order matters:** Define intermediate variables (like Total FCF, then FCF Per Share) BEFORE using them in the final formula.\n"
                    "4. **Naming:** In 'pandas_eval_expression' and 'target_metric_name', replace ALL spaces with underscores '_'.\n"
                    "5. **Standard Definitions** (Use these if specific tags are missing):\n"
                    "   - Free Cash Flow = Net Cash provided by (used in) operating activities - Payments for (proceeds from) capital expenditures\n"
                    "   - Working Capital = Current assets - Current liabilities\n"
                    "   - Shares: Look for 'Weighted Average Shares', 'Common Stock Shares Outstanding', or similar.\n\n"
                    "EXAMPLE CALCULATION PLAN (Price to Free Cash Flow):\n"
                    "1. free_cash_flow = Cash_Flow_from_Operations - Capital_Expenditures\n"
                    "2. free_cash_flow_per_share = free_cash_flow / Weighted_Average_Shares_Outstanding\n"
                    "3. price_to_free_cash_flow = stock_price / free_cash_flow_per_share",
                ),
                (
                    "human",
                    f"Available Concepts (Raw Data): {available_concepts}\n\n"
                    f"Metrics to decompose: {targets}\n\n"
                    "Provide a calculation plan. Ensure you normalize total metrics to 'per share' before comparing them to 'stock_price'.",
                ),
            ]
        )

        llm = service_manager.get_agent(temperature=0)
        structured_llm = llm.with_structured_output(DecompositionPlan)
        result: DecompositionPlan = await (prompt | structured_llm).ainvoke({})

        # Extract dependencies (both raw data and newly created intermediates)
        # Note: Depending on your architecture, you might need to filter 'new_dependencies'
        # to ensure you only fetch raw data, not the intermediate metrics you just invented.
        # usually, the fetcher will ignore metrics it can't find in the DB,
        # but it is safer if the LLM output distinguishes between 'fetch' and 'calculate'.

        new_dependencies = [
            dep for calc in result.calculations for dep in calc.dependencies
        ]

        # Clean up dependencies: Only fetch things that are NOT being calculated in this very plan
        calculated_vars = {c.target_metric_name for c in result.calculations}
        metrics_to_fetch = [d for d in new_dependencies if d not in calculated_vars]
        metrics_to_fetch.extend([t for t in targets if t not in calculated_vars])

        logger.info(f"   -> New dependencies to fetch: ")
        for m in metrics_to_fetch:
            logger.info(f"      - {m}")

        logger.info(f"   -> Calculations to perform:")
        for i, calc in enumerate(result.calculations, 1):
            logger.info(
                f"      {i}. {calc.target_metric_name} = {calc.pandas_eval_expression}"
            )

        return {
            "calculations_to_run": result.calculations,
            "metrics_to_fetch": metrics_to_fetch,
        }

    async def _fetch_data_node(self, state: _AgentState) -> dict:
        """
        Retrieves the actual data values from the database.
        """
        # REWRITTEN: Use state.attribute access
        logger.info(f"--- [Node] Fetcher ---")
        metrics_to_query = list(set(state.metrics_to_fetch))

        if not metrics_to_query:
            return {"financial_data": pd.DataFrame()}

        financial_df = await self.db.search_label(
            state.ticker, metrics_to_query, state.start_date, state.end_date
        )

        if any("stock_price" in m.lower() for m in metrics_to_query):
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
                    price_t.columns = financial_df.columns

                financial_df = pd.concat([financial_df, price_t])

        return {"financial_data": financial_df}

    async def _calculator_node(self, state: _AgentState) -> dict:
        """
        Executes the formulas from the decomposer using pandas eval.
        """
        # REWRITTEN: Use state.attribute access
        logger.info(f"--- [Node] Calculator ---")
        df = state.financial_data
        calculations = state.calculations_to_run

        if df is None or df.empty or not calculations:
            return {}

        df_eval = df.T

        for calc in calculations:
            try:
                expr = calc.pandas_eval_expression
                target = calc.target_metric_name
                logger.info(f"[Calculator] {target} = {expr}")

                missing_deps = [
                    d for d in calc.dependencies if d not in df_eval.columns
                ]
                if missing_deps:
                    logger.warning(
                        f"[Calculator] Warning: Missing dependencies for {target}: {missing_deps}"
                    )
                    continue

                df_eval.eval(f"{target} = {expr}", inplace=True)
            except Exception as e:
                logger.error(f"[Calculator] Error calculating {calc.target_metric_name}: {e}")

        return {"financial_data": df_eval.T}

    async def _generate_analysis(self, state: _AgentState) -> FundamentalAnalysisOutput:
        logger.info(f"--- [Node] Analyst ---")

        # Format dataframe for readability
        if state.financial_data is None or state.financial_data.empty:
            return FundamentalAnalysisOutput(
                financial_data=None, analysis="No data found."
            )

        human_readable = state.financial_data.map(
            lambda x: add_units(x) if isinstance(x, (int, float)) else x
        ).to_string()

        msg = (
            f"Analyze the following financial data for {state.ticker}.\n"
            f"Data (Rows=Metrics, Cols=Dates):\n{human_readable}\n\n"
            f"User Question: {state.query}\n\n"
            "### Analysis Instructions:\n"
            "1. Highlight key trends, risks, and positives.\n"
            "2. **Number Formatting:** When citing numbers from the data, you MUST automatically convert "
            "scientific notation (e.g., 1.5e9) or large raw integers into human-readable denominations "
            "(Million, Billion, Trillion). For example, convert '1.5e9' to '1.5 Billion'."
        )

        # Use ainvoke for async LLM call
        response = await service_manager.get_agent(temperature=0.7).ainvoke(msg)

        return FundamentalAnalysisOutput(
            financial_data=state.financial_data, analysis=response.content
        )


def add_units(x):
    if (x > 0 and x >= 1_000_000_000) or (x < 0 and x <= -1_000_000_000):
        return f"{x/1_000_000_000:.2f} Billion"
    elif (x > 0 and x >= 1_000_000) or (x < 0 and x <= -1_000_000):
        return f"{x/1_000_000:.2f} Million"
    elif (x > 0 and x >= 1_000) or (x < 0 and x <= -1_000):
        return f"{x/1_000:.2f} Thousand"

    return f"{x:.2f}"


if __name__ == "__main__":

    async def main():
        agent = FundamentalAnalysisAgent()

        logger.info("--- Running Scenario: Complex Metric (Net Profit Margin) ---")
        input_data = BaseAgentInput(
            ticker="MSFT",
            metrics=["stock_price"],
            vector_query="MSFT",
            start_date="2020-01-01",
            end_date="2022-12-31",
            query="Get MSFT's price from 2020 to 2022.",
        )
        output_complex = await agent.run(input_data)
        logger.info("\n--- RAW AGENT OUTPUT ---\n")

        logger.info("[analysis]")
        logger.info(output_complex.analysis)

        logger.info("-" * 60)

        logger.info("\n[financial_data]")
        human_readable = output_complex.financial_data.map(
            lambda x: add_units(x) if isinstance(x, (int, float)) else x
        ).to_string()
        logger.info(human_readable)

    asyncio.run(main())

    # agent = FundamentalAnalysisAgent()
    # graph = agent._graph.get_graph()
    # png_bytes = graph.draw_mermaid_png(
    #     frontmatter_config={"chartOrientation": "horizontal"}
    # )

    # with open("statisstical_analysis.png", "wb") as f:
    #     f.write(png_bytes)

    # logger.info("Saved graph as graph.png")
