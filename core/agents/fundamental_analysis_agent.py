import asyncio
import logging
import operator
from datetime import datetime
from typing import Annotated, Any, List, Optional, Type

import pandas as pd
from core.agents.base_agent import AbstractAgent, AgentOutput
from core.services import service_manager
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FundamentalAnalysisAgent")


# --- 1. Internal Structured Output Models ---


class CalculatedMetric(BaseModel):
    """
    Represents a single metric to be calculated.
    """

    target_metric_name: str = Field(
        description="The name of the new metric to calculate (e.g., 'net_profit_margin'). Use snake_case."
    )
    pandas_eval_expression: str = Field(
        description="The mathematical formula compatible with pandas.eval(). Example: 'net_income / revenue'"
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

    # Note: We don't use Pydantic to validate the DataFrame structure deeply
    # as it changes dynamically, but we type hint it.
    financial_data: Any = Field(default_factory=pd.DataFrame)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- 2. Public Input Schema ---


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
    Refactored Agent using Structured Outputs and Async IO,
    optimizing for one bulk database insert.
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

    @classmethod
    def get_output_schema_class(cls) -> Type[BaseModel]:
        return AgentOutput

    async def run(self, input_data: FundamentalAnalysisInput) -> AgentOutput:
        logger.info(f"🚀 [Agent Start] Ticker: {input_data.ticker}")

        # Defaults
        current_year = datetime.now().year
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

        final_state = await self._graph.ainvoke(initial_state)
        output_content = final_state["messages"][-1].content

        logger.info("🏁 [Agent Finish] Analysis Complete.")
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

    # --- Helper: CPU/IO Bound Wrappers ---

    async def _run_blocking(self, func, *args, **kwargs):
        """Helper to run blocking DB/API/Pandas calls in a thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # --- Node Implementations ---

    async def _parser_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Separates metrics into 'fetchable' (exist in DB) and 'complex' (need formulas).
        """
        logger.info("🔍 [Step: Parser] Resolving metrics against database...")
        db = service_manager.get_financial_database()

        def _resolve_metrics():
            f, u = [], []
            for metric in state.metrics:
                resolved = db.resolve_concept(state.ticker, metric)
                if resolved:
                    f.append(resolved)
                else:
                    u.append(metric)
            return f, u

        to_fetch, unknown_metrics = await self._run_blocking(_resolve_metrics)

        logger.info(
            f"   -> Found {len(to_fetch)} direct metrics, {len(unknown_metrics)} require calculation."
        )

        # Create placeholders for unknown metrics
        placeholders = [
            CalculatedMetric(
                target_metric_name=m, pandas_eval_expression="", dependencies=[]
            )
            for m in unknown_metrics
        ]

        return {"metrics_to_fetch": to_fetch, "calculations_to_run": placeholders}

    async def _decomposer_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Uses LLM with Structured Output to break down complex metrics into formulas.
        """
        targets = [
            c.target_metric_name
            for c in state.calculations_to_run
            if not c.pandas_eval_expression
        ]

        if not targets:
            return {}

        logger.info(f"🧠 [Step: Decomposer] Deriving formulas for: {targets}")

        db = service_manager.get_financial_database()
        # Blocking call to get concepts
        available_concepts = await self._run_blocking(
            db.get_all_concepts_for_company, state.ticker
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert financial quant. Your task is to build a calculation dependency tree.\n"
                    "Start by decomposing the requested target metric into a mathematical formula.\n"
                    "If the constituents of that formula are not in the available database concepts, "
                    "you must generate additional formulas to decompose those constituents.\n"
                    "Repeat this process recursively until every final input variable exists within the available database concepts.",
                ),
                (
                    "human",
                    f"Available Database Concepts: {available_concepts}\n\n"
                    f"Target Metric to decompose: {targets}\n\n"
                    "Output the calculation plan as a set of step-by-step formulas.\n"
                    "Ensure that the final level of the decomposition uses ONLY the available concepts.",
                ),
            ]
        )

        llm = service_manager.get_agent(temperature=0)
        structured_llm = llm.with_structured_output(DecompositionPlan)

        chain = prompt | structured_llm
        # Async Invoke
        result: DecompositionPlan = await chain.ainvoke({})

        # Extract new dependencies to fetch
        new_dependencies = []
        for calc in result.calculations:
            new_dependencies.extend(calc.dependencies)

        logger.info(f"   -> Plan generated with {len(result.calculations)} formulas.")
        return {
            "calculations_to_run": result.calculations,
            "metrics_to_fetch": new_dependencies,
        }

    async def _fetch_data_node(self, state: _AgentState) -> dict[str, Any]:
        logger.info("💾 [Step: Fetcher] Retrieving and caching data...")
        db = service_manager.get_financial_database()

        # Unique metrics only
        metrics = list(set(state.metrics_to_fetch))

        # 1. Fetch New Data (Network IO)
        # Fetch for the entire date range to ensure DB completeness
        def _fetch_new():
            return db.fetch_new_filings(
                state.ticker, state.start_year, state.end_year + 1
            )

        new_data_df = await self._run_blocking(_fetch_new)

        if not new_data_df.empty:
            logger.info(f"   -> Fetched {len(new_data_df)} new data points from API.")
        else:
            logger.info("   -> No new data found from API.")

        # 2. Fetch Existing Data (Disk IO)
        def _fetch_existing():
            return db.get_concept(
                state.ticker, metrics, state.start_year, state.end_year, exact=True
            )

        existing_data_df = await self._run_blocking(_fetch_existing)
        logger.info(f"   -> Found {existing_data_df.shape} existing data points.")

        # 3. Combine Data for Agent State (CPU Bound)
        pivoted_new_df = pd.DataFrame()
        if not new_data_df.empty:
            # We must pivot the new data before combining with existing pivoted data
            pivoted_new_df = await self._run_blocking(db.pivot_data, new_data_df)

        combined_df = existing_data_df
        if not pivoted_new_df.empty:
            if combined_df.empty:
                combined_df = pivoted_new_df
            else:
                # Use combine_first to take existing data over new data if years overlap
                combined_df = combined_df.combine_first(pivoted_new_df)

        # 4. Bulk Save New Data (Disk IO) - SINGLE TRANSACTION
        if not new_data_df.empty:
            await self._run_blocking(db.bulk_insert_financials, new_data_df)

        logger.info(f"   -> Final Data Shape for Agent: {combined_df.shape}")
        return {"financial_data": combined_df}

    async def _calculator_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Executes the formulas using pandas eval and saves the results.
        """
        if state.financial_data.empty or not state.calculations_to_run:
            return {}

        logger.info("🧮 [Step: Calculator] Computing derived metrics...")

        # Define the heavy calculation logic
        def _compute(df_input, calculations, ticker):
            df_eval = df_input.copy()

            # 1. Flatten MultiIndex and Transpose
            try:
                df_eval.index = [c[1] for c in df_input.index]
            except IndexError:
                pass
            df_eval = df_eval.T

            # 2. Execute
            for calc in calculations:
                try:
                    df_eval.eval(
                        f"{calc.target_metric_name} = {calc.pandas_eval_expression}",
                        inplace=True,
                    )
                except Exception as e:
                    logger.error(
                        f"      Error calculating {calc.target_metric_name}: {e}"
                    )

            # 3. Filter & Restore to MultiIndex Pivot
            calculated_cols = [
                c.target_metric_name
                for c in calculations
                if c.target_metric_name in df_eval.columns
            ]
            if not calculated_cols:
                return None

            result_df = df_eval[calculated_cols].T
            result_df.index = pd.MultiIndex.from_tuples(
                [(ticker, col) for col in result_df.index],
                names=["company", "concept"],
            )
            return result_df

        # Run computation in thread
        result_df = await self._run_blocking(
            _compute, state.financial_data, state.calculations_to_run, state.ticker
        )

        if result_df is not None:
            # Merge results back into the state data
            final_df = state.financial_data.combine_first(result_df)

            # Convert calculated pivoted metrics to long format for DB save
            db_save_df = self._melt_calculated(result_df, state.ticker)

            # Save calculated metrics (Bulk IO) - The second potential transaction
            db = service_manager.get_financial_database()
            await self._run_blocking(db.bulk_insert_financials, db_save_df)

            return {"financial_data": final_df}

        return {}

    def _melt_calculated(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Helper to convert pivoted calculated metrics back to long format for DB save."""
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Reset index so 'concept' becomes a column
        work_df = df.copy().reset_index()

        # 2. Melt (Unpivot) the DataFrame
        melted = work_df.melt(
            id_vars=["company", "concept"], var_name="year", value_name="value"
        )

        # 3. Clean and Add Metadata
        melted.dropna(subset=["value", "year"], inplace=True)
        melted["company"] = ticker
        melted["statement_type"] = "calculated"

        return melted[["company", "year", "statement_type", "concept", "value"]]

    async def _analyst_node(self, state: _AgentState) -> dict[str, Any]:
        logger.info("✍️  [Step: Analyst] Generating insights...")
        llm = service_manager.get_agent(temperature=0.7)

        data_str = (
            state.financial_data.to_string()
            if not state.financial_data.empty
            else "No data."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert financial analyst. Your task is to provide a concise and insightful quantitative analysis based ONLY on the provided financial data and the user's question. Focus on trends, year-over-year changes, and comparative analysis of the metrics. Your output must be a professional and easy-to-read report. Highlight key trends, risks, and positives.",
                ),
                (
                    "human",
                    f"Analyze the following financial data for {state.ticker}.\n"
                    f"Data:\n{data_str}\n\n"
                    f"User Question: {state.messages[0].content}",
                ),
            ]
        )

        response = await llm.ainvoke(prompt.format_prompt())
        return {"messages": [response]}


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
            metrics=["revenue", "net_income"],
            start_year=2020,
            end_year=2022,
            raw_input="What were Apple's revenues and net income from 2020 to 2022?",
        )
        output_basic = await agent.run(input_basic)
        print(f"Scenario 1 Output: {output_basic.output}\n")

        # Scenario 2: Complex metric (Net Profit Margin)
        print("--- Running Scenario 2: Complex Metric (Net Profit Margin) ---")
        input_complex = FundamentalAnalysisInput(
            ticker="MSFT",
            metrics=["net_profit_margin"],
            start_year=2020,
            end_year=2022,
            raw_input="Calculate Microsoft's net profit margin for the last three years.",
        )
        output_complex = await agent.run(input_complex)
        print(f"Scenario 2 Output: {output_complex.output}\n")

    asyncio.run(main())  # Uncomment to test with actual setup
