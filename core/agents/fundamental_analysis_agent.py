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
        description="The mathematical formula compatible with pandas.eval(). Example: 'price / revenue'"
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
    Refactored Agent using Structured Outputs and Async IO.
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
    def get_input_schema_class(cls) -> Type[BaseModel]:
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

    async def _run_blocking_db(self, func, *args, **kwargs):
        """Helper to run blocking DB calls in a thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _run_cpu_bound(self, func, *args, **kwargs):
        """Helper to run blocking Pandas operations in a thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # --- Node Implementations ---

    async def _parser_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Separates metrics into 'fetchable' (exist in DB) and 'complex' (need formulas).
        """
        logger.info("🔍 [Step: Parser] Resolving metrics against database...")
        db = service_manager.get_financial_database()

        to_fetch = []
        unknown_metrics = []

        # Run resolution in thread as it might involve DB lookups
        def _resolve_metrics():
            f, u = [], []
            for metric in state.metrics:
                resolved = db.resolve_concept(state.ticker, metric)
                if resolved:
                    f.append(resolved)
                else:
                    u.append(metric)
            return f, u

        to_fetch, unknown_metrics = await self._run_blocking_db(_resolve_metrics)

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
        available_concepts = await self._run_blocking_db(
            db.get_all_concepts_for_company, state.ticker
        )

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
        logger.info("💾 [Step: Fetcher] Retrieving data from Financial DB...")

        # Unique metrics only
        metrics = list(set(state.metrics_to_fetch))

        if not metrics:
            logger.warning("   -> No metrics to fetch.")
            return {}

        db = service_manager.get_financial_database()

        # Define the blocking IO operations
        def _fetch_op():
            # Update data (might trigger API calls)
            db.update_company_data(
                state.ticker, num_years=state.end_year - state.start_year + 1
            )
            # Query data (SQL/Pandas)
            return db.get_concept(
                state.ticker, metrics, state.start_year, state.end_year, exact=True
            )

        df = await self._run_blocking_db(_fetch_op)

        # Merge with existing if any (CPU bound)
        def _merge_op(existing, new_df):
            if not existing.empty:
                return existing.combine_first(new_df)
            return new_df

        combined_df = await self._run_cpu_bound(_merge_op, state.financial_data, df)

        logger.info(f"   -> Data shape: {combined_df.shape}")
        return {"financial_data": combined_df}

    async def _calculator_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Executes the formulas using pandas eval.
        """
        if state.financial_data.empty or not state.calculations_to_run:
            return {}

        logger.info("🧮 [Step: Calculator] Computing derived metrics...")

        # Define the heavy calculation logic
        def _compute(df_input, calculations, ticker):
            df_eval = df_input.copy()

            # 1. Flatten MultiIndex for eval (Ticker, Metric) -> Metric
            try:
                df_eval.index = [c[1] for c in df_input.index]
            except IndexError:
                pass

            # 2. Transpose for Eval (Rows=Years, Cols=Metrics)
            df_eval = df_eval.T

            # 3. Execute
            for calc in calculations:
                try:
                    logger.debug(
                        f"      Eval: {calc.target_metric_name} = {calc.pandas_eval_expression}"
                    )
                    df_eval.eval(
                        f"{calc.target_metric_name} = {calc.pandas_eval_expression}",
                        inplace=True,
                    )
                except Exception as e:
                    logger.error(
                        f"      Error calculating {calc.target_metric_name}: {e}"
                    )

            # 4. Filter & Restore
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
        result_df = await self._run_cpu_bound(
            _compute, state.financial_data, state.calculations_to_run, state.ticker
        )

        if result_df is not None:
            # Merge results back
            final_df = await self._run_cpu_bound(
                lambda old, new: old.combine_first(new), state.financial_data, result_df
            )

            # Save back to DB (IO bound)
            db = service_manager.get_financial_database()
            await self._run_blocking_db(
                db.save_calculated_metric, state.ticker, result_df
            )

            return {"financial_data": final_df}

        return {}

    async def _analyst_node(self, state: _AgentState) -> dict[str, Any]:
        logger.info("✍️  [Step: Analyst] Generating insights...")
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

        response = await llm.ainvoke(msg)
        return {"messages": [response]}


if __name__ == "__main__":
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

        # Scenario 3: Mix of basic and complex
        print("--- Running Scenario 3: Mixed Metrics ---")
        input_mixed = FundamentalAnalysisInput(
            ticker="GOOG",
            metrics=["revenue", "gross_profit_margin"],
            start_year=2021,
            end_year=2023,
            raw_input="Tell me about Google's revenue and gross profit margin from 2021 to 2023.",
        )
        output_mixed = await agent.run(input_mixed)
        print(f"Scenario 3 Output: {output_mixed.output}\n")

        # Scenario 4: No specific years, default to last 5
        print("--- Running Scenario 4: Default Years ---")
        input_default_years = FundamentalAnalysisInput(
            ticker="AMZN",
            metrics=["free_cash_flow"],
            raw_input="What is Amazon's free cash flow?",
            start_year=None,
            end_year=None,
        )
        output_default_years = await agent.run(input_default_years)
        print(f"Scenario 4 Output: {output_default_years.output}\n")

    asyncio.run(main())
