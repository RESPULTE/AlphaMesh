import datetime
import operator
import re
from typing import Annotated, Any, List, Optional, Type

import pandas as pd
from core.agents.base_agent import AbstractAgent, AgentOutput
from core.services import service_manager
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

# --- Internal State Models ---


class _AgentState(BaseModel):
    """
    Internal state for the agent.
    Includes fields from FundamentalAnalysisInput plus internal processing fields.
    """

    # Base message history
    messages: Annotated[List[BaseMessage], operator.add]

    # Input fields mapped from FundamentalAnalysisInput
    ticker: str = Field(default="")
    metrics: List[str] = Field(default_factory=list)
    start_year: int = Field(default=0)
    end_year: int = Field(default=0)

    # Internal processing fields
    metrics_to_process: Annotated[list[str], operator.add] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
    financial_data: pd.DataFrame = Field(default_factory=pd.DataFrame)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class _BatchResponse(BaseModel):
    formulas: List[str]


# --- Public Input Schema ---


class FundamentalAnalysisInput(BaseModel):
    """Input schema for the Fundamental Analysis Agent."""

    ticker: str = Field(description="The stock ticker symbol to analyze.")
    metrics: List[str] = Field(
        description="The list of financial metrics to analyze (e.g., 'revenue', 'net income')."
    )
    start_year: Optional[int] = Field(
        default=None,
        description="The starting year for the analysis. Defaults to 5 years ago.",
    )
    end_year: Optional[int] = Field(
        default=None,
        description="The ending year for the analysis. Defaults to the current year.",
    )
    raw_input: str = Field(
        description="The original user query for context in the final analysis."
    )


# --- Agent Definition ---


class FundamentalAnalysisAgent(AbstractAgent):
    """Agent for deep-dive quantitative analysis of company financials."""

    def __init__(self):
        super().__init__()
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "fundamentals_agent"

    @property
    def description(self) -> str:
        return (
            "Focuses on quantitative data: financial statements, balance sheets, "
            "revenue numbers, margins, and growth ratios. Use this for questions "
            "about a company's profitability, financial health, and valuation metrics."
        )

    @classmethod
    def get_input_schema_class(cls) -> Type[BaseModel]:
        return FundamentalAnalysisInput

    def run(self, input_data: FundamentalAnalysisInput) -> AgentOutput:
        """Executes the fundamental analysis workflow."""
        print(
            f"--- [Agent: {self.name}] Executing with input: {input_data.model_dump()} ---"
        )

        # Calculate default years if not provided
        current_year = datetime.datetime.now().year
        s_year = input_data.start_year if input_data.start_year else (current_year - 5)
        e_year = input_data.end_year if input_data.end_year else current_year

        # Construct the initial state to match _AgentState structure
        initial_state = {
            "messages": [HumanMessage(content=input_data.raw_input)],
            "ticker": input_data.ticker.upper(),
            "metrics": input_data.metrics,
            "start_year": s_year,
            "end_year": e_year,
            # Initialize internal fields
            "metrics_to_process": [],
            "formulas": [],
            "financial_data": pd.DataFrame(),
        }

        # Invoke the graph
        final_state = self._graph.invoke(initial_state)

        # The final output is in the 'messages' of the final state
        output_content = final_state["messages"][-1].content

        return AgentOutput(agent_name=self.name, output=output_content)

    def _build_graph(self):
        """Builds and compiles the LangGraph workflow."""
        workflow = StateGraph(_AgentState)

        workflow.add_node("parser", self._parser_node)
        workflow.add_node("fetch_data", self._fetch_data_node)
        workflow.add_node("decomposer", self._decomposer_node)
        workflow.add_node("calculator", self._calculator_node)
        workflow.add_node("analyst", self._analyst_node)

        workflow.add_edge(START, "parser")
        workflow.add_edge("parser", "decomposer")
        workflow.add_edge("decomposer", "fetch_data")

        workflow.add_conditional_edges(
            "fetch_data",
            lambda state: "calculator" if state.formulas else "analyst",
            {"calculator": "calculator", "analyst": "analyst"},
        )
        workflow.add_edge("calculator", "analyst")
        workflow.add_edge("analyst", END)

        return workflow.compile()

    # --- Node Implementations ---

    def _parser_node(self, state: _AgentState) -> dict[str, Any]:
        """
        Parses the inputs in the state to prepare for data fetching.
        """
        print(f"\n--- [Node] Parser ---")

        # Access data directly from the state object
        ticker = state.ticker
        start_year = state.start_year
        end_year = state.end_year
        metrics = state.metrics

        print(
            f"[Parser] Processing - Ticker: {ticker}, Years: {start_year}-{end_year}, Metrics: {metrics}"
        )

        db = service_manager.get_financial_database()
        db.update_company_data(ticker, num_years=end_year - start_year + 1)

        metrics_to_process = []
        formulas = []

        for metric in metrics:
            resolved_metric = db.resolve_concept(ticker, metric)
            print(f"[Parser] Resolving metric '{metric}' -> '{resolved_metric}'")

            if resolved_metric is None:
                print(
                    f"[Parser] Could not resolve concept for '{metric}', treating as formula input."
                )
                formulas.append(metric)
                continue
            metrics_to_process.append(resolved_metric)

        # Return updates to the state
        return {
            "metrics_to_process": metrics_to_process,
            "formulas": formulas,
        }

    def _decomposer_node(self, state: _AgentState) -> dict[str, Any]:
        if not state.formulas:
            return {}

        print(f"--- [Decomposer] Batch processing: {state.formulas} ---")
        db = service_manager.get_financial_database()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial decomposition assistant. "
                    "For each requested metric, provide ONLY an explicit mathematical formula using exclusively the concepts listed below:\n"
                    f"{db.get_all_concepts_for_company(state.ticker)}\n"
                    "Rules:\n"
                    "1. The formula must strictly follow this style: metric = (concept_1) + (concept_2)\n"
                    "2. Use brackets '()' around each constituent concept.\n"
                    "3. Use only +, -, *, / operators.\n"
                    "4. Output exactly one formula per metric, nothing else.",
                ),
                ("human", "Metrics to decompose: {metrics}"),
            ]
        )

        llm = service_manager.get_agent(temperature=0)
        res: _BatchResponse = (
            prompt | llm.with_structured_output(_BatchResponse)
        ).invoke({"metrics": ", ".join(state.formulas)})

        updated_metrics = []
        updated_formulas = []

        for formula in res.formulas:
            updated_metrics.extend(re.findall(r"\((.*?)\)", formula))
            lhs, rhs = formula.split("=")
            new_lhs = lhs.strip().replace(" ", "_")
            new_rhs = rhs.replace("(", "").replace(")", "")
            updated_formulas.append(f"{new_lhs} = {new_rhs}")

        return {"formulas": updated_formulas, "metrics_to_process": updated_metrics}

    def _fetch_data_node(self, state: _AgentState) -> dict[str, Any]:
        print(f"\n--- [Node] Fetcher ---")
        metrics = state.metrics_to_process or []

        if not metrics:
            return {}

        print(f"Fetching data for ticker: {state.ticker} -> {metrics}")
        db = service_manager.get_financial_database()

        new_data = db.get_concept(
            state.ticker,
            tuple(metrics),
            state.start_year,
            state.end_year,
            exact=True,
        )

        current_data = state.financial_data
        combined_data = (
            current_data.combine_first(new_data) if not current_data.empty else new_data
        )

        # Clear metrics_to_process as they are now in financial_data
        return {"financial_data": combined_data, "metrics_to_process": []}

    def _calculator_node(self, state: _AgentState) -> dict[str, Any]:
        print(f"\n--- [Node] Calculator ---")
        # Transpose for easier eval calculation (columns as variables)
        work_df = state.financial_data.T

        # Flatten MultiIndex if necessary or ensure simple column names for eval
        # Assuming get_concept returns MultiIndex (Ticker, Metric)
        work_df.columns = [concept for _, concept in work_df.columns]

        for f in state.formulas:
            try:
                work_df.eval(f, inplace=True)
            except Exception as e:
                print(f"Could not process formula: {f}, due to {e}")

        # Reconstruct MultiIndex columns
        work_df.columns = pd.MultiIndex.from_tuples(
            [(state.ticker, concept) for concept in work_df.columns],
            names=state.financial_data.index.names,
        )

        calculated_data = work_df.T
        db = service_manager.get_financial_database()
        db.save_calculated_metric(state.ticker, calculated_data)

        return {"financial_data": calculated_data, "formulas": []}

    def _analyst_node(self, state: _AgentState) -> dict[str, Any]:
        print(f"--- [Node] Analyst ---")
        llm = service_manager.get_agent(temperature=0.7)
        data_str = (
            (state.financial_data).to_string()
            if not state.financial_data.empty
            else "No data available."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior financial analyst. Provide a clear, concise analysis based on the provided data. "
                    "Explain the key trends and what they mean for the company.",
                ),
                (
                    "human",
                    f"Original Query: {state.messages[-1].content}\n\nData:\n{data_str}",
                ),
            ]
        )
        response = (prompt | llm).invoke({})
        return {"messages": [response]}


if __name__ == "__main__":
    # 1. Create an instance of the agent
    fundamental_agent = FundamentalAnalysisAgent()

    # 2. Define the structured input for the agent
    user_request = FundamentalAnalysisInput(
        ticker="META",
        metrics=["revenue", "net income", "free cash flow", "debt to asset ratio"],
        start_year=2020,
        end_year=2023,
        raw_input="Analyze revenue, earnings and free cash flow, and debt to asset ratio for META for last 3 years.",
    )

    # 3. Execute the agent's run method
    print("Starting Agent...")
    final_output = fundamental_agent.run(user_request)

    # 4. Print the result
    print("\n" + "=" * 40)
    print(f"Agent '{final_output.agent_name}' completed its analysis.")
    print("Final Answer:")
    print(final_output.output)
