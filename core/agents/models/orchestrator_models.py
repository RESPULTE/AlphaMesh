from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from core.agents.models.base_agent_models import BaseAgentInput, BaseAgentOutput
from core.agents.models.fundamental_agent_models import VisualizationPlan
from core.agents.models.news_agent_models import CitedSource


class UserInterestEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity_name: str
    entity_type: Literal[
        "Company",
        "FinancialEvent",
        "FinancialConcept",
        "Sector",
    ]


class InvestmentSignalDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Literal["Bought", "Interested", "Sold", "Avoids"]
    target_entities: List[UserInterestEntity] = Field(default_factory=list)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Inference certainty (0.0â€“1.0). "
            "1.0 = explicit stance verb; 0.7â€“0.9 = strong implicit; "
            "0.4â€“0.6 = inferred; omit signal entirely if below 0.4."
        ),
    )


class LearningSignalDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Literal["Interested", "Understood", "Confused", "Not Interested"]
    target_entities: List[UserInterestEntity] = Field(default_factory=list)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Inference certainty (0.0â€“1.0). "
            "1.0 = explicit learning request; 0.7â€“0.9 = strong implicit; "
            "0.4â€“0.6 = inferred; omit signal entirely if below 0.4."
        ),
    )


class OrchestratorPlan(BaseAgentInput):
    """
    The LLM planner's output.

    Inherits all agent-dispatch fields from BaseAgentInput
    (query, ticker, start_date, end_date, metrics, granularity, conversation_id)
    so those fields are declared exactly once and _execute_node can call
    plan.model_copy(update={"query": agent_query}) directly â€” no dict-filtering
    hack required.

    Orchestrator-only fields are declared below.
    """

    # BaseAgentInput.query has no default, but the planner must be able to
    # produce a plan even for trivial/direct-answer cases where `query` might
    # be empty.  Override with a default here.
    query: str = Field(
        default="",
        description="The original (or lightly cleaned) user query, for orchestrator continuity.",
    )
    goal: str = Field(
        default="",
        description=(
            "Reserved orchestrator-level goal field. Per-agent goals should be set via "
            "`per_agent_goals`."
        ),
    )

    # â”€â”€ Orchestrator-only fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    ticker: Optional[str] = Field(
        default=None,
        description="Derived from tickers[0]; do not populate directly.",
    )

    # Authoritative list of all tickers identified in the query (up to 3).
    tickers: List[str] = Field(
        default_factory=list,
        description=(
            "All ticker symbols identified in the query, up to 3. "
            "Always populate this list instead of the legacy `ticker` field."
        ),
    )

    final_answer: Optional[str] = Field(
        default=None,
        description=(
            "Set ONLY when the question can be answered directly without any agent, "
            "memory, or financial data."
        ),
    )
    needs_memory: bool = Field(default=False)
    target_agents: List[str] = Field(default_factory=list)
    target_entities: List[str] = Field(default_factory=list)

    per_agent_goals: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "A plain-text execution goal tailored to each target agent's job. "
            "Keys are agent names (e.g. 'news_agent', 'fundamentals_agent'); "
            "values are goal strings. Every agent listed in "
            "`target_agents` should have an entry here.  Falls back to `query` "
            "for any agent without an explicit entry."
        ),
    )
    per_agent_queries: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Deprecated legacy field. Kept for backward compatibility only; "
            "sub-agent dispatch should use `per_agent_goals`."
        ),
    )

    detected_investment_signals: List[InvestmentSignalDetection] = Field(
        default_factory=list
    )
    detected_learning_signals: List[LearningSignalDetection] = Field(
        default_factory=list
    )


class OrchestratorState(BaseModel):
    """
    LangGraph state for the orchestrator pipeline.

    `user_context_block` is populated once â€” synchronously from the
    UserContextService in-memory cache â€” at the start of `run()`, before the
    graph is invoked.  There is no `load_context` node: context loading is
    handled externally (on session start) and the cache read is O(1).
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    messages: List[BaseMessage] = Field(default_factory=list)
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: Dict[str, BaseAgentOutput] = Field(default_factory=dict)
    final_response: Optional["FinalResponse"] = None

    conversation_id: Optional[str] = None
    user_email: Optional[str] = None
    history_turns: List[dict] = Field(default_factory=list)
    agent_memory_summaries: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Populated synchronously in run() from the UserContextService cache.
    user_context_block: str = ""
    # Portfolio snapshot loaded once in run() from per-user JSON holdings.
    portfolio_block: str = "[]"

    # Keyed by ticker symbol; values are formatted context blocks from yfinance.
    # Built by _validate_and_enrich_node and read by _execute_node.
    company_context_blocks: Dict[str, str] = Field(default_factory=dict)

    summary: str = ""
    fundamental_data: Optional[pd.DataFrame] = Field(default=None, exclude=True)
    fundamentals_visualization: Optional[VisualizationPlan] = Field(default=None)
    fundamentals_raw_display_data: Optional[pd.DataFrame] = Field(
        default=None, exclude=True
    )
    fundamentals_task_completed: bool = Field(default=True)
    fundamentals_task_completion_reason: str = Field(default="")
    sources: List[CitedSource] = Field(default_factory=list)

    turn_id: str = Field(
        default="",
        description="UUID generated once per run() call for turn-level provenance.",
    )
    ticker_metadata: Dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "Per-ticker enrichment from yfinance, keyed by ticker symbol. "
            "Values: {long_name, sector, industry, description}. "
            "Populated by _validate_and_enrich_node; current turn only."
        ),
    )


class FinalResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    summary: str = ""
    fundamental_data: Optional[pd.DataFrame] = Field(default=None, exclude=True)
    fundamentals_visualization: Optional[VisualizationPlan] = Field(default=None)
    fundamentals_raw_display_data: Optional[pd.DataFrame] = Field(
        default=None, exclude=True
    )
    fundamentals_task_completed: bool = Field(default=True)
    fundamentals_task_completion_reason: str = Field(default="")
    sources: List[CitedSource] = Field(default_factory=list)
    agent_analyses: Dict[str, str] = Field(default_factory=dict)
    agent_memory_summaries: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tickers: List[str] = Field(
        default_factory=list,
        description="Ticker symbols processed in this turn (populated by orchestrator).",
    )
    turn_id: str = Field(
        default="",
        description="Orchestrator-generated turn UUID used across sub-agents.",
    )
