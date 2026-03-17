from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import pandas as pd
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from core.agents.models import BaseAgentInput, BaseAgentOutput, CitedSource
from core.memory.graph.models import RelationshipType
from core.memory.user_context_service import UserContext


class UserInterestEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity_name: str
    entity_type: str


class InvestmentSignalDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Literal["Bought", "Interested", "Sold", "Avoids"]
    target_entities: List[UserInterestEntity] = Field(default_factory=list)


class LearningSignalDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Literal["Interested", "Understood", "Confused", "Not Interested"]
    target_entities: List[UserInterestEntity] = Field(default_factory=list)


class OrchestratorPlan(BaseAgentInput):
    """
    The LLM planner's output.

    Inherits all agent-dispatch fields from BaseAgentInput
    (query, ticker, start_date, end_date, metrics, granularity, conversation_id)
    so those fields are declared exactly once and _execute_node can call
    plan.model_copy(update={"query": agent_query}) directly — no dict-filtering
    hack required.

    Orchestrator-only fields are declared below.
    """

    # BaseAgentInput.query has no default, but the planner must be able to
    # produce a plan even for trivial/direct-answer cases where `query` might
    # be empty.  Override with a default here.
    query: str = Field(
        default="",
        description="The original (or lightly cleaned) user query, for downstream context.",
    )

    # ── Orchestrator-only fields ──────────────────────────────────────────────

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

    per_agent_queries: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "A rewritten query tailored to each target agent's job and retrieval strategy. "
            "Keys are agent names (e.g. 'news_agent', 'fundamentals_agent'); "
            "values are the rewritten query string.  Every agent listed in "
            "`target_agents` should have an entry here.  Falls back to `query` "
            "for any agent without an explicit entry."
        ),
    )

    detected_investment_signals: List[InvestmentSignalDetection] = Field(
        default_factory=list
    )
    detected_learning_signals: List[LearningSignalDetection] = Field(
        default_factory=list
    )


class OrchestratorState(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    messages: List[BaseMessage] = Field(default_factory=list)
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: Dict[str, BaseAgentOutput] = Field(default_factory=dict)
    final_response: Optional["FinalResponse"] = None

    conversation_id: Optional[str] = None
    user_email: Optional[str] = None
    user_context: Optional[UserContext] = None
    user_context_block: str = ""
    user_context_loaded: bool = False

    summary: str = ""
    fundamental_data: Optional[pd.DataFrame] = None
    sources: List[CitedSource] = Field(default_factory=list)


class FinalResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    summary: str = ""
    fundamental_data: Optional[pd.DataFrame] = Field(default=None, exclude=True)
    sources: List[CitedSource] = Field(default_factory=list)
    agent_analyses: Dict[str, str] = Field(default_factory=dict)


class CrossDomainRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")
    from_name: str
    from_type: Literal["Company", "FinancialEvent", "FinancialConcept", "Sector"]
    relation: RelationshipType
    to_name: str
    to_type: Literal["Company", "FinancialEvent", "FinancialConcept", "Sector"]
    confidence: Literal["high", "low"]
    reason: str
    source_agent_from: Literal["news_agent", "fundamentals_agent"]
    source_agent_to: Literal["news_agent", "fundamentals_agent"]


# ──────────────────────────────────────────────────────────────────────────────
# SynthesisResult — returned by _parse_synthesis_output
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SynthesisResult:
    """Typed container for all blocks parsed from a single synthesis LLM call."""

    user_response: str
    cross_relationships: List[dict] = field(default_factory=list)
    interest_edges: List[dict] = field(default_factory=list)
