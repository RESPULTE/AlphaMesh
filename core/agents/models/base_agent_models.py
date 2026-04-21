from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentSentiment(BaseModel):
    """Structured sentiment produced by the LLM at the end of each agent's analysis."""

    model_config = ConfigDict(extra="ignore")

    score: int = Field(
        default=50,
        ge=0,
        le=100,
        description="0 = maximally bearish, 50 = neutral, 100 = maximally bullish.",
    )
    label: str = Field(
        default="NEUTRAL",
        description="One of: STRONG BUY | BUY | NEUTRAL | SELL | STRONG SELL",
    )
    rationale: str = Field(
        default="",
        description="1-2 sentence justification grounded in the retrieved evidence.",
    )


class BaseAgentInput(BaseModel):
    """
    The unified input schema shared by the Orchestrator and all Sub-Agents.

    """

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        description=(
            "The query for this agent, already rewritten by the orchestrator to be "
            "optimal for this agent's job.  Used for both LLM context and vector retrieval."
        )
    )
    ticker: Optional[str] = Field(
        default=None, description="The stock ticker symbol (e.g., AAPL)."
    )
    metrics: Optional[List[str]] = Field(
        default_factory=list,
        description="List of financial metrics to analyze (if applicable).",
    )
    start_date: Optional[date] = Field(
        default=None, description="Start date (format: YYYY-MM-DD)."
    )
    end_date: Optional[date] = Field(
        default=None, description="End date (format: YYYY-MM-DD)."
    )

    # ── Internal (still serialized into agent state) ──────────────────────────
    # NOTE: Do not exclude from serialization; LangGraph initial state is built
    # from model_dump() and downstream agents need this for graph queue writes.
    conversation_id: Optional[str] = Field(default=None)

    sentiment: Optional[AgentSentiment] = Field(default=None)

    granularity: Optional[Literal["yearly", "quarterly"]] = Field(
        default="yearly",
        description=(
            "Data granularity for the fundamental agent. "
            "'yearly' fetches 10-K annual filings over a 5-year window (default). "
            "'quarterly' fetches 10-Q filings — the orchestrator should set this "
            "when the user explicitly asks for quarterly trends or TTM figures."
        ),
    )

    company_context: Optional[str] = Field(default=None)

    @field_validator("granularity", mode="before")
    @classmethod
    def _default_granularity(cls, v: Any) -> str:
        return v if v in ("yearly", "quarterly") else "yearly"


class BaseAgentOutput(BaseModel, ABC):
    """
    An abstract base class for agent outputs. It enforces that each output
    type must know how to format itself into a string for the final LLM analyst.
    """

    model_config = ConfigDict(extra="ignore")

    agent_name: str = Field(
        description="The name of the agent that produced this output."
    )
    analysis: str = Field(
        description="The detailed analysis or primary output of the agent."
    )
    entities_enriched: List[Any] = Field(
        default_factory=list,
        description=(
            "List of enriched DataPoint objects resolved by this agent. "
            "Must use classes from core.memory.graph.models. "
            "Populated by each agent's final node before returning."
        ),
    )
    sentiment: Optional[AgentSentiment] = Field(default=None)
    subgraph_id: Optional[str] = Field(default=None, exclude=True)
    relationships_extracted: bool = Field(default=False)

    @abstractmethod
    def get_llm_context_str(self) -> str:
        """Formats the output into a string suitable for LLM context."""
        raise NotImplementedError
