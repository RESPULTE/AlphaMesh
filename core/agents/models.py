from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.memory.graph.models import EntityNode
from core.memory.retrieval.models import RetrievedChunk


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
    start_date: Optional[datetime] = Field(
        default=None, description="Start date (format: YYYY-MM-DD)."
    )
    end_date: Optional[datetime] = Field(
        default=None, description="End date (format: YYYY-MM-DD)."
    )

    # ── Internal / excluded from serialisation ────────────────────────────────
    conversation_id: Optional[str] = Field(default=None, exclude=True)

    granularity: Optional[Literal["yearly", "quarterly"]] = Field(
        default="yearly",
        description=(
            "Data granularity for the fundamental agent. "
            "'yearly' fetches 10-K annual filings over a 5-year window (default). "
            "'quarterly' fetches 10-Q filings — the orchestrator should set this "
            "when the user explicitly asks for quarterly trends or TTM figures."
        ),
    )

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
    # subgraph_task removed: SubgraphExtractionService.schedule() manages its
    # own background task internally and returns only the subgraph_id string.
    subgraph_id: Optional[str] = Field(default=None, exclude=True)
    relationships_extracted: bool = Field(default=False)

    @abstractmethod
    def get_llm_context_str(self) -> str:
        """Formats the output into a string suitable for LLM context."""
        raise NotImplementedError


class CitedSource(BaseModel):
    """Citable source metadata for news analysis output."""

    model_config = ConfigDict(extra="ignore")

    source_id: int = Field(description="The numeric ID used in the text, e.g., 1.")
    title: str = Field(description="The title of the article.")
    url: str = Field(description="The URL of the article.")
    page_content: str = Field(description="The content of the article.")


class NewsAgentState(BaseModel):
    """State container for the news analysis agent."""

    model_config = ConfigDict(extra="ignore")

    # ── Inputs set by the agent's run() method ────────────────────────────────
    query: str
    ticker: str
    # date (not datetime): _constrain_date_range returns date objects.
    # These are used only as ISO strings in fetch_articles.
    start_date: date
    end_date: date
    conversation_id: Optional[str] = Field(default=None)

    # ── Internal pipeline state ───────────────────────────────────────────────
    # memory_task is created in _rewrite_queries_node and awaited in
    # _rendezvous_node. Excluded from Pydantic serialisation; carried through
    # the LangGraph state dict directly.
    memory_task: Optional[Any] = Field(default=None, exclude=True)

    raw_articles: List[dict] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    final_chunks: List[RetrievedChunk] = Field(default_factory=list)

    # ── Output fields ─────────────────────────────────────────────────────────
    analysis: Optional[str] = None
    sources: List[CitedSource] = Field(default_factory=list)
    entities_enriched: List[EntityNode] = Field(default_factory=list)


class NewsAgentOutput(BaseAgentOutput):
    """Output schema for the news analysis agent."""

    model_config = ConfigDict(extra="ignore")

    agent_name: str = Field(default="news_agent")
    sources: List[CitedSource] = Field(default_factory=list)
    entities_enriched: List[EntityNode] = Field(default_factory=list)

    def get_llm_context_str(self) -> str:
        """Return analysis and sources formatted for LLM context."""
        if not self.sources:
            return f"[news_agent]\n{self.analysis}"
        sources_block = "\n".join(
            f"[{s.source_id}] {s.title} — {s.url}" for s in self.sources
        )
        return f"[news_agent]\n{self.analysis}\n\nSources:\n{sources_block}"
