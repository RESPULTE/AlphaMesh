from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.memory.graph.models import ChunkExtractionResult, EntityNode
from core.memory.retrieval.models import RetrievedChunk


class BaseAgentInput(BaseModel):
    """
    The unified input schema shared by the Orchestrator and all Sub-Agents.
    """

    model_config = ConfigDict(extra="ignore")

    query: str = Field(description="The original user query for context.")
    vector_query: str = Field(
        description="The query optimized for vector store retrieval (e.g., 'news about AAPL stock price drop')."
    )
    ticker: Optional[str] = Field(description="The stock ticker symbol (e.g., AAPL).")
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
    memory_task: Optional[Any] = Field(default=None, exclude=True)

    # @field_validator("start_date", "end_date", mode="before")
    # def parse_dates(cls, v):
    #     if v is None:
    #         return None
    #     if isinstance(v, datetime):
    #         return v
    #     if isinstance(v, str):
    #         try:
    #             return datetime.strptime(v, "%Y-%m-%d")
    #         except ValueError:
    #             return datetime.fromisoformat(v)
    #     return v


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

    @abstractmethod
    def get_llm_context_str(self) -> str:
        """
        Formats the output's data into a string suitable for an LLM context.
        """
        raise NotImplementedError


class CitedSource(BaseModel):
    """Citable source metadata for news analysis output."""

    model_config = ConfigDict(extra="ignore")

    source_id: int = Field(description="The numeric ID used in the text, e.g., 1.")
    title: str = Field(description="The title of the article.")
    url: str = Field(description="The URL of the article.")
    page_content: str = Field(description="The content of the article.")


class NewsAgentState(BaseModel):
    """State container for the refactored news analysis agent."""

    model_config = ConfigDict(extra="ignore")

    query: str
    ticker: str
    start_date: datetime
    end_date: datetime
    raw_articles: List[dict] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    extraction_results: List[ChunkExtractionResult] = Field(default_factory=list)
    memory_task: Optional[Any] = Field(default=None, exclude=True)

    final_chunks: List[RetrievedChunk] = Field(default_factory=list)
    analysis: Optional[str] = None
    sources: List[CitedSource] = Field(default_factory=list)
    entities_enriched: List[EntityNode] = Field(default_factory=list)


class NewsAgentOutput(BaseAgentOutput):
    """Output schema for the refactored news analysis agent."""

    model_config = ConfigDict(extra="ignore")

    agent_name: str = Field(default="news_agent")
    sources: List[CitedSource] = Field(default_factory=list)
    entities_enriched: List[EntityNode] = Field(default_factory=list)

    def get_llm_context_str(self) -> str:
        """Return analysis and sources formatted for LLM context."""
        header = "### REPORT FROM news_agent\n"
        if not self.sources:
            return f"{header}{self.analysis}"

        sources_block = "\n".join(
            [
                f"[{s.source_id}] {s.title}\n{getattr(s, 'url', '')}".strip()
                for s in self.sources
            ]
        )
        return f"{header}{self.analysis}\n\n### SOURCES\n{sources_block}"


