from __future__ import annotations

from datetime import date
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.agents.models.base_agent_models import BaseAgentOutput
from core.memory.graph.models import EntityNode
from core.memory.retrieval.models import RetrievedChunk


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
    company_context: Optional[str] = Field(default=None)


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
