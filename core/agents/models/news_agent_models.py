from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.agents.models.base_agent_models import BaseAgentInput, BaseAgentOutput
from core.memory.graph.models import EntityNode
from core.memory.retrieval.models import RetrievedChunk


class CitedSource(BaseModel):
    """Citable source metadata for news analysis output."""

    model_config = ConfigDict(extra="ignore")

    source_id: int = Field(description="The numeric ID used in the text, e.g., 1.")
    title: str = Field(description="The title of the article.")
    url: str = Field(description="The URL of the article.")
    page_content: str = Field(description="The content of the article.")


class ResearchStepPlan(BaseModel):
    """Planner decision for one research iteration."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["newsapi", "web_search", "proceed"] = Field(
        default="proceed",
        description="Research action to take in this iteration.",
    )
    query: str = Field(
        default="",
        description="Tool query to execute for this iteration.",
    )
    rationale: str = Field(
        default="",
        description="Short reason for the selected action.",
    )
    include_domains: List[str] = Field(
        default_factory=list,
        description="Domain allow-list for web search tools (if any).",
    )
    exclude_domains: List[str] = Field(
        default_factory=list,
        description="Domain block-list for web search tools (if any).",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum result count for the selected tool call.",
    )


class ResearchStepLog(BaseModel):
    """Execution log for one research iteration."""

    model_config = ConfigDict(extra="ignore")

    iteration: int
    action: Literal["newsapi", "web_search", "proceed", "none"]
    query: str = ""
    rationale: str = ""
    fetched_articles: int = 0
    newly_added_articles: int = 0


class NewsAgentState(BaseAgentInput):
    """State container for the news analysis agent."""

    model_config = ConfigDict(extra="ignore")

    # ── Internal pipeline state ───────────────────────────────────────────────
    # memory_task is created in _rewrite_queries_node and awaited in
    # _rendezvous_node. Excluded from Pydantic serialisation; carried through
    # the LangGraph state dict directly.
    memory_task: Optional[Any] = Field(default=None, exclude=True)
    research_plan: Optional[ResearchStepPlan] = None
    research_logs: List[ResearchStepLog] = Field(default_factory=list)
    latest_articles: List[dict] = Field(default_factory=list)
    seen_urls: List[str] = Field(default_factory=list)
    research_iteration: int = 0
    max_research_iterations: int = 3
    is_information_sufficient: bool = False

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
