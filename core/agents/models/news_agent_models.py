from __future__ import annotations

from typing import List, Literal, Optional

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
    """Unified planner decision for one research iteration.

    The planner has two responsibilities:
    1. Rewrite the user query into domain-specific retrieval strings.
    2. Decide whether to fetch more data or proceed to analysis.

    When action != "proceed", BOTH the online fetch branch and the semantic
    memory branch are always triggered. The planner only controls *what* to
    search, not *whether* to search.
    """

    model_config = ConfigDict(extra="ignore")

    action: Literal["newsapi", "web_search", "proceed"] = Field(
        default="proceed",
        description=(
            "Online fetch tool to use, or 'proceed' to skip all fetching "
            "and go directly to analysis."
        ),
    )
    query: str = Field(
        default="",
        description=(
            "Primary search query for the selected online tool. "
            "Should be a fully self-contained, domain-rewritten query. "
            "Empty only when action='proceed'."
        ),
    )
    rationale: str = Field(
        default="",
        description="Short reason for the decisions made in this plan.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum result count for the selected online tool call.",
    )

    # -- Domain-specific memory retrieval queries --------------------------------
    # All four are produced each iteration when action != "proceed".
    # Leave a field None only when that domain is genuinely irrelevant.
    company_query: Optional[str] = Field(
        default=None,
        description="Narrow retrieval string targeting the specific company/ticker.",
    )
    sector_query: Optional[str] = Field(
        default=None,
        description="Broadened retrieval string targeting the company's sector/industry.",
    )
    market_query: Optional[str] = Field(
        default=None,
        description="Macro/market-wide retrieval string for systemic context.",
    )
    knowledge_query: Optional[str] = Field(
        default=None,
        description="General knowledge retrieval string for definitions or concepts.",
    )

    # -- Tavily domain scoping (only relevant when action="web_search") ----------
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="Restrict Tavily results to these domains.",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Exclude these domains from Tavily results.",
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
    research_plan: Optional[ResearchStepPlan] = None
    research_logs: List[ResearchStepLog] = Field(default_factory=list)
    seen_urls: List[str] = Field(default_factory=list)
    research_iteration: int = 0
    max_research_iterations: int = 3
    is_information_sufficient: bool = False

    # Chunks accumulated from the online fetch+ingest branch across all iterations.
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    # Chunks returned by retrieve_memory for the current iteration.
    # Reset to [] by rendezvous after merging into final_chunks.
    memory_chunks: List[RetrievedChunk] = Field(default_factory=list)
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
