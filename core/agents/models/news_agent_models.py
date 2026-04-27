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


# ---------------------------------------------------------------------------
# Query rewriting models
# ---------------------------------------------------------------------------


class DomainQuery(BaseModel):
    """A single domain-specific retrieval string produced by the query rewriter."""

    model_config = ConfigDict(extra="ignore")

    domain: Literal["company", "sector", "market", "knowledge"] = Field(
        description=(
            "The retrieval scope this query targets:\n"
            "  company   — narrow, ticker/company-focused\n"
            "  sector    — industry/sector-wide context\n"
            "  market    — macro/systemic factors\n"
            "  knowledge — definitions or general financial concepts"
        )
    )
    query: str = Field(
        description="A fully self-contained retrieval string for this domain."
    )


class QueryRewritePlan(BaseModel):
    """Output of the query-rewrite node: a list of domain-specific search strings."""

    model_config = ConfigDict(extra="ignore")

    queries: List[DomainQuery] = Field(
        description=(
            "Domain-specific queries to execute in parallel this iteration. "
            "At least one entry is required. Always include a 'company' entry "
            "when a ticker is present."
        )
    )
    rationale: str = Field(
        default="",
        description="Brief explanation of the chosen domains and query formulations.",
    )


# ---------------------------------------------------------------------------
# Planner model
# ---------------------------------------------------------------------------


class ResearchStepPlan(BaseModel):
    """Planner decision for one research iteration.

    The planner's sole responsibility is to assess information sufficiency
    and select the online fetch tool. Query rewriting is handled separately
    by the query-rewrite node.
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
            "Base search query handed to the query-rewrite node as a starting point. "
            "Should capture the core information need. Empty only when action='proceed'."
        ),
    )
    rationale: str = Field(
        default="",
        description="Short reason for the action chosen (1-2 sentences).",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum result count per query for the selected online tool.",
    )

    # Tavily domain scoping — only relevant when action="web_search".
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="Restrict Tavily results to these domains.",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Exclude these domains from Tavily results.",
    )


# ---------------------------------------------------------------------------
# Remaining state / output models
# ---------------------------------------------------------------------------


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
    rewrite_plan: Optional[QueryRewritePlan] = None
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
