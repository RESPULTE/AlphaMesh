from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.agents.models.base_agent_models import BaseAgentInput, BaseAgentOutput
from core.memory.graph.models import EntityNode
from core.memory.retrieval.models import CitedSource, RetrievedChunk


class DomainQuery(BaseModel):
    """A single domain-specific retrieval string."""

    model_config = ConfigDict(extra="ignore")

    domain: Literal["company", "sector", "market", "knowledge"] = Field(
        description=(
            "The retrieval scope this query targets:\n"
            "  company   - narrow, ticker/company-focused\n"
            "  sector    - industry/sector-wide context\n"
            "  market    - macro/systemic factors\n"
            "  knowledge - definitions or general financial concepts"
        )
    )
    query: str = Field(
        description="A fully self-contained retrieval string for this domain."
    )


class PlannerDecision(BaseModel):
    """Fetch-only planner output for one research iteration."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["newsapi", "web_search"] = Field(default="newsapi")
    queries: List[DomainQuery] = Field(default_factory=list)


class ResearchStepLog(BaseModel):
    """Execution log for one research iteration."""

    model_config = ConfigDict(extra="ignore")

    iteration: int
    action: Literal["newsapi", "web_search"]
    queries: List[DomainQuery] = Field(default_factory=list)
    total_fetched_articles: int = 0
    newly_fetched_articles: int = 0
    merged_chunk_count: int = 0


class NewsAgentState(BaseAgentInput):
    """State container for the news analysis agent."""

    model_config = ConfigDict(extra="ignore")

    planner_decision: Optional[PlannerDecision] = None
    research_logs: Annotated[List[ResearchStepLog], operator.add] = Field(
        default_factory=list
    )
    seen_urls: Annotated[List[str], operator.add] = Field(default_factory=list)
    research_iteration: int = 0

    is_context_sufficient: bool = False
    missing_information_goal: Optional[str] = None
    persist_chunk_ids: List[str] = Field(default_factory=list)
    grouped_query_context_block: str = ""
    working_memory_context_block: str = ""

    retrieved_chunks: Annotated[List[RetrievedChunk], operator.add] = Field(
        default_factory=list
    )
    memory_chunks: Annotated[List[RetrievedChunk], operator.add] = Field(
        default_factory=list
    )
    final_chunks: List[RetrievedChunk] = Field(default_factory=list)

    sources: List[CitedSource] = Field(default_factory=list)


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
            f"[{s.source_id}] {s.title} - {s.url}" for s in self.sources
        )
        return f"[news_agent]\n{self.analysis}\n\nSources:\n{sources_block}"
