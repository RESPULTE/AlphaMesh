from __future__ import annotations

from typing import List, Literal, Optional

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


class RelevantChunkSelection(BaseModel):
    """Planner-selected relevant chunk for analysis."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    reason: str = ""


class PlannerDecision(BaseModel):
    """Single planner output for one research iteration."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["newsapi", "web_search", "proceed"] = Field(default="proceed")
    proceed_to_analysis: bool = Field(default=False)
    queries: List[DomainQuery] = Field(default_factory=list)
    rationale: str = Field(default="")
    max_results: int = Field(default=5, ge=1, le=20)
    relevant_chunks: List[RelevantChunkSelection] = Field(default_factory=list)


class ResearchStepLog(BaseModel):
    """Execution log for one research iteration."""

    model_config = ConfigDict(extra="ignore")

    iteration: int
    action: Literal["newsapi", "web_search", "proceed", "none"]
    query: str = ""
    queries: List[DomainQuery] = Field(default_factory=list)
    rationale: str = ""
    total_fetched_articles: int = 0
    newly_fetched_articles: int = 0
    relevant_chunk_count: int = 0
    relevant_source_count: int = 0
    score_unavailable: bool = False
    no_relevant_note: str = ""

    @staticmethod
    def _history_block(logs: List[ResearchStepLog], limit: int = 6) -> str:
        if not logs:
            return "(none)"
        lines: List[str] = []
        for row in logs[-limit:]:
            query_lines = (
                ", ".join(f"{q.domain}:{q.query}" for q in row.queries)
                or row.query
                or "(none)"
            )
            lines.append(
                f"Iteration {row.iteration}\n"
                f"  action: {row.action}\n"
                f"  queries: {query_lines}\n"
                f"  Total fetched articles: {row.total_fetched_articles}\n"
                f"  newly fetched articles: {row.newly_fetched_articles}\n"
                f"  relevant chunks: {row.relevant_chunk_count}\n"
                f"  relevant sources: {row.relevant_source_count}\n"
                f"  score unavailable: {row.score_unavailable}\n"
                f"  note: {row.no_relevant_note or '(none)'}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _list_unique_actions(logs: List[ResearchStepLog]) -> List[str]:
        actions: List[str] = []
        for row in logs:
            action = str(getattr(row, "action", "") or "").strip()
            if not action or action in {"proceed", "none"} or action in actions:
                continue
            actions.append(action)
        return actions


class NewsAgentState(BaseAgentInput):
    """State container for the news analysis agent."""

    model_config = ConfigDict(extra="ignore")

    planner_decision: Optional[PlannerDecision] = None
    research_logs: List[ResearchStepLog] = Field(default_factory=list)
    seen_urls: List[str] = Field(default_factory=list)
    research_iteration: int = 0
    is_information_sufficient: bool = False

    # Rendezvous status exposed to planner.
    rendezvous_has_minimum_sources: bool = False
    rendezvous_score_unavailable: bool = False
    rendezvous_relevant_chunk_count: int = 0
    rendezvous_relevant_source_count: int = 0

    # Chunks accumulated from the online fetch+ingest branch across all iterations.
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    # Chunks returned by retrieve_memory for the current iteration.
    memory_chunks: List[RetrievedChunk] = Field(default_factory=list)
    # Definitive chunks considered by planner for this iteration.
    final_chunks: List[RetrievedChunk] = Field(default_factory=list)

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
            f"[{s.source_id}] {s.title} - {s.url}" for s in self.sources
        )
        return f"[news_agent]\n{self.analysis}\n\nSources:\n{sources_block}"
