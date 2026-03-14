"""Models for the dual-store retriever."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class NodeSelectionOutput(BaseModel):
    """Structured output for selecting nodes to expand."""

    selected_entity_ids: List[str] = Field(default_factory=list)


class RetrieverState(TypedDict):
    """LangGraph state for the dual-store retriever."""

    query: str
    accumulated_chunks: List[RetrievedChunk]
    visited_entity_ids: List[str]
    visited_chunk_ids: List[str]
    current_frontier: List[str]
    candidate_neighbors: List[dict]
    iteration: int
    should_continue: bool


class RewrittenQueries(BaseModel):
    company_query: Optional[str] = None
    sector_query: Optional[str] = None
    market_query: Optional[str] = None
    knowledge_query: Optional[str] = None
    active_domains: List[Literal["company", "sector", "market", "knowledge"]] = Field(
        default_factory=list
    )


class RetrievedChunk(BaseModel):
    """Represents a retrieved chunk from either vector or graph store."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    text: str
    score: Optional[float] = None
    source: Literal["vector", "graph"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryChunk(RetrievedChunk):
    """RetrievedChunk enriched with domain context and composite scoring."""

    domain: str
    embedding_score: float = 0.0
    graph_depth: int = 0
    composite_score: float = 0.0

    @classmethod
    def from_retrieved(cls, chunk: RetrievedChunk, domain: str) -> "MemoryChunk":
        return cls(
            **chunk.model_dump(),
            domain=domain,
            embedding_score=chunk.score or 0.0,
            graph_depth=0 if chunk.source == "vector" else 1,
        )


class MemoryContext(BaseModel):
    chunks: List[MemoryChunk]
    rewritten_queries: RewrittenQueries
