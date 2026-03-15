"""Models for the dual-store retriever."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document

from pydantic import BaseModel, ConfigDict, Field


class NodeSelectionOutput(BaseModel):
    """Structured output for selecting nodes to expand."""

    selected_entity_ids: List[str] = Field(default_factory=list)


class RetrieverState(TypedDict):
    """LangGraph state for the dual-store retriever."""

    query: str
    accumulated_chunks: List["RetrievedChunk"]
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

    # Enrichment fields used during ranking/analysis.
    domain: Optional[str] = None
    embedding_score: float = 0.0
    graph_depth: int = 0
    composite_score: float = 0.0

    @classmethod
    def from_document(
        cls,
        document: Document,
        *,
        score: Optional[float] = None,
        source: Literal["vector", "graph"] = "vector",
        domain: Optional[str] = None,
    ) -> "RetrievedChunk":
        """Build a RetrievedChunk from a LangChain Document."""
        metadata = document.metadata or {}
        chunk_id = document.id or metadata.get("chunk_id") or ""
        return cls(
            chunk_id=chunk_id,
            text=document.page_content,
            score=score,
            source=source,
            metadata=metadata,
            domain=domain,
        )

    @classmethod
    def from_raw_chunk(
        cls,
        chunk: "RetrievedChunk",
        domain: str,
    ) -> "RetrievedChunk":
        """Normalize a RetrievedChunk with ranking fields populated."""
        embedding_score = float(chunk.score) if chunk.score is not None else 0.0
        graph_depth = 0
        if chunk.source == "graph":
            embedding_score = 0.0
            graph_depth = 1
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=chunk.score,
            source=chunk.source,
            metadata=chunk.metadata or {},
            domain=domain,
            embedding_score=embedding_score,
            graph_depth=graph_depth,
            composite_score=chunk.composite_score,
        )

class MemoryContext(BaseModel):
    chunks: List[RetrievedChunk]
    rewritten_queries: RewrittenQueries



