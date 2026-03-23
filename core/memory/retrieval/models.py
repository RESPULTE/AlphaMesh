"""Models for the dual-store retriever."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


class NodeSelectionOutput(BaseModel):
    """Structured output for selecting nodes to expand."""

    selected_entity_ids: List[str] = Field(default_factory=list)


class RetrieverState(TypedDict):
    """LangGraph state for the dual-store retriever.

    candidate_neighbors was intentionally removed: it was written by
    select_neighbor_frontier but never read by any downstream node.
    Keeping unread fields in LangGraph state causes unnecessary serialisation
    overhead and misleads readers about what data is actually consumed.
    """

    query: str
    accumulated_chunks: List["RetrievedChunk"]
    visited_entity_ids: List[str]
    visited_chunk_ids: List[str]
    current_frontier: List[str]
    iteration: int
    should_continue: bool


class RewrittenQueries(BaseModel):
    # Domain-specific retrieval strings produced by the query-rewrite LLM call.
    company_query: Optional[str] = None
    sector_query: Optional[str] = None
    market_query: Optional[str] = None
    knowledge_query: Optional[str] = None
    active_domains: List[Literal["company", "sector", "market", "knowledge"]] = Field(
        default_factory=list
    )
    # Set programmatically after the LLM call — not produced by the LLM.
    # Carries the agent-level query (already orchestrator-rewritten) for use
    # as the Jina reranker's query string in _rendezvous_node.
    original_query: Optional[str] = None


class RetrievedChunk(BaseModel):
    """Represents a retrieved chunk from either vector or graph store."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    text: str
    score: Optional[float] = None
    source: Literal["vector", "graph"]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    article_title: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[datetime] = None
    nodeset_ids: List[str] = Field(default_factory=list)
    extraction_status: Literal["PENDING", "EXTRACTED"] = "PENDING"
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
        nodeset_ids = metadata.get("nodeset_ids") or []
        if isinstance(nodeset_ids, str):
            nodeset_ids = [nodeset_ids]
        return cls(
            chunk_id=chunk_id,
            text=document.page_content,
            score=score,
            source=source,
            metadata=metadata,
            domain=domain,
            document_id=metadata.get("document_id"),
            chunk_index=metadata.get("chunk_index"),
            article_title=metadata.get("article_title"),
            source_url=metadata.get("source_url"),
            published_at=metadata.get("published_at"),
            nodeset_ids=nodeset_ids,
            extraction_status=metadata.get("extraction_status", "PENDING"),
        )

    @classmethod
    def normalize_for_reranking(
        cls,
        chunk: "RetrievedChunk",
        domain: str,
    ) -> "RetrievedChunk":
        """
        Return a copy of *chunk* with reranking fields properly populated.

        For vector chunks: embedding_score is taken from chunk.score.
        For graph chunks:  embedding_score is 0.0 (no similarity score available);
                           graph_depth is preserved from the chunk if already set by
                           the traversal (hop-aware), otherwise defaults to 1.

        Previously named from_raw_chunk.  Renamed because the input is always
        a fully-typed RetrievedChunk, not a raw/untyped source.
        """
        if chunk.source == "graph":
            embedding_score = 0.0
            # Preserve the hop depth set by _fetch_frontier_chunks_node so the
            # reranker's depth_bonus reflects actual traversal distance.
            graph_depth = chunk.graph_depth if chunk.graph_depth > 0 else 1
        else:
            embedding_score = float(chunk.score) if chunk.score is not None else 0.0
            graph_depth = 0

        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=chunk.score,
            source=chunk.source,
            metadata=chunk.metadata or {},
            domain=domain,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            article_title=chunk.article_title,
            source_url=chunk.source_url,
            published_at=chunk.published_at,
            nodeset_ids=chunk.nodeset_ids,
            extraction_status=chunk.extraction_status,
            embedding_score=embedding_score,
            graph_depth=graph_depth,
            composite_score=chunk.composite_score,
        )

    # Keep the old name as a deprecated alias so any external callers that
    # haven't been updated yet continue to work without modification.
    from_raw_chunk = normalize_for_reranking


class MemoryContext(BaseModel):
    chunks: List[RetrievedChunk]
    rewritten_queries: RewrittenQueries


@dataclass
class NeighborCandidate:
    """
    Typed representation of one row returned by Neo4jAdapter.get_entity_neighbors().

    Confines string-key access to _parse_neighbor() so node logic works with
    typed attributes instead of raw dict lookups.
    """

    source_entity_id: str
    neighbor_entity_id: str
    neighbor_name: str
    neighbor_type: str
    relationship_type: str


@dataclass
class GraphChunkRow:
    """
    Typed representation of one row returned by Neo4jAdapter.get_chunks_for_entities().

    extraction_status is included so PENDING vs EXTRACTED state flows correctly
    into RetrievedChunk objects and avoids spurious re-extraction queuing.
    """

    chunk_id: str
    chunk_text: str
    chunk_index: Optional[int] = None
    document_id: Optional[str] = None
    article_title: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[Any] = None
    extraction_status: str = "PENDING"
