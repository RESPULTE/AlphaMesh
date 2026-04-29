"""Models for the dual-store retriever."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from core.agents.utils import trim_text


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
    iteration: int
    should_continue: bool
    run_id: str
    parent_run_id: Optional[str]
    domain: str


class RewrittenQueries(BaseModel):
    company_query: Optional[str] = None
    sector_query: Optional[str] = None
    market_query: Optional[str] = None
    knowledge_query: Optional[str] = None
    active_domains: List[Literal["company", "sector", "market", "knowledge"]] = Field(
        default_factory=list
    )
    original_query: Optional[str] = None


class CitedSource(BaseModel):
    """Citable source metadata for news analysis output."""

    model_config = ConfigDict(extra="ignore")

    source_id: int = Field(description="The numeric ID used in the text, e.g., 1.")
    title: str = Field(description="The title of the article.")
    url: str = Field(description="The URL of the article.")
    page_content: str = Field(description="The content of the article.")


def _parse_published_at(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    elif hasattr(value, "to_native"):
        try:
            dt = value.to_native()
        except Exception:
            return None
    elif hasattr(value, "to_datetime"):
        try:
            dt = value.to_datetime()
        except Exception:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_date_tag(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.strftime("%d-%m-%Y")


@dataclass
class RetrievedChunk:
    """Canonical retrieved chunk shared by ingestion, retrieval, and agent nodes."""

    chunk_id: str
    text: str
    source: Literal["vector", "graph"]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Unified relevance surface used by the agent.
    relevance_score: Optional[float] = None
    relevance_source: Optional[Literal["jina", "tavily", "vector", "composite"]] = None

    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    article_title: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[datetime] = None
    date_tag: str = ""

    nodeset_ids: List[str] = field(default_factory=list)
    extraction_status: Literal["PENDING", "EXTRACTED"] = "PENDING"
    domain: Optional[str] = None

    # Composite prefilter internals.
    embedding_score: float = 0.0
    graph_depth: int = 0
    composite_score: float = 0.0

    def __post_init__(self) -> None:
        self.published_at = _parse_published_at(self.published_at)
        if not self.date_tag:
            self.date_tag = _format_date_tag(self.published_at)
        if isinstance(self.nodeset_ids, str):
            self.nodeset_ids = [self.nodeset_ids]
        if self.relevance_score is not None:
            self.relevance_score = float(self.relevance_score)

    def model_dump(self) -> Dict[str, Any]:
        """Compatibility helper for call sites previously using pydantic models."""
        return asdict(self)

    def model_copy(self, update: Optional[Dict[str, Any]] = None) -> "RetrievedChunk":
        """Compatibility helper for call sites previously using pydantic models."""
        if not update:
            return replace(self)
        return replace(self, **update)

    @classmethod
    def from_document(
        cls,
        document: Document,
        *,
        score: Optional[float] = None,
        source: Literal["vector", "graph"] = "vector",
        domain: Optional[str] = None,
        relevance_source: Optional[Literal["jina", "tavily", "vector", "composite"]] = None,
    ) -> "RetrievedChunk":
        metadata = document.metadata or {}
        chunk_id = document.id or metadata.get("chunk_id") or ""
        nodeset_ids = metadata.get("nodeset_ids") or []
        if isinstance(nodeset_ids, str):
            nodeset_ids = [nodeset_ids]
        published_at = metadata.get("published_at")
        parsed_published_at = _parse_published_at(published_at)

        chosen_relevance_source = relevance_source
        if chosen_relevance_source is None and score is not None:
            chosen_relevance_source = "vector"

        return cls(
            chunk_id=chunk_id,
            text=document.page_content,
            source=source,
            metadata=metadata,
            relevance_score=score,
            relevance_source=chosen_relevance_source,
            domain=domain,
            document_id=metadata.get("document_id"),
            chunk_index=metadata.get("chunk_index"),
            article_title=metadata.get("article_title"),
            source_url=metadata.get("source_url"),
            published_at=parsed_published_at,
            date_tag=_format_date_tag(parsed_published_at),
            nodeset_ids=nodeset_ids,
            extraction_status=metadata.get("extraction_status", "PENDING"),
        )

    @classmethod
    def normalize_for_reranking(
        cls,
        chunk: "RetrievedChunk",
        domain: str,
    ) -> "RetrievedChunk":
        if chunk.source == "graph":
            embedding_score = 0.0
            graph_depth = chunk.graph_depth if chunk.graph_depth > 0 else 1
        else:
            embedding_score = (
                float(chunk.relevance_score) if chunk.relevance_score is not None else 0.0
            )
            graph_depth = 0

        normalized = chunk.model_copy(
            update={
                "domain": domain,
                "embedding_score": embedding_score,
                "graph_depth": graph_depth,
            }
        )
        if normalized.relevance_source is None and normalized.relevance_score is not None:
            normalized.relevance_source = "vector"
        return normalized

    def _source_key(self) -> str:
        metadata = self.metadata or {}
        return str(metadata.get("source_url") or self.source_url or "").strip()

    @staticmethod
    def _chunks_block(chunks: List["RetrievedChunk"], *, limit: int = 12) -> str:
        if not chunks:
            return "(none)"
        lines: List[str] = []
        for chunk in chunks[:limit]:
            title = (
                chunk.article_title
                or (chunk.metadata or {}).get("article_title")
                or "Unknown"
            )
            url = chunk._source_key() or "no-url"
            relevance = chunk.relevance_score
            relevance_text = "N/A" if relevance is None else f"{relevance:.4f}"
            preview = (chunk.text or "").replace("\n", " ")
            preview = trim_text(preview, max_chars=160)
            lines.append(
                f"- chunk_id={chunk.chunk_id} | title={title} | date={chunk.date_tag or 'N/A'} | source={url} | "
                f"relevance_score={relevance_text}\n  text={preview}"
            )
        if len(chunks) > limit:
            lines.append(f"... and {len(chunks) - limit} more chunk(s)")
        return "\n".join(lines)

    @staticmethod
    def _build_deduplicated_sources(
        chunks: List["RetrievedChunk"],
    ) -> Tuple[List[CitedSource], Dict[int, int]]:
        article_map: Dict[Tuple[str, str], Tuple[int, List[str]]] = {}
        chunk_to_source_id: Dict[int, int] = {}
        next_id = 1

        for chunk_idx, chunk in enumerate(chunks):
            metadata = chunk.metadata or {}
            title = (
                metadata.get("article_title") or chunk.article_title or "Unknown Title"
            )
            url = metadata.get("source_url") or chunk.source_url or ""
            key = (title, url)

            if key not in article_map:
                article_map[key] = (next_id, [chunk.text])
                next_id += 1
            else:
                sid, texts = article_map[key]
                if chunk.text not in texts:
                    texts.append(chunk.text)

            chunk_to_source_id[chunk_idx] = article_map[key][0]

        sources: List[CitedSource] = []
        for (title, url), (source_id, texts) in article_map.items():
            sources.append(
                CitedSource(
                    source_id=source_id,
                    title=title,
                    url=url,
                    page_content="\n\n".join(texts),
                )
            )
        return sources, chunk_to_source_id

    @staticmethod
    def _build_chunk_alias_maps(
        chunks: List["RetrievedChunk"],
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        chunk_id_to_alias: Dict[str, str] = {}
        alias_to_chunk_id: Dict[str, str] = {}
        alias_index = 1
        for chunk in chunks:
            chunk_id = (chunk.chunk_id or "").strip()
            if not chunk_id or chunk_id in chunk_id_to_alias:
                continue
            alias = str(alias_index)
            alias_index += 1
            chunk_id_to_alias[chunk_id] = alias
            alias_to_chunk_id[alias] = chunk_id
        return chunk_id_to_alias, alias_to_chunk_id

    @staticmethod
    def _render_candidate_chunks(
        chunks: List["RetrievedChunk"],
        chunk_id_to_alias: Dict[str, str],
    ) -> str:
        if not chunks:
            return "(none)"
        lines: List[str] = []
        for chunk in chunks:
            chunk_id = (chunk.chunk_id or "").strip()
            normalized_chunk_id = chunk_id_to_alias.get(chunk_id, "?")
            relevance = chunk.relevance_score
            relevance_text = "N/A" if relevance is None else f"{relevance:.4f}"
            chunk_text = str(chunk.text or "").strip() or "(empty)"
            lines.append(
                f"- chunk_id={normalized_chunk_id} | date={chunk.date_tag or 'N/A'} | relevance_score={relevance_text}\n"
                f"  text={chunk_text}"
            )
        return "\n".join(lines)


class MemoryContext(BaseModel):
    chunks: List[RetrievedChunk]
    rewritten_queries: RewrittenQueries


@dataclass
class NeighborCandidate:
    """Typed representation of one row returned by Neo4jAdapter.get_entity_neighbors()."""

    source_entity_id: str
    neighbor_entity_id: str
    neighbor_name: str
    neighbor_type: str
    relationship_type: str


@dataclass
class GraphChunkRow:
    """Typed representation of one row returned by Neo4jAdapter.get_chunks_for_entities()."""

    chunk_id: str
    chunk_text: str
    chunk_index: Optional[int] = None
    document_id: Optional[str] = None
    article_title: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[Any] = None
    extraction_status: str = "PENDING"
    supporting_entity_id: Optional[str] = None
