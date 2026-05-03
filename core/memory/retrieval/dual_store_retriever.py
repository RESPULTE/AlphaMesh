"""
LangGraph-powered dual-store retriever: vector seed + iterative graph traversal.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from core.config import settings
from core.logger import get_logger
from core.memory.retrieval.models import (
    GraphChunkRow,
    MemoryContext,
    NeighborCandidate,
    RetrievedChunk,
    RetrieverState,
    RewrittenQueries,
)
from core.memory.retrieval.reranker import CompositePrefilter, TwoStageReranker
from core.memory.retrieval.tracing import (
    NetworkXRetrievalTraceSink,
    NullRetrievalTraceSink,
    PrefilterTraceContext,
    RetrievalTraceEvent,
    RetrievalTraceSink,
)
from core.memory.retrieval.traversal_policy import TraversalPolicy
from core.memory.stores.chroma_adapter import ChromaDBAdapter
from core.memory.stores.neo4j_adapter import Neo4jAdapter

RELATIONSHIP_DETAIL_KEY = "reason"
SELECTED_NEIGHBOR_RELATIONSHIPS_KEY = "selected_neighbor_relationships"


def _parse_neighbor(row: dict) -> Optional[NeighborCandidate]:
    """Normalize a raw Neo4j neighbor row."""
    sid = (row.get("source_entity_id") or "").strip()
    nid = (row.get("neighbor_entity_id") or "").strip()
    if not sid or not nid:
        return None
    return NeighborCandidate(
        source_entity_id=sid,
        source_entity_name=(row.get("source_entity_name") or "").strip(),
        neighbor_entity_id=nid,
        neighbor_name=(row.get("neighbor_name") or "").strip(),
        neighbor_type=(row.get("neighbor_type") or "FinancialConcept").strip(),
        relationship_type=(row.get("relationship_type") or "RELATED_TO").strip(),
        relationship_reason=(
            str(row.get(RELATIONSHIP_DETAIL_KEY) or "").strip() or None
        ),
    )


def _normalize_published_at(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_native"):
        try:
            return value.to_native()
        except Exception:
            pass
    if hasattr(value, "to_datetime"):
        try:
            return value.to_datetime()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return datetime.fromisoformat(value.isoformat())
        except Exception:
            return None
    return None


def _parse_chunk_row(row: dict) -> Optional[GraphChunkRow]:
    """Normalize a raw Neo4j chunk row."""
    chunk_id = (row.get("chunk_id") or "").strip()
    if not chunk_id:
        return None
    return GraphChunkRow(
        chunk_id=chunk_id,
        chunk_text=row.get("chunk_text") or "",
        chunk_index=row.get("chunk_index"),
        document_id=row.get("document_id"),
        article_title=row.get("article_title"),
        source_url=row.get("source_url"),
        published_at=_normalize_published_at(row.get("published_at")),
        extraction_status=row.get("extraction_status") or "PENDING",
        supporting_entity_id=(row.get("supporting_entity_id") or "").strip() or None,
    )


def _extend_unique(existing: List[str], new_items: Sequence[str]) -> List[str]:
    """Append items from new_items that are not already in existing."""
    seen = set(existing)
    updated = list(existing)
    for item in new_items:
        if item not in seen:
            seen.add(item)
            updated.append(item)
    return updated


class DualStoreRetriever:
    """
    Retrieve chunks by seeding vector search then expanding through the graph.
    """

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        chroma_adapter: ChromaDBAdapter,
        prefilter: CompositePrefilter,
        reranker: TwoStageReranker,
        trace_sink: RetrievalTraceSink | None = None,
    ) -> None:
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._prefilter = prefilter
        self._reranker = reranker
        if trace_sink is not None:
            self._trace_sink = trace_sink
        elif settings.RETRIEVAL_TRACE_ENABLED:
            self._trace_sink = NetworkXRetrievalTraceSink(
                max_runs=settings.RETRIEVAL_TRACE_MAX_RUNS
            )
        else:
            self._trace_sink = NullRetrievalTraceSink()

        self._logger = get_logger(__name__)

        self._max_iterations = settings.RETRIEVER_MAX_ITERATIONS
        self._seed_top_k = settings.RETRIEVER_SEED_TOP_K

        self._policy = TraversalPolicy(
            max_parallel_nodes=settings.RETRIEVER_MAX_PARALLEL_NODES,
            max_neighbor_candidates=settings.RETRIEVER_MAX_NEIGHBOR_CANDIDATES,
            max_iterations=self._max_iterations,
        )

        self._graph = self._build_graph()

    @staticmethod
    def _safe_slug(value: str) -> str:
        cleaned = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value
        )
        return cleaned.strip("_") or "trace"

    def _maybe_auto_export_trace_graph(self, *, run_id: str, domain: str) -> None:
        """
        Optionally auto-export retrieval trace artifacts based on settings.

        This is intentionally best-effort and non-blocking for retrieval.
        """
        if not settings.RETRIEVAL_TRACE_AUTO_EXPORT:
            return

        output_dir = Path(settings.RETRIEVAL_TRACE_AUTO_EXPORT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_domain = self._safe_slug(domain)
        filename_base = f"{safe_domain}_{run_id}"
        html_path = output_dir / f"{filename_base}.html"
        graphml_path = output_dir / f"{filename_base}.graphml"

        try:
            self._trace_sink.export_html(run_id, str(html_path))  # type: ignore[attr-defined]
            self._trace_sink.export_graphml(run_id, str(graphml_path))  # type: ignore[attr-defined]
        except Exception:
            self._logger.warning(
                "Auto-export of retrieval trace failed for run_id=%s domain=%s",
                run_id,
                domain,
                exc_info=True,
            )

    def _trace_event(
        self,
        *,
        state: RetrieverState,
        stage: str,
        hop: int,
        layer: int,
        payload: Dict[str, Any],
    ) -> None:
        """Emit retrieval trace event, never raising into retrieval flow."""
        try:
            self._trace_sink.record(
                RetrievalTraceEvent(
                    run_id=state["run_id"],
                    parent_run_id=state["parent_run_id"],
                    domain=state["domain"],
                    stage=stage,
                    hop=hop,
                    layer=layer,
                    payload=payload,
                )
            )
        except Exception:
            self._logger.warning("Trace sink emission failed", exc_info=True)

    @staticmethod
    def _normalize_relationship_detail(value: Any) -> Optional[str]:
        detail = str(value or "").strip()
        return detail or None

    @classmethod
    def _dedupe_relationship_records(
        cls, records: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: Set[tuple[str, str, str, str]] = set()
        for record in records:
            frontier_name = str(record.get("frontier_name") or "").strip()
            neighbor_name = str(record.get("selected_neighbor_name") or "").strip()
            relationship_type = str(record.get("relationship_type") or "").strip()
            detail = cls._normalize_relationship_detail(
                record.get(RELATIONSHIP_DETAIL_KEY)
            )
            key = (frontier_name, neighbor_name, relationship_type, detail or "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "frontier_name": frontier_name,
                    "selected_neighbor_name": neighbor_name,
                    "relationship_type": relationship_type,
                    RELATIONSHIP_DETAIL_KEY: detail,
                }
            )
        return deduped

    @classmethod
    def _build_selected_relationship_map(
        cls,
        candidates: Sequence[NeighborCandidate],
        selected_ids: Set[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        relationship_map: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            selected_id = (candidate.neighbor_entity_id or "").strip()
            if not selected_id or selected_id not in selected_ids:
                continue
            relationship_map.setdefault(selected_id, []).append(
                {
                    "frontier_name": (candidate.source_entity_name or "").strip()
                    or candidate.source_entity_id,
                    "selected_neighbor_name": (candidate.neighbor_name or "").strip()
                    or selected_id,
                    "relationship_type": (candidate.relationship_type or "").strip()
                    or "RELATED_TO",
                    RELATIONSHIP_DETAIL_KEY: cls._normalize_relationship_detail(
                        candidate.relationship_reason
                    ),
                }
            )
        return {
            neighbor_id: cls._dedupe_relationship_records(records)
            for neighbor_id, records in relationship_map.items()
        }

    @staticmethod
    def _upsert_best_chunk(
        chunk_map: Dict[str, RetrievedChunk], incoming: RetrievedChunk
    ) -> None:
        existing = chunk_map.get(incoming.chunk_id)
        if (
            existing is None
            or (
                incoming.relevance_score is not None
                and (
                    existing.relevance_score is None
                    or incoming.relevance_score > existing.relevance_score
                )
            )
        ):
            chunk_map[incoming.chunk_id] = incoming

    @staticmethod
    def _normalize_domain_chunks(
        domain: str, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        return [
            RetrievedChunk.normalize_for_reranking(chunk, domain)
            for chunk in chunks
            if isinstance(chunk, RetrievedChunk)
        ]

    async def _rerank_domain_chunks(
        self,
        domain: str,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> tuple[str, List[RetrievedChunk]]:
        if not chunks:
            return domain, []
        try:
            ranked = await self._reranker.rank(query, chunks)
        except Exception as exc:
            self._logger.error(
                "Jina rerank failed for domain '%s': %s; fallback to prefilter only.",
                domain,
                exc,
            )
            ranked = self._prefilter.score(chunks)
        return domain, ranked

    # -- Graph wiring --------------------------------------------------------

    def _build_graph(self):
        workflow = StateGraph(RetrieverState)

        workflow.add_node("vector_seed", self._vector_seed_node)
        workflow.add_node("extract_seed_entities", self._extract_seed_entities_node)
        workflow.add_node(
            "select_neighbor_frontier", self._select_neighbor_frontier_node
        )
        workflow.add_node("fetch_frontier_chunks", self._fetch_frontier_chunks_node)

        workflow.add_edge(START, "vector_seed")
        workflow.add_edge("vector_seed", "extract_seed_entities")
        workflow.add_edge("extract_seed_entities", "select_neighbor_frontier")
        workflow.add_edge("select_neighbor_frontier", "fetch_frontier_chunks")

        workflow.add_conditional_edges(
            "fetch_frontier_chunks",
            self._should_continue_routing,
            {
                "select_neighbor_frontier": "select_neighbor_frontier",
                END: END,
            },
        )

        return workflow.compile()

    # -- Routing -------------------------------------------------------------

    def _should_continue_routing(self, state: RetrieverState) -> str:
        if self._policy.should_continue(state):
            return "select_neighbor_frontier"
        return END

    # -- Nodes ---------------------------------------------------------------

    async def _vector_seed_node(self, state: RetrieverState) -> dict:
        query = state["query"]
        try:
            results = await self._chroma_adapter.query(
                query_text=query,
                n_results=self._seed_top_k,
                search_type="similarity",
            )
        except Exception as exc:
            self._logger.error(
                "Vector seed query failed for query='%.80s': %s", query, exc
            )
            results = []

        retrieved: List[RetrievedChunk] = []
        visited_chunk_ids: List[str] = []

        for doc, score in results:
            chunk = RetrievedChunk.from_document(doc, score=score, source="vector")
            if not chunk.chunk_id or chunk.chunk_id in visited_chunk_ids:
                continue
            retrieved.append(chunk)
            visited_chunk_ids.append(chunk.chunk_id)

        self._trace_event(
            state=state,
            stage="vector_seed",
            hop=0,
            layer=0,
            payload={
                "query": query,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "score": chunk.relevance_score,
                        "chunk_text": chunk.text,
                        "article_title": chunk.article_title
                        or (chunk.metadata or {}).get("article_title")
                        or "",
                        "source_url": chunk.source_url
                        or (chunk.metadata or {}).get("source_url")
                        or "",
                    }
                    for chunk in retrieved
                ],
            },
        )

        self._logger.info(
            "Vector seed: %d chunks for query='%.60s'", len(retrieved), query
        )
        return {
            "accumulated_chunks": retrieved,
            "visited_chunk_ids": visited_chunk_ids,
            "iteration": 0,
        }

    async def _extract_seed_entities_node(self, state: RetrieverState) -> dict:
        chunk_ids = [c.chunk_id for c in state["accumulated_chunks"]]
        try:
            entity_rows = await self._neo4j_adapter.get_entities_for_chunks(chunk_ids)
        except Exception as exc:
            self._logger.error(
                "Seed entity lookup failed (chunk_ids=%d): %s", len(chunk_ids), exc
            )
            entity_rows = []

        entity_ids: List[str] = []
        seen: Set[str] = set()
        trace_links: List[Dict[str, Any]] = []
        for row in entity_rows:
            eid = str(row.get("entity_id") or "").strip()
            if not eid:
                continue
            if eid not in seen:
                seen.add(eid)
                entity_ids.append(eid)

            trace_links.append(
                {
                    "entity_id": eid,
                    "entity_name": row.get("entity_name") or "",
                    "entity_type": row.get("entity_type") or "",
                    "source_chunk_id": row.get("source_chunk_id") or "",
                }
            )

        self._trace_event(
            state=state,
            stage="seed_entities",
            hop=0,
            layer=0,
            payload={"links": trace_links},
        )

        self._logger.info(
            "Seed entities: %d from %d chunks", len(entity_ids), len(chunk_ids)
        )
        return {
            "current_frontier": entity_ids,
            "visited_entity_ids": list(entity_ids),
        }

    async def _select_neighbor_frontier_node(self, state: RetrieverState) -> dict:
        frontier = state["current_frontier"]
        if not frontier:
            return {
                "current_frontier": [],
                "selected_neighbor_relationships": {},
                "should_continue": False,
            }

        hop_depth = state["iteration"] + 1
        try:
            raw_rows = await self._neo4j_adapter.get_entity_neighbors(
                frontier, state["visited_entity_ids"]
            )
        except Exception as exc:
            self._logger.error(
                "Neighbor fetch failed at hop %d (frontier=%d): %s",
                hop_depth,
                len(frontier),
                exc,
            )
            return {
                "current_frontier": [],
                "selected_neighbor_relationships": {},
                "should_continue": False,
            }

        candidates: List[NeighborCandidate] = [
            parsed for row in raw_rows if (parsed := _parse_neighbor(row)) is not None
        ]
        capped = self._policy.cap_per_source(candidates)
        if not capped:
            self._logger.info(
                "Hop %d: no valid neighbor candidates - stopping.", hop_depth
            )
            return {
                "current_frontier": [],
                "selected_neighbor_relationships": {},
                "should_continue": False,
            }

        selected = self._policy.select_frontier(capped, state["query"], hop_depth)
        if not selected:
            self._logger.info(
                "Hop %d: frontier selection returned empty - stopping.", hop_depth
            )
            return {
                "current_frontier": [],
                "selected_neighbor_relationships": {},
                "should_continue": False,
            }

        selected_set = set(selected)
        selected_relationship_map = self._build_selected_relationship_map(
            capped, selected_set
        )
        scored_candidates = []
        for candidate in capped:
            score = self._policy.score_neighbor(candidate, state["query"], hop_depth)
            scored_candidates.append(
                {
                    "source_entity_id": candidate.source_entity_id,
                    "source_entity_name": candidate.source_entity_name,
                    "neighbor_entity_id": candidate.neighbor_entity_id,
                    "neighbor_name": candidate.neighbor_name,
                    "neighbor_type": candidate.neighbor_type,
                    "relationship_type": candidate.relationship_type,
                    RELATIONSHIP_DETAIL_KEY: self._normalize_relationship_detail(
                        candidate.relationship_reason
                    ),
                    "score": score,
                    "selected": candidate.neighbor_entity_id in selected_set,
                }
            )
        self._trace_event(
            state=state,
            stage="neighbor_expansion",
            hop=hop_depth,
            layer=hop_depth,
            payload={
                "frontier": list(frontier),
                "candidates": scored_candidates,
                "selected_frontier": list(selected),
            },
        )

        updated_visited = _extend_unique(state["visited_entity_ids"], selected)
        self._logger.info(
            "Frontier hop %d: %d raw -> %d capped -> %d selected",
            hop_depth,
            len(candidates),
            len(capped),
            len(selected),
        )
        return {
            "current_frontier": selected,
            "selected_neighbor_relationships": selected_relationship_map,
            "visited_entity_ids": updated_visited,
            "should_continue": True,
        }

    async def _fetch_frontier_chunks_node(self, state: RetrieverState) -> dict:
        frontier = state["current_frontier"]
        if not frontier:
            return {}

        selected_relationships_map = state.get("selected_neighbor_relationships") or {}
        hop_depth = state["iteration"] + 1
        try:
            rows = await self._neo4j_adapter.get_chunks_for_entities(
                frontier, state["visited_chunk_ids"]
            )
        except Exception as exc:
            self._logger.error(
                "Chunk fetch failed at hop %d (frontier=%d): %s",
                hop_depth,
                len(frontier),
                exc,
            )
            return {"iteration": state["iteration"] + 1}

        new_chunks_by_id: Dict[str, RetrievedChunk] = {}
        new_chunk_ids: List[str] = []
        previously_visited_chunk_ids = set(state["visited_chunk_ids"])
        trace_links: Set[tuple[str, str, str]] = set()

        for row in rows:
            parsed = _parse_chunk_row(row)
            if parsed is None:
                continue
            if parsed.chunk_id in previously_visited_chunk_ids:
                # Chunk already accumulated in previous iterations/runs for this
                # state, so skip creation and provenance expansion.
                continue

            support_id = parsed.supporting_entity_id or ""
            chunk = new_chunks_by_id.get(parsed.chunk_id)
            if chunk is None:
                chunk = RetrievedChunk(
                    chunk_id=parsed.chunk_id,
                    text=parsed.chunk_text,
                    source="graph",
                    graph_depth=hop_depth,
                    extraction_status=parsed.extraction_status,
                    document_id=parsed.document_id,
                    chunk_index=parsed.chunk_index,
                    article_title=parsed.article_title,
                    source_url=parsed.source_url,
                    published_at=parsed.published_at,
                    metadata={
                        "chunk_index": parsed.chunk_index,
                        "document_id": parsed.document_id,
                        "article_title": parsed.article_title,
                        "source_url": parsed.source_url,
                        "published_at": parsed.published_at,
                        "supporting_entity_ids": [],
                        SELECTED_NEIGHBOR_RELATIONSHIPS_KEY: [],
                    },
                )
                new_chunks_by_id[parsed.chunk_id] = chunk
                new_chunk_ids.append(parsed.chunk_id)

            if support_id:
                support_ids = chunk.metadata.setdefault("supporting_entity_ids", [])
                if support_id not in support_ids:
                    support_ids.append(support_id)

                support_relationships = selected_relationships_map.get(support_id) or []
                supporting_entity_name = support_id
                for relationship in support_relationships:
                    candidate_name = str(
                        relationship.get("selected_neighbor_name") or ""
                    ).strip()
                    if candidate_name:
                        supporting_entity_name = candidate_name
                        break

                existing_relationships = chunk.metadata.setdefault(
                    SELECTED_NEIGHBOR_RELATIONSHIPS_KEY, []
                )
                chunk.metadata[SELECTED_NEIGHBOR_RELATIONSHIPS_KEY] = (
                    self._dedupe_relationship_records(
                        [*existing_relationships, *support_relationships]
                    )
                )
                trace_links.add(
                    (support_id, supporting_entity_name, parsed.chunk_id)
                )

        new_chunks = list(new_chunks_by_id.values())
        accumulated = state["accumulated_chunks"] + new_chunks
        updated_chunk_ids = _extend_unique(state["visited_chunk_ids"], new_chunk_ids)

        self._trace_event(
            state=state,
            stage="frontier_chunks",
            hop=hop_depth,
            layer=hop_depth,
            payload={
                "frontier": list(frontier),
                "links": [
                    {
                        "supporting_entity_id": supporting_entity_id,
                        "supporting_entity_name": supporting_entity_name,
                        "chunk_id": chunk_id,
                    }
                    for supporting_entity_id, supporting_entity_name, chunk_id in sorted(
                        trace_links
                    )
                ],
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_text": chunk.text,
                        "article_title": chunk.article_title
                        or (chunk.metadata or {}).get("article_title")
                        or "",
                        "source_url": chunk.source_url
                        or (chunk.metadata or {}).get("source_url")
                        or "",
                    }
                    for chunk in new_chunks
                ],
                "new_chunk_ids": list(new_chunk_ids),
            },
        )

        self._logger.info(
            "Fetch hop %d: %d new chunks from %d entities (total=%d)",
            hop_depth,
            len(new_chunks),
            len(frontier),
            len(accumulated),
        )
        return {
            "accumulated_chunks": accumulated,
            "visited_chunk_ids": updated_chunk_ids,
            "iteration": state["iteration"] + 1,
        }

    # -- Public API ----------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        run_id: str | None = None,
        domain: str = "single",
        parent_run_id: str | None = None,
    ) -> List[RetrievedChunk]:
        """
        Run full vector-seed + graph-expansion workflow for a single query.
        """
        retrieval_run_id = run_id or str(uuid4())
        initial_state: RetrieverState = {
            "query": query,
            "accumulated_chunks": [],
            "visited_entity_ids": [],
            "visited_chunk_ids": [],
            "current_frontier": [],
            "iteration": 0,
            "should_continue": False,
            "selected_neighbor_relationships": {},
            "run_id": retrieval_run_id,
            "parent_run_id": parent_run_id,
            "domain": domain,
        }
        final_state = await self._graph.ainvoke(initial_state)
        self._maybe_auto_export_trace_graph(run_id=retrieval_run_id, domain=domain)
        return final_state["accumulated_chunks"]

    async def comprehensive_retrieve(
        self, rewritten_queries: RewrittenQueries
    ) -> MemoryContext:
        """
        Fan out retrieval across active domain queries, per-domain rerank, then merge.
        """
        domain_map = {
            "company": rewritten_queries.company_query,
            "sector": rewritten_queries.sector_query,
            "market": rewritten_queries.market_query,
            "knowledge": rewritten_queries.knowledge_query,
        }
        active_queries: Dict[str, str] = {
            domain: query
            for domain, query in domain_map.items()
            if domain in rewritten_queries.active_domains and query is not None
        }

        if not active_queries:
            return MemoryContext(
                chunks=[],
                rewritten_queries=rewritten_queries,
            )

        parent_run_id = str(uuid4())
        task_items: List[tuple[str, asyncio.Task[List[RetrievedChunk]]]] = []
        for domain, query in active_queries.items():
            domain_run_id = str(uuid4())
            task_items.append(
                (
                    domain,
                    asyncio.create_task(
                        self.retrieve(
                            query,
                            run_id=domain_run_id,
                            domain=domain,
                            parent_run_id=parent_run_id,
                        )
                    ),
                )
            )

        results = await asyncio.gather(
            *(task for _, task in task_items), return_exceptions=True
        )

        domain_inputs: List[tuple[str, str, List[RetrievedChunk]]] = []
        for (domain, _task), result in zip(task_items, results):
            if isinstance(result, Exception):
                self._logger.error(
                    "Memory retrieval failed for domain '%s': %s", domain, result
                )
                continue
            if not isinstance(result, list):
                continue
            normalized_domain_chunks = self._normalize_domain_chunks(domain, result)
            if not normalized_domain_chunks:
                continue
            domain_inputs.append(
                (domain, active_queries.get(domain, ""), normalized_domain_chunks)
            )

        rerank_tasks = [
            asyncio.create_task(self._rerank_domain_chunks(domain, query, chunks))
            for domain, query, chunks in domain_inputs
        ]
        rerank_results = await asyncio.gather(*rerank_tasks, return_exceptions=True)

        chunk_map: Dict[str, RetrievedChunk] = {}
        for rerank_result in rerank_results:
            if isinstance(rerank_result, Exception):
                self._logger.error(
                    "Domain rerank task failed unexpectedly: %s", rerank_result
                )
                continue
            _domain, reranked_domain_chunks = rerank_result
            for reranked in reranked_domain_chunks:
                self._upsert_best_chunk(chunk_map, reranked)

        merged_count = len(chunk_map)
        merged = self._prefilter.score(
            list(chunk_map.values()),
            trace_context=PrefilterTraceContext(
                run_id=parent_run_id,
                parent_run_id=None,
                domain="comprehensive",
                sink=self._trace_sink,
                hop=0,
                layer=0,
            ),
        )

        self._logger.info(
            "Comprehensive retrieve: domains=%s unique_chunks=%d merged=%d",
            list(active_queries.keys()),
            merged_count,
            len(merged),
        )
        self._maybe_auto_export_trace_graph(
            run_id=parent_run_id,
            domain="comprehensive",
        )

        return MemoryContext(
            chunks=merged,
            rewritten_queries=rewritten_queries,
        )
