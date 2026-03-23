"""
core/memory/retrieval/dual_store_retriever.py

LangGraph-powered dual-store retriever: vector seed + iterative graph traversal,
fanned out across rewritten domain queries and reranked.

Design
──────
DualStoreRetriever
    Public entrypoints: retrieve() and comprehensive_retrieve().
    Builds and drives a LangGraph workflow; delegates all policy decisions to
    TraversalPolicy so orchestration and scoring concerns never mix.

TraversalPolicy
    Encapsulates neighbor scoring, frontier selection, and stopping logic.
    Fully extractable for unit testing without touching the LangGraph wiring.
    Scoring uses a three-factor composite:
        structural_weight × hop_decay × (1 + query_relevance_bonus)

NeighborCandidate / GraphChunkRow
    Typed dataclasses that normalise raw Neo4j dict rows at the adapter
    boundary, eliminating scattered string-key access inside node logic.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Sequence

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
from core.memory.retrieval.reranker import CompositePrefilter
from core.memory.retrieval.traversal_policy import TraversalPolicy
from core.memory.stores.chroma_adapter import ChromaDBAdapter
from core.memory.stores.neo4j_adapter import Neo4jAdapter

# ─────────────────────────────────────────────────────────────────────────────
# Row normalisers  (dict → typed model, adapter boundary)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_neighbor(row: dict) -> Optional[NeighborCandidate]:
    """Normalise a raw Neo4j neighbor row.  Returns None if mandatory IDs are absent."""
    sid = (row.get("source_entity_id") or "").strip()
    nid = (row.get("neighbor_entity_id") or "").strip()
    if not sid or not nid:
        return None
    return NeighborCandidate(
        source_entity_id=sid,
        neighbor_entity_id=nid,
        neighbor_name=(row.get("neighbor_name") or "").strip(),
        neighbor_type=(row.get("neighbor_type") or "FinancialConcept").strip(),
        relationship_type=(row.get("relationship_type") or "RELATED_TO").strip(),
    )


def _parse_chunk_row(row: dict) -> Optional[GraphChunkRow]:
    """Normalise a raw Neo4j chunk row.  Returns None if chunk_id is absent."""
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
        published_at=row.get("published_at"),
        extraction_status=row.get("extraction_status") or "PENDING",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers (module-level, stateless)
# ─────────────────────────────────────────────────────────────────────────────


def _extend_unique(existing: List[str], new_items: Sequence[str]) -> List[str]:
    """Append items from new_items that are not already in existing, preserving order."""
    seen = set(existing)
    updated = list(existing)
    for item in new_items:
        if item not in seen:
            seen.add(item)
            updated.append(item)
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# DualStoreRetriever
# ─────────────────────────────────────────────────────────────────────────────


class DualStoreRetriever:
    """
    Retrieve chunks by seeding vector search then expanding through the graph.

    Graph topology
    ──────────────
    START
      → vector_seed              (Chroma similarity search)
      → extract_seed_entities    (Neo4j entity lookup for seed chunks)
      → select_neighbor_frontier (TraversalPolicy: score + select next frontier)
      → fetch_frontier_chunks    (Neo4j: chunks for frontier entities)
      → [conditional routing]
            ├─ select_neighbor_frontier  (if should_continue)
            └─ END

    Public API
    ──────────
    retrieve(query)                           → List[RetrievedChunk]
    comprehensive_retrieve(rewritten_queries) → MemoryContext
    """

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        chroma_adapter: ChromaDBAdapter,
        prefilter: CompositePrefilter,
    ) -> None:
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._prefilter = prefilter
        self._logger = get_logger(__name__)

        self._max_iterations = settings.RETRIEVER_MAX_ITERATIONS
        self._seed_top_k = settings.RETRIEVER_SEED_TOP_K

        self._policy = TraversalPolicy(
            max_parallel_nodes=settings.RETRIEVER_MAX_PARALLEL_NODES,
            max_neighbor_candidates=settings.RETRIEVER_MAX_NEIGHBOR_CANDIDATES,
            max_iterations=self._max_iterations,
        )

        self._graph = self._build_graph()

    # ── Graph wiring ──────────────────────────────────────────────────────────

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

    # ── Routing ───────────────────────────────────────────────────────────────

    def _should_continue_routing(self, state: RetrieverState) -> str:
        """Route back to frontier selection or terminate the traversal."""
        if self._policy.should_continue(state):
            return "select_neighbor_frontier"
        return END

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def _vector_seed_node(self, state: RetrieverState) -> dict:
        """
        Seed retrieval via Chroma vector similarity search.

        Initialises accumulated_chunks and visited_chunk_ids.  The iteration
        counter starts at 0 — it only advances when a non-empty graph frontier
        is actually expanded in fetch_frontier_chunks.
        """
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

        self._logger.info(
            "Vector seed: %d chunks for query='%.60s'", len(retrieved), query
        )
        return {
            "accumulated_chunks": retrieved,
            "visited_chunk_ids": visited_chunk_ids,
            "iteration": 0,
        }

    async def _extract_seed_entities_node(self, state: RetrieverState) -> dict:
        """
        Look up entities linked to seed chunks in Neo4j and set the initial
        graph frontier.  Failures degrade gracefully to an empty frontier.
        """
        chunk_ids = [c.chunk_id for c in state["accumulated_chunks"]]
        try:
            entity_rows = await self._neo4j_adapter.get_entities_for_chunks(chunk_ids)
        except Exception as exc:
            self._logger.error(
                "Seed entity lookup failed (chunk_ids=%d): %s", len(chunk_ids), exc
            )
            entity_rows = []

        entity_ids: List[str] = []
        seen: set = set()
        for row in entity_rows:
            eid = row.get("entity_id")
            if eid and eid not in seen:
                seen.add(eid)
                entity_ids.append(eid)

        self._logger.info(
            "Seed entities: %d from %d chunks", len(entity_ids), len(chunk_ids)
        )
        return {
            "current_frontier": entity_ids,
            "visited_entity_ids": list(entity_ids),
        }

    async def _select_neighbor_frontier_node(self, state: RetrieverState) -> dict:
        """
        Fetch neighbors of the current frontier, score them via TraversalPolicy,
        and select the next set of entities to expand.

        Selection is fully deterministic (no LLM).  The scoring model is:
            structural_weight × hop_decay × (1 + query_relevance_bonus)

        Sets should_continue=False if no valid neighbors remain, which causes
        the routing function to terminate the traversal immediately.
        """
        frontier = state["current_frontier"]
        if not frontier:
            return {"current_frontier": [], "should_continue": False}

        # hop_depth is 1-based: on the first pass iteration=0 so hop_depth=1.
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
            return {"current_frontier": [], "should_continue": False}

        # Normalise raw dicts → typed NeighborCandidate at the adapter boundary.
        candidates: List[NeighborCandidate] = [
            parsed for row in raw_rows if (parsed := _parse_neighbor(row)) is not None
        ]

        capped = self._policy.cap_per_source(candidates)
        if not capped:
            self._logger.info(
                "Hop %d: no valid neighbor candidates — stopping.", hop_depth
            )
            return {"current_frontier": [], "should_continue": False}

        selected = self._policy.select_frontier(capped, state["query"], hop_depth)
        if not selected:
            self._logger.info(
                "Hop %d: frontier selection returned empty — stopping.", hop_depth
            )
            return {"current_frontier": [], "should_continue": False}

        updated_visited = _extend_unique(state["visited_entity_ids"], selected)

        self._logger.info(
            "Frontier hop %d: %d raw → %d capped → %d selected",
            hop_depth,
            len(candidates),
            len(capped),
            len(selected),
        )
        return {
            "current_frontier": selected,
            "visited_entity_ids": updated_visited,
            "should_continue": True,
        }

    async def _fetch_frontier_chunks_node(self, state: RetrieverState) -> dict:
        """
        Retrieve Neo4j chunks for the current frontier entities and accumulate them.

        Provenance metadata written to each chunk:
            hop_depth             — 1-based expansion round when this chunk was found.
                                    The reranker uses this as graph_depth for its
                                    depth_bonus, so deeper expansions are penalised.
            supporting_entity_ids — frontier entities that led to this chunk,
                                    enabling future explainability / path tracing.
            extraction_status     — propagated from Neo4j so only genuinely PENDING
                                    chunks are re-queued for entity extraction.

        iteration increments ONLY when the frontier is non-empty.  On an empty
        frontier the node returns immediately without touching iteration, so the
        routing function terminates the traversal cleanly without burning budget.
        """
        frontier = state["current_frontier"]
        if not frontier:
            # Empty frontier — routing will terminate; do not advance iteration.
            return {}

        hop_depth = state["iteration"] + 1  # 1-based, consistent with select node

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
            # Advance iteration so we don't retry the same broken frontier endlessly.
            return {"iteration": state["iteration"] + 1}

        new_chunks: List[RetrievedChunk] = []
        new_chunk_ids: List[str] = []
        visited_set = set(state["visited_chunk_ids"])

        for row in rows:
            parsed = _parse_chunk_row(row)
            if parsed is None or parsed.chunk_id in visited_set:
                continue
            visited_set.add(parsed.chunk_id)
            new_chunk_ids.append(parsed.chunk_id)
            new_chunks.append(
                RetrievedChunk(
                    chunk_id=parsed.chunk_id,
                    text=parsed.chunk_text,
                    score=None,
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
                        # Provenance: which entities caused this chunk to be retrieved.
                        "supporting_entity_ids": list(frontier),
                    },
                )
            )

        accumulated = state["accumulated_chunks"] + new_chunks
        updated_chunk_ids = _extend_unique(state["visited_chunk_ids"], new_chunk_ids)

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

    # ── Public entrypoints ────────────────────────────────────────────────────

    async def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Run the full vector-seed + graph-expansion workflow for a single query.

        Returns the accumulated list of RetrievedChunk objects (vector seeds +
        all graph-expanded chunks).  No reranking is applied here; reranking
        happens in comprehensive_retrieve after cross-domain deduplication.
        """
        initial_state: RetrieverState = {
            "query": query,
            "accumulated_chunks": [],
            "visited_entity_ids": [],
            "visited_chunk_ids": [],
            "current_frontier": [],
            "iteration": 0,
            # False is the safer sentinel: should_continue is always written by
            # select_neighbor_frontier before the routing check fires, but
            # defaulting to False prevents accidental infinite loops if the graph
            # topology is ever changed and the check fires earlier than expected.
            "should_continue": False,
        }
        final_state = await self._graph.ainvoke(initial_state)
        return final_state["accumulated_chunks"]

    async def comprehensive_retrieve(
        self, rewritten_queries: RewrittenQueries
    ) -> MemoryContext:
        """
        Fan out retrieval across all active domain queries, deduplicate, and rerank.

        Deduplication strategy
        ──────────────────────
        Chunks are deduplicated by chunk_id before being passed to the reranker.
        When the same chunk is found in multiple domain traversals, the copy with
        the highest embedding_score is kept.  This means:
        - The reranker receives a clean, non-redundant input.
        - The reranker's composite_score is computed from the best available
          embedding signal for that chunk across all domains.

        This is strictly better than relying solely on the reranker's internal
        dedup (which silently discards the lower-scoring duplicate rather than
        selecting the best one).
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
            return MemoryContext(chunks=[], rewritten_queries=rewritten_queries)

        # Concurrent retrieval — one full graph traversal per domain.
        tasks = [self.retrieve(query) for query in active_queries.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Normalise + deduplicate by chunk_id before reranking.
        chunk_map: Dict[str, RetrievedChunk] = {}
        for (domain, _), result in zip(active_queries.items(), results):
            if isinstance(result, Exception):
                self._logger.error(
                    "Memory retrieval failed for domain '%s': %s", domain, result
                )
                continue
            if not isinstance(result, list):
                continue
            for chunk in result:
                if not isinstance(chunk, RetrievedChunk):
                    continue
                normalized = RetrievedChunk.normalize_for_reranking(chunk, domain)
                existing = chunk_map.get(normalized.chunk_id)
                if (
                    existing is None
                    or normalized.embedding_score > existing.embedding_score
                ):
                    chunk_map[normalized.chunk_id] = normalized

        pre_rerank_count = len(chunk_map)
        prefiltered = self._prefilter.score(list(chunk_map.values()))

        self._logger.info(
            "Comprehensive retrieve: domains=%s unique_chunks=%d prefiltered=%d",
            list(active_queries.keys()),
            pre_rerank_count,
            len(prefiltered),
        )
        return MemoryContext(chunks=prefiltered, rewritten_queries=rewritten_queries)
