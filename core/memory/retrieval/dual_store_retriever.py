"""LangGraph-powered dual-store retriever for vector + graph traversal, fanned out across domains."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Sequence

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from core.config import settings
from core.logger import get_logger
from core.memory.retrieval.models import (
    MemoryContext,
    NodeSelectionOutput,
    RetrievedChunk,
    RetrieverState,
    RewrittenQueries,
)
from core.memory.retrieval.reranker import CompositeReranker
from core.memory.retrieval.retrieval_prompts import build_node_selection_prompt
from core.memory.stores.chroma_adapter import ChromaDBAdapter
from core.memory.stores.neo4j_adapter import Neo4jAdapter


class DualStoreRetriever:
    """Retrieve chunks by seeding vector search and expanding through the graph."""

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        chroma_adapter: ChromaDBAdapter,
        llm: ChatGoogleGenerativeAI,
        reranker: CompositeReranker,
    ) -> None:
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._reranker = reranker
        self._llm = llm

        self._logger = get_logger(__name__)

        self._max_iterations = settings.RETRIEVER_MAX_ITERATIONS
        self._seed_top_k = settings.RETRIEVER_SEED_TOP_K
        self._max_parallel_nodes = settings.RETRIEVER_MAX_PARALLEL_NODES
        self._max_neighbor_candidates = settings.RETRIEVER_MAX_NEIGHBOR_CANDIDATES

        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RetrieverState)

        workflow.add_node("vector_seed", self._vector_seed_node)
        workflow.add_node("extract_seed_entities", self._extract_seed_entities_node)
        workflow.add_node("agent_select_nodes", self._agent_select_nodes_node)
        workflow.add_node("expand_nodes", self._expand_and_fetch_node)

        workflow.add_edge(START, "vector_seed")
        workflow.add_edge("vector_seed", "extract_seed_entities")
        workflow.add_edge("extract_seed_entities", "agent_select_nodes")
        workflow.add_edge("agent_select_nodes", "expand_nodes")

        workflow.add_conditional_edges(
            "expand_nodes",
            self._should_continue,
            {"agent_select_nodes": "agent_select_nodes", END: END},
        )

        return workflow.compile()

    async def _vector_seed_node(self, state: RetrieverState) -> dict:
        query = state["query"]
        results = await self._chroma_adapter.query(
            query_text=query,
            n_results=self._seed_top_k,
            search_type="similarity",
        )

        retrieved: List[RetrievedChunk] = []
        visited_chunk_ids: List[str] = []

        for doc, score in results:
            chunk = RetrievedChunk.from_document(doc, score=score, source="vector")
            if not chunk.chunk_id or chunk.chunk_id in visited_chunk_ids:
                continue
            retrieved.append(chunk)
            visited_chunk_ids.append(chunk.chunk_id)

        self._logger.info("Vector seed retrieved %d chunks.", len(retrieved))

        return {
            "accumulated_chunks": retrieved,
            "visited_chunk_ids": visited_chunk_ids,
            "iteration": 0,
        }

    async def _extract_seed_entities_node(self, state: RetrieverState) -> dict:
        chunk_ids = [chunk.chunk_id for chunk in state["accumulated_chunks"]]
        entities = await self._neo4j_adapter.get_entities_for_chunks(chunk_ids)
        entity_ids = []
        seen = set()
        for record in entities:
            entity_id = record.get("entity_id")
            if entity_id and entity_id not in seen:
                seen.add(entity_id)
                entity_ids.append(entity_id)

        self._logger.info("Seed entity count: %d", len(entity_ids))

        return {
            "current_frontier": entity_ids,
            "visited_entity_ids": list(entity_ids),
        }

    async def _agent_select_nodes_node(self, state: RetrieverState) -> dict:
        frontier = state["current_frontier"]
        if not frontier:
            return {
                "candidate_neighbors": [],
                "current_frontier": [],
                "should_continue": False,
            }

        neighbors = await self._neo4j_adapter.get_entity_neighbors(
            frontier, state["visited_entity_ids"]
        )

        filtered_neighbors = self._limit_neighbors_per_source(neighbors)
        if not filtered_neighbors:
            return {
                "candidate_neighbors": [],
                "current_frontier": [],
                "should_continue": False,
            }

        prompt = build_node_selection_prompt()
        structured_llm = self._llm.with_structured_output(NodeSelectionOutput)
        formatted_candidates = self._format_candidates(filtered_neighbors)

        result: NodeSelectionOutput = await (prompt | structured_llm).ainvoke(
            {
                "query": state["query"],
                "iteration": state["iteration"],
                "max_iterations": self._max_iterations,
                "candidate_neighbors": formatted_candidates,
                "already_retrieved_count": len(state["accumulated_chunks"]),
                "max_parallel_nodes": self._max_parallel_nodes,
            }
        )

        candidate_ids = {c.get("neighbor_entity_id") for c in filtered_neighbors}
        selected = [
            entity_id
            for entity_id in result.selected_entity_ids
            if entity_id in candidate_ids
        ]
        selected = self._dedupe_keep_order(selected)
        selected = selected[: self._max_parallel_nodes]

        updated_visited = self._extend_unique(state["visited_entity_ids"], selected)

        return {
            "candidate_neighbors": filtered_neighbors,
            "current_frontier": selected,
            "visited_entity_ids": updated_visited,
            "should_continue": len(selected) > 0,
        }

    async def _expand_and_fetch_node(self, state: RetrieverState) -> dict:
        frontier = state["current_frontier"]
        if not frontier:
            return {"iteration": state["iteration"] + 1}

        rows = await self._neo4j_adapter.get_chunks_for_entities(
            frontier, state["visited_chunk_ids"]
        )

        new_chunks: List[RetrievedChunk] = []
        new_chunk_ids: List[str] = []
        visited_chunk_ids = set(state["visited_chunk_ids"])

        for row in rows:
            chunk_id = row.get("chunk_id")
            if not chunk_id or chunk_id in visited_chunk_ids:
                continue
            visited_chunk_ids.add(chunk_id)
            new_chunk_ids.append(chunk_id)
            new_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=row.get("chunk_text", ""),
                    score=None,
                    source="graph",
                    metadata={
                        "chunk_index": row.get("chunk_index"),
                        "document_id": row.get("document_id"),
                        "article_title": row.get("article_title"),
                        "source_url": row.get("source_url"),
                        "published_at": row.get("published_at"),
                    },
                )
            )

        accumulated = state["accumulated_chunks"] + new_chunks
        updated_chunk_ids = state["visited_chunk_ids"] + new_chunk_ids

        self._logger.info(
            "Expanded %d nodes, retrieved %d new chunks (total=%d).",
            len(frontier),
            len(new_chunks),
            len(accumulated),
        )

        return {
            "accumulated_chunks": accumulated,
            "visited_chunk_ids": updated_chunk_ids,
            "iteration": state["iteration"] + 1,
        }

    def _should_continue(self, state: RetrieverState) -> str:
        if state["should_continue"] and state["iteration"] < self._max_iterations:
            return "agent_select_nodes"
        return END

    async def retrieve(self, query: str) -> List[RetrievedChunk]:
        initial_state: RetrieverState = {
            "query": query,
            "accumulated_chunks": [],
            "visited_entity_ids": [],
            "visited_chunk_ids": [],
            "current_frontier": [],
            "candidate_neighbors": [],
            "iteration": 0,
            "should_continue": True,
        }
        final_state = await self._graph.ainvoke(initial_state)
        return final_state["accumulated_chunks"]

    async def comprehensive_retrieve(
        self, rewritten_queries: RewrittenQueries
    ) -> MemoryContext:
        """Fans out the LangGraph traversal across active domains and reranks results."""
        domain_map = {
            "company": rewritten_queries.company_query,
            "sector": rewritten_queries.sector_query,
            "market": rewritten_queries.market_query,
            "knowledge": rewritten_queries.knowledge_query,
        }

        active_queries = {
            domain: query
            for domain, query in domain_map.items()
            if domain in rewritten_queries.active_domains and query is not None
        }

        # Concurrently run the graph traversal for each domain
        tasks = [self.retrieve(query) for query in active_queries.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_chunks: List[RetrievedChunk] = []
        for (domain, _query), result in zip(active_queries.items(), results):
            if isinstance(result, Exception):
                self._logger.error(
                    "Memory retrieval failed for domain %s: %s", domain, result
                )
                continue
            if not isinstance(result, list):
                continue

            for chunk in result:
                if isinstance(chunk, RetrievedChunk):
                    all_chunks.append(RetrievedChunk.from_raw_chunk(chunk, domain))

        # Rerank and return combined context
        ranked = self._reranker.rank(all_chunks)
        return MemoryContext(chunks=ranked, rewritten_queries=rewritten_queries)

    def _limit_neighbors_per_source(self, neighbors: Sequence[dict]) -> List[dict]:
        if not neighbors:
            return []
        capped: List[dict] = []
        counts: Dict[str, int] = {}
        for neighbor in neighbors:
            source_id = neighbor.get("source_entity_id")
            if not source_id:
                continue
            counts.setdefault(source_id, 0)
            if counts[source_id] >= self._max_neighbor_candidates:
                continue
            counts[source_id] += 1
            capped.append(neighbor)
        return capped

    def _format_candidates(self, neighbors: Sequence[dict]) -> str:
        lines = []
        for idx, neighbor in enumerate(neighbors, start=1):
            lines.append(
                f"{idx}. {neighbor.get('neighbor_name')} "
                f"({neighbor.get('neighbor_type')}) "
                f"| relationship={neighbor.get('relationship_type')} "
                f"| source_entity_id={neighbor.get('source_entity_id')} "
                f"| neighbor_entity_id={neighbor.get('neighbor_entity_id')}"
            )
        return "\n".join(lines) if lines else "None."

    @staticmethod
    def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @staticmethod
    def _extend_unique(existing: List[str], new_items: Sequence[str]) -> List[str]:
        seen = set(existing)
        updated = list(existing)
        for item in new_items:
            if item in seen:
                continue
            seen.add(item)
            updated.append(item)
        return updated
