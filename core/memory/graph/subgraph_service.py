"""
core/memory/graph/subgraph_service.py

Single-responsibility service that owns the full lifecycle of
relationship-level subgraph extraction:

    text → LLM extraction → in-memory graph → Neo4j persistence

Replaces the scattered InMemorySubgraphBuilder.schedule_subgraph_extraction()
+ relationship_extractor.build_and_store() + relationship_extractor.retry_*
pattern.

Design decisions
────────────────
- SubgraphExtractionService is a singleton managed by service_manager.
  Agents receive it via service_manager.get_subgraph_service(); they never
  instantiate it directly.

- Callers supply their own system_prompt. The service enforces no default
  because extraction quality is domain-specific: a news agent's prompt
  differs from a fundamentals agent's prompt. Prompts live in
  core/agents/prompts.py and core/memory/graph/extraction_prompts.py as
  before.

- build() (graph construction) and extract() (LLM call) are independent
  public methods so callers that already have relationships (e.g.
  NewsAnalysisAgent's first-pass extraction) can skip the LLM call.

- schedule() is a single fire-and-forget entry point that covers both
  paths (pre-extracted relationships and text-only) behind one interface.

- extract_entities_for_chunks (chunk-level entity ingestion) is NOT part
  of this service. That function belongs to DualStoreIngestor because it
  reads from ChromaDB and writes Entity nodes — it is ingestion pipeline
  logic, not relationship enrichment logic.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from rapidfuzz import fuzz
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from core.config import settings
from core.logger import get_logger
from core.memory.graph.utils import canonical_entity_id

logger = get_logger(__name__)

# Regex for parsing <relationships> XML from LLM output
import re

_REL_RE = re.compile(r"<relationships>(.*?)</relationships>", re.DOTALL | re.IGNORECASE)


class SubgraphExtractionService:
    """
    Owns: LLM relationship extraction → in-memory graph construction
          → Neo4j persistence.

    Dependencies are injected at construction time by service_manager,
    never fetched via service_manager inside methods.  This keeps the
    service testable without a running service registry.
    """

    def __init__(
        self,
        ingestor,  # DualStoreIngestor
        embedding_func,
        fuzzy_threshold: float,
        semantic_threshold: float,
    ) -> None:
        self._ingestor = ingestor
        self._embedding_func = embedding_func
        self._fuzzy_threshold = fuzzy_threshold
        self._semantic_threshold = semantic_threshold
        # Shared cache across all agents — no duplicated work across turns
        self._entity_name_cache: Dict[Tuple[str, str], str] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LLM extraction
    # ──────────────────────────────────────────────────────────────────────────

    async def extract_relationships(
        self,
        *,
        text: str,
        llm,
        system_prompt: str,
        max_attempts: int = settings.EXTRACTION_LLM_RETRY_ATTEMPTS,
    ) -> List[dict]:
        """
        Call the LLM with the caller-supplied system_prompt and parse the
        <relationships> XML block from the response.

        Returns an empty list (never raises) so callers can always proceed
        to build_graph() without conditional guards.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=False,
        ):
            with attempt:
                from langchain_core.messages import HumanMessage, SystemMessage

                response = await llm.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=text),
                    ]
                )
                match = _REL_RE.search(response.content or "")
                if not match:
                    logger.warning(
                        "extract_relationships: no <relationships> block found"
                    )
                    return []
                rel_text = match.group(1).strip()
                if not rel_text:
                    return []
                import json

                try:
                    relationships = json.loads(rel_text)
                    return relationships if isinstance(relationships, list) else []
                except json.JSONDecodeError:
                    logger.warning("extract_relationships: JSON parse failed")
                    return []
        return []

    # ──────────────────────────────────────────────────────────────────────────
    # 2. In-memory graph construction
    # ──────────────────────────────────────────────────────────────────────────

    async def build_graph(
        self, relationships: List[dict], *, source_agent: str
    ) -> nx.DiGraph:
        """
        Deduplicate entity names (fuzzy then semantic) and build an nx.DiGraph.
        No LLM calls, no I/O.  Idempotent — safe to call multiple times.

        Each relationship dict may carry optional `from_node_props` / `to_node_props`
        dicts with extra entity attributes (e.g. ticker, description sourced from
        yfinance).  These are attached as `node_props` on the nx node and forwarded
        to _resolve_entity during persistence, so canonical attributes survive the
        full dedup pipeline.
        """
        graph = nx.DiGraph()
        if not relationships:
            return graph

        # Collect all entity (name, type) pairs
        entities: List[Tuple[str, str]] = []
        for rel in relationships:
            for key in ("from_name", "to_name"):
                name = str(rel.get(key, "") or "").strip()
                type_ = str(
                    rel.get("from_type" if key == "from_name" else "to_type", "") or ""
                ).strip()
                if name and type_:
                    entities.append((name, type_))

        alias_to_canon = dict(self._entity_name_cache)
        canonical_by_type: Dict[str, List[str]] = {}
        for (_, entity_type), canon in alias_to_canon.items():
            canonical_by_type.setdefault(entity_type, [])
            if canon not in canonical_by_type[entity_type]:
                canonical_by_type[entity_type].append(canon)

        unresolved: List[Tuple[str, str]] = []
        for name, entity_type in entities:
            key = (name.lower(), entity_type)
            if key in alias_to_canon:
                continue
            candidates = canonical_by_type.get(entity_type, [])
            matched = next(
                (
                    c
                    for c in candidates
                    if fuzz.token_sort_ratio(name, c) >= self._fuzzy_threshold
                ),
                None,
            )
            if matched is None:
                unresolved.append((name, entity_type))
            else:
                alias_to_canon[key] = matched

        if unresolved:
            unique_names = list(
                {name for name, _ in unresolved}
                | {c for v in canonical_by_type.values() for c in v}
            )
            embeddings = await asyncio.to_thread(
                self._embedding_func.embed_documents, unique_names
            )
            emb_map = {
                name: np.array(vec)
                for name, vec in zip(unique_names, embeddings, strict=False)
            }
            for name, entity_type in unresolved:
                key = (name.lower(), entity_type)
                candidates = canonical_by_type.get(entity_type, [])
                matched = None
                vec = emb_map.get(name)
                if vec is not None:
                    for canon in candidates:
                        canon_vec = emb_map.get(canon)
                        if canon_vec is None:
                            continue
                        denom = np.linalg.norm(vec) * np.linalg.norm(canon_vec)
                        if denom > 0:
                            sim = float(np.dot(vec, canon_vec) / denom)
                            if sim >= self._semantic_threshold:
                                matched = canon
                                break
                if matched is None:
                    canonical_by_type.setdefault(entity_type, []).append(name)
                    alias_to_canon[key] = name
                else:
                    alias_to_canon[key] = matched

        self._entity_name_cache = alias_to_canon

        for rel in relationships:
            from_name_raw = str(rel.get("from_name", "") or "").strip()
            to_name_raw = str(rel.get("to_name", "") or "").strip()
            from_type = str(rel.get("from_type", "") or "").strip()
            to_type = str(rel.get("to_type", "") or "").strip()

            from_name = alias_to_canon.get((from_name_raw.lower(), from_type), "")
            to_name = alias_to_canon.get((to_name_raw.lower(), to_type), "")
            if not from_name or not to_name or not from_type or not to_type:
                continue

            from_id = canonical_entity_id(from_name, from_type)
            to_id = canonical_entity_id(to_name, to_type)

            from_node_props = rel.get("from_node_props") or {}
            to_node_props = rel.get("to_node_props") or {}

            # Add nodes — preserve existing non-empty node_props on subsequent
            # encounters of the same entity (e.g. Sector appears as both `to` in
            # Industry→Sector and `from` in Sector→Market; props from either pass
            # should not clobber each other).
            if from_id not in graph:
                graph.add_node(
                    from_id,
                    name=from_name,
                    entity_type=from_type,
                    source_agent=source_agent,
                    node_props=from_node_props,
                )
            elif from_node_props and not graph.nodes[from_id].get("node_props"):
                graph.nodes[from_id]["node_props"] = from_node_props

            if to_id not in graph:
                graph.add_node(
                    to_id,
                    name=to_name,
                    entity_type=to_type,
                    source_agent=source_agent,
                    node_props=to_node_props,
                )
            elif to_node_props and not graph.nodes[to_id].get("node_props"):
                graph.nodes[to_id]["node_props"] = to_node_props

            edge_attrs = {
                "relation_type": str(rel.get("relation", "RELATED_TO")).strip(),
                "confidence": str(rel.get("confidence", "low")).strip(),
                "reason": str(rel.get("reason", "")).strip(),
                "source_agent": source_agent,
            }
            extra = rel.get("extra_props")
            if isinstance(extra, dict):
                edge_attrs.update(extra)

            graph.add_edge(from_id, to_id, **edge_attrs)

        return graph

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Persistence
    # ──────────────────────────────────────────────────────────────────────────

    async def persist_graph(self, graph: nx.DiGraph, *, conversation_id: str) -> None:
        """Write the graph to Neo4j via the ingestor. No-ops on empty graph."""
        if graph is None or graph.number_of_edges() == 0:
            return
        await self._ingestor._upsert_graph_to_neo4j(graph, conversation_id)
        logger.debug(
            "persist_graph: wrote %d edges for conversation '%s'",
            graph.number_of_edges(),
            conversation_id,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Fire-and-forget scheduling — single entry point for all agents
    # ──────────────────────────────────────────────────────────────────────────

    async def schedule(
        self,
        *,
        agent_name: str,
        conversation_id: str,
        analysis_text: str,
        llm,
        system_prompt: str,
        relationships: Optional[List[dict]] = None,
        bypass_guards: bool = False,
    ) -> Optional[str]:
        """
        Schedule relationship extraction + graph persistence as a background task.

        Two paths:
        relationships is not None → skip LLM call, build + persist directly.
        relationships is None     → extract from analysis_text, build + persist.

        bypass_guards=True skips the EXTRACTION_ENABLED and conversation_id checks.
        Use this for system-internal upserts (taxonomy bootstrap, ticker enrichment)
        that must always run regardless of user-facing extraction settings.

        Returns the subgraph_id (a SubgraphStore key) or None when disabled
        or conversation_id is absent.
        """
        if not bypass_guards and (
            not settings.EXTRACTION_ENABLED or not conversation_id
        ):
            return None

        from core.memory.stores.subgraph_store import SubgraphStore

        subgraph_id = SubgraphStore.make_key(agent_name, conversation_id)

        async def _run() -> None:
            try:
                if relationships is not None:
                    rels = relationships
                else:
                    rels = await self.extract_relationships(
                        text=analysis_text,
                        llm=llm,
                        system_prompt=system_prompt,
                    )
                graph = await self.build_graph(rels, source_agent=agent_name)
                await self.persist_graph(graph, conversation_id=conversation_id)
                logger.info(
                    "schedule: completed '%s' (%d edges)",
                    subgraph_id,
                    graph.number_of_edges(),
                )
            except Exception:
                logger.exception(
                    "schedule: background task failed for '%s'", subgraph_id
                )

        task = asyncio.create_task(_run())

        if settings.EXTRACTION_IMMEDIATE or bypass_guards:
            await task

        return subgraph_id
