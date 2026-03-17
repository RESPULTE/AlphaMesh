"""
core/memory/graph/subgraph_builder.py

Build and merge in-memory subgraphs with fuzzy/semantic deduplication.

Added in this revision
──────────────────────
schedule_subgraph_extraction() — a module-level async helper that
encapsulates the fire-and-forget subgraph build/store pattern that was
previously duplicated verbatim in NewsAnalysisAgent and
FundamentalAnalysisAgent.  It lives here because this module already owns
everything it needs: InMemorySubgraphBuilder, the SubgraphStore key
convention, and the retry path for failed extractions.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from rapidfuzz import fuzz

from core.config import settings
from core.memory.graph.utils import canonical_entity_id
from core.services import service_manager


class InMemorySubgraphBuilder:
    def __init__(self) -> None:
        self._embedding_func = service_manager.get_embedding_func()
        self._fuzzy_threshold = settings.EXTRACTION_FUZZY_THRESHOLD
        self._semantic_threshold = settings.EXTRACTION_SEMANTIC_THRESHOLD
        self._entity_name_cache: Dict[Tuple[str, str], str] = {}

    async def build(self, relationships: List[dict], source_agent: str) -> nx.DiGraph:
        graph = nx.DiGraph()
        if not relationships:
            return graph

        entities = []
        for rel in relationships:
            entities.append(
                (
                    str(rel.get("from_name", "")).strip(),
                    str(rel.get("from_type", "")).strip(),
                )
            )
            entities.append(
                (
                    str(rel.get("to_name", "")).strip(),
                    str(rel.get("to_type", "")).strip(),
                )
            )

        canonical_by_type: Dict[str, List[str]] = {}
        alias_to_canon: Dict[Tuple[str, str], str] = dict(self._entity_name_cache)

        for (_, entity_type), canon in alias_to_canon.items():
            canonical_by_type.setdefault(entity_type, [])
            if canon not in canonical_by_type[entity_type]:
                canonical_by_type[entity_type].append(canon)

        unresolved: List[Tuple[str, str]] = []

        for name, entity_type in entities:
            if not name or not entity_type:
                continue
            key = (name.lower(), entity_type)
            if key in alias_to_canon:
                continue
            candidates = canonical_by_type.get(entity_type, [])
            matched = None
            for canon in candidates:
                score = fuzz.token_sort_ratio(name, canon)
                if score >= self._fuzzy_threshold:
                    matched = canon
                    break
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
                if vec is not None and candidates:
                    for canon in candidates:
                        canon_vec = emb_map.get(canon)
                        if canon_vec is None:
                            continue
                        denom = np.linalg.norm(vec) * np.linalg.norm(canon_vec)
                        if denom == 0:
                            continue
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
            from_name = alias_to_canon.get(
                (
                    str(rel.get("from_name", "")).strip().lower(),
                    str(rel.get("from_type", "")).strip(),
                ),
                "",
            )
            to_name = alias_to_canon.get(
                (
                    str(rel.get("to_name", "")).strip().lower(),
                    str(rel.get("to_type", "")).strip(),
                ),
                "",
            )
            from_type = str(rel.get("from_type", "")).strip()
            to_type = str(rel.get("to_type", "")).strip()
            if not from_name or not to_name or not from_type or not to_type:
                continue

            from_id = canonical_entity_id(from_name, from_type)
            to_id = canonical_entity_id(to_name, to_type)

            graph.add_node(
                from_id,
                name=from_name,
                entity_type=from_type,
                source_agent=source_agent,
            )
            graph.add_node(
                to_id,
                name=to_name,
                entity_type=to_type,
                source_agent=source_agent,
            )

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

    async def schedule_subgraph_extraction(
        self,
        *,
        agent_name: str,
        conversation_id: str,
        analysis_text: str,
        relationships: List[dict],
        relationships_extracted: bool,
        llm: object,
    ) -> Optional[str]:
        """
        Build and persist a relationship subgraph for one agent turn.

        Encapsulates the fire-and-forget pattern shared by every agent that
        produces relationship data (NewsAnalysisAgent, FundamentalAnalysisAgent).

        Parameters
        ----------
        agent_name:
            Calling agent's canonical name (e.g. ``"news_agent"``).
            Used as the SubgraphStore key prefix and the ``source_agent``
            label on every graph edge.
        conversation_id:
            Current conversation ID.  Returns None immediately if empty.
        analysis_text:
            The agent's written analysis.  Passed to ``retry_relationships_only``
            when ``relationships_extracted`` is False for a second-pass LLM
            extraction from the prose alone.
        relationships:
            Relationship dicts from ``extract_with_retry``.  May be empty when
            first-pass extraction failed — the retry path handles that.
        relationships_extracted:
            True  → first-pass succeeded; build directly from ``relationships``.
            False → first-pass failed; schedule ``retry_relationships_only``.
        llm:
            LLM instance for the retry path.  Callers supply their own so that
            temperature / model choices stay with the agent, not this helper.

        Returns
        -------
        The ``subgraph_id`` string scheduled for storage, or ``None`` when
        extraction is disabled or ``conversation_id`` is absent.

        Notes
        -----
        - Never raises.  All exceptions inside the background task are caught
          and logged so a failed write never surfaces to the user.
        - When ``settings.EXTRACTION_IMMEDIATE`` is True the task is awaited
          inline (useful for tests and synchronous debugging).
        """
        # Deferred imports to avoid a circular dependency:
        # relationship_extractor imports InMemorySubgraphBuilder from this
        # module, so importing relationship_extractor at the top level here
        # would create a cycle.
        from core.config import settings
        from core.logger import get_logger
        from core.memory.graph.extraction_prompts import (
            ANALYSIS_ONLY_RELATIONSHIP_PROMPT,
        )
        from core.memory.graph.relationship_extractor import retry_relationships_only
        from core.memory.stores.subgraph_store import SubgraphStore
        from core.services import service_manager

        logger = get_logger(__name__)

        if not settings.EXTRACTION_ENABLED or not conversation_id:
            return None

        store = service_manager.get_subgraph_store()
        subgraph_id = SubgraphStore.make_key(agent_name, conversation_id)

        async def _build_and_store() -> None:
            try:
                graph = await self.build(relationships, source_agent=agent_name)
                await store.save(subgraph_id, graph)
                logger.debug(
                    "schedule_subgraph_extraction: saved '%s' (%d edges)",
                    subgraph_id,
                    graph.number_of_edges(),
                )
            except Exception:
                logger.exception(
                    "schedule_subgraph_extraction: _build_and_store failed for '%s'",
                    subgraph_id,
                )

        if relationships_extracted:
            task = asyncio.create_task(_build_and_store())
        else:
            task = asyncio.create_task(
                retry_relationships_only(
                    llm,
                    analysis_text,
                    agent_name,
                    conversation_id,
                    self,
                    store,
                    subgraph_id,
                    ANALYSIS_ONLY_RELATIONSHIP_PROMPT,
                )
            )

        if settings.EXTRACTION_IMMEDIATE:
            await task

        return subgraph_id


async def merge_subgraphs(
    g1: nx.DiGraph, g2: nx.DiGraph, cross_edges: List[dict]
) -> nx.DiGraph:
    merged = nx.compose(g1, g2)
    for edge in cross_edges or []:
        src = edge.get("source_id")
        tgt = edge.get("target_id")
        if not src or not tgt:
            continue
        merged.add_edge(src, tgt, **edge.get("props", {}))
    return merged
