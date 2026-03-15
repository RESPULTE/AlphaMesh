"""Build and merge in-memory subgraphs with fuzzy/semantic deduplication."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from rapidfuzz import fuzz

from core.memory.graph.utils import canonical_entity_id


class InMemorySubgraphBuilder:
    def __init__(
        self, embedding_func, fuzzy_threshold: float, semantic_threshold: float
    ) -> None:
        self._embedding_func = embedding_func
        self._fuzzy_threshold = fuzzy_threshold
        self._semantic_threshold = semantic_threshold
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
