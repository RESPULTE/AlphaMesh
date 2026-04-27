"""
Entity and relationship endpoint resolution for graph writes.

Resolution order for domain entities:
1) Positive resolution cache (in-memory LRU+TTL)
2) Negative cache short-circuit (avoid re-querying known-unresolvable entities)
3) Neo4j exact canonical ID
4) Neo4j exact normalised name + type
5) Neo4j fuzzy candidates  →  company-alias match  →  strongest-score match
6) Local entity vector similarity search
7) Create a new entity (when allowed)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from core.config import settings
from core.logger import get_logger
from core.memory.graph.utils import (
    canonical_entity_id,
    entity_key,
    normalize_entity_description,
    normalize_entity_name,
    normalize_entity_type,
    normalize_relationship_type,
)

from .cache import ResolutionCache
from .fuzzy import match_company_alias, pick_strongest_fuzzy_candidate
from .props import build_entity_node, merge_confidence, merge_props, props_to_dict
from .types import EntityResolution, ResolvedEdgeBatch, ResolverThresholds

logger = get_logger(__name__)

_MAX_CACHE_SIZE = 10_000
_CACHE_TTL_SECONDS = 3600
_DEFAULT_STRONG_FUZZY_THRESHOLD = 0.90
_DEFAULT_RESOLVE_CONCURRENCY = 10


def _normalize_neo4j_fuzzy_threshold(raw: float) -> float:
    """Ensure the Neo4j fuzzy threshold is in [0, 1] regardless of input scale."""
    value = float(raw)
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _normalize_vector_distance_threshold(semantic_threshold: float) -> float:
    """Convert a [0, 1] similarity threshold to a cosine distance threshold."""
    similarity = max(0.0, min(1.0, float(semantic_threshold)))
    return 1.0 - similarity


class EntityResolver:
    """Resolves entity endpoints and prepares deduplicated relationship batches.

    Public API:
    - ``resolve_entity(name, entity_type, props, allow_create)``
    - ``resolve_entities(entities, allow_create)``
    - ``resolve_relationship_edges(relationships, allow_create)``
    """

    def __init__(
        self,
        neo4j_adapter,
        entity_chroma_adapter,
        *,
        neo4j_fuzzy_threshold: float = settings.EXTRACTION_FUZZY_THRESHOLD,
        rapidfuzz_threshold: float = settings.EXTRACTION_FUZZY_THRESHOLD,
        vector_distance_threshold: Optional[float] = None,
        cache_max_size: int = _MAX_CACHE_SIZE,
        cache_ttl_seconds: int = _CACHE_TTL_SECONDS,
        resolve_concurrency: int = _DEFAULT_RESOLVE_CONCURRENCY,
    ) -> None:
        self._neo4j = neo4j_adapter
        self._chroma = entity_chroma_adapter

        if vector_distance_threshold is None:
            vector_distance_threshold = _normalize_vector_distance_threshold(
                settings.EXTRACTION_SEMANTIC_THRESHOLD
            )

        self._thresholds = ResolverThresholds(
            neo4j_fuzzy_threshold=_normalize_neo4j_fuzzy_threshold(
                neo4j_fuzzy_threshold
            ),
            # Stored as [0, 1]; multiplied by 100 when passed to rapidfuzz
            # (which returns scores in [0, 100]).
            rapidfuzz_threshold=max(0.0, min(1.0, float(rapidfuzz_threshold))),
            vector_distance_threshold=max(0.0, min(1.0, vector_distance_threshold)),
            strong_fuzzy_threshold=_DEFAULT_STRONG_FUZZY_THRESHOLD,
        )

        self._cache = ResolutionCache(
            max_size=max(int(cache_max_size), 1),
            ttl_seconds=max(int(cache_ttl_seconds), 1),
        )
        self._resolve_semaphore = asyncio.Semaphore(
            max(int(resolve_concurrency), 1)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve_entity(
        self,
        *,
        name: str,
        entity_type: str,
        props: Optional[Any] = None,
        allow_create: bool = True,
    ) -> EntityResolution:
        """Resolve a single entity, returning an :class:`EntityResolution`."""
        clean_name = normalize_entity_name(name)
        clean_type = normalize_entity_type(entity_type)
        if not clean_name or not clean_type:
            return EntityResolution(entity_id=None, match_stage="invalid")

        cache_key = entity_key(clean_name, clean_type)

        # 1. Positive cache fast-path.
        cached_id = await self._cache.get(cache_key)
        if cached_id:
            return EntityResolution(entity_id=cached_id, match_stage="cache")

        # 2. Negative cache short-circuit (only when creation is not allowed).
        if not allow_create and await self._cache.is_negative(cache_key):
            return EntityResolution(entity_id=None, match_stage="unresolved")

        # 3. Single-flight: serialise concurrent resolutions of the same key.
        lock = await self._cache.get_lock(cache_key)
        try:
            async with lock:
                # Re-check positive cache after waiting on the lock.
                cached_id = await self._cache.get(cache_key)
                if cached_id:
                    return EntityResolution(entity_id=cached_id, match_stage="cache")

                resolution = await self._resolve_uncached(
                    name=clean_name,
                    entity_type=clean_type,
                    props=props,
                    allow_create=allow_create,
                )

                if resolution.entity_id:
                    await self._cache.set(cache_key, resolution.entity_id)
                    # A successful resolution invalidates any stale negative entry.
                    await self._cache.clear_negative(cache_key)
                elif not allow_create:
                    # Cache the negative result to avoid re-querying on every call.
                    await self._cache.set_negative(cache_key)

                return resolution
        finally:
            # Clean up the per-key lock so the dict does not grow unboundedly.
            await self._cache.release_lock(cache_key)

    async def resolve_entities(
        self,
        entities: List[Tuple[str, str, Optional[Any]]],
        *,
        allow_create: bool = True,
    ) -> Dict[Tuple[str, str], EntityResolution]:
        """Resolve a batch of entities, deduplicating inputs by (name, type)."""
        if not entities:
            return {}

        merged_inputs: Dict[Tuple[str, str], dict] = {}
        for raw_name, raw_type, raw_props in entities:
            clean_name = normalize_entity_name(raw_name)
            clean_type = normalize_entity_type(raw_type)
            if not clean_name or not clean_type:
                continue
            key = entity_key(clean_name, clean_type)
            if key not in merged_inputs:
                merged_inputs[key] = {
                    "name": clean_name,
                    "entity_type": clean_type,
                    "props": props_to_dict(raw_props),
                }
            else:
                merged_inputs[key]["props"] = merge_props(
                    merged_inputs[key]["props"],
                    props_to_dict(raw_props),
                )

        if not merged_inputs:
            return {}

        async def _resolve_one(key: Tuple[str, str], payload: dict) -> EntityResolution:
            async with self._resolve_semaphore:
                return await self.resolve_entity(
                    name=payload["name"],
                    entity_type=payload["entity_type"],
                    props=payload["props"],
                    allow_create=allow_create,
                )

        tasks = [_resolve_one(k, p) for k, p in merged_inputs.items()]
        results = await asyncio.gather(*tasks)
        return {
            key: result
            for (key, _payload), result in zip(merged_inputs.items(), results)
        }

    async def resolve_relationship_edges(
        self,
        relationships: List[dict],
        *,
        allow_create: bool,
    ) -> ResolvedEdgeBatch:
        """Normalise, resolve endpoints, and deduplicate a batch of relationship dicts."""
        if not relationships:
            return ResolvedEdgeBatch(
                relationships=[], entity_cache={}, skipped_relationships=0
            )

        normalized_rels: List[dict] = []
        endpoint_inputs: List[Tuple[str, str, Optional[Any]]] = []

        for rel in relationships:
            from_name = normalize_entity_name(str(rel.get("from_name") or ""))
            to_name = normalize_entity_name(str(rel.get("to_name") or ""))
            from_type = normalize_entity_type(str(rel.get("from_type") or "").strip())
            to_type = normalize_entity_type(str(rel.get("to_type") or "").strip())
            if not from_name or not to_name or not from_type or not to_type:
                continue

            relation_type = normalize_relationship_type(
                str(rel.get("relation") or rel.get("relation_type") or "RELATED_TO")
            )
            confidence = rel.get("confidence", "low")
            reason = str(rel.get("reason") or "").strip()
            extra_props = rel.get("extra_props")
            if not isinstance(extra_props, dict):
                extra_props = {}

            normalized_rels.append(
                {
                    "from_name": from_name,
                    "from_type": from_type,
                    "to_name": to_name,
                    "to_type": to_type,
                    "relation": relation_type,
                    "confidence": confidence,
                    "reason": reason,
                    "extra_props": dict(extra_props),
                }
            )
            endpoint_inputs.append((from_name, from_type, rel.get("from_node_props")))
            endpoint_inputs.append((to_name, to_type, rel.get("to_node_props")))

        if not normalized_rels:
            return ResolvedEdgeBatch(
                relationships=[], entity_cache={}, skipped_relationships=0
            )

        resolved_endpoints = await self.resolve_entities(
            endpoint_inputs,
            allow_create=allow_create,
        )

        entity_cache: Dict[Tuple[str, str], str] = {
            key: res.entity_id
            for key, res in resolved_endpoints.items()
            if res.entity_id
        }

        merged_edges: Dict[Tuple[str, str, str], dict] = {}
        skipped = 0

        for rel in normalized_rels:
            from_key = entity_key(rel["from_name"], rel["from_type"])
            to_key = entity_key(rel["to_name"], rel["to_type"])
            source_id = entity_cache.get(from_key)
            target_id = entity_cache.get(to_key)
            if not source_id or not target_id:
                skipped += 1
                continue

            dedup_key = (source_id, rel["relation"], target_id)
            if dedup_key not in merged_edges:
                merged_edges[dedup_key] = {
                    "from_name": rel["from_name"],
                    "from_type": rel["from_type"],
                    "to_name": rel["to_name"],
                    "to_type": rel["to_type"],
                    "relation": rel["relation"],
                    "confidence": rel["confidence"],
                    "reason": rel["reason"],
                    "extra_props": dict(rel["extra_props"]),
                }
                continue

            existing = merged_edges[dedup_key]
            existing["confidence"] = merge_confidence(
                existing.get("confidence"),
                rel.get("confidence"),
            )
            if not existing.get("reason") and rel.get("reason"):
                existing["reason"] = rel["reason"]
            for prop_key, prop_value in rel["extra_props"].items():
                if prop_key not in existing["extra_props"]:
                    existing["extra_props"][prop_key] = prop_value

        return ResolvedEdgeBatch(
            relationships=list(merged_edges.values()),
            entity_cache=entity_cache,
            skipped_relationships=skipped,
        )

    # ------------------------------------------------------------------
    # Internal resolution pipeline
    # ------------------------------------------------------------------

    async def _resolve_uncached(
        self,
        *,
        name: str,
        entity_type: str,
        props: Optional[Any],
        allow_create: bool,
    ) -> EntityResolution:
        canonical_id = canonical_entity_id(name, entity_type)

        if await self._neo4j.entity_exists(canonical_id):
            return EntityResolution(
                entity_id=canonical_id,
                match_stage="exact_id",
                score=1.0,
                created=False,
            )

        exact_match = await self._find_exact_name_match(name=name, entity_type=entity_type)
        if exact_match:
            return EntityResolution(
                entity_id=exact_match,
                match_stage="exact_name",
                score=1.0,
                created=False,
            )

        fuzzy_candidates = await self._find_fuzzy_candidates(
            name=name, entity_type=entity_type
        )

        alias_match = match_company_alias(
            name=name,
            entity_type=entity_type,
            candidates=fuzzy_candidates,
        )
        if alias_match:
            return EntityResolution(
                entity_id=alias_match,
                match_stage="fuzzy_alias",
                score=1.0,
                created=False,
            )

        strongest_fuzzy = pick_strongest_fuzzy_candidate(fuzzy_candidates)
        if (
            strongest_fuzzy is not None
            and strongest_fuzzy[1] >= self._thresholds.strong_fuzzy_threshold
        ):
            return EntityResolution(
                entity_id=strongest_fuzzy[0],
                match_stage="fuzzy",
                score=strongest_fuzzy[1],
                created=False,
            )

        vector_match = await self._find_vector_match(name=name, entity_type=entity_type)
        if vector_match is not None:
            return EntityResolution(
                entity_id=vector_match[0],
                match_stage="vector",
                score=1.0 - vector_match[1],
                created=False,
            )

        if not allow_create:
            return EntityResolution(entity_id=None, match_stage="unresolved", created=False)

        node = build_entity_node(
            name=name,
            entity_type=entity_type,
            entity_id=canonical_id,
            props=props,
        )
        await self._persist_entity(node)
        return EntityResolution(
            entity_id=canonical_id,
            match_stage="created",
            score=None,
            created=True,
        )

    # ------------------------------------------------------------------
    # Backend lookups
    # ------------------------------------------------------------------

    async def _find_exact_name_match(
        self, *, name: str, entity_type: str
    ) -> Optional[str]:
        if not hasattr(self._neo4j, "find_entity_by_name"):
            return None
        try:
            record = await self._neo4j.find_entity_by_name(
                entity_type=entity_type,
                name=name,
            )
        except Exception:
            logger.exception(
                "EntityResolver: exact name lookup failed for '%s' (%s)",
                name,
                entity_type,
            )
            return None
        if not isinstance(record, dict):
            return None
        entity_id = record.get("id")
        return str(entity_id) if entity_id else None

    async def _find_fuzzy_candidates(
        self, *, name: str, entity_type: str
    ) -> List[dict]:
        try:
            candidates = await self._neo4j.find_fuzzy_entity_candidates(
                entity_type=entity_type,
                name=name,
                threshold=self._thresholds.neo4j_fuzzy_threshold,
                limit=settings.MEMORY_VECTOR_TOP_K,
            )
        except Exception:
            logger.exception(
                "EntityResolver: fuzzy lookup failed for '%s' (%s)",
                name,
                entity_type,
            )
            return []

        # rapidfuzz returns [0, 100]; our threshold is stored as [0, 1].
        # Bug fix: previously compared 0–100 score against a 0–1 threshold,
        # making this filter a no-op.  Now we scale correctly.
        rapidfuzz_cutoff = self._thresholds.rapidfuzz_threshold * 100
        filtered: List[dict] = []
        for candidate in candidates:
            candidate_name = str(candidate.get("name") or "")
            if fuzz.token_sort_ratio(name, candidate_name) >= rapidfuzz_cutoff:
                filtered.append(candidate)
        return filtered

    async def _find_vector_match(
        self,
        *,
        name: str,
        entity_type: str,
    ) -> Optional[Tuple[str, float]]:
        if self._chroma is None:
            return None

        text = f"{name}. {normalize_entity_description(None, name)}"
        try:
            results = await self._chroma.query_entity_similar(
                text=text,
                entity_type=entity_type,
                n_results=settings.MEMORY_VECTOR_TOP_K,
            )
        except Exception:
            logger.exception(
                "EntityResolver: vector lookup failed for '%s' (%s)",
                name,
                entity_type,
            )
            return None

        best_id: Optional[str] = None
        best_distance: Optional[float] = None
        for doc, score in results:
            candidate_id = doc.id or (doc.metadata or {}).get("entity_id")
            distance = _score_to_distance(score)
            if not candidate_id or distance is None:
                continue
            if distance > self._thresholds.vector_distance_threshold:
                continue
            if best_distance is None or distance < best_distance:
                best_id = str(candidate_id)
                best_distance = distance

        if best_id is None or best_distance is None:
            return None
        if not await self._neo4j.entity_exists(best_id):
            return None
        return best_id, best_distance

    async def _persist_entity(self, node) -> None:
        await self._neo4j.merge_entity_node(node)
        if self._chroma is None:
            return
        try:
            await self._chroma.upsert_entity_embedding(
                entity_id=node.id,
                name=node.name,
                description=node.description,
                entity_type=node.entity_type,
            )
        except Exception:
            logger.exception(
                "EntityResolver: vector upsert failed for '%s' (%s)",
                node.name,
                node.entity_type,
            )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _score_to_distance(score: Optional[float]) -> Optional[float]:
    """Convert a Chroma relevance score to a distance value (lower = more similar).

    Chroma relevance scores are usually in [0, 1] with higher = better.
    Some backends return raw distance where lower = better.
    """
    if score is None:
        return None
    if not isinstance(score, (int, float)):
        return None
    numeric = float(score)
    if numeric < 0:
        return None
    if numeric <= 1.0:
        return 1.0 - numeric
    return numeric
