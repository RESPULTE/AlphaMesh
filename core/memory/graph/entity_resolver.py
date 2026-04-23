"""
Entity and relationship endpoint resolution for graph writes.

Resolution order for domain entities:
1) Neo4j exact (canonical ID, then exact normalized name+type)
2) Neo4j fuzzy candidates
3) Local entity vector similarity search
4) Create a new entity (when allowed)
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from core.config import settings
from core.logger import get_logger
from core.memory.graph.models import EntityNode
from core.memory.graph.utils import (
    canonical_entity_id,
    entity_key,
    normalize_entity_description,
    normalize_entity_name,
    normalize_entity_type,
    normalize_relationship_type,
)

logger = get_logger(__name__)

_MAX_CACHE_SIZE = 10_000
_CACHE_TTL_SECONDS = 3600
_DEFAULT_STRONG_FUZZY_THRESHOLD = 0.90

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

_CORP_SUFFIXES = {
    "inc",
    "inc.",
    "incorporated",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "ltd",
    "ltd.",
    "limited",
    "plc",
    "llc",
    "l.l.c.",
    "sa",
    "ag",
    "nv",
    "bv",
    "gmbh",
}


def _normalize_company_alias(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    if not cleaned:
        return ""
    tokens = [t for t in cleaned.split() if t and t not in _CORP_SUFFIXES]
    return " ".join(tokens)


def _normalize_neo4j_fuzzy_threshold(raw: float) -> float:
    value = float(raw)
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _normalize_vector_distance_threshold(semantic_threshold: float) -> float:
    similarity = max(0.0, min(1.0, float(semantic_threshold)))
    return 1.0 - similarity


@dataclass(frozen=True)
class ResolverThresholds:
    neo4j_fuzzy_threshold: float
    rapidfuzz_threshold: float
    vector_distance_threshold: float
    strong_fuzzy_threshold: float


@dataclass(frozen=True)
class EntityResolution:
    entity_id: Optional[str]
    match_stage: str
    score: Optional[float] = None
    created: bool = False

    @property
    def resolved(self) -> bool:
        return bool(self.entity_id)


@dataclass(frozen=True)
class ResolvedEdgeBatch:
    relationships: List[dict]
    entity_cache: Dict[Tuple[str, str], str]
    skipped_relationships: int


class EntityResolver:
    """
    Resolves entity endpoints and prepares deduplicated relationship batches.

    Public API:
    - resolve_entity(...)
    - resolve_entities(...)
    - resolve_relationship_edges(...)
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
    ) -> None:
        self._neo4j = neo4j_adapter
        self._chroma = entity_chroma_adapter
        self._cache_max_size = max(int(cache_max_size), 1)
        self._cache_ttl_seconds = max(int(cache_ttl_seconds), 1)

        if vector_distance_threshold is None:
            vector_distance_threshold = _normalize_vector_distance_threshold(
                settings.EXTRACTION_SEMANTIC_THRESHOLD
            )

        self._thresholds = ResolverThresholds(
            neo4j_fuzzy_threshold=_normalize_neo4j_fuzzy_threshold(
                neo4j_fuzzy_threshold
            ),
            rapidfuzz_threshold=float(rapidfuzz_threshold),
            vector_distance_threshold=max(0.0, min(1.0, vector_distance_threshold)),
            strong_fuzzy_threshold=_DEFAULT_STRONG_FUZZY_THRESHOLD,
        )

        self._cache: "OrderedDict[Tuple[str, str], Tuple[str, float]]" = OrderedDict()
        self._cache_lock = asyncio.Lock()

        self._inflight_lock_guard = asyncio.Lock()
        self._inflight_locks: Dict[Tuple[str, str], asyncio.Lock] = {}

    async def resolve_entity(
        self,
        *,
        name: str,
        entity_type: str,
        props: Optional[Any] = None,
        allow_create: bool = True,
    ) -> EntityResolution:
        clean_name = normalize_entity_name(name)
        clean_type = normalize_entity_type(entity_type)
        if not clean_name or not clean_type:
            return EntityResolution(entity_id=None, match_stage="invalid")

        cache_key = entity_key(clean_name, clean_type)
        cached_id = await self._cache_get(cache_key)
        if cached_id:
            return EntityResolution(entity_id=cached_id, match_stage="cache")

        lock = await self._get_inflight_lock(cache_key)
        async with lock:
            # Check cache again after waiting on the per-key single-flight lock.
            cached_id = await self._cache_get(cache_key)
            if cached_id:
                return EntityResolution(entity_id=cached_id, match_stage="cache")

            resolution = await self._resolve_uncached(
                name=clean_name,
                entity_type=clean_type,
                props=props,
                allow_create=allow_create,
            )
            if resolution.entity_id:
                await self._cache_set(cache_key, resolution.entity_id)
            return resolution

    async def resolve_entities(
        self,
        entities: List[Tuple[str, str, Optional[Any]]],
        *,
        allow_create: bool = True,
    ) -> Dict[Tuple[str, str], EntityResolution]:
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
                    "props": self._props_to_dict(raw_props),
                }
            else:
                merged_inputs[key]["props"] = self._merge_props(
                    merged_inputs[key]["props"],
                    self._props_to_dict(raw_props),
                )

        if not merged_inputs:
            return {}

        async def _resolve_one(key: Tuple[str, str], payload: dict) -> EntityResolution:
            return await self.resolve_entity(
                name=payload["name"],
                entity_type=payload["entity_type"],
                props=payload["props"],
                allow_create=allow_create,
            )

        tasks = [_resolve_one(key, payload) for key, payload in merged_inputs.items()]
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

            normalized = {
                "from_name": from_name,
                "from_type": from_type,
                "to_name": to_name,
                "to_type": to_type,
                "relation": relation_type,
                "confidence": confidence,
                "reason": reason,
                "extra_props": dict(extra_props),
            }
            normalized_rels.append(normalized)
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

        entity_cache: Dict[Tuple[str, str], str] = {}
        for key, resolution in resolved_endpoints.items():
            if resolution.entity_id:
                entity_cache[key] = resolution.entity_id

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
            existing["confidence"] = self._merge_confidence(
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

        exact_match = await self._find_exact_name_match(
            name=name, entity_type=entity_type
        )
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
        alias_match = self._match_company_alias(
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

        strongest_fuzzy = self._pick_strongest_fuzzy_candidate(fuzzy_candidates)
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
            return EntityResolution(
                entity_id=None, match_stage="unresolved", created=False
            )

        node = self._build_entity_node(
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
                "resolve_entity: exact name lookup failed for '%s' (%s)",
                name,
                entity_type,
            )
            return None
        if not isinstance(record, dict):
            return None
        entity_id = record.get("id")
        if entity_id:
            return str(entity_id)
        return None

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
            filtered: List[dict] = []
            for candidate in candidates:
                candidate_name = str(candidate.get("name") or "")
                if (
                    fuzz.token_sort_ratio(name, candidate_name)
                    >= self._thresholds.rapidfuzz_threshold
                ):
                    filtered.append(candidate)
            return filtered
        except Exception:
            logger.exception(
                "resolve_entity: fuzzy lookup failed for '%s' (%s)",
                name,
                entity_type,
            )
            return []

    @staticmethod
    def _match_company_alias(
        *,
        name: str,
        entity_type: str,
        candidates: List[dict],
    ) -> Optional[str]:
        if entity_type != "Company" or not candidates:
            return None
        normalized = _normalize_company_alias(name)
        if not normalized:
            return None
        for candidate in candidates:
            candidate_id = candidate.get("id")
            candidate_name = str(candidate.get("name") or "")
            if candidate_id and _normalize_company_alias(candidate_name) == normalized:
                return str(candidate_id)
        return None

    @staticmethod
    def _pick_strongest_fuzzy_candidate(
        candidates: List[dict],
    ) -> Optional[Tuple[str, float]]:
        best_id: Optional[str] = None
        best_score: float = -1.0
        for candidate in candidates:
            candidate_id = candidate.get("id")
            similarity = candidate.get("similarity")
            if not candidate_id:
                continue
            if not isinstance(similarity, (int, float)):
                continue
            score = float(similarity)
            if score > best_score:
                best_score = score
                best_id = str(candidate_id)
        if best_id is None:
            return None
        return best_id, best_score

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
                "resolve_entity: vector lookup failed for '%s' (%s)",
                name,
                entity_type,
            )
            return None

        best_candidate_id: Optional[str] = None
        best_distance: Optional[float] = None
        for doc, score in results:
            candidate_id = doc.id or (doc.metadata or {}).get("entity_id")
            distance = self._score_to_distance(score)
            if not candidate_id or distance is None:
                continue
            if distance > self._thresholds.vector_distance_threshold:
                continue
            if best_distance is None or distance < best_distance:
                best_candidate_id = str(candidate_id)
                best_distance = distance

        if best_candidate_id is None or best_distance is None:
            return None
        if not await self._neo4j.entity_exists(best_candidate_id):
            return None
        return best_candidate_id, best_distance

    @staticmethod
    def _score_to_distance(score: Optional[float]) -> Optional[float]:
        if score is None:
            return None
        if not isinstance(score, (int, float)):
            return None
        numeric = float(score)
        if numeric < 0:
            return None
        if numeric <= 1.0:
            # Chroma relevance scores are usually in [0,1] with higher=better.
            return 1.0 - numeric
        # Some backends return distance where lower=better.
        return numeric

    async def _persist_entity(self, node: EntityNode) -> None:
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
                "resolve_entity: vector upsert failed for '%s' (%s)",
                node.name,
                node.entity_type,
            )

    async def _cache_get(self, key: Tuple[str, str]) -> Optional[str]:
        now = time.time()
        async with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            entity_id, written_at = entry
            if now - written_at > self._cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return entity_id

    async def _cache_set(self, key: Tuple[str, str], entity_id: str) -> None:
        async with self._cache_lock:
            self._cache[key] = (entity_id, time.time())
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_size:
                self._cache.popitem(last=False)

    async def _get_inflight_lock(self, key: Tuple[str, str]) -> asyncio.Lock:
        async with self._inflight_lock_guard:
            lock = self._inflight_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._inflight_locks[key] = lock
            return lock

    @staticmethod
    def _props_to_dict(props: Optional[Any]) -> dict:
        if isinstance(props, dict):
            return dict(props)
        if props is None:
            return {}
        output: dict = {}
        for field in ("description", "ticker", "nodeset_ids"):
            value = getattr(props, field, None)
            if value is not None:
                output[field] = value
        return output

    @staticmethod
    def _merge_props(existing: dict, incoming: dict) -> dict:
        if not incoming:
            return dict(existing)
        merged = dict(existing)

        incoming_desc = str(incoming.get("description") or "").strip()
        existing_desc = str(merged.get("description") or "").strip()
        if incoming_desc and len(incoming_desc) > len(existing_desc):
            merged["description"] = incoming_desc

        if not merged.get("ticker") and incoming.get("ticker"):
            merged["ticker"] = incoming["ticker"]

        existing_nodeset_ids = list(merged.get("nodeset_ids") or [])
        for nodeset_id in list(incoming.get("nodeset_ids") or []):
            if nodeset_id not in existing_nodeset_ids:
                existing_nodeset_ids.append(nodeset_id)
        if existing_nodeset_ids:
            merged["nodeset_ids"] = existing_nodeset_ids

        return merged

    @staticmethod
    def _merge_confidence(existing: Any, incoming: Any) -> Any:
        existing_numeric = EntityResolver._try_float(existing)
        incoming_numeric = EntityResolver._try_float(incoming)
        if existing_numeric is not None and incoming_numeric is not None:
            return max(existing_numeric, incoming_numeric)
        if existing_numeric is not None:
            return existing
        if incoming_numeric is not None:
            return incoming

        existing_text = str(existing or "low").strip().lower()
        incoming_text = str(incoming or "low").strip().lower()
        if _CONFIDENCE_RANK.get(incoming_text, 0) > _CONFIDENCE_RANK.get(
            existing_text, 0
        ):
            return incoming
        return existing

    @staticmethod
    def _try_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _build_entity_node(
        *,
        name: str,
        entity_type: str,
        entity_id: str,
        props: Optional[Any],
    ) -> EntityNode:
        if isinstance(props, dict):
            description = props.get("description")
            ticker = props.get("ticker") or None
            nodeset_ids = list(props.get("nodeset_ids") or [])
        elif props is not None:
            description = getattr(props, "description", None)
            ticker = getattr(props, "ticker", None)
            nodeset_ids = list(getattr(props, "nodeset_ids", []) or [])
        else:
            description = None
            ticker = None
            nodeset_ids = []

        return EntityNode(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            description=normalize_entity_description(description, name),
            ticker=ticker,
            nodeset_ids=nodeset_ids,
        )
