"""
core/memory/graph/entity_resolver.py

Centralised entity resolution: canonical ID lookup, fuzzy+semantic dedup,
Neo4j + Chroma persistence.  Extracted from DualStoreIngestor so a single,
lock-protected LRU cache is shared across all conversations and users.

Design
──────
- resolve()       → single entity, creates Neo4j/Chroma node if new
- resolve_batch() → concurrent batch, type-scoped dedup, returns id map
- resolve_user_node() → deterministic user-scoped nodes (no dedup, no cache)

Thread-safety
─────────────
_cache is protected by _cache_lock.  The lock is only held during cache
reads/writes — never during I/O (Neo4j, Chroma, embeddings).  This means
concurrent callers may redundantly resolve the same new entity, but that is
safe because Neo4j MERGE is idempotent and the cache will converge.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

from core.config import settings
from core.logger import get_logger
from core.memory.graph.models import _USER_SCOPED_TYPES, EntityNode
from core.memory.graph.utils import (
    canonical_entity_id,
    entity_key,
    normalize_entity_description,
    normalize_entity_name,
    normalize_entity_type,
)

logger = get_logger(__name__)

_MAX_CACHE_SIZE = 10_000
_EVICT_FRACTION = 0.20  # evict oldest 20% when cap reached
_FUZZY_CANDIDATE_THRESHOLD = 0.50
_SEMANTIC_MERGE_THRESHOLD = 0.85
_VECTOR_TOP_K = 10


class EntityResolver:
    """
    Resolves entity names to canonical IDs, creating Entity nodes as needed.

    Injected dependencies (set at construction, never fetched from service_manager):
        neo4j_adapter          — for MERGE and existence checks
        entity_chroma_adapter  — for semantic similarity search + embedding upsert
        embedding_func         — for embedding new entity names during dedup
        fuzzy_threshold        — minimum rapidfuzz token_sort_ratio to consider a match
        semantic_threshold     — minimum cosine similarity to consider a merge
    """

    def __init__(
        self,
        neo4j_adapter,
        entity_chroma_adapter,
        embedding_func,
        fuzzy_threshold: float = settings.EXTRACTION_FUZZY_THRESHOLD,
        semantic_threshold: float = settings.EXTRACTION_SEMANTIC_THRESHOLD,
    ) -> None:
        self._neo4j = neo4j_adapter
        self._chroma = entity_chroma_adapter
        self._embedding_func = embedding_func
        self._fuzzy_threshold = fuzzy_threshold
        self._semantic_threshold = semantic_threshold

        # Global LRU alias cache: (name.lower(), entity_type) → canonical_id
        self._cache: Dict[Tuple[str, str], str] = {}
        self._cache_lock = asyncio.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Public: single entity resolution
    # ──────────────────────────────────────────────────────────────────────────

    async def resolve(
        self,
        name: str,
        entity_type: str,
        props: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Resolve a single domain entity to a canonical ID.
        Creates the Entity node in Neo4j and Chroma if it does not exist.
        Returns None if name or entity_type is invalid.
        """
        name = normalize_entity_name(name)
        if not name or not entity_type:
            return None

        etype = normalize_entity_type(entity_type)
        if not etype:
            return None

        cache_key = entity_key(name, etype)

        async with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        entity_id = await self._resolve_internal(name, etype, props)

        if entity_id:
            await self._cache_set(cache_key, entity_id)

        return entity_id

    # ──────────────────────────────────────────────────────────────────────────
    # Public: batch resolution with type-scoped dedup
    # ──────────────────────────────────────────────────────────────────────────

    async def resolve_batch(
        self,
        entities: List[Tuple[str, str, Optional[dict]]],
    ) -> Dict[Tuple[str, str], str]:
        """
        Resolve multiple (name, entity_type, props) tuples concurrently.

        Dedup is TYPE-SCOPED: fuzzy + semantic matching only runs between
        entities of the same type.  Returns {(name, entity_type) → canonical_id}.
        Empty or invalid entries are silently skipped.
        """
        if not entities:
            return {}

        # Deduplicate input list — same (name, type) should not be resolved twice
        seen: set = set()
        unique: List[Tuple[str, str, Optional[dict]]] = []
        for name, etype, props in entities:
            name = normalize_entity_name(name)
            etype_norm = normalize_entity_type(etype) if etype else None
            if not name or not etype_norm:
                continue
            key = (name.lower(), etype_norm)
            if key not in seen:
                seen.add(key)
                unique.append((name, etype_norm, props))

        # Check cache first (under one lock acquisition)
        result: Dict[Tuple[str, str], str] = {}
        uncached: List[Tuple[str, str, Optional[dict]]] = []

        async with self._cache_lock:
            for name, etype, props in unique:
                ck = entity_key(name, etype)
                if ck in self._cache:
                    result[(name, etype)] = self._cache[ck]
                else:
                    uncached.append((name, etype, props))

        if not uncached:
            return result

        # Run type-scoped fuzzy+semantic dedup on uncached entities
        alias_map = await self._type_scoped_dedup(uncached)

        # Resolve concurrently (each resolution is idempotent)
        async def _resolve_one(name: str, etype: str, props: Optional[dict]) -> None:
            canonical_name = alias_map.get((name.lower(), etype), name)
            entity_id = await self._resolve_internal(canonical_name, etype, props)
            if entity_id:
                result[(name, etype)] = entity_id
                # Cache both the original name and the canonical alias
                updates = {entity_key(name, etype): entity_id}
                if canonical_name != name:
                    updates[entity_key(canonical_name, etype)] = entity_id
                await self._cache_set_many(updates)

        await asyncio.gather(*[_resolve_one(n, e, p) for n, e, p in uncached])
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Public: user-scoped node resolution (no dedup, no cache)
    # ──────────────────────────────────────────────────────────────────────────

    async def resolve_user_node(
        self,
        name: str,
        entity_type: str,
        props: dict,
    ) -> Optional[str]:
        """
        Persist a user-scoped node (UserInterestDomain, UserInterestEdge, TurnNode).
        IDs are deterministic UUIDs — no fuzzy/semantic dedup, not cached.
        Delegates to the appropriate Neo4j merge method by entity_type.
        Returns the node ID or None on failure.
        """
        if entity_type not in _USER_SCOPED_TYPES:
            logger.warning(
                "resolve_user_node called for non-user-scoped type: %s", entity_type
            )
            return None

        node_id: Optional[str] = None

        try:
            if entity_type == "UserInterestDomain":
                node_id = props.get("id") or name
                await self._neo4j.merge_user_interest_domain(
                    domain_id=node_id, props=props
                )

            elif entity_type == "UserInterestEdge":
                node_id = props.get("id") or name
                operation = props.get("operation", "reinforce")
                weight_delta = float(props.get("weight_delta", 1.0))
                await self._neo4j.merge_user_interest_edge(
                    edge_id=node_id,
                    props=props,
                    operation=operation,
                    weight_delta=weight_delta,
                )

            elif entity_type == "TurnNode":
                node_id = props.get("id") or name
                await self._neo4j.merge_turn_node(turn_id=node_id, props=props)

        except Exception:
            logger.exception(
                "resolve_user_node: failed to persist %s '%s'", entity_type, name
            )
            return None

        return node_id

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: single-entity resolution pipeline
    # ──────────────────────────────────────────────────────────────────────────

    async def _resolve_internal(
        self,
        name: str,
        entity_type: str,
        props: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Resolve or create one domain entity.  Never touches the cache directly.
        """
        entity_id = canonical_entity_id(name, entity_type)

        # Fast path: entity already exists by canonical ID
        if await self._neo4j.entity_exists(entity_id):
            return entity_id

        # Semantic similarity fallback (finds renamed/aliased entities)
        similar_id = await self._find_similar(name, entity_type, entity_id)
        if similar_id:
            return similar_id

        # Create new entity
        node = self._build_entity_node(name, entity_type, entity_id, props)
        await self._persist_entity(node)
        return entity_id

    async def _find_similar(
        self, name: str, entity_type: str, exclude_id: str
    ) -> Optional[str]:
        """Check Neo4j fuzzy candidates, then Chroma semantic similarity."""
        if self._chroma is None:
            return None

        try:
            fuzzy_candidates = await self._neo4j.find_fuzzy_entity_candidates(
                entity_type=entity_type,
                name=name,
                exclude_id=exclude_id,
                threshold=_FUZZY_CANDIDATE_THRESHOLD,
                limit=_VECTOR_TOP_K,
            )
            candidate_ids = set(fuzzy_candidates) if fuzzy_candidates else None
        except Exception:
            logger.exception(
                "_find_similar: fuzzy candidate query failed for '%s'", name
            )
            candidate_ids = None

        description = normalize_entity_description(None, name)
        text = f"{name}. {description}"
        try:
            results = await self._chroma.query_entity_similar(
                text=text,
                entity_type=entity_type,
                n_results=_VECTOR_TOP_K,
            )
        except Exception:
            logger.exception(
                "_find_similar: entity embedding search failed for '%s'", name
            )
            return None

        for doc, distance in results:
            candidate_id = doc.id or (doc.metadata or {}).get("entity_id")
            if not candidate_id or distance is None:
                continue
            if candidate_ids is not None and candidate_id not in candidate_ids:
                continue
            if (1.0 - float(distance)) >= _SEMANTIC_MERGE_THRESHOLD:
                return str(candidate_id)

        return None

    async def _persist_entity(self, node: EntityNode) -> None:
        """Write entity to Neo4j and optionally Chroma."""
        await self._neo4j.merge_entity_node(node)
        if self._chroma is not None:
            try:
                await self._chroma.upsert_entity_embedding(
                    entity_id=node.id,
                    name=node.name,
                    description=node.description,
                    entity_type=node.entity_type,
                )
            except Exception:
                logger.exception(
                    "_persist_entity: Chroma upsert failed for '%s'", node.name
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: type-scoped fuzzy + semantic dedup
    # ──────────────────────────────────────────────────────────────────────────

    async def _type_scoped_dedup(
        self,
        entities: List[Tuple[str, str, Optional[dict]]],
    ) -> Dict[Tuple[str, str], str]:
        """
        Build an alias_map: (name.lower(), entity_type) → canonical_name.

        Only entities of the SAME entity_type are compared against each other.
        Uses rapidfuzz fuzzy matching first (cheap), then cosine similarity
        on embeddings (expensive, only for unresolved candidates).

        Returns a dict mapping (name.lower(), type) → canonical_name.
        Entities that are their own canonical form are not in the dict
        (callers should default to the original name if key missing).
        """
        # Group by entity_type
        by_type: Dict[str, List[str]] = {}
        for name, etype, _ in entities:
            by_type.setdefault(etype, []).append(name)

        alias_map: Dict[Tuple[str, str], str] = {}

        for etype, names in by_type.items():
            # Build canonicals list: first name of each cluster is canonical
            canonicals: List[str] = []
            for name in names:
                key = (name.lower(), etype)
                # Fuzzy match against existing canonicals of this type
                matched = next(
                    (
                        c
                        for c in canonicals
                        if fuzz.token_sort_ratio(name, c) >= self._fuzzy_threshold
                    ),
                    None,
                )
                if matched:
                    alias_map[key] = matched
                else:
                    canonicals.append(name)
                    # key maps to itself implicitly (no entry needed)

            # Semantic pass: re-check unresolved names against canonicals
            unresolved_keys = [
                (name, etype)
                for name in names
                if (name.lower(), etype) not in alias_map
                and sum(1 for c in canonicals if c != name) > 0
            ]
            if not unresolved_keys or self._embedding_func is None:
                continue

            all_names = list({name for name, _ in unresolved_keys} | set(canonicals))
            try:
                embeddings = await asyncio.to_thread(
                    self._embedding_func.embed_documents, all_names
                )
                emb_map = {n: np.array(v) for n, v in zip(all_names, embeddings)}
            except Exception:
                logger.exception(
                    "_type_scoped_dedup: embedding call failed for type '%s'", etype
                )
                continue

            for name, _ in unresolved_keys:
                key = (name.lower(), etype)
                if key in alias_map:
                    continue
                vec = emb_map.get(name)
                if vec is None:
                    continue
                for canon in canonicals:
                    if canon == name:
                        continue
                    canon_vec = emb_map.get(canon)
                    if canon_vec is None:
                        continue
                    denom = np.linalg.norm(vec) * np.linalg.norm(canon_vec)
                    if (
                        denom > 0
                        and np.dot(vec, canon_vec) / denom >= self._semantic_threshold
                    ):
                        alias_map[key] = canon
                        break

        return alias_map

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: cache helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _cache_set(self, key: Tuple[str, str], value: str) -> None:
        async with self._cache_lock:
            self._cache[key] = value
            if len(self._cache) >= _MAX_CACHE_SIZE:
                self._evict()

    async def _cache_set_many(self, updates: Dict[Tuple[str, str], str]) -> None:
        async with self._cache_lock:
            self._cache.update(updates)
            if len(self._cache) >= _MAX_CACHE_SIZE:
                self._evict()

    def _evict(self) -> None:
        """Evict oldest 20% of entries (FIFO — Python dict preserves insertion order)."""
        evict_count = int(_MAX_CACHE_SIZE * _EVICT_FRACTION)
        keys_to_evict = list(self._cache.keys())[:evict_count]
        for k in keys_to_evict:
            del self._cache[k]
        logger.debug("EntityResolver: evicted %d cache entries", evict_count)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: entity node builder
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_entity_node(
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
