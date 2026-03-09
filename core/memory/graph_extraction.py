"""
core/memory/graph_extraction.py

Two-pass financial graph extraction pipeline.

Section 1 — Parallel shallow pass (LLM Call 1 per chunk)
    Each chunk independently extracts entity names + types only.
    Uses FINANCIAL_NODE_EXTRACTION_PROMPT.

Deduplication (0 LLM calls — across all chunks after Section 1)
    Combines all extracted entities across chunks.
    Validates Sector names against ALL_MAIN_SECTORS (immutable whitelist).
    Detects near-duplicates via APOC fuzzy matching (Neo4j) + vector
    semantic confirmation — reusing the core logic from entity_merger.py
    without writing new nodes to the database.
    If an existing Neo4j node is matched, its cognee_id is reused so that
    Section 2 updates the existing node rather than creating a duplicate.
    Builds a stable canonical_entity_pool + per-chunk entity map.

Section 2 — Parallel per-chunk attribute + relationship pass (LLM Call 2 per chunk)
    Schema is sliced to only include entity types present in the chunk.
    The canonical entity list (names + types) is injected into the user
    message so the LLM populates attributes without hallucinating new names.
    Uses FINANCIAL_ATTRIBUTE_EXTRACTION_PROMPT.
    Returns chunk.contains = FinancialKnowledgeGraph (ready for add_data_points).

The original DocumentChunk objects are carried through unchanged; only
chunk.contains is populated at the end.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from pydantic import BaseModel, Field, create_model

from core.memory.graph_models import (
    ALL_MAIN_SECTORS,
    Company,
    FinancialConcept,
    FinancialEvent,
    FinancialKnowledgeGraph,
    Industry,
    UserInvestmentInterest,
    UserLearningInterest,
)
from core.memory.prompts import (
    FINANCIAL_NODE_EXTRACTION_PROMPT,
    build_attribute_extraction_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (mirrors entity_merger.py)
# ---------------------------------------------------------------------------

FUZZY_CANDIDATE_THRESHOLD = 0.50
SEMANTIC_MERGE_THRESHOLD = 0.85
# Higher threshold for in-batch difflib matching — prevents over-merging
# of descriptive user-entity names or similar-sounding-but-distinct concepts.
IN_BATCH_MERGE_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Maps entity type name → the DataPoint subclass (for schema slicing)
# ---------------------------------------------------------------------------

_ENTITY_TYPE_MAP: Dict[str, Type[DataPoint]] = {
    "Company": Company,
    "FinancialConcept": FinancialConcept,
    "FinancialEvent": FinancialEvent,
    "Industry": Industry,
    "UserInvestmentInterest": UserInvestmentInterest,
    "UserLearningInterest": UserLearningInterest,
}

# Types that appear as relationship *targets* in other entities.
# Always included in every sliced schema so nested relationships resolve.
_ALWAYS_INCLUDE_IN_SCHEMA = {
    "Company",
    "FinancialEvent",
    "FinancialConcept",
    "Industry",
}

_ALL_TYPE_NAMES = Literal[
    "Company",
    "FinancialConcept",
    "FinancialEvent",
    "Industry",
    "UserInvestmentInterest",
    "UserLearningInterest",
    "Sector",
]

# ---------------------------------------------------------------------------
# Section 1 — intermediary models (never persisted to DB)
# ---------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    """Shallow entity from Section 1 LLM call — name and type only."""

    name: str = Field(description="Canonical entity name.")
    entity_type: _ALL_TYPE_NAMES = Field(  # type: ignore[valid-type]
        description="Entity type from the allowed list."
    )


class ChunkNodeList(BaseModel):
    """Structured output for Section 1."""

    entities: List[ExtractedEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical entity pool — shared across chunks after deduplication
# ---------------------------------------------------------------------------


@dataclass
class CanonicalEntity:
    """A deduplicated, stable entity reference used after Section 1."""

    name: str
    entity_type: str
    stable_id: uuid.UUID  # uuid5 derived from canonical name
    # If an existing Neo4j node was matched, reuse its cognee_id so that
    # Section 2 updates the existing node instead of creating a new one.
    existing_cognee_id: Optional[str] = None

    @property
    def effective_id(self) -> uuid.UUID:
        if self.existing_cognee_id:
            return uuid.UUID(self.existing_cognee_id)
        return self.stable_id


def _stable_id(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, name.upper())


# ---------------------------------------------------------------------------
# Section 1 — extract node names + types per chunk (LLM Call 1)
# ---------------------------------------------------------------------------


async def _extract_preliminary_nodes(
    chunk: DocumentChunk,
) -> Tuple[DocumentChunk, List[ExtractedEntity]]:
    """LLM Call 1: extract entity names and types from a single chunk."""
    try:
        response: ChunkNodeList = await LLMGateway.acreate_structured_output(
            text_input=chunk.text,
            system_prompt=FINANCIAL_NODE_EXTRACTION_PROMPT,
            response_model=ChunkNodeList,
        )
        logger.debug(
            "Section 1 [chunk %s]: extracted %d candidate entities.",
            chunk.id,
            len(response.entities),
        )
        return chunk, response.entities
    except Exception as exc:
        logger.warning(
            "Section 1 [chunk %s]: LLM call failed — %s. Returning empty list.",
            chunk.id,
            exc,
        )
        return chunk, []


# ---------------------------------------------------------------------------
# Deduplication helpers (adapted from entity_merger.py)
# ---------------------------------------------------------------------------


def _validate_sectors(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
    """
    Hard-filter Sector entities against ALL_MAIN_SECTORS.
    Non-matching sectors are dropped — they must be Industry or Company instead.
    """
    validated: List[ExtractedEntity] = []
    for e in entities:
        if e.entity_type == "Sector":
            if e.name in ALL_MAIN_SECTORS or e.name == "Market":
                validated.append(e)
            else:
                logger.debug(
                    "Dedup: dropped invalid Sector '%s' (not in ALL_MAIN_SECTORS).",
                    e.name,
                )
        else:
            validated.append(e)
    return validated


async def _neo4j_fuzzy_lookup(
    graph_client: Any,
    name: str,
    entity_type: str,
) -> List[Dict[str, Any]]:
    """
    Query Neo4j with APOC sorensenDiceSimilarity to find candidate matches.
    Returns rows with cognee_id, name, sim score.
    """
    fuzzy_query = """
    MATCH (n:`__Node__`)
    WHERE n.name IS NOT NULL AND n.type = $entity_type
    WITH n, apoc.text.sorensenDiceSimilarity(
            toLower(n.name), toLower($name)
         ) AS sim
    WHERE sim >= $threshold
    RETURN n.id AS cognee_id, n.name AS name, sim
    ORDER BY sim DESC
    LIMIT 5
    """
    try:
        rows = await graph_client.query(
            fuzzy_query,
            {
                "name": name,
                "entity_type": entity_type,
                "threshold": FUZZY_CANDIDATE_THRESHOLD,
            },
        )
        results = []
        for row in rows:
            data = (
                row.data()
                if hasattr(row, "data")
                else (dict(row) if isinstance(row, dict) else {})
            )
            results.append(data)
        return results
    except Exception as exc:
        logger.warning("Dedup: APOC fuzzy query failed for '%s': %s", name, exc)
        return []


async def _confirm_semantic_match(
    vector_engine: Any,
    name: str,
    entity_type: str,
    match_cognee_id: str,
    fuzzy_score: float,
) -> bool:
    """
    Confirm a fuzzy candidate using vector engine search.
    Short-circuits to True when fuzzy_score >= SEMANTIC_MERGE_THRESHOLD.
    """
    if fuzzy_score >= SEMANTIC_MERGE_THRESHOLD:
        return True

    collection = f"{entity_type}_name"
    try:
        results = await vector_engine.search(
            collection_name=collection,
            query_text=name,
            limit=10,
        )
    except Exception as exc:
        logger.warning("Dedup: vector search failed for '%s': %s", name, exc)
        return False

    for sr in results:
        if str(sr.id) == match_cognee_id and sr.score <= (1 - SEMANTIC_MERGE_THRESHOLD):
            return True
    return False


async def _resolve_entity_pool(
    all_chunk_entities: List[Tuple[DocumentChunk, List[ExtractedEntity]]],
) -> Tuple[Dict[str, CanonicalEntity], Dict[str, List[str]]]:
    """
    Cross-chunk deduplication.

    Returns:
        canonical_pool: maps canonical_name → CanonicalEntity
        chunk_canonical_map: maps chunk_id → list[canonical_name]
    """
    graph_engine = await get_graph_engine()
    vector_engine = get_vector_engine()

    # 1. Collect all unique entities across chunks (case-insensitive key)
    raw_pool: Dict[str, ExtractedEntity] = {}  # lower_name → first seen
    chunk_entity_names: Dict[str, List[str]] = {}  # chunk_id → [names]

    for chunk, entities in all_chunk_entities:
        validated = _validate_sectors(entities)
        chunk_entity_names[str(chunk.id)] = []
        for e in validated:
            key = e.name.lower()
            if key not in raw_pool:
                raw_pool[key] = e
            chunk_entity_names[str(chunk.id)].append(e.name)

    # 2. Build canonical pool — check against Neo4j to reuse existing IDs
    #    and collapse in-batch near-duplicates
    canonical_pool: Dict[str, CanonicalEntity] = {}  # canonical_lower → CanonicalEntity
    # Maps any seen name (lower) → canonical_lower so chunks can resolve
    alias_map: Dict[str, str] = {}

    for lower_name, extracted in raw_pool.items():
        # Already aliased to another canonical entity
        if lower_name in alias_map:
            continue

        canon_name = extracted.name
        entity_type = extracted.entity_type
        stable = _stable_id(canon_name)
        existing_cid: Optional[str] = None

        # 2a. Check Neo4j for existing match
        rows = await _neo4j_fuzzy_lookup(graph_engine, canon_name, entity_type)
        for row in rows:
            match_cid = str(row.get("cognee_id", ""))
            match_name = str(row.get("name", ""))
            sim = float(row.get("sim", 0.0))
            if not match_cid:
                continue
            confirmed = await _confirm_semantic_match(
                vector_engine, canon_name, entity_type, match_cid, sim
            )
            if confirmed:
                # Reuse existing Neo4j node's ID
                existing_cid = match_cid
                # Prefer the existing canonical name in the graph
                canon_name = match_name
                logger.debug(
                    "Dedup: '%s' → matched existing Neo4j node '%s' (id=%s, sim=%.3f)",
                    extracted.name,
                    match_name,
                    match_cid,
                    sim,
                )
                break

        # 2b. In-batch near-duplicate collapse (compare against already-resolved canonicals)
        canon_lower = canon_name.lower()
        if canon_lower not in canonical_pool:
            # Check if any already-resolved canonical is a near match
            merged_into: Optional[str] = None
            for existing_lower, existing_ce in canonical_pool.items():
                if existing_ce.entity_type != entity_type:
                    continue
                import difflib

                ratio = difflib.SequenceMatcher(
                    None, canon_lower, existing_lower
                ).ratio()
                if ratio >= IN_BATCH_MERGE_THRESHOLD:
                    merged_into = existing_lower
                    break

            if merged_into:
                alias_map[lower_name] = merged_into
                alias_map[canon_lower] = merged_into
                logger.debug(
                    "Dedup: '%s' → in-batch merged into '%s'",
                    canon_name,
                    canonical_pool[merged_into].name,
                )
            else:
                ce = CanonicalEntity(
                    name=canon_name,
                    entity_type=entity_type,
                    stable_id=stable,
                    existing_cognee_id=existing_cid,
                )
                canonical_pool[canon_lower] = ce
                alias_map[lower_name] = canon_lower
                alias_map[canon_lower] = canon_lower
        else:
            alias_map[lower_name] = canon_lower

    # 3. Rebuild chunk_canonical_map using resolved aliases
    chunk_canonical_map: Dict[str, List[str]] = {}
    for chunk_id, names in chunk_entity_names.items():
        resolved: List[str] = []
        seen: Set[str] = set()
        for name in names:
            key = alias_map.get(name.lower())
            if key and key in canonical_pool and key not in seen:
                resolved.append(canonical_pool[key].name)
                seen.add(key)
        chunk_canonical_map[chunk_id] = resolved

    logger.info(
        "Dedup: %d raw entities → %d canonical entities across %d chunks.",
        len(raw_pool),
        len(canonical_pool),
        len(all_chunk_entities),
    )
    return canonical_pool, chunk_canonical_map


# ---------------------------------------------------------------------------
# Section 2 — schema slicing
# ---------------------------------------------------------------------------


def _build_sliced_graph_model(entity_types: Set[str]) -> Type[BaseModel]:
    """
    Build a FinancialKnowledgeGraph variant with a union restricted to the
    types present in this chunk.

    Relationship-target types (Company, FinancialEvent, FinancialConcept,
    Industry) are always added to the union even if not directly present,
    so that nested relationship fields in user-interest entities can resolve.
    Falls back to the full FinancialKnowledgeGraph if the type set is empty.
    """
    # Expand with always-include types so nested relationship fields work
    expanded_types = entity_types | _ALWAYS_INCLUDE_IN_SCHEMA
    type_classes = [
        _ENTITY_TYPE_MAP[t] for t in expanded_types if t in _ENTITY_TYPE_MAP
    ]
    if not type_classes:
        return FinancialKnowledgeGraph

    # Deduplicate (already unique since expanded_types is a set)
    if len(type_classes) == 1:
        union_annotation = List[type_classes[0]]  # type: ignore[valid-type]
    else:
        from typing import Union

        union_annotation = List[Union[tuple(type_classes)]]  # type: ignore[valid-type]

    SlicedGraph = create_model(
        "SlicedFinancialKnowledgeGraph",
        entities=(
            union_annotation,
            Field(
                default_factory=list, description="Extracted entities for this chunk."
            ),
        ),
    )
    return SlicedGraph


def _build_section2_user_message(
    chunk_text: str,
    canonical_entities: List[CanonicalEntity],
) -> str:
    """
    Construct the user-facing message for LLM Call 2.
    Injects the canonical entity list so the LLM knows the primary entities
    to populate. Relationship target fields (targets, supporting_events, etc.)
    may reference entities from this list OR other known financial entities
    named explicitly in the source text.
    """
    entity_lines = "\n".join(
        f"  - [{e.entity_type}] {e.name}" for e in canonical_entities
    )
    return (
        f"PRIMARY ENTITIES (create full attribute objects for each of these):\n"
        f"{entity_lines}\n\n"
        f"NOTE: Relationship fields (targets, supporting_events, positively_impacted, etc.) "
        f"may reference entities from the primary list above OR other Company/Sector/ "
        f"FinancialEvent/FinancialConcept entities explicitly named in the source text.\n\n"
        f"SOURCE TEXT:\n{chunk_text}"
    )


# ---------------------------------------------------------------------------
# Section 2 — per-chunk attribute + relationship extraction (LLM Call 2)
# ---------------------------------------------------------------------------


async def _extract_full_graph_for_chunk(
    chunk: DocumentChunk,
    canonical_entities: List[CanonicalEntity],
    sliced_model: Type[BaseModel],
) -> Tuple[DocumentChunk, FinancialKnowledgeGraph]:
    """LLM Call 2: extract full attributes + relationships for canonical entities."""
    user_message = _build_section2_user_message(chunk.text, canonical_entities)

    # Build a chunk-specific system prompt that injects the canonical entity
    # roster directly, so the LLM sees the closed-world constraint in BOTH
    # the system prompt and the user message.
    entity_dicts = [
        {"name": ce.name, "entity_type": ce.entity_type} for ce in canonical_entities
    ]
    system_prompt = build_attribute_extraction_prompt(entity_dicts)

    try:
        raw_result = await LLMGateway.acreate_structured_output(
            text_input=user_message,
            system_prompt=system_prompt,
            response_model=sliced_model,
        )
        entities = getattr(raw_result, "entities", [])
        logger.debug(
            "Section 2 [chunk %s]: extracted %d full entities.",
            chunk.id,
            len(entities),
        )
    except Exception as exc:
        logger.warning(
            "Section 2 [chunk %s]: LLM call failed — %s. Returning empty graph.",
            chunk.id,
            exc,
        )
        entities = []

    # Remap IDs to canonical stable IDs so the graph nodes are consistent.
    # UserInvestmentInterest / UserLearningInterest have no `name` field —
    # match them by `reason` text to the canonical entity's name,
    # or simply assign a fresh stable uuid5 based on reason to avoid collisions.
    name_to_canonical = {ce.name.lower(): ce for ce in canonical_entities}
    for entity in entities:
        entity_name = getattr(entity, "name", None)
        if entity_name:
            # Named entities (Company, FinancialEvent, etc.)
            ce = name_to_canonical.get(entity_name.lower())
            if ce:
                entity.id = ce.effective_id
        else:
            # User-specific entities: derive a stable ID from their reason text
            # so repeated ingestion of the same conversation converges to one node.
            reason = getattr(entity, "reason", None)
            if reason:
                entity.id = uuid.uuid5(uuid.NAMESPACE_DNS, reason.upper()[:200])

    graph = FinancialKnowledgeGraph(entities=entities)
    return chunk, graph


# ---------------------------------------------------------------------------
# Orchestrator — wires the three sections together
# ---------------------------------------------------------------------------


async def extract_financial_graph(
    data_chunks: List[DocumentChunk],
) -> List[DocumentChunk]:
    """
    Drop-in replacement for cognee's extract_graph_from_data.

    Runs the 3-section extraction pipeline:
      1. Section 1: LLM Call 1 per chunk (parallel) — entity names + types.
      2. Deduplication: cross-chunk fuzzy + semantic dedup (0 LLM calls).
      3. Section 2: LLM Call 2 per chunk (parallel, schema-sliced) — full attrs + rels.

    Args:
        data_chunks: List of DocumentChunk objects from extract_chunks_from_documents.

    Returns:
        The same list of DocumentChunk objects with chunk.contains populated
        as FinancialKnowledgeGraph instances, ready for assign_nodesets.
    """
    if not data_chunks:
        return data_chunks

    # --- Section 1: parallel shallow extraction ---
    logger.info(
        "Section 1: extracting entity names+types from %d chunks.", len(data_chunks)
    )
    section1_results: List[Tuple[DocumentChunk, List[ExtractedEntity]]] = list(
        await asyncio.gather(*[_extract_preliminary_nodes(c) for c in data_chunks])
    )

    # --- Deduplication: cross-chunk, 0 LLM calls ---
    logger.info("Dedup: resolving canonical entity pool across all chunks.")
    canonical_pool, chunk_canonical_map = await _resolve_entity_pool(section1_results)

    # --- Section 2: parallel per-chunk attribute + relationship extraction ---
    logger.info("Section 2: extracting full attributes + relationships per chunk.")

    async def _process_chunk(
        chunk: DocumentChunk,
    ) -> Tuple[DocumentChunk, FinancialKnowledgeGraph]:
        canonical_names = chunk_canonical_map.get(str(chunk.id), [])
        canonical_entities = [
            ce for ce in canonical_pool.values() if ce.name in canonical_names
        ]
        if not canonical_entities:
            logger.debug(
                "Section 2 [chunk %s]: no canonical entities — skipping.", chunk.id
            )
            return chunk, FinancialKnowledgeGraph(entities=[])

        entity_types = {ce.entity_type for ce in canonical_entities}
        sliced_model = _build_sliced_graph_model(entity_types)

        return await _extract_full_graph_for_chunk(
            chunk, canonical_entities, sliced_model
        )

    section2_results = await asyncio.gather(*[_process_chunk(c) for c in data_chunks])

    # --- Assign results back to the original chunks ---
    for chunk, kg in section2_results:
        chunk.contains = kg

    logger.info("extract_financial_graph: completed for %d chunks.", len(data_chunks))
    return data_chunks
