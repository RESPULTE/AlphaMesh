"""
core/memory/conversation_writeback.py

Async write-back of enriched entities and relationships to the knowledge graph.
Called fire-and-forget from OrchestratorAgent._synthesize_node.

Write order (strict):
  1. Resolve NodeSets for enriched entities (via assign_nodesets logic)
  2. Dedup enriched entities against existing graph nodes
  3. Write enriched entities via add_data_points
  4. Resolve relationship endpoints against written entities
  5. Write EntityRelationship DataPoints

A failure at any step is logged but NEVER propagated — this is a background task.
The user response has already been delivered before this runs.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage import add_data_points

from core.memory.entity_merger import find_and_merge_candidates
from core.memory.graph.models import (
    GLOBAL_FINANCIAL_EVENT_NODESET,
    GLOBAL_FINANCIAL_WISDOM_NODESET,
)
from core.memory.nodeset_manager import (
    get_or_create_global_nodeset,
    get_or_create_nodeset,
    get_or_create_user_nodeset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EntityRelationship DataPoint — written as a graph edge
# ---------------------------------------------------------------------------


class EntityRelationship(DataPoint):
    """
    A directed relationship between two named entities.
    Written to the graph by the write-back system.
    canonical ID: uuid5 of "{from_name}::{relation}::{to_name}"
    """

    __tablename__ = "entity_relationship"

    from_name: str
    from_type: str
    relation_type: str
    to_name: str
    to_type: str
    confidence: str = "low"
    source_conversation_id: str = ""
    metadata: dict = {"index_fields": ["from_name", "relation_type", "to_name"]}


def _relationship_id(from_name: str, relation: str, to_name: str) -> uuid.UUID:
    key = f"{from_name.upper()}::{relation}::{to_name.upper()}"
    return uuid.uuid5(uuid.NAMESPACE_DNS, key)


# ---------------------------------------------------------------------------
# NodeSet resolution for enriched entities
# ---------------------------------------------------------------------------


async def _resolve_entity_nodeset(entity: Any) -> None:
    """
    Assign belongs_to_set on an enriched entity using the same logic as
    assign_nodesets() in pipeline_tasks.py, but without requiring a
    DocumentChunk wrapper.

    This is a simplified version that handles the entity types produced
    by downstream agents during conversation write-back.
    """
    entity_type = type(entity).__name__

    try:
        entity.belongs_to_set = getattr(entity, "belongs_to_set", []) or []

        if entity_type == "FinancialConcept":
            ns = await get_or_create_nodeset(GLOBAL_FINANCIAL_WISDOM_NODESET)
            entity.belongs_to_set.append(ns)

        elif entity_type == "FinancialEvent":
            ns = await get_or_create_nodeset(GLOBAL_FINANCIAL_EVENT_NODESET)
            entity.belongs_to_set.append(ns)

        elif entity_type == "Company":
            global_ns = await get_or_create_global_nodeset()
            entity.belongs_to_set.append(global_ns)
            # If sector is known, also resolve sector NodeSet
            if getattr(entity, "sector", None):
                try:
                    sector_ns = await get_or_create_nodeset(entity.sector)
                    entity.belongs_to_set.append(sector_ns)
                except Exception:
                    pass  # sector resolution failure is non-fatal

    except Exception as exc:
        logger.warning(
            "write_back: failed to resolve NodeSet for %s '%s': %s",
            entity_type,
            getattr(entity, "name", getattr(entity, "ticker", "?")),
            exc,
        )


# ---------------------------------------------------------------------------
# Entity deduplication check
# ---------------------------------------------------------------------------


async def _should_write_entity(entity: Any) -> bool:
    """
    Check if entity should be written or skipped.
    Mirrors the dedup logic from graph_extraction._resolve_entity_pool but
    operates against the live graph rather than a batch.

    Returns True if entity should be written (new or unenriched stub exists).
    Returns False if an already-enriched node with same canonical ID exists.
    """

    canonical_id = str(getattr(entity, "id", None) or "")
    if not canonical_id:
        return True

    table = getattr(entity, "__tablename__", None)
    if not table:
        return True

    try:
        engine = get_relational_engine()
        # Check if an enriched node already exists with this ID
        result = await engine.fetch_one(
            f"SELECT id FROM {table} WHERE id = :cid LIMIT 1",
            {"cid": canonical_id},
        )
        # If node doesn't exist → write it
        return result is None
    except Exception:
        # If check fails, write anyway — add_data_points handles upserts
        return True


# ---------------------------------------------------------------------------
# Relationship DataPoint construction
# ---------------------------------------------------------------------------


def _build_relationship_datapoints(
    relationships: List[dict],
    conversation_id: str,
) -> List[EntityRelationship]:
    """
    Convert the synthesiser's <relationships> JSON array into
    EntityRelationship DataPoints ready for add_data_points.

    Skips malformed entries silently — never raises.
    """
    datapoints = []
    for rel in relationships:
        try:
            from_name = rel["from_name"]
            relation = rel["relation"]
            to_name = rel["to_name"]
        except KeyError:
            logger.debug("write_back: skipping malformed relationship: %s", rel)
            continue

        datapoints.append(
            EntityRelationship(
                id=_relationship_id(from_name, relation, to_name),
                from_name=from_name,
                from_type=rel.get("from_type", "unknown"),
                relation_type=relation,
                to_name=to_name,
                to_type=rel.get("to_type", "unknown"),
                confidence=rel.get("confidence", "low"),
                source_conversation_id=conversation_id,
            )
        )

    return datapoints


# ---------------------------------------------------------------------------
# Main write-back entry point
# ---------------------------------------------------------------------------


async def run_conversation_writeback(
    relationships: List[dict],
    enriched_entities: List[Any],
    conversation_id: str,
    user_email: Optional[str] = None,
) -> None:
    """
    Write enriched entities and synthesiser-derived relationships to the graph.

    Called fire-and-forget from OrchestratorAgent._synthesize_node.
    All exceptions are caught and logged — this function NEVER raises.

    Write order:
      1. Resolve NodeSets for all enriched entities
      2. Filter to entities that need writing (dedup check)
      3. Write enriched entities via add_data_points
      4. Run entity merger on written entities (APOC fuzzy + semantic dedup)
      5. Build and write EntityRelationship DataPoints

    Args:
        relationships:      Parsed <relationships> JSON list from synthesiser.
        enriched_entities:  DataPoint objects from all downstream agents.
        conversation_id:    Unique ID for this conversation turn (for traceability).
        user_email:         If provided, user-specific entities get USER NodeSet.
    """
    try:
        if not enriched_entities and not relationships:
            logger.debug("write_back [%s]: nothing to write.", conversation_id)
            return

        # --- Step 1: Resolve NodeSets ---
        for entity in enriched_entities:
            await _resolve_entity_nodeset(entity)

        # If user_email provided, also tag user-specific entities
        if user_email:
            from core.memory.graph.models import (
                UserInvestmentInterest,
                UserLearningInterest,
            )

            _, user_ns = await get_or_create_user_nodeset(user_email)
            for entity in enriched_entities:
                if isinstance(entity, (UserInvestmentInterest, UserLearningInterest)):
                    entity.belongs_to_set = getattr(entity, "belongs_to_set", []) or []
                    entity.belongs_to_set.append(user_ns)

        # --- Step 2: Filter to entities that need writing ---
        to_write = []
        for entity in enriched_entities:
            if await _should_write_entity(entity):
                to_write.append(entity)
            else:
                logger.debug(
                    "write_back [%s]: skipping already-enriched entity %s '%s'.",
                    conversation_id,
                    type(entity).__name__,
                    getattr(entity, "name", getattr(entity, "ticker", "?")),
                )

        # --- Step 3: Write enriched entities ---
        if to_write:
            await add_data_points(to_write)
            logger.info(
                "write_back [%s]: wrote %d enriched entities.",
                conversation_id,
                len(to_write),
            )

        # --- Step 4: Run entity merger on written entities ---
        # This catches any near-duplicates introduced by the write-back
        # and merges them using the existing APOC fuzzy + semantic pipeline.
        if to_write:
            try:
                graph_engine = await get_graph_engine()
                await find_and_merge_candidates(graph_engine, to_write)
            except Exception as merge_exc:
                # Merger failure is non-fatal — entities are written, just not merged
                logger.warning(
                    "write_back [%s]: entity merger failed (non-fatal): %s",
                    conversation_id,
                    merge_exc,
                )

        # --- Step 5: Build and write relationship DataPoints ---
        if relationships:
            rel_datapoints = _build_relationship_datapoints(
                relationships, conversation_id
            )
            if rel_datapoints:
                await add_data_points(rel_datapoints)
                logger.info(
                    "write_back [%s]: wrote %d relationship edges.",
                    conversation_id,
                    len(rel_datapoints),
                )

    except Exception as exc:
        # CRITICAL: This function must NEVER raise — it is fire-and-forget
        logger.error(
            "write_back [%s]: unhandled error (user response unaffected): %s",
            conversation_id,
            exc,
            exc_info=True,
        )
