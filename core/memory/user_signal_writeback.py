"""
core/memory/user_signal_writeback.py
=====================================

Owns all graph-write logic that persists user investment/learning signals
detected during a conversation turn.

Previously this logic lived in OrchestratorAgent._write_user_signals, which
was a violation of separation of concerns — the orchestrator should only
orchestrate; writing to the memory graph is the memory module's job.

Public surface
--------------
    write_user_signals(payload: UserSignalPayload) -> None

        Fire-and-forget coroutine.  Accepts a plain data payload (no agent
        state, no LLM) and:

          1. Upserts UserInvestmentInterestNode / UserLearningInterestNode
             records into Neo4j via UserContextService.

          2. Builds a user-scoped interest subgraph from pre-extracted
             interest edges (produced by the synthesis LLM call — see
             SynthesisResult.interest_edges) and persists it to Neo4j via
             SubgraphExtractionService.

          3. Updates last_analysis_summary on each target entity.

Design notes
------------
- No LLM calls here. The interest edges are extracted once, inside
  _run_synthesis_chain (the synthesis prompt includes a third output block).
  This module is pure write logic.

- SubgraphExtractionService is used directly (build_graph + persist_graph)
  rather than through schedule() because the relationships are already
  extracted — there is no LLM call to defer. _upsert_interest_subgraph is
  itself already called from a background coroutine, so nesting another
  fire-and-forget task inside it would cause failures to be silently
  swallowed. We await both operations directly.

- All exceptions are caught and logged. This function NEVER raises —
  a failure here must never surface to the user.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.logger import get_logger
from core.memory.graph.models import (
    UserInvestmentInterestNode,
    UserLearningInterestNode,
)
from core.services import service_manager

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Payload dataclasses — decouples this module from OrchestratorState
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DetectedEntity:
    """A single entity inside an investment or learning signal."""

    entity_name: str
    entity_type: str


@dataclass
class InvestmentSignal:
    status: str  # "Bought" | "Interested" | "Sold" | "Avoids"
    target_entities: List[DetectedEntity] = field(default_factory=list)
    confidence: float = 0.5  # float from planner, 0.0–1.0


@dataclass
class LearningSignal:
    status: str
    target_entities: List[DetectedEntity] = field(default_factory=list)
    confidence: float = 0.5  # carried for symmetry; not stored on LearningInterestNode


@dataclass
class InterestEdge:
    """
    A single relationship edge extracted by the synthesis LLM.
    Mirrors the InterestEdge Pydantic model on the orchestrator side but uses
    a plain dataclass so this module stays free of Pydantic/LangChain imports.
    """

    entity_name: str
    entity_type: str
    user_signal_type: str  # "investment" | "learning"
    target_entity_name: str
    relationship: str  # "THREATENS" | "SUPPORTS" | "CLARIFIES" | …
    reason: str
    confidence: str  # "high" | "low"


@dataclass
class UserSignalPayload:
    """
    Everything write_user_signals needs. Built by the orchestrator from
    OrchestratorState + SynthesisResult and passed in as a plain value object.
    """

    user_email: str
    conversation_id: str
    user_message: str  # last human turn, used as "reason"
    investment_signals: List[InvestmentSignal] = field(default_factory=list)
    learning_signals: List[LearningSignal] = field(default_factory=list)
    interest_edges: List[InterestEdge] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_create_task(coro) -> Optional[asyncio.Task]:
    """Schedule a coroutine as a fire-and-forget task if a loop is running."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        logger.warning("user_signal_writeback: no running event loop — task skipped.")
        return None


def _first_sentence(text: str) -> str:
    text = (text or "").strip()
    sentence = text.split(".")[0].strip()
    return f"{sentence}." if sentence else ""


def _build_target_type_lookup(payload: UserSignalPayload) -> Dict[str, str]:
    """
    Index entity_type by lowercased entity_name across all signal target lists.
    Used by both _upsert_interest_subgraph and _update_analysis_summaries to
    resolve the type of a target entity given only its name.
    """
    lookup: Dict[str, str] = {}
    for signal in payload.investment_signals + payload.learning_signals:
        for entity in signal.target_entities:
            if entity.entity_name and entity.entity_type:
                lookup[entity.entity_name.lower()] = entity.entity_type
    return lookup


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — upsert interest nodes
# ──────────────────────────────────────────────────────────────────────────────


async def _upsert_interest_nodes(payload: UserSignalPayload) -> None:
    ingestor = service_manager.get_ingestor()
    user_context_service = service_manager.get_user_context_service()
    entity_cache: Dict[str, Any] = {}

    for signal in payload.investment_signals:
        for entity in signal.target_entities:
            try:
                resolved_id = await ingestor.resolve_entity_id(
                    entity.entity_name,
                    entity.entity_type,
                    entity_cache=entity_cache,
                )
                if not resolved_id:
                    continue
                node = UserInvestmentInterestNode(
                    id="",
                    user_email=payload.user_email,
                    status=signal.status,
                    reason=payload.user_message,
                    confidence=signal.confidence,  # float stored directly
                    updated_at=datetime.now(timezone.utc),
                    target_entity_ids=[resolved_id],
                )
                user_context_service.schedule_upsert_fire_and_forget(
                    node, payload.user_email
                )
            except Exception:
                logger.exception(
                    "user_signal_writeback: investment node upsert failed for '%s'",
                    entity.entity_name,
                )

    for signal in payload.learning_signals:
        for entity in signal.target_entities:
            try:
                resolved_id = await ingestor.resolve_entity_id(
                    entity.entity_name,
                    entity.entity_type,
                    entity_cache=entity_cache,
                )
                if not resolved_id:
                    continue
                node = UserLearningInterestNode(
                    id="",
                    user_email=payload.user_email,
                    status=signal.status,
                    reason=payload.user_message,
                    updated_at=datetime.now(timezone.utc),
                    confidence=signal.confidence,
                    target_entity_ids=[resolved_id],
                )
                user_context_service.schedule_upsert_fire_and_forget(
                    node, payload.user_email
                )
            except Exception:
                logger.exception(
                    "user_signal_writeback: learning node upsert failed for '%s'",
                    entity.entity_name,
                )


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — build and persist the user-scoped interest subgraph
# ──────────────────────────────────────────────────────────────────────────────


async def _upsert_interest_subgraph(payload: UserSignalPayload) -> None:
    """
    Convert interest edges (from the synthesis LLM) into a Neo4j subgraph
    tagged with derived_for_user_email.

    Uses SubgraphExtractionService.build_graph() + persist_graph() directly
    rather than schedule(), because:
      - The relationships are already extracted; no LLM call is needed.
      - This coroutine is itself executing inside a background task
        (write_user_signals is fire-and-forget from the orchestrator).
        Nesting another task via schedule() would cause any exception in the
        inner task to bypass the exception handlers in write_user_signals.
    """
    if not payload.interest_edges:
        return

    target_type_lookup = _build_target_type_lookup(payload)

    rels = []
    for edge in payload.interest_edges:
        target_type = target_type_lookup.get(edge.target_entity_name.lower())
        if not target_type:
            logger.debug(
                "_upsert_interest_subgraph: skipping edge — unknown type for '%s'",
                edge.target_entity_name,
            )
            continue
        rels.append(
            {
                "from_name": edge.entity_name,
                "from_type": edge.entity_type,
                "relation": edge.relationship,
                "to_name": edge.target_entity_name,
                "to_type": target_type,
                "confidence": edge.confidence,
                "reason": edge.reason,
                "extra_props": {"derived_for_user_email": payload.user_email},
            }
        )

    if not rels:
        return

    try:
        subgraph_svc = service_manager.get_subgraph_service()
        interest_graph = await subgraph_svc.build_graph(
            rels, source_agent="orchestrator"
        )
        await subgraph_svc.persist_graph(
            interest_graph, conversation_id=payload.conversation_id
        )
        logger.info(
            "_upsert_interest_subgraph: persisted %d edges for user '%s'",
            interest_graph.number_of_edges(),
            payload.user_email,
        )
    except Exception:
        logger.exception(
            "user_signal_writeback: interest subgraph build/persist failed "
            "for user '%s'",
            payload.user_email,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — update last_analysis_summary on target entities
# ──────────────────────────────────────────────────────────────────────────────


async def _update_analysis_summaries(payload: UserSignalPayload) -> None:
    """
    Write a one-sentence summary onto each entity that was targeted by an
    interest edge, so the user graph surfaces fresh context per entity.
    """
    if not payload.interest_edges:
        return

    target_type_lookup = _build_target_type_lookup(payload)

    try:
        ingestor = service_manager.get_ingestor()
        neo4j = service_manager.get_neo4j_adapter()
        entity_cache: Dict[str, Any] = {}

        for edge in payload.interest_edges:
            target_type = target_type_lookup.get(edge.target_entity_name.lower())
            if not target_type:
                continue
            try:
                target_id = await ingestor.resolve_entity_id(
                    edge.target_entity_name,
                    target_type,
                    entity_cache=entity_cache,
                )
                if not target_id:
                    continue
                summary = _first_sentence(edge.reason)
                if summary:
                    await neo4j.update_targets_last_analysis_summary(
                        payload.user_email, target_id, summary
                    )
            except Exception:
                logger.exception(
                    "user_signal_writeback: summary update failed for '%s'",
                    edge.target_entity_name,
                )
    except Exception:
        logger.exception("user_signal_writeback: summary update loop failed")


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


async def write_user_signals(payload: UserSignalPayload) -> None:
    """
    Persist all user interest signals produced during a conversation turn.

    Steps (all exceptions caught internally — never raises):
      1. Upsert UserInvestmentInterestNode / UserLearningInterestNode records.
      2. Build and persist the user-scoped interest subgraph.
      3. Update last_analysis_summary on each targeted entity.

    Intended to be called fire-and-forget from the orchestrator:

        _safe_create_task(write_user_signals(payload))
    """
    logger.info(
        "write_user_signals: started for user '%s' with %d investment signals, "
        "%d learning signals, and %d interest edges",
        payload.user_email,
        len(payload.investment_signals),
        len(payload.learning_signals),
        len(payload.interest_edges),
    )
    if not payload.user_email or not payload.conversation_id:
        logger.warning(
            "write_user_signals: missing user_email or conversation_id — skipping"
        )
        return

    try:
        await _upsert_interest_nodes(payload)
    except Exception:
        logger.exception("write_user_signals: _upsert_interest_nodes failed")

    try:
        await _upsert_interest_subgraph(payload)
    except Exception:
        logger.exception("write_user_signals: _upsert_interest_subgraph failed")

    try:
        await _update_analysis_summaries(payload)
    except Exception:
        logger.exception("write_user_signals: _update_analysis_summaries failed")


def build_signal_payload(
    detected_investment_signals: List[InvestmentSignal],
    detected_learning_signals: List[LearningSignal],
    interest_edges: List[dict],
    user_message: str,
    user_email: Optional[str],
    conversation_id: Optional[str],
) -> UserSignalPayload:
    investment_signals = [
        InvestmentSignal(
            status=s.status,
            confidence=getattr(s, "confidence", 0.5),
            target_entities=[
                DetectedEntity(entity_name=e.entity_name, entity_type=e.entity_type)
                for e in s.target_entities
            ],
        )
        for s in (detected_investment_signals or [])
    ]
    learning_signals = [
        LearningSignal(
            status=s.status,
            confidence=getattr(s, "confidence", 0.5),
            target_entities=[
                DetectedEntity(entity_name=e.entity_name, entity_type=e.entity_type)
                for e in s.target_entities
            ],
        )
        for s in (detected_learning_signals or [])
    ]
    edges = [
        InterestEdge(
            entity_name=e.get("entity_name", ""),
            entity_type=e.get("entity_type", ""),
            user_signal_type=e.get("user_signal_type", "investment"),
            target_entity_name=e.get("target_entity_name", ""),
            relationship=e.get("relationship", "RELATED_TO"),
            reason=e.get("reason", ""),
            confidence=e.get("confidence", "low"),
        )
        for e in interest_edges
        if e.get("entity_name") and e.get("target_entity_name")
    ]
    return UserSignalPayload(
        user_email=user_email or "",
        conversation_id=conversation_id or "",
        user_message=user_message,
        investment_signals=investment_signals,
        learning_signals=learning_signals,
        interest_edges=edges,
    )
