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
             SynthesisResult.interest_edges) and upserts it to Neo4j.

          3. Updates last_analysis_summary on each target entity.

Design notes
------------
- No LLM calls here.  The interest edges are extracted once, inside
  _run_synthesis_chain (the synthesis prompt now includes a third output
  block).  This module is pure write logic.

- All exceptions are caught and logged.  This function NEVER raises —
  a failure here must never surface to the user.

- _safe_create_task is duplicated from orchestrator_agent.py deliberately
  so this module has no import dependency on agents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


from core.config import settings
from core.logger import get_logger
from core.memory.graph.models import (
    UserInvestmentInterestNode,
    UserLearningInterestNode,
)
from core.memory.graph.subgraph_builder import InMemorySubgraphBuilder
from core.services import service_manager

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Payload dataclass — decouples this module from OrchestratorState
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


@dataclass
class LearningSignal:
    status: str  # "Interested" | "Understood" | "Confused" | …
    target_entities: List[DetectedEntity] = field(default_factory=list)


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
    Everything write_user_signals needs.  Built by the orchestrator from
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


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — upsert interest nodes
# ──────────────────────────────────────────────────────────────────────────────


async def _upsert_interest_nodes(payload: UserSignalPayload) -> None:
    """
    Persist UserInvestmentInterestNode / UserLearningInterestNode records.
    Each signal entity is resolved to a graph entity ID first; if resolution
    fails the entry is skipped silently.
    """
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
                    confidence="high",
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
# Step 2 — build and upsert the user-scoped interest subgraph
# ──────────────────────────────────────────────────────────────────────────────


async def _upsert_interest_subgraph(payload: UserSignalPayload) -> None:
    """
    Convert interest edges (from the synthesis LLM) into a Neo4j subgraph
    tagged with derived_for_user_email.
    """
    if not payload.interest_edges:
        return

    # Build type lookup from all signal entities so we can resolve target types
    target_type_lookup: Dict[str, str] = {}
    for signal in payload.investment_signals + payload.learning_signals:
        for entity in signal.target_entities:
            if entity.entity_name and entity.entity_type:
                target_type_lookup[entity.entity_name.lower()] = entity.entity_type

    rels = []
    for edge in payload.interest_edges:
        target_type = target_type_lookup.get(edge.target_entity_name.lower())
        if not target_type:
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
        builder = InMemorySubgraphBuilder(
            embedding_func=service_manager.get_embedding_func(),
            fuzzy_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
            semantic_threshold=settings.EXTRACTION_SEMANTIC_THRESHOLD,
        )
        interest_graph = await builder.build(rels, source_agent="orchestrator")
        _safe_create_task(
            service_manager.get_ingestor()._upsert_graph_to_neo4j(
                interest_graph, payload.conversation_id
            )
        )
    except Exception:
        logger.exception("user_signal_writeback: interest subgraph upsert failed")


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

    target_type_lookup: Dict[str, str] = {}
    for signal in payload.investment_signals + payload.learning_signals:
        for entity in signal.target_entities:
            if entity.entity_name and entity.entity_type:
                target_type_lookup[entity.entity_name.lower()] = entity.entity_type

    try:
        ingestor = service_manager.get_ingestor()
        user_graph = service_manager.get_neo4j_adapter()
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
                    await user_graph.update_targets_last_analysis_summary(
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
      2. Build and upsert the user-scoped interest subgraph.
      3. Update last_analysis_summary on each targeted entity.

    Intended to be called fire-and-forget from the orchestrator:

        _safe_create_task(write_user_signals(payload))
    """
    if not payload.user_email or not payload.conversation_id:
        logger.warning(
            "write_user_signals: missing user_email or conversation_id, skipping"
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
