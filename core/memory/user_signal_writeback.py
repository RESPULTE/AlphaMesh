"""
core/memory/user_signal_writeback.py
=====================================

Unified user signal write-back via SubgraphExtractionService.schedule().

Graph schema written here
─────────────────────────
NodeSet (USER_{hash})
  ←─BELONGS_TO_NODESET─── UserInterestDomain {domain_type, category, user_email}
                               └─HAS_INTEREST_IN──> UserInterestEdge {weight, status, ...}
                                                         ├─TARGETS──────────> Entity
                                                         ├─SOURCED_FROM─────> TurnNode (reinforce)
                                                         └─INVALIDATED_BY──> TurnNode (invalidate)

All graph writes go through SubgraphExtractionService.schedule(bypass_guards=True)
so that entity deduplication runs before any Neo4j write.

Cache is updated synchronously after schedule() dispatches the background task
so the current session sees new signals immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core.logger import get_logger
from core.memory.user_context_service import InterestCacheEntry

logger = get_logger(__name__)


# ── Payload dataclasses ───────────────────────────────────────────────────────


@dataclass
class DetectedEntity:
    entity_name: str
    entity_type: str


@dataclass
class InvestmentSignal:
    status: str  # "Bought" | "Interested" | "Sold" | "Avoids"
    target_entities: List[DetectedEntity] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class LearningSignal:
    status: str  # "Interested" | "Understood" | "Confused" | "Not Interested"
    target_entities: List[DetectedEntity] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class InterestEdge:
    """Pre-extracted cross-domain interest edge from synthesis LLM."""

    entity_name: str
    entity_type: str
    user_signal_type: str
    target_entity_name: str
    relationship: str
    reason: str
    confidence: str


@dataclass
class UserSignalPayload:
    user_email: str
    conversation_id: str
    turn_id: str  # uuid4 per OrchestratorAgent.run()
    user_message: str
    ticker_metadata: Dict[str, dict] = field(default_factory=dict)
    # {ticker: {long_name, sector, industry, description}}
    investment_signals: List[InvestmentSignal] = field(default_factory=list)
    learning_signals: List[LearningSignal] = field(default_factory=list)
    interest_edges: List[InterestEdge] = field(default_factory=list)


# ── Domain category derivation ────────────────────────────────────────────────


def _derive_investment_category(
    entity_name: str,
    entity_type: str,
    ticker_metadata: Dict[str, dict],
) -> str:
    """
    Derive the domain category for an investment signal.

    Company targets: look up sector from ticker_metadata (keyed by ticker symbol
    and also checked against long_name for robustness).
    Sector targets: use the sector name directly.
    """
    if entity_type == "Sector":
        return entity_name
    if entity_type == "Company":
        entity_lower = entity_name.lower()
        for ticker, meta in ticker_metadata.items():
            if (
                ticker.lower() == entity_lower
                or meta.get("long_name", "").lower() == entity_lower
            ):
                sector = meta.get("sector", "")
                if sector:
                    return sector
    return "general"


async def _derive_learning_category(
    entity_type: str,
    entity_id: str,
    neo4j,
) -> str:
    """
    Derive the domain category for a learning signal.

    FinancialConcept: fetch category from Neo4j (valuation, risk, etc.).
    FinancialEvent:   fixed category "market_events".
    Fallback:         "general".
    """
    if entity_type == "FinancialConcept":
        try:
            category = await neo4j.get_entity_category(entity_id)
            if category:
                return category
        except Exception:
            pass
    if entity_type == "FinancialEvent":
        return "market_events"
    return "general"


# ── Relationship list builder ─────────────────────────────────────────────────


async def _build_interest_relationships(
    payload: UserSignalPayload,
    ingestor,
    neo4j,
    nodeset_id: str,
) -> Tuple[List[dict], List[InterestCacheEntry]]:
    """
    Pre-resolve all domain entity IDs, then build the full relationship list
    expressing the UserInterestDomain → UserInterestEdge → Entity + TurnNode chain.

    Pre-resolution (Option A) is necessary to compute deterministic UUIDs for
    UserInterestEdge nodes, which depend on the resolved entity ID.

    Returns (relationships, cache_entries) so the caller can dispatch to
    schedule() and update_cache() in one pass.
    """
    from core.memory.graph.nodeset_manager import (
        canonical_nodeset_id,
        get_user_nodeset_name,
    )
    from core.memory.graph.utils import generate_uuid5

    now = datetime.now(timezone.utc)
    now_str = now.isoformat()
    relationships: List[dict] = []
    cache_entries: List[InterestCacheEntry] = []
    entity_cache: Dict = {}

    turn_node_props = {
        "id": payload.turn_id,
        "conversation_id": payload.conversation_id,
        "user_message_excerpt": payload.user_message[:200],
        "created_at": now_str,
    }

    def _build_rels_for_edge(
        edge_id: str,
        edge_props: dict,
        domain_id: str,
        domain_props: dict,
        entity_name: str,
        entity_type: str,
        is_invalidation: bool,
    ) -> None:
        """Append the three standard relationships for one UserInterestEdge."""
        # 1. Edge ─HAS_INTEREST_IN─> Domain
        relationships.append(
            {
                "from_name": edge_id,
                "from_type": "UserInterestEdge",
                "relation": "HAS_INTEREST_IN",
                "to_name": domain_id,
                "to_type": "UserInterestDomain",
                "confidence": "high",
                "reason": "",
                "from_node_props": edge_props,
                "to_node_props": domain_props,
            }
        )
        # 2. Edge ─TARGETS─> Entity
        relationships.append(
            {
                "from_name": edge_id,
                "from_type": "UserInterestEdge",
                "relation": "TARGETS",
                "to_name": entity_name,
                "to_type": entity_type,
                "confidence": "high",
                "reason": "",
                "from_node_props": edge_props,
                "to_node_props": {},
            }
        )
        # 3. Edge ─SOURCED_FROM / INVALIDATED_BY─> TurnNode
        rel_type = "INVALIDATED_BY" if is_invalidation else "SOURCED_FROM"
        relationships.append(
            {
                "from_name": edge_id,
                "from_type": "UserInterestEdge",
                "relation": rel_type,
                "to_name": payload.turn_id,
                "to_type": "TurnNode",
                "confidence": "high",
                "reason": "",
                "from_node_props": edge_props,
                "to_node_props": turn_node_props,
            }
        )

    async def _process_investment(signal: InvestmentSignal) -> None:
        is_invalidation = signal.status in ("Avoids", "Sold")
        operation = "invalidate" if is_invalidation else "reinforce"

        for entity in signal.target_entities:
            resolved_id = await ingestor.resolve_entity_id(
                entity.entity_name,
                entity.entity_type,
                entity_cache=entity_cache,
            )
            if not resolved_id:
                continue

            category = _derive_investment_category(
                entity.entity_name, entity.entity_type, payload.ticker_metadata
            )
            domain_id = generate_uuid5(f"{payload.user_email}::investment::{category}")
            edge_id = generate_uuid5(
                f"{payload.user_email}::investment::{category}::{resolved_id}"
            )

            domain_props = {
                "id": domain_id,
                "user_email": payload.user_email,
                "domain_type": "investment",
                "category": category,
                "created_at": now_str,
                "nodeset_id": nodeset_id,
            }
            edge_props = {
                "id": edge_id,
                "user_email": payload.user_email,
                "domain_type": "investment",
                "category": category,
                "entity_id": resolved_id,
                "weight_delta": signal.confidence,
                "operation": operation,
                "created_at": now_str,
                "last_updated_at": now_str,
            }

            _build_rels_for_edge(
                edge_id,
                edge_props,
                domain_id,
                domain_props,
                entity.entity_name,
                entity.entity_type,
                is_invalidation,
            )
            cache_entries.append(
                InterestCacheEntry(
                    kind="investment",
                    category=category,
                    entity_name=entity.entity_name,
                    entity_type=entity.entity_type,
                    weight=signal.confidence,
                    status="Invalidated" if is_invalidation else "Active",
                    invalidated=is_invalidation,
                    cached_at=now,
                    reason=payload.user_message[:200],
                )
            )

    async def _process_learning(signal: LearningSignal) -> None:
        is_invalidation = signal.status == "Not Interested"
        operation = "invalidate" if is_invalidation else "reinforce"

        for entity in signal.target_entities:
            resolved_id = await ingestor.resolve_entity_id(
                entity.entity_name,
                entity.entity_type,
                entity_cache=entity_cache,
            )
            if not resolved_id:
                continue

            category = await _derive_learning_category(
                entity.entity_type, resolved_id, neo4j
            )
            domain_id = generate_uuid5(f"{payload.user_email}::learning::{category}")
            edge_id = generate_uuid5(
                f"{payload.user_email}::learning::{category}::{resolved_id}"
            )

            domain_props = {
                "id": domain_id,
                "user_email": payload.user_email,
                "domain_type": "learning",
                "category": category,
                "created_at": now_str,
                "nodeset_id": nodeset_id,
            }
            edge_props = {
                "id": edge_id,
                "user_email": payload.user_email,
                "domain_type": "learning",
                "category": category,
                "entity_id": resolved_id,
                "weight_delta": signal.confidence,
                "operation": operation,
                "created_at": now_str,
                "last_updated_at": now_str,
            }

            _build_rels_for_edge(
                edge_id,
                edge_props,
                domain_id,
                domain_props,
                entity.entity_name,
                entity.entity_type,
                is_invalidation,
            )
            cache_entries.append(
                InterestCacheEntry(
                    kind="learning",
                    category=category,
                    entity_name=entity.entity_name,
                    entity_type=entity.entity_type,
                    weight=signal.confidence,
                    status="Invalidated" if is_invalidation else "Active",
                    invalidated=is_invalidation,
                    cached_at=now,
                    reason=payload.user_message[:200],
                )
            )

    for signal in payload.investment_signals:
        try:
            await _process_investment(signal)
        except Exception:
            logger.exception("_build_interest_relationships: investment signal failed")

    for signal in payload.learning_signals:
        try:
            await _process_learning(signal)
        except Exception:
            logger.exception("_build_interest_relationships: learning signal failed")

    return relationships, cache_entries


# ── Public entry point ────────────────────────────────────────────────────────


async def write_user_signals(payload: UserSignalPayload) -> None:
    """
    Persist all user interest signals from a conversation turn.

    Steps:
      1. Resolve domain entity IDs + build full relationship list.
      2. Dispatch to schedule(bypass_guards=True) for unified graph write
         (dedup → Neo4j + Chroma via _upsert_graph_to_neo4j).
      3. Update in-memory cache synchronously so the current session sees
         new signals immediately without waiting for Neo4j.

    Never raises — all exceptions are caught and logged.
    """
    logger.info(
        "write_user_signals: user='%s' turn='%s' " "investment=%d learning=%d edges=%d",
        payload.user_email,
        payload.turn_id,
        len(payload.investment_signals),
        len(payload.learning_signals),
        len(payload.interest_edges),
    )

    if not payload.user_email or not payload.conversation_id:
        logger.warning(
            "write_user_signals: missing user_email or conversation_id — skipping"
        )
        return

    if not payload.investment_signals and not payload.learning_signals:
        return

    try:
        from core.memory.graph.nodeset_manager import (
            canonical_nodeset_id,
            get_user_nodeset_name,
        )
        from core.services import service_manager

        ingestor = service_manager.get_ingestor()
        neo4j = service_manager.get_neo4j_adapter()
        subgraph_svc = service_manager.get_subgraph_service()
        user_ctx_svc = service_manager.get_user_context_service()

        nodeset_id = canonical_nodeset_id(get_user_nodeset_name(payload.user_email))

        # ── Step 1: build relationship list with pre-resolved entity IDs ──────
        relationships, cache_entries = await _build_interest_relationships(
            payload, ingestor, neo4j, nodeset_id
        )

        if not relationships:
            logger.info(
                "write_user_signals: no relationships built — skipping graph write"
            )
            return

        # ── Step 2: unified graph write via schedule() ────────────────────────
        await subgraph_svc.schedule(
            agent_name="user_signal_writeback",
            conversation_id=payload.conversation_id,
            analysis_text="",
            llm=None,
            system_prompt="",
            relationships=relationships,
            bypass_guards=True,
        )

        # ── Step 3: update in-memory cache immediately ────────────────────────
        if cache_entries:
            user_ctx_svc.update_cache(cache_entries, payload.user_email)

    except Exception:
        logger.exception("write_user_signals: failed for user '%s'", payload.user_email)


def build_signal_payload(
    detected_investment_signals: List[InvestmentSignal],
    detected_learning_signals: List[LearningSignal],
    interest_edges: List[dict],
    user_message: str,
    user_email: Optional[str],
    conversation_id: Optional[str],
    turn_id: str,
    ticker_metadata: Dict[str, dict],
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
        turn_id=turn_id,
        user_message=user_message,
        ticker_metadata=ticker_metadata,
        investment_signals=investment_signals,
        learning_signals=learning_signals,
        interest_edges=edges,
    )
