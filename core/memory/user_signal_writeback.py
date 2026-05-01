"""
Unified user-signal writeback for investment + learning interests.

Graph pattern:
NodeSet <-[:BELONGS_TO_NODESET]- UserInterestDomain
       -[:HAS_INTEREST_IN]-> UserInterestEdge
       -[:TARGETS]-> Entity
       -[:HAS_EVENT]-> UserInterestEvent
UserInterestEvent -[:OBSERVED_IN]-> SessionNode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from core.logger import get_logger
from core.memory.user_context_service import InterestCacheEntry

logger = get_logger(__name__)


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
class UserSignalPayload:
    user_email: str
    conversation_id: str
    turn_id: str
    user_message: str
    ticker_metadata: Dict[str, dict] = field(default_factory=dict)
    investment_signals: List[InvestmentSignal] = field(default_factory=list)
    learning_signals: List[LearningSignal] = field(default_factory=list)
    session_started_at: Optional[str] = None


@dataclass
class UserSignalWritebackResult:
    relationships: List[dict] = field(default_factory=list)
    cache_entries: List[InterestCacheEntry] = field(default_factory=list)


def _derive_investment_category(
    entity_name: str,
    entity_type: str,
    ticker_metadata: Dict[str, dict],
) -> str:
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


async def _derive_learning_category(entity_type: str, entity_id: str, neo4j) -> str:
    if entity_type == "FinancialConcept":
        try:
            category = await neo4j.get_entity_category(entity_id)
            if category:
                return category
        except Exception:
            logger.exception("_derive_learning_category: get_entity_category failed")
    if entity_type == "FinancialEvent":
        return "market_events"
    return "general"


def _signal_stance(kind: Literal["investment", "learning"], status: str) -> Literal["positive", "negative"]:
    if kind == "investment":
        return "negative" if status in {"Avoids", "Sold"} else "positive"
    return "negative" if status == "Not Interested" else "positive"


@dataclass
class _NormalizedSignal:
    kind: Literal["investment", "learning"]
    status: str
    confidence: float
    target_entities: Sequence[DetectedEntity]


def _iter_signals(payload: UserSignalPayload) -> List[_NormalizedSignal]:
    normalized: List[_NormalizedSignal] = []
    for signal in payload.investment_signals:
        normalized.append(
            _NormalizedSignal(
                kind="investment",
                status=signal.status,
                confidence=float(signal.confidence or 0.0),
                target_entities=signal.target_entities,
            )
        )
    for signal in payload.learning_signals:
        normalized.append(
            _NormalizedSignal(
                kind="learning",
                status=signal.status,
                confidence=float(signal.confidence or 0.0),
                target_entities=signal.target_entities,
            )
        )
    return normalized


async def _build_interest_relationships(
    payload: UserSignalPayload,
    entity_resolver,
    neo4j,
    nodeset_id: str,
) -> Tuple[List[dict], List[InterestCacheEntry]]:
    from core.memory.graph.utils import generate_uuid5

    now = datetime.now(timezone.utc)
    now_str = now.isoformat()
    session_id = payload.conversation_id
    session_started_at = payload.session_started_at or now_str
    relationships: List[dict] = []
    cache_entries: List[InterestCacheEntry] = []
    entity_cache: Dict[Tuple[str, str], str] = {}

    session_node_props = {
        "id": session_id,
        "user_email": payload.user_email,
        "started_at": session_started_at,
    }

    def _append_rels(
        *,
        domain_id: str,
        domain_props: dict,
        edge_id: str,
        edge_props: dict,
        entity_name: str,
        entity_type: str,
        event_id: str,
        event_props: dict,
    ) -> None:
        relationships.append(
            {
                "from_name": domain_id,
                "from_type": "UserInterestDomain",
                "relation": "HAS_INTEREST_IN",
                "to_name": edge_id,
                "to_type": "UserInterestEdge",
                "confidence": "high",
                "reason": "",
                "from_node_props": domain_props,
                "to_node_props": edge_props,
            }
        )
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
        relationships.append(
            {
                "from_name": edge_id,
                "from_type": "UserInterestEdge",
                "relation": "HAS_EVENT",
                "to_name": event_id,
                "to_type": "UserInterestEvent",
                "confidence": "high",
                "reason": "",
                "from_node_props": edge_props,
                "to_node_props": event_props,
            }
        )
        relationships.append(
            {
                "from_name": event_id,
                "from_type": "UserInterestEvent",
                "relation": "OBSERVED_IN",
                "to_name": session_id,
                "to_type": "SessionNode",
                "confidence": "high",
                "reason": "",
                "from_node_props": event_props,
                "to_node_props": session_node_props,
            }
        )

    event_ordinal = 0
    for signal in _iter_signals(payload):
        stance = _signal_stance(signal.kind, signal.status)
        operation = "invalidate" if stance == "negative" else "reinforce"

        for entity in signal.target_entities:
            key = (entity.entity_name, entity.entity_type)
            resolved_id = entity_cache.get(key)
            if resolved_id is None:
                resolution = await entity_resolver.resolve_entity(
                    name=entity.entity_name,
                    entity_type=entity.entity_type,
                )
                if resolution.entity_id:
                    resolved_id = resolution.entity_id
                    entity_cache[key] = resolved_id
            if not resolved_id:
                continue

            if signal.kind == "investment":
                category = _derive_investment_category(
                    entity.entity_name,
                    entity.entity_type,
                    payload.ticker_metadata,
                )
            else:
                category = await _derive_learning_category(
                    entity.entity_type,
                    resolved_id,
                    neo4j,
                )

            domain_id = generate_uuid5(f"{payload.user_email}::{signal.kind}::{category}")
            edge_id = generate_uuid5(
                f"{payload.user_email}::{signal.kind}::{category}::{resolved_id}"
            )
            event_id = generate_uuid5(
                f"{payload.turn_id}::{signal.kind}::{category}::{resolved_id}::{stance}::{event_ordinal}"
            )
            event_ordinal += 1

            domain_props = {
                "id": domain_id,
                "user_email": payload.user_email,
                "domain_type": signal.kind,
                "category": category,
                "created_at": now_str,
                "nodeset_id": nodeset_id,
            }
            edge_props = {
                "id": edge_id,
                "user_email": payload.user_email,
                "domain_type": signal.kind,
                "category": category,
                "entity_id": resolved_id,
                "weight_delta": max(0.0, signal.confidence),
                "operation": operation,
                "event_observed_at": now_str,
                "created_at": now_str,
                "last_updated_at": now_str,
            }
            event_props = {
                "id": event_id,
                "user_email": payload.user_email,
                "domain_type": signal.kind,
                "category": category,
                "entity_id": resolved_id,
                "stance": stance,
                "confidence": max(0.0, signal.confidence),
                "observed_at": now_str,
                "source_excerpt": payload.user_message[:200],
            }

            _append_rels(
                domain_id=domain_id,
                domain_props=domain_props,
                edge_id=edge_id,
                edge_props=edge_props,
                entity_name=entity.entity_name,
                entity_type=entity.entity_type,
                event_id=event_id,
                event_props=event_props,
            )
            cache_entries.append(
                InterestCacheEntry(
                    kind=signal.kind,
                    category=category,
                    entity_id=resolved_id,
                    entity_name=entity.entity_name,
                    entity_type=entity.entity_type,
                    cumulative_weight=max(0.0, signal.confidence),
                    reinforcement_count=1 if stance == "positive" else 0,
                    invalidation_count=1 if stance == "negative" else 0,
                    current_stance=stance,
                    previous_stance=None,
                    last_changed_at=now,
                    cached_at=now,
                    reason=payload.user_message[:200],
                )
            )

    return relationships, cache_entries


async def _build_user_signal_relationships(
    payload: UserSignalPayload,
) -> Tuple[List[dict], List[InterestCacheEntry]]:
    """
    Build all user-interest relationships for one turn/session writeback.

    Returns (relationships, cache_entries). No graph writes here.
    """
    logger.info(
        "_build_user_signal_relationships: user='%s' turn='%s' investment=%d learning=%d",
        payload.user_email,
        payload.turn_id,
        len(payload.investment_signals),
        len(payload.learning_signals),
    )

    if not payload.user_email or not payload.conversation_id:
        logger.warning(
            "_build_user_signal_relationships: missing user_email or conversation_id; skipping"
        )
        return [], []

    if not payload.investment_signals and not payload.learning_signals:
        return [], []

    try:
        from core.memory.graph.nodeset_manager import canonical_nodeset_id, get_user_nodeset_name
        from core.services import service_manager

        entity_resolver = service_manager.get_entity_resolver()
        neo4j = service_manager.get_neo4j_adapter()
        nodeset_id = canonical_nodeset_id(get_user_nodeset_name(payload.user_email))
        return await _build_interest_relationships(
            payload=payload,
            entity_resolver=entity_resolver,
            neo4j=neo4j,
            nodeset_id=nodeset_id,
        )
    except Exception:
        logger.exception(
            "_build_user_signal_relationships: failed for user '%s'",
            payload.user_email,
        )
        return [], []


def _update_user_signal_cache(
    cache_entries: List[InterestCacheEntry], user_email: str
) -> None:
    if not cache_entries or not user_email:
        return
    try:
        from core.services import service_manager

        user_ctx_svc = service_manager.get_user_context_service()
        user_ctx_svc.update_cache(cache_entries, user_email)
    except Exception:
        logger.exception("_update_user_signal_cache: failed for user '%s'", user_email)


def _build_signal_payload(
    detected_investment_signals,
    detected_learning_signals,
    user_message: str,
    user_email: Optional[str],
    conversation_id: Optional[str],
    turn_id: str,
    ticker_metadata: Dict[str, dict],
    session_started_at: Optional[str] = None,
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
    return UserSignalPayload(
        user_email=user_email or "",
        conversation_id=conversation_id or "",
        turn_id=turn_id,
        user_message=user_message,
        ticker_metadata=ticker_metadata,
        investment_signals=investment_signals,
        learning_signals=learning_signals,
        session_started_at=session_started_at,
    )


async def process_user_signal_writeback(
    *,
    user_email: Optional[str],
    conversation_id: Optional[str],
    turn_id: str,
    user_message: str,
    ticker_metadata: Optional[Dict[str, dict]] = None,
    detected_investment_signals: Optional[Sequence[object]] = None,
    detected_learning_signals: Optional[Sequence[object]] = None,
    session_started_at: Optional[str] = None,
) -> UserSignalWritebackResult:
    """
    Single entry-point for user-signal writeback.

    Normalizes planner signals, builds graph relationships, and updates the
    in-memory user-context cache. Graph persistence is intentionally left to
    the caller via the returned relationships.
    """
    payload = _build_signal_payload(
        detected_investment_signals=detected_investment_signals or [],
        detected_learning_signals=detected_learning_signals or [],
        user_message=user_message,
        user_email=user_email,
        conversation_id=conversation_id,
        turn_id=turn_id,
        ticker_metadata=ticker_metadata or {},
        session_started_at=session_started_at,
    )
    relationships, cache_entries = await _build_user_signal_relationships(payload)
    if cache_entries and payload.user_email:
        _update_user_signal_cache(cache_entries, payload.user_email)
    return UserSignalWritebackResult(
        relationships=relationships,
        cache_entries=cache_entries,
    )
