"""
core/memory/graph/graph_writer.py

Pure write layer for Neo4j relationship edges.

Receives already-resolved entity IDs (from EntityResolver) and relationship
dicts, builds the relationship properties, and calls neo4j_adapter.merge_relationship().

No dedup, no LLM, no state.  Callers are:
  - ConversationQueue._process_batch()   (queue consumer path)
  - GraphQueueManager.write_immediate()  (bypass path for system tasks)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.logger import get_logger
from core.memory.graph.models import _USER_SCOPED_TYPES
from core.memory.graph.utils import (
    entity_key,
    normalize_entity_name,
    normalize_entity_type,
)

logger = get_logger(__name__)


class GraphWriter:
    """
    Writes relationship edges to Neo4j given pre-resolved entity IDs.

    Injected at construction:
        neo4j_adapter — the async Neo4j adapter (core.memory.stores.neo4j_adapter)
    """

    def __init__(self, neo4j_adapter) -> None:
        self._neo4j = neo4j_adapter

    async def write_relationships(
        self,
        relationships: List[dict],
        conversation_id: str,
        source_agent: str,
        entity_cache: Dict[Tuple[str, str], str],
    ) -> int:
        """
        Write edges to Neo4j.  Entity IDs must be pre-resolved in entity_cache.

        entity_cache maps (name.lower(), entity_type) → canonical_id.
        Any edge whose from/to entity is not in entity_cache is skipped with a warning.

        Returns the number of edges successfully written.
        """
        written = 0

        for rel in relationships:
            from_name_raw = str(rel.get("from_name") or "").strip()
            to_name_raw = str(rel.get("to_name") or "").strip()
            raw_from_type = str(rel.get("from_type") or "").strip()
            raw_to_type = str(rel.get("to_type") or "").strip()

            # User-scoped types bypass normalize_entity_type validation
            from_type = (
                raw_from_type
                if raw_from_type in _USER_SCOPED_TYPES
                else normalize_entity_type(raw_from_type)
            )
            to_type = (
                raw_to_type
                if raw_to_type in _USER_SCOPED_TYPES
                else normalize_entity_type(raw_to_type)
            )

            from_name = normalize_entity_name(from_name_raw)
            to_name = normalize_entity_name(to_name_raw)

            if not from_name or not to_name or not from_type or not to_type:
                logger.debug(
                    "write_relationships: skipping incomplete rel from=%r to=%r",
                    from_name_raw,
                    to_name_raw,
                )
                continue

            from_key = entity_key(from_name, from_type)
            to_key = entity_key(to_name, to_type)

            resolved_source = entity_cache.get(from_key)
            resolved_target = entity_cache.get(to_key)

            if not resolved_source or not resolved_target:
                logger.warning(
                    "write_relationships: unresolved entity — "
                    "from='%s' (%s) resolved=%s | to='%s' (%s) resolved=%s",
                    from_name,
                    from_type,
                    resolved_source,
                    to_name,
                    to_type,
                    resolved_target,
                )
                continue

            relation_type = str(
                rel.get("relation") or rel.get("relation_type") or "RELATED_TO"
            ).strip()
            confidence = str(rel.get("confidence") or "low").strip() or "low"
            reason = str(rel.get("reason") or "").strip() or None

            # Collect any extra_props forwarded by the agent
            extra_props: dict = {}
            if isinstance(rel.get("extra_props"), dict):
                extra_props.update(rel["extra_props"])

            props = self._build_relationship_props(
                relation_type=relation_type,
                confidence=confidence,
                conversation_id=conversation_id,
                from_type=from_type,
                to_type=to_type,
                reason=reason,
                source_agent=source_agent,
                extra_props=extra_props,
            )

            try:
                await self._neo4j.merge_relationship(
                    resolved_source, resolved_target, relation_type, props
                )
                written += 1
            except Exception:
                logger.exception(
                    "write_relationships: Neo4j merge_relationship failed "
                    "for %s -[%s]-> %s",
                    from_name,
                    relation_type,
                    to_name,
                )

        return written

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_relationship_props(
        relation_type: str,
        confidence: str,
        conversation_id: str,
        from_type: str,
        to_type: str,
        reason: Optional[str] = None,
        source_agent: Optional[str] = None,
        extra_props: Optional[dict] = None,
    ) -> Dict[str, Any]:
        props: Dict[str, Any] = {
            "relationship_type": relation_type,
            "confidence": confidence,
            "source_conversation_id": conversation_id,
            "from_type": from_type,
            "to_type": to_type,
        }
        if reason:
            props["reason"] = reason
        if source_agent:
            props["source_agent"] = source_agent
        if extra_props:
            # Extra props cannot override the standard keys
            for k, v in extra_props.items():
                if k not in props:
                    props[k] = v
        return props
