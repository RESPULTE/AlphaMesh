"""Async Neo4j adapter for graph I/O."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from core.logger import get_logger
from core.memory.graph.models import (
    _USER_SCOPED_TYPES,
    _USER_SCOPED_RELATIONSHIP_TYPES,
    ALLOWED_ENTITY_TYPES,
    DocumentNode,
    EntityNode,
)
from core.memory.graph.utils import (
    entity_key,
    normalize_entity_name,
    normalize_entity_type,
)
from core.memory.retrieval.models import RetrievedChunk


class RelationshipType(Enum):
    TARGETS = "TARGETS"
    BELONGS_TO_NODESET = "BELONGS_TO_NODESET"
    MENTIONS_ENTITY = "MENTIONS_ENTITY"
    RELATED_TO = "RELATED_TO"
    BELONGS_TO_DOCUMENT = "BELONGS_TO_DOCUMENT"


class Neo4jAdapter:
    """Encapsulates all Neo4j operations for AlphaMesh."""

    def __init__(
        self, uri: str, username: str, password: str, database: str = "neo4j"
    ) -> None:
        """Initialize the adapter with connection details."""
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver: Optional[AsyncDriver] = None
        self._logger = get_logger(__name__)

    async def _get_driver(self) -> AsyncDriver:
        """Lazily initialize and return the async Neo4j driver."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._username, self._password)
            )
        return self._driver

    async def _execute_write(self, cypher: str, params: Dict[str, object]) -> None:
        """Execute a write query with the provided parameters."""
        driver = await self._get_driver()

        async def _tx_run(tx) -> None:
            await tx.run(cypher, **params)

        try:
            async with driver.session(database=self._database) as session:
                await session.execute_write(_tx_run)
        except (Neo4jError, ServiceUnavailable) as exc:
            self._logger.exception("Neo4j write failed.")
            raise

    async def _execute_read(self, cypher: str, params: Dict[str, object]):
        """Execute a read query and return the result data (not cursor)."""
        driver = await self._get_driver()

        async def _tx_run(tx):
            result = await tx.run(cypher, **params)
            return await result.data()

        try:
            async with driver.session(database=self._database) as session:
                return await session.execute_read(_tx_run)
        except (Neo4jError, ServiceUnavailable) as exc:
            self._logger.exception("Neo4j read failed.")
            raise

    async def entity_exists(self, entity_id: str) -> bool:
        if not entity_id:
            return False
        cypher = "MATCH (e:Entity {id: $id}) RETURN e.id AS id LIMIT 1"
        records = await self._execute_read(cypher, {"id": entity_id})
        return bool(records)

    async def find_entity_by_name(self, entity_type: str, name: str) -> Optional[dict]:
        """
        Return an exact normalized name match for a type, if present.

        Normalization is done in Cypher via lower+trim so callers can pass
        user-provided values without pre-processing concerns.
        """
        if not entity_type or not name:
            return None
        cypher = (
            "MATCH (e:Entity) "
            "WHERE e.entity_type = $entity_type "
            "AND e.name IS NOT NULL "
            "AND toLower(trim(e.name)) = toLower(trim($name)) "
            "RETURN e.id AS id, e.name AS name "
            "LIMIT 1"
        )
        records = await self._execute_read(
            cypher,
            {
                "entity_type": entity_type,
                "name": name,
            },
        )
        if not records:
            return None
        record = records[0]
        entity_id = record.get("id")
        if not entity_id:
            return None
        return {"id": entity_id, "name": record.get("name")}

    async def find_fuzzy_entity_candidates(
        self,
        entity_type: str,
        name: str,
        exclude_id: str = "",
        threshold: float = 0.50,
        limit: int = 10,
    ) -> List[dict]:
        cypher = (
            "MATCH (e:Entity) "
            "WHERE e.entity_type = $entity_type AND e.name IS NOT NULL AND e.id <> $exclude_id "
            "WITH e, apoc.text.sorensenDiceSimilarity(toLower(e.name), toLower($name)) AS sim "
            "WHERE sim >= $threshold "
            "RETURN e.id AS id, e.name AS name, sim AS similarity "
            "ORDER BY sim DESC "
            "LIMIT $limit"
        )
        records = await self._execute_read(
            cypher,
            {
                "entity_type": entity_type,
                "name": name,
                "exclude_id": exclude_id or "",
                "threshold": threshold,
                "limit": limit,
            },
        )
        return [
            {
                "id": record.get("id"),
                "name": record.get("name"),
                "similarity": record.get("similarity"),
            }
            for record in records
            if record.get("id")
        ]

    async def merge_document_node(self, node: DocumentNode) -> None:
        """Merge a document node and update its properties."""
        cypher = "MERGE (d:Document {id: $id}) SET d += $props"
        props = node.model_dump()
        await self._execute_write(cypher, {"id": node.id, "props": props})

    async def merge_chunk_node(self, node: RetrievedChunk) -> None:
        """Merge a chunk node and connect it to its document."""
        belongs_to = RelationshipType.BELONGS_TO_DOCUMENT.value
        cypher = (
            "MERGE (d:Document {id: $doc_id}) "
            "MERGE (c:Chunk {id: $id}) "
            "SET c += $props "
            f"MERGE (c)-[:{belongs_to}]->(d)"
        )
        props = {
            "text": node.text,
            "chunk_index": node.chunk_index,
            "document_id": node.document_id,
            "article_title": node.article_title,
            "source_url": node.source_url,
            "published_at": node.published_at,
            "nodeset_ids": node.nodeset_ids,
            "extraction_status": node.extraction_status,
        }
        await self._execute_write(
            cypher,
            {
                "id": node.chunk_id,
                "doc_id": node.document_id,
                "props": props,
            },
        )

    async def merge_entity_node(self, node: EntityNode) -> None:
        """
        Merge an entity node with a dynamic label.

        ON CREATE: all properties are written in full.
        ON MATCH:  only mutable metadata (e.g. nodeset_ids, ticker) is updated.
                name and description are intentionally preserved to protect
                canonical values sourced from yfinance for Company, Sector,
                Industry and Market entities.
        """
        if node.entity_type not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"Unsupported entity_type: {node.entity_type}")

        props = {k: v for k, v in node.model_dump().items() if k != "local_id"}

        cypher = (
            f"MERGE (e:Entity {{id: $id}}) "
            "ON CREATE SET e += $props "
            "ON MATCH SET "
            "  e.nodeset_ids = CASE WHEN $props.nodeset_ids IS NOT NULL AND size($props.nodeset_ids) > 0 "
            "                   THEN $props.nodeset_ids ELSE e.nodeset_ids END, "
            "  e.ticker = CASE WHEN $props.ticker IS NOT NULL THEN $props.ticker ELSE e.ticker END "
            f"SET e:{node.entity_type}"
        )
        await self._execute_write(cypher, {"id": node.id, "props": props})

    async def entity_exists_by_ticker(self, ticker: str) -> Optional[str]:
        """
        Return the entity ID if a Company entity with this ticker already exists
        in the graph, otherwise None.
        """
        cypher = (
            "MATCH (e:Entity:Company) "
            "WHERE e.ticker = $ticker "
            "RETURN e.id AS id LIMIT 1"
        )
        records = await self._execute_read(cypher, {"ticker": ticker.upper()})
        return records[0]["id"] if records else None

    async def merge_relationship(
        self, source_id: str, target_id: str, rel_type: str, props: Dict[str, object]
    ) -> None:
        """Merge a relationship between two entities."""
        if not rel_type.isidentifier():
            raise ValueError(f"Invalid relationship type: {rel_type}")

        cypher = (
            "MATCH (s {id: $source_id}) "
            "MATCH (t {id: $target_id}) "
            f"MERGE (s)-[r:{rel_type}]->(t) "
            "SET r += $props"
        )
        await self._execute_write(
            cypher,
            {"source_id": source_id, "target_id": target_id, "props": props},
        )

    async def write_relationships(
        self,
        relationships: List[dict],
        conversation_id: str,
        source_agent: str,
        entity_cache: Dict[Tuple[str, str], str],
    ) -> int:
        """
        Write edges to Neo4j. Entity IDs must be pre-resolved in entity_cache.

        entity_cache maps (name.lower(), entity_type) ? canonical_id.
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
                self._logger.debug(
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
                self._logger.warning(
                    "write_relationships: unresolved entity "
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
                await self.merge_relationship(
                    resolved_source, resolved_target, relation_type, props
                )
                written += 1
            except Exception:
                self._logger.exception(
                    "write_relationships: Neo4j merge_relationship failed "
                    "for %s -[%s]-> %s",
                    from_name,
                    relation_type,
                    to_name,
                )

        return written

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

    async def get_chunk_extraction_status(self, chunk_ids: List[str]) -> Dict[str, str]:
        """Fetch extraction status values for a list of chunk IDs."""
        cypher = (
            "MATCH (c:Chunk) "
            "WHERE c.id IN $chunk_ids "
            "RETURN c.id AS id, c.extraction_status AS status"
        )
        records = await self._execute_read(cypher, {"chunk_ids": chunk_ids})
        return {record["id"]: record.get("status", "PENDING") for record in records}

    async def update_chunk_extraction_status(self, chunk_id: str, status: str) -> None:
        """Update the extraction status on a chunk node."""
        cypher = "MATCH (c:Chunk {id: $id}) SET c.extraction_status = $status"
        await self._execute_write(cypher, {"id": chunk_id, "status": status})

    async def merge_nodeset_node(
        self, nodeset_id: str, name: str, description: str
    ) -> None:
        """Merge a NodeSet node and update its properties."""
        cypher = "MERGE (n:NodeSet {id: $id}) SET n += $props"
        props = {"id": nodeset_id, "name": name, "description": description}
        await self._execute_write(cypher, {"id": nodeset_id, "props": props})

    async def get_entities_for_chunks(self, chunk_ids: List[str]) -> List[dict]:
        """Return entities mentioned by the provided chunk IDs."""
        if not chunk_ids:
            return []
        mentions = RelationshipType.MENTIONS_ENTITY.value
        cypher = (
            f"MATCH (c:Chunk)-[:{mentions}]->(e:Entity) "
            "WHERE c.id IN $chunk_ids "
            "RETURN e.id AS entity_id, e.name AS entity_name, "
            "e.entity_type AS entity_type, c.id AS source_chunk_id"
        )
        records = await self._execute_read(cypher, {"chunk_ids": chunk_ids})
        self._logger.info(
            "Fetched %d entities for %d chunks.", len(records), len(chunk_ids)
        )
        return records

    async def get_entity_neighbors(
        self, entity_ids: List[str], exclude_ids: List[str]
    ) -> List[dict]:
        """Return neighboring entities connected by edges."""
        if not entity_ids:
            return []
        blocked_relationship_types = sorted(_USER_SCOPED_RELATIONSHIP_TYPES)
        cypher = (
            "MATCH (e:Entity)-[r]-(neighbor:Entity) "
            "WHERE e.id IN $entity_ids "
            "AND NOT type(r) IN $blocked_relationship_types "
            "AND coalesce(trim(e.user_email), '') = '' "
            "AND coalesce(trim(neighbor.user_email), '') = '' "
            "AND ($exclude_ids IS NULL OR NOT neighbor.id IN $exclude_ids) "
            "RETURN e.id AS source_entity_id, "
            "e.name AS source_entity_name, "
            "neighbor.id AS neighbor_entity_id, "
            "neighbor.name AS neighbor_name, "
            "neighbor.entity_type AS neighbor_type, "
            "r.relationship_type AS relationship_type, "
            "r.reason AS reason"
        )
        records = await self._execute_read(
            cypher,
            {
                "entity_ids": entity_ids,
                "exclude_ids": exclude_ids,
                "blocked_relationship_types": blocked_relationship_types,
            },
        )
        self._logger.info(
            "Fetched %d neighbors for %d entities.",
            len(records),
            len(entity_ids),
        )
        return records

    async def get_chunks_for_entities(
        self, entity_ids: List[str], exclude_chunk_ids: List[str]
    ) -> List[dict]:
        """
        Return chunks that mention the provided entities.

        extraction_status is included so that the retriever can correctly
        propagate PENDING vs EXTRACTED state to RetrievedChunk objects � without
        this field, all graph-expanded chunks defaulted to PENDING and were
        unnecessarily re-queued for entity extraction on every retrieval.
        """
        if not entity_ids:
            return []
        cypher = (
            "MATCH (c:Chunk)-[:MENTIONS_ENTITY]->(e:Entity) "
            "WHERE e.id IN $entity_ids "
            "AND coalesce(trim(e.user_email), '') = '' "
            "AND ($exclude_chunk_ids IS NULL OR NOT c.id IN $exclude_chunk_ids) "
            "RETURN c.id AS chunk_id, "
            "c.text AS chunk_text, "
            "c.article_title AS article_title, "
            "c.source_url AS source_url, "
            "c.chunk_index AS chunk_index, "
            "c.document_id AS document_id, "
            "c.published_at AS published_at, "
            "c.extraction_status AS extraction_status, "
            "e.id AS supporting_entity_id"
        )
        records = await self._execute_read(
            cypher,
            {"entity_ids": entity_ids, "exclude_chunk_ids": exclude_chunk_ids},
        )
        self._logger.info(
            "Fetched %d chunks for %d entities.",
            len(records),
            len(entity_ids),
        )
        return records

    async def close(self) -> None:
        """Close the underlying Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def run_traversal(self, cypher: str, params: dict) -> List[dict]:
        """Run a read-only traversal query."""
        return await self._execute_read(cypher, params)

    async def merge_user_interest_domain(self, domain_id: str, props: dict) -> None:
        """
        Merge a UserInterestDomain node.
        ON CREATE: full props written.
        ON MATCH: only last_seen_at updated; category and domain_type are immutable.
        """
        belongs_to = RelationshipType.BELONGS_TO_NODESET.value
        cypher = (
            "MERGE (d:UserInterestDomain {id: $id}) "
            "ON CREATE SET d += $props "
            "ON MATCH SET d.last_seen_at = $now "
            "WITH d "
            "FOREACH (_ IN CASE WHEN $nodeset_id IS NULL THEN [] ELSE [1] END | "
            "  MERGE (s:NodeSet {id: $nodeset_id}) "
            f"  MERGE (d)-[:{belongs_to}]->(s)"
            ")"
        )
        nodeset_id = str(props.get("nodeset_id") or "").strip() or None
        clean = {k: v for k, v in props.items() if k != "nodeset_id"}
        await self._execute_write(
            cypher,
            {
                "id": domain_id,
                "props": clean,
                "nodeset_id": nodeset_id,
                "now": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def merge_user_interest_edge(
        self,
        edge_id: str,
        props: dict,
        operation: str,
        weight_delta: float,
    ) -> None:
        """Merge a UserInterestEdge aggregate from immutable events."""
        observed_at = str(props.get("event_observed_at") or "").strip()
        if not observed_at:
            observed_at = datetime.now(timezone.utc).isoformat()

        if operation == "reinforce":
            cypher = (
                "MERGE (e:UserInterestEdge {id: $id}) "
                "ON CREATE SET e += $props, "
                "              e.cumulative_weight = $weight_delta, "
                "              e.reinforcement_count = 1, "
                "              e.invalidation_count = 0, "
                "              e.current_stance = 'positive', "
                "              e.last_changed_at = $observed_at "
                "ON MATCH SET  e.cumulative_weight = coalesce(e.cumulative_weight, 0.0) + $weight_delta, "
                "              e.reinforcement_count = coalesce(e.reinforcement_count, 0) + 1, "
                "              e.last_updated_at = $now, "
                "              e.last_changed_at = $observed_at, "
                "              e.current_stance = 'positive'"
            )
        else:  # invalidate
            cypher = (
                "MERGE (e:UserInterestEdge {id: $id}) "
                "ON CREATE SET e += $props, "
                "              e.cumulative_weight = $weight_delta, "
                "              e.reinforcement_count = 0, "
                "              e.invalidation_count = 1, "
                "              e.current_stance = 'negative', "
                "              e.last_changed_at = $observed_at "
                "ON MATCH SET  e.cumulative_weight = coalesce(e.cumulative_weight, 0.0) + $weight_delta, "
                "              e.invalidation_count = coalesce(e.invalidation_count, 0) + 1, "
                "              e.current_stance = 'negative', "
                "              e.last_updated_at = $now, "
                "              e.last_changed_at = $observed_at"
            )
        clean_props = {
            k: v for k, v in props.items() if k not in ("weight_delta", "operation")
        }
        await self._execute_write(
            cypher,
            {
                "id": edge_id,
                "props": clean_props,
                "weight_delta": weight_delta,
                "observed_at": observed_at,
                "now": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def merge_user_interest_event(self, node_id: str, props: dict) -> None:
        """Merge immutable user interest event by deterministic ID."""
        cypher = "MERGE (e:UserInterestEvent {id: $id}) ON CREATE SET e += $props"
        await self._execute_write(cypher, {"id": node_id, "props": props})

    async def merge_session_node(self, node_id: str, props: dict) -> None:
        """Merge one lightweight session node per conversation."""
        belongs_to = RelationshipType.BELONGS_TO_NODESET.value
        cypher = (
            "MERGE (s:SessionNode {id: $id}) "
            "ON CREATE SET s += $props "
            "ON MATCH SET s.last_seen_at = $now "
            "WITH s "
            "FOREACH (_ IN CASE WHEN $nodeset_id IS NULL THEN [] ELSE [1] END | "
            "  MERGE (ns:NodeSet {id: $nodeset_id}) "
            f"  MERGE (s)-[:{belongs_to}]->(ns)"
            ")"
        )
        nodeset_id = str(props.get("nodeset_id") or "").strip() or None
        clean = {k: v for k, v in props.items() if k != "nodeset_id"}
        await self._execute_write(
            cypher,
            {
                "id": node_id,
                "props": clean,
                "nodeset_id": nodeset_id,
                "now": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def get_entity_category(self, entity_id: str) -> Optional[str]:
        """Return the category field of an entity (used for FinancialConcept category)."""
        cypher = (
            "MATCH (e:Entity {id: $id}) "
            "OPTIONAL MATCH (e)-[:BELONGS_TO]->(c:Entity:FinancialConceptCategory) "
            "WITH e, collect(c.name) AS categories "
            "RETURN coalesce(e.category, categories[0]) AS category LIMIT 1"
        )
        records = await self._execute_read(cypher, {"id": entity_id})
        return records[0].get("category") if records else None

    async def get_user_interest_domain_summary(
        self,
        user_email: str,
        nodeset_id: str,
        limit: int = 3,
    ) -> List[dict]:
        """
        Return top user-interest domains only (no edge/entity expansion).

        Ranked by positive stance weight then recency so broad fallback context
        remains compact and avoids over-weighting stale negative edges.
        """
        safe_limit = max(1, min(10, int(limit or 3)))
        cypher = (
            "MATCH (ns:NodeSet {id: $nodeset_id})"
            "<-[:BELONGS_TO_NODESET]-(d:UserInterestDomain {user_email: $user_email})"
            "-[:HAS_INTEREST_IN]->(e:UserInterestEdge) "
            "WITH d, "
            "     count(e) AS edge_count, "
            "     sum(CASE WHEN coalesce(e.current_stance, 'positive') = 'positive' "
            "         THEN coalesce(e.cumulative_weight, 0.0) ELSE 0.0 END) AS positive_weight, "
            "     sum(CASE WHEN coalesce(e.current_stance, 'positive') = 'negative' "
            "         THEN coalesce(e.cumulative_weight, 0.0) ELSE 0.0 END) AS negative_weight, "
            "     max(coalesce(e.last_changed_at, e.last_updated_at, e.created_at)) AS last_changed_at "
            "RETURN d AS domain, edge_count, positive_weight, negative_weight, last_changed_at "
            "ORDER BY positive_weight DESC, last_changed_at DESC "
            "LIMIT $limit"
        )
        return await self._execute_read(
            cypher,
            {
                "nodeset_id": nodeset_id,
                "user_email": user_email,
                "limit": safe_limit,
            },
        )

    async def query_user_interest_context(
        self,
        *,
        user_email: str,
        nodeset_id: str,
        domain_type: Optional[str] = None,
        category: Optional[str] = None,
        target_entities: Optional[List[dict]] = None,
        hops: int = 0,
        risk_or_avoidance_intent: bool = False,
        domain_limit: int = 3,
        edge_limit: int = 8,
        expanded_entity_limit: int = 12,
    ) -> List[dict]:
        """
        Targeted read for orchestrator personalization context.

        Filters by domain type/category and optional entities, then returns
        interest edges + target entities + hop-based expansion neighbors.
        """
        safe_hops = max(0, min(2, int(hops or 0)))
        safe_domain_limit = max(1, min(10, int(domain_limit or 3)))
        safe_edge_limit = max(1, min(20, int(edge_limit or 8)))
        safe_expanded_limit = max(1, min(40, int(expanded_entity_limit or 12)))

        normalized_domain_type = str(domain_type or "").strip().lower() or None
        normalized_category = str(category or "").strip() or None
        include_negative = bool(risk_or_avoidance_intent)

        target_filters: List[dict] = []
        for row in list(target_entities or []):
            if not isinstance(row, dict):
                continue
            entity_name = str(row.get("entity_name") or row.get("name") or "").strip()
            entity_type = str(row.get("entity_type") or "").strip()
            if not entity_name or not entity_type:
                continue
            target_filters.append(
                {
                    "name_lower": entity_name.lower(),
                    "entity_type": entity_type,
                }
            )

        blocked_relationship_types = sorted(_USER_SCOPED_RELATIONSHIP_TYPES)
        blocked_node_labels = sorted(_USER_SCOPED_TYPES)
        cypher = (
            "MATCH (ns:NodeSet {id: $nodeset_id})"
            "<-[:BELONGS_TO_NODESET]-(d:UserInterestDomain {user_email: $user_email}) "
            "WHERE ($domain_type IS NULL OR d.domain_type = $domain_type) "
            "  AND ($category IS NULL OR toLower(trim(d.category)) = toLower(trim($category))) "
            "MATCH (d)-[:HAS_INTEREST_IN]->(rank_edge:UserInterestEdge) "
            "WITH d, "
            "     max(coalesce(rank_edge.last_changed_at, rank_edge.last_updated_at, rank_edge.created_at)) AS domain_last_changed, "
            "     sum(CASE WHEN coalesce(rank_edge.current_stance, 'positive') = 'positive' "
            "         THEN coalesce(rank_edge.cumulative_weight, 0.0) ELSE 0.0 END) AS domain_positive_weight "
            "ORDER BY domain_positive_weight DESC, domain_last_changed DESC "
            "LIMIT $domain_limit "
            "MATCH (d)-[:HAS_INTEREST_IN]->(e:UserInterestEdge)-[:TARGETS]->(entity:Entity) "
            "WHERE ($include_negative OR coalesce(e.current_stance, 'positive') = 'positive') "
            "  AND (size($target_filters) = 0 "
            "       OR any(tf IN $target_filters "
            "              WHERE toLower(trim(entity.name)) = tf.name_lower "
            "                AND entity.entity_type = tf.entity_type)) "
            "WITH d, e, entity, "
            "     coalesce(e.current_stance, 'positive') AS stance, "
            "     coalesce(e.last_changed_at, e.last_updated_at, e.created_at) AS edge_last_changed "
            "ORDER BY CASE WHEN stance = 'positive' THEN 0 ELSE 1 END, "
            "         edge_last_changed DESC, "
            "         coalesce(e.cumulative_weight, 0.0) DESC "
            "WITH collect({d: d, e: e, entity: entity, stance: stance, edge_last_changed: edge_last_changed})[..$edge_limit] AS edge_rows, "
            "     $hops AS hops, "
            "     $expanded_entity_limit AS expanded_entity_limit "
            "UNWIND edge_rows AS row "
            "WITH row.d AS d, row.e AS e, row.entity AS entity, row.stance AS stance, row.edge_last_changed AS edge_last_changed, "
            "     hops, expanded_entity_limit "
            "CALL { "
            "  WITH entity, hops, expanded_entity_limit, $blocked_relationship_types AS blocked_relationship_types, "
            "       $blocked_node_labels AS blocked_node_labels, $user_email AS user_email "
            "  WITH entity, hops, expanded_entity_limit, blocked_relationship_types, blocked_node_labels, user_email WHERE hops > 0 "
            "  MATCH path=(entity)-[*1..2]-(neighbor:Entity) "
            "  WHERE length(path) <= hops "
            "    AND all(rel IN relationships(path) WHERE NOT type(rel) IN blocked_relationship_types) "
            "    AND all(path_node IN tail(nodes(path)) WHERE none(label IN labels(path_node) WHERE label IN blocked_node_labels)) "
            "    AND all(path_node IN tail(nodes(path)) WHERE coalesce(trim(path_node.user_email), '') IN ['', user_email]) "
            "  WITH collect(DISTINCT {id: neighbor.id, name: neighbor.name, entity_type: neighbor.entity_type}) AS all_expanded_neighbors, "
            "       expanded_entity_limit "
            "  RETURN all_expanded_neighbors[..expanded_entity_limit] AS expanded_neighbors "
            "  UNION "
            "  WITH entity "
            "  RETURN [] AS expanded_neighbors "
            "} "
            "RETURN d AS domain, e AS edge, entity AS entity, stance, edge_last_changed, expanded_neighbors"
        )
        return await self._execute_read(
            cypher,
            {
                "nodeset_id": nodeset_id,
                "user_email": user_email,
                "domain_type": normalized_domain_type,
                "category": normalized_category,
                "target_filters": target_filters,
                "include_negative": include_negative,
                "domain_limit": safe_domain_limit,
                "edge_limit": safe_edge_limit,
                "hops": safe_hops,
                "expanded_entity_limit": safe_expanded_limit,
                "blocked_relationship_types": blocked_relationship_types,
                "blocked_node_labels": blocked_node_labels,
            },
        )

    async def get_user_interest_data(
        self, user_email: str, nodeset_id: str
    ) -> List[dict]:
        """
        Full read: NodeSet -> Domain -> Edge -> Entity with latest event snapshots.
        Returns all edges including conflicting/negative stances.
        """
        cypher = (
            "MATCH (ns:NodeSet {id: $nodeset_id})"
            "<-[:BELONGS_TO_NODESET]-(d:UserInterestDomain {user_email: $user_email})"
            "-[:HAS_INTEREST_IN]->(e:UserInterestEdge)"
            "-[:TARGETS]->(entity:Entity) "
            "OPTIONAL MATCH (e)-[:HAS_EVENT]->(ev:UserInterestEvent) "
            "WITH d, e, entity, ev "
            "ORDER BY ev.observed_at DESC "
            "WITH d, e, entity, [x IN collect(ev) WHERE x IS NOT NULL] AS events "
            "RETURN d, e, entity, "
            "       CASE WHEN size(events) > 0 THEN events[0] ELSE NULL END AS latest_event, "
            "       CASE WHEN size(events) > 1 THEN events[1] ELSE NULL END AS previous_event "
            "ORDER BY d.category ASC, "
            "         coalesce(e.last_changed_at, e.last_updated_at, e.created_at) DESC, "
            "         coalesce(e.cumulative_weight, 0.0) DESC"
        )
        return await self._execute_read(
            cypher,
            {
                "nodeset_id": nodeset_id,
                "user_email": user_email,
            },
        )


