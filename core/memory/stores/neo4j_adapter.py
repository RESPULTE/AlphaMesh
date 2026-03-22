"""Async Neo4j adapter for graph I/O."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from core.logger import get_logger
from core.memory.graph.models import ALLOWED_ENTITY_TYPES, DocumentNode, EntityNode
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

    async def find_fuzzy_entity_candidates(
        self,
        entity_type: str,
        name: str,
        exclude_id: str = "",
        threshold: float = 0.50,
        limit: int = 10,
    ) -> List[str]:
        cypher = (
            "MATCH (e:Entity) "
            "WHERE e.entity_type = $entity_type AND e.name IS NOT NULL AND e.id <> $exclude_id "
            "WITH e, apoc.text.sorensenDiceSimilarity(toLower(e.name), toLower($name)) AS sim "
            "WHERE sim >= $threshold "
            "RETURN e.id AS id "
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
        return [record.get("id") for record in records if record.get("id")]

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
        ON MATCH:  only mutable metadata (aliases, nodeset_ids, ticker) is updated.
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
            "  e.aliases = CASE WHEN $props.aliases IS NOT NULL AND size($props.aliases) > 0 "
            "               THEN $props.aliases ELSE e.aliases END, "
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
        cypher = (
            "MATCH (e:Entity)-[r]-(neighbor:Entity) "
            "WHERE e.id IN $entity_ids "
            "AND ($exclude_ids IS NULL OR NOT neighbor.id IN $exclude_ids) "
            "RETURN e.id AS source_entity_id, "
            "neighbor.id AS neighbor_entity_id, "
            "neighbor.name AS neighbor_name, "
            "neighbor.entity_type AS neighbor_type, "
            "r.relationship_type AS relationship_type"
        )
        records = await self._execute_read(
            cypher, {"entity_ids": entity_ids, "exclude_ids": exclude_ids}
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
        """Return chunks that mention the provided entities."""
        if not entity_ids:
            return []
        cypher = (
            "MATCH (c:Chunk)-[:MENTIONS_ENTITY]->(e:Entity) "
            "WHERE e.id IN $entity_ids "
            "AND ($exclude_chunk_ids IS NULL OR NOT c.id IN $exclude_chunk_ids) "
            "RETURN c.id AS chunk_id, c.text AS chunk_text, c.article_title AS article_title, c.source_url AS source_url, "
            "c.chunk_index AS chunk_index, c.document_id AS document_id, "
            "c.published_at AS published_at"
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

    async def update_targets_last_analysis_summary(
        self, user_email: str, target_entity_id: str, summary: str
    ) -> None:
        if not user_email or not target_entity_id:
            return
        cypher = (
            "MATCH (u)-[r:TARGETS]->(t:Entity {id: $target_id}) "
            "WHERE (u:UserInvestmentInterestNode OR u:UserLearningInterestNode) "
            "AND u.user_email = $user_email "
            "SET r.last_analysis_summary = $summary"
        )
        await self._execute_write(
            cypher,
            {
                "user_email": user_email,
                "target_id": target_entity_id,
                "summary": summary,
            },
        )

    async def close(self) -> None:
        """Close the underlying Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def run_traversal(self, cypher: str, params: dict) -> List[dict]:
        """Run a read-only traversal query."""
        return await self._execute_read(cypher, params)

    async def get_user_investment_interests(self, user_email: str) -> List[dict]:
        """Fetch user investment interests with targets."""
        targets = RelationshipType.TARGETS.value
        cypher = (
            "MATCH (u:UserInvestmentInterestNode {user_email: $user_email}) "
            f"OPTIONAL MATCH (u)-[:{targets}]->(t:Entity) "
            "RETURN u AS node, "
            "collect({id: t.id, name: t.name, entity_type: t.entity_type}) AS targets "
            "ORDER BY u.updated_at DESC"
        )
        records = await self._execute_read(cypher, {"user_email": user_email})
        return records

    async def get_user_learning_interests(self, user_email: str) -> List[dict]:
        """Fetch user learning interests with targets."""
        targets = RelationshipType.TARGETS.value
        cypher = (
            "MATCH (u:UserLearningInterestNode {user_email: $user_email}) "
            f"OPTIONAL MATCH (u)-[:{targets}]->(t:Entity) "
            "RETURN u AS node, "
            "collect({id: t.id, name: t.name, entity_type: t.entity_type}) AS targets "
            "ORDER BY u.updated_at DESC"
        )
        records = await self._execute_read(cypher, {"user_email": user_email})
        return records

    async def upsert_user_connected_nodes(self, node: Any, nodeset_id: str) -> None:
        """Upsert a user-connected node and its relationships."""
        label = node.__class__.__name__
        belongs_to = RelationshipType.BELONGS_TO_NODESET.value
        targets = RelationshipType.TARGETS.value

        cypher = (
            f"MERGE (u:{label} {{id: $id}}) "
            "SET u += $props "
            "WITH u "
            f"MERGE (s:NodeSet {{id: $nodeset_id}}) "
            f"MERGE (u)-[:{belongs_to}]->(s) "
            "WITH u "
            f"FOREACH (tid IN $target_ids | "
            f"  MERGE (t:Entity {{id: tid}}) "
            f"  MERGE (u)-[:{targets}]->(t)"
            ")"
        )
        props = {k: v for k, v in node.model_dump().items() if k != "id"}
        await self._execute_write(
            cypher,
            {
                "id": str(node.id),
                "props": props,
                "nodeset_id": nodeset_id,
                "target_ids": getattr(node, "target_entity_ids", []) or [],
            },
        )

    async def merge_user_interest_domain(self, domain_id: str, props: dict) -> None:
        """
        Merge a UserInterestDomain node.
        ON CREATE: full props written.
        ON MATCH:  only last_seen_at updated — category and domain_type are immutable.
        """
        cypher = (
            "MERGE (d:UserInterestDomain {id: $id}) "
            "ON CREATE SET d += $props "
            "ON MATCH SET d.last_seen_at = $now"
        )
        clean = {k: v for k, v in props.items() if k != "nodeset_id"}
        await self._execute_write(
            cypher,
            {
                "id": domain_id,
                "props": clean,
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
        """
        Merge a UserInterestEdge with operation-specific MATCH behaviour.

        operation="reinforce": increments weight, ensures Active status (unless
                            already invalidated by user — in that case status
                            is left alone so invalidation is not silently reversed).
        operation="invalidate": sets invalidated=True and status=Invalidated.
        """
        if operation == "reinforce":
            cypher = (
                "MERGE (e:UserInterestEdge {id: $id}) "
                "ON CREATE SET e += $props, e.weight = $weight_delta, "
                "              e.status = 'Active', e.invalidated = false "
                "ON MATCH SET  e.weight = e.weight + $weight_delta, "
                "              e.last_updated_at = $now, "
                "              e.status = CASE WHEN e.invalidated THEN e.status "
                "                              ELSE 'Active' END"
            )
        else:  # invalidate
            cypher = (
                "MERGE (e:UserInterestEdge {id: $id}) "
                "ON CREATE SET e += $props, e.weight = 0.0, "
                "              e.status = 'Invalidated', e.invalidated = true "
                "ON MATCH SET  e.status = 'Invalidated', e.invalidated = true, "
                "              e.last_updated_at = $now"
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
                "now": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def merge_turn_node(self, turn_id: str, props: dict) -> None:
        """
        Merge a TurnNode. Idempotent — the same turn may source multiple edges
        so this is called once per turn per signal, not per relationship.
        """
        cypher = "MERGE (t:TurnNode {id: $id}) " "ON CREATE SET t += $props"
        await self._execute_write(cypher, {"id": turn_id, "props": props})

    async def get_entity_category(self, entity_id: str) -> Optional[str]:
        """Return the category field of an entity (used for FinancialConcept category)."""
        cypher = "MATCH (e:Entity {id: $id}) " "RETURN e.category AS category LIMIT 1"
        records = await self._execute_read(cypher, {"id": entity_id})
        return records[0].get("category") if records else None

    async def get_user_interest_data(
        self, user_email: str, nodeset_id: str
    ) -> List[dict]:
        """
        Full 3-hop read: NodeSet → Domain → Edge → Entity + provenance turns.
        Returns all edges including invalidated ones; caller filters as needed.
        """
        cypher = (
            "MATCH (ns:NodeSet {id: $nodeset_id})"
            "<-[:BELONGS_TO_NODESET]-(d:UserInterestDomain {user_email: $user_email})"
            "-[:HAS_INTEREST_IN]->(e:UserInterestEdge)"
            "-[:TARGETS]->(entity:Entity) "
            "OPTIONAL MATCH (e)-[:SOURCED_FROM]->(t:TurnNode) "
            "OPTIONAL MATCH (e)-[:INVALIDATED_BY]->(it:TurnNode) "
            "RETURN d, e, entity, "
            "       collect(DISTINCT t)  AS source_turns, "
            "       collect(DISTINCT it) AS invalidating_turns "
            "ORDER BY d.category ASC, e.weight DESC"
        )
        return await self._execute_read(
            cypher,
            {
                "nodeset_id": nodeset_id,
                "user_email": user_email,
            },
        )
