"""Async Neo4j adapter for graph I/O."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from core.logger import get_logger
from core.memory.graph.models import (
    ALLOWED_ENTITY_TYPES,
    ChunkNode,
    DocumentNode,
    EntityNode,
)


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

    async def merge_chunk_node(self, node: ChunkNode) -> None:
        """Merge a chunk node and connect it to its document."""
        belongs_to = RelationshipType.BELONGS_TO_DOCUMENT.value
        cypher = (
            "MERGE (d:Document {id: $doc_id}) "
            "MERGE (c:Chunk {id: $id}) "
            "SET c += $props "
            f"MERGE (c)-[:{belongs_to}]->(d)"
        )
        props = {k: v for k, v in node.model_dump().items() if k != "id"}
        await self._execute_write(
            cypher,
            {
                "id": node.id,
                "doc_id": node.document_id,
                "props": props,
            },
        )

    async def merge_entity_node(self, node: EntityNode) -> None:
        """Merge an entity node with a dynamic label."""
        if node.entity_type not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"Unsupported entity_type: {node.entity_type}")

        cypher = (
            "MERGE (e:Entity {id: $id}) " "SET e += $props " f"SET e:{node.entity_type}"
        )
        props = node.model_dump()
        await self._execute_write(cypher, {"id": node.id, "props": props})

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
