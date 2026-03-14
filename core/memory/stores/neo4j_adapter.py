"""Async Neo4j adapter for graph I/O."""

from __future__ import annotations

from typing import Dict, List, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from core.logger import get_logger
from core.memory.graph.models import ChunkNode, DocumentNode, EntityNode

_ALLOWED_ENTITY_TYPES = {
    "Company",
    "Person",
    "MacroIndicator",
    "Event",
    "GeoPoliticalRegion",
    "Instrument",
}


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

    async def merge_document_node(self, node: DocumentNode) -> None:
        """Merge a document node and update its properties."""
        cypher = "MERGE (d:Document {id: $id}) SET d += $props"
        props = node.model_dump()
        await self._execute_write(cypher, {"id": node.id, "props": props})

    async def merge_chunk_node(self, node: ChunkNode) -> None:
        """Merge a chunk node and connect it to its document."""
        cypher = (
            "MERGE (d:Document {id: $doc_id}) "
            "MERGE (c:Chunk {id: $id}) "
            "SET c += $props "
            "MERGE (c)-[:BELONGS_TO_DOCUMENT]->(d)"
        )
        props = node.model_dump()
        await self._execute_write(
            cypher, {"id": node.id, "doc_id": node.document_id, "props": props}
        )

    async def merge_entity_node(self, node: EntityNode) -> None:
        """Merge an entity node with a dynamic label."""
        if node.entity_type not in _ALLOWED_ENTITY_TYPES:
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
        cypher = (
            "MATCH (c:Chunk)-[:MENTIONS_ENTITY]->(e:Entity) "
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
            "RETURN c.id AS chunk_id, c.text AS chunk_text, "
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
