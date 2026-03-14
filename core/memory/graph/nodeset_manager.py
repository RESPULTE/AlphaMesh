"""NodeSet manager for deterministic NodeSet creation and assignment."""

from __future__ import annotations

import uuid
from typing import Dict, Optional

from core.logger import get_logger
from core.memory.graph.models import (
    ALL_MAIN_SECTORS,
    ENTITY_NAMESPACE,
    GLOBAL_ENTITY_NODESETS,
)
from core.memory.stores.neo4j_adapter import Neo4jAdapter


class NodeSetManager:
    """Manages NodeSet creation, lookup, and assignment."""

    def __init__(self, neo4j_adapter: Neo4jAdapter) -> None:
        """Initialize the manager with a Neo4j adapter."""
        self._neo4j_adapter = neo4j_adapter
        self._registry: Dict[str, str] = {}
        self._initialized = False
        self._logger = get_logger(__name__)

    async def _ensure_initialized(self) -> None:
        """Synchronize in-memory registry with existing NodeSets."""
        if self._initialized:
            return

        cypher = "MATCH (n:NodeSet) RETURN n.id AS id, n.name AS name"
        try:
            records = await self._neo4j_adapter._execute_read(cypher, {})
            if hasattr(records, "data"):
                records = await records.data()
            for record in records:
                if record.get("name") and record.get("id"):
                    self._registry[record["name"]] = record["id"]
            self._initialized = True
        except Exception:
            self._logger.exception("Failed to initialize NodeSet registry.")
            raise

    async def get_or_create(self, name: str, description: str = "") -> str:
        """Get an existing NodeSet ID or create it deterministically."""
        await self._ensure_initialized()
        nodeset_id = str(uuid.uuid5(ENTITY_NAMESPACE, name))
        await self._neo4j_adapter.merge_nodeset_node(nodeset_id, name, description)
        self._registry[name] = nodeset_id
        return nodeset_id

    async def get_id(self, name: str) -> Optional[str]:
        """Return a NodeSet ID from the registry if present."""
        await self._ensure_initialized()
        return self._registry.get(name)

    async def assign_to_node(
        self, node_id: str, node_label: str, nodeset_id: str
    ) -> None:
        """Assign a NodeSet to a graph node via BELONGS_TO_NODESET edge."""
        await self._ensure_initialized()
        cypher = (
            f"MATCH (n:{node_label} {{id: $node_id}}) "
            "MERGE (s:NodeSet {id: $nodeset_id}) "
            "MERGE (n)-[:BELONGS_TO_NODESET]->(s)"
        )
        try:
            await self._neo4j_adapter._execute_write(
                cypher, {"node_id": node_id, "nodeset_id": nodeset_id}
            )
        except Exception:
            self._logger.exception("Failed to assign NodeSet to node.")
            raise

    def assign_to_chunk_metadata(self, chunk_metadata: dict, nodeset_id: str) -> dict:
        """Append the NodeSet ID to a chunk metadata dict."""
        existing = chunk_metadata.get("nodeset_ids")
        if existing is None:
            chunk_metadata["nodeset_ids"] = [nodeset_id]
        elif isinstance(existing, list):
            if nodeset_id not in existing:
                existing.append(nodeset_id)
        else:
            chunk_metadata["nodeset_ids"] = [nodeset_id]
        return chunk_metadata

    async def get_global_financial_events_id(self) -> str:
        """Get or create the GlobalFinancialEvents NodeSet ID."""
        description = "Global anchor NodeSet for financial news ingestion."
        return await self.get_or_create("GlobalFinancialEvents", description)

    async def initialize_default_nodesets(self) -> None:
        """
        Batch initialize the global nodesets and all main sector nodesets.
        This adapts the archived Cognee logic to the current Neo4j adapter cleanly.
        """
        self._logger.info("Initializing default global and sector nodesets...")

        # 1. Global anchor nodesets
        for name, description in GLOBAL_ENTITY_NODESETS.items():
            await self.get_or_create(name, description)

        # Ensure the specific financial events node exists for backward compatibility
        await self.get_global_financial_events_id()

        # 2. The 12 primary sectors
        for sector_name, description in ALL_MAIN_SECTORS.items():
            await self.get_or_create(sector_name, description)

        self._logger.info("Default nodeset initialization complete.")
