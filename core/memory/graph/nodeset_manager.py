"""NodeSet manager for deterministic NodeSet creation and assignment."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Dict, Optional

from core.logger import get_logger
from core.memory.graph.models import (
    ALL_MAIN_SECTORS,
    FINANCIAL_CONCEPT_CATEGORIES,
    GLOBAL_ENTITY_NODESETS,
    EntityNode,
)
from core.memory.graph.utils import canonical_entity_id, canonical_nodeset_id
from core.memory.stores.neo4j_adapter import Neo4jAdapter


def hash_user_email(email: str) -> str:
    """Deterministically hash a user email to a stable short identifier."""
    if not email or not isinstance(email, str):
        raise ValueError(f"Invalid email for hashing: {email!r}")
    normalized = email.strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:16]


def get_user_nodeset_name(user_email: str) -> str:
    """Return the canonical NodeSet name for a given user email."""
    return f"USER_{hash_user_email(user_email)}"


# Canonical description for the global Market anchor entity.
_MARKET_DESCRIPTION = (
    "Global equity market — top-level anchor node for the sector taxonomy. "
    "All sectors belong to this node."
)


class NodeSetManager:
    """Manages NodeSet creation, lookup, assignment, and canonical entity taxonomy bootstrap."""

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        entity_chroma_adapter=None,  # Optional ChromaDBAdapter for entity embeddings
    ) -> None:
        """
        Initialize the manager.

        Args:
            neo4j_adapter: Adapter for all Neo4j operations.
            entity_chroma_adapter: Optional Chroma adapter for the entity
                embeddings collection. When provided, Market and Sector entity
                nodes are embedded at bootstrap so they participate in semantic
                retrieval. Pass None to skip embedding (e.g. in unit tests).
        """
        self._neo4j_adapter = neo4j_adapter
        self._entity_chroma_adapter = entity_chroma_adapter
        self._registry: Dict[str, str] = {}
        self._initialized = False
        self._logger = get_logger(__name__)

    # ── Registry bootstrap ────────────────────────────────────────────────────

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

    # ── NodeSet CRUD ──────────────────────────────────────────────────────────

    async def get_or_create(self, name: str, description: str = "") -> str:
        """Get an existing NodeSet ID or create it deterministically."""
        await self._ensure_initialized()
        nodeset_id = canonical_nodeset_id(name)
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

    async def get_global_financial_wisdom_id(self) -> str:
        """Get or create the Global Financial Wisdom NodeSet ID."""
        description = GLOBAL_ENTITY_NODESETS.get(
            "Global Financial Wisdom",
            "Global anchor NodeSet for FinancialConcept taxonomy.",
        )
        return await self.get_or_create("Global Financial Wisdom", description)

    async def get_or_create_user_nodeset(self, user_email: str) -> tuple[str, str]:
        """Get or create the private NodeSet for a user."""
        nodeset_name = get_user_nodeset_name(user_email)
        nodeset_id = await self.get_or_create(nodeset_name)
        return nodeset_name, nodeset_id

    # ── Canonical taxonomy bootstrap ─────────────────────────────────────────

    async def _upsert_entity_with_embedding(self, node: EntityNode) -> None:
        """
        Upsert a single entity node to Neo4j and, if the Chroma adapter is
        available, embed it into the entity vector store.
        """
        await self._neo4j_adapter.merge_entity_node(node)
        if self._entity_chroma_adapter is not None:
            await self._entity_chroma_adapter.upsert_entity_embedding(
                entity_id=node.id,
                name=node.name,
                description=node.description,
                entity_type=node.entity_type,
            )

    async def _bootstrap_market_entity(self) -> str:
        """
        Ensure the global Market entity node exists in both stores.
        Returns the canonical Market entity ID.
        """
        market_id = canonical_entity_id("Market", "Market")
        market_exists = await self._neo4j_adapter.entity_exists(market_id)
        if not market_exists:
            market_node = EntityNode(
                id=market_id,
                name="Market",
                entity_type="Market",
                description=_MARKET_DESCRIPTION,
            )
            await self._upsert_entity_with_embedding(market_node)
            self._logger.info("Bootstrapped Market entity node.")
        return market_id

    async def _bootstrap_sector_entities(self, market_id: str) -> None:
        """
        Ensure all canonical Sector entity nodes exist in both stores and
        have a BELONGS_TO edge pointing to the Market entity.

        Runs all existence checks concurrently, then creates missing sectors.
        """
        # Collect which sectors are missing — batch the existence checks.
        sector_ids = {
            name: canonical_entity_id(name, "Sector") for name in ALL_MAIN_SECTORS
        }

        existence_results = await asyncio.gather(
            *[self._neo4j_adapter.entity_exists(sid) for sid in sector_ids.values()],
            return_exceptions=True,
        )

        missing_sectors = [
            name
            for (name, _sid), exists in zip(sector_ids.items(), existence_results)
            if not isinstance(exists, Exception) and not exists
        ]

        if not missing_sectors:
            return

        self._logger.info(
            "Bootstrapping %d missing Sector entity nodes: %s",
            len(missing_sectors),
            missing_sectors,
        )

        async def _create_sector(name: str) -> None:
            sector_id = sector_ids[name]
            description = ALL_MAIN_SECTORS[name]
            sector_node = EntityNode(
                id=sector_id,
                name=name,
                entity_type="Sector",
                description=description,
            )
            await self._upsert_entity_with_embedding(sector_node)
            # Sector → Market taxonomy edge
            await self._neo4j_adapter.merge_relationship(
                sector_id,
                market_id,
                "BELONGS_TO",
                {
                    "relationship_type": "BELONGS_TO",
                    "source_agent": "taxonomy_bootstrap",
                },
            )
            self._logger.debug("Bootstrapped Sector entity: %s", name)

        await asyncio.gather(*[_create_sector(name) for name in missing_sectors])
        self._logger.info("Sector entity bootstrap complete.")

    async def _bootstrap_financial_concept_categories(
        self, wisdom_nodeset_id: str
    ) -> None:
        """
        Ensure all FinancialConceptCategory entity nodes exist and are linked to
        the Global Financial Wisdom NodeSet.
        """
        category_ids = {
            name: canonical_entity_id(name, "FinancialConceptCategory")
            for name in FINANCIAL_CONCEPT_CATEGORIES
        }

        existence_results = await asyncio.gather(
            *[self._neo4j_adapter.entity_exists(cid) for cid in category_ids.values()],
            return_exceptions=True,
        )

        missing_categories = [
            name
            for (name, _cid), exists in zip(category_ids.items(), existence_results)
            if not isinstance(exists, Exception) and not exists
        ]

        if not missing_categories:
            return

        self._logger.info(
            "Bootstrapping %d FinancialConceptCategory nodes: %s",
            len(missing_categories),
            missing_categories,
        )

        async def _create_category(name: str) -> None:
            category_id = category_ids[name]
            description = FINANCIAL_CONCEPT_CATEGORIES[name]
            category_node = EntityNode(
                id=category_id,
                name=name,
                entity_type="FinancialConceptCategory",
                description=description,
                nodeset_ids=[wisdom_nodeset_id],
            )
            await self._upsert_entity_with_embedding(category_node)
            await self.assign_to_node(category_id, "Entity", wisdom_nodeset_id)
            self._logger.debug("Bootstrapped FinancialConceptCategory: %s", name)

        await asyncio.gather(*[_create_category(name) for name in missing_categories])
        self._logger.info("FinancialConceptCategory bootstrap complete.")

    # ── Public initializer (called at application startup) ───────────────────

    async def initialize_default_nodesets(self) -> None:
        """
        Idempotent startup routine that:
          1. Creates global anchor NodeSets (existing behaviour).
          2. Creates per-sector NodeSets (existing behaviour).
          3. Bootstraps FinancialConceptCategory nodes and links them to
             Global Financial Wisdom via BELONGS_TO_NODESET.
          4. Bootstraps the Market entity node in Neo4j + Chroma.
          5. Bootstraps all 11 Sector entity nodes in Neo4j + Chroma,
             each linked to Market via a BELONGS_TO edge.

        Safe to call multiple times all operations are idempotent.
        """
        self._logger.info(
            "Initializing default nodesets and canonical entity taxonomy..."
        )

        # ── 1. Global anchor nodesets ─────────────────────────────────────────
        for name, description in GLOBAL_ENTITY_NODESETS.items():
            await self.get_or_create(name, description)

        wisdom_nodeset_id = await self.get_global_financial_wisdom_id()
        await self.get_global_financial_events_id()

        # ── 2. Sector nodesets (membership containers, separate from entities) ─
        for sector_name, description in ALL_MAIN_SECTORS.items():
            await self.get_or_create(sector_name, description)

        await self._bootstrap_financial_concept_categories(wisdom_nodeset_id)

        # ── 3 & 4. Market + Sector entity nodes (taxonomy graph) ──────────────
        market_id = await self._bootstrap_market_entity()
        await self._bootstrap_sector_entities(market_id)

        self._logger.info("Default nodeset initialization complete.")
