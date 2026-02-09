# core/memory/graphiti_memory.py
"""
Financial Knowledge Memory Module using Graphiti.

This module provides a dual-namespace memory system:
- GLOBAL namespace: Shared financial knowledge (companies, ETFs, news, events, concepts)
- User namespaces: Personal preferences, portfolios, watchlists, and references to global entities

The module supports:
- Adding episodes to global or user-specific namespaces
- Searching across both namespaces with configurable scope
- Creating lightweight references from user entities to global entities
- CRUD operations on entities and relationships
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Union

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.search.search_config import SearchConfig, SearchResults
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_RRF,
)

from core.memory.entities import (
    ALL_ENTITY_TYPES,
    GLOBAL_ENTITY_TYPES,
    USER_ENTITY_TYPES,
    EntityReference,
)
from core.memory.extraction_prompts import (
    GLOBAL_EXTRACTION_PROMPT,
    get_user_extraction_prompt,
)
from core.memory.relationships import (
    ALL_EDGE_TYPES,
    GLOBAL_EDGE_TYPE_MAP,
    GLOBAL_EDGE_TYPES,
    USER_EDGE_TYPE_MAP,
    USER_EDGE_TYPES,
)

logger = logging.getLogger(__name__)




class FinancialKnowledgeMemory:
    """
    A dual-namespace memory module for financial knowledge using Graphiti.

    Provides isolated knowledge graphs for each user while maintaining a
    shared global knowledge base for common financial information.

    Attributes:
        GLOBAL_NAMESPACE: The group_id for the shared global knowledge base.
        USER_NAMESPACE_PREFIX: Prefix for user-specific namespaces.
    """

    GLOBAL_NAMESPACE = "GLOBAL"
    USER_NAMESPACE_PREFIX = "user_"

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        api_key: str,
        llm_model: str = "gemini-2.5-flash-lite",
        embedding_model: str = "gemini-embedding-001",
        embedding_dim: int = 3072,
        max_coroutines: int = 5,
    ):
        """
        Initialize the Financial Knowledge Memory module.

        Args:
            neo4j_uri: Neo4j database URI (e.g., "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            api_key: Google API key for Gemini models
            llm_model: LLM model name for text generation
            embedding_model: Embedding model name
            embedding_dim: Embedding dimension
            max_coroutines: Maximum concurrent Graphiti operations
        """
        self._api_key = api_key
        self._llm_model = llm_model

        # Initialize LLM client
        self._llm_client = GeminiClient(
            config=LLMConfig(api_key=api_key, model=llm_model)
        )

        # Initialize embedder
        self._embedder = GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=api_key,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
            )
        )

        # Initialize cross-encoder for reranking
        self._cross_encoder = GeminiRerankerClient(
            config=LLMConfig(api_key=api_key, model=llm_model)
        )

        # Initialize Graphiti client
        self._graphiti = Graphiti(
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            llm_client=self._llm_client,
            embedder=self._embedder,
            cross_encoder=self._cross_encoder,
            max_coroutines=max_coroutines,
        )

        logger.info("FinancialKnowledgeMemory initialized successfully")

    @staticmethod
    def get_user_namespace(user_id: str) -> str:
        """Generate the namespace (group_id) for a specific user."""
        return f"{FinancialKnowledgeMemory.USER_NAMESPACE_PREFIX}{user_id}"

    # ==========================================================================
    # EPISODE METHODS
    # ==========================================================================

    async def add_global_episode(
        self,
        name: str,
        episode_body: Union[str, dict[str, Any]],
        source: EpisodeType = EpisodeType.text,
        source_description: str = "Financial knowledge",
        reference_time: Optional[datetime] = None,
    ) -> AddEpisodeResults:
        """
        Add an episode to the GLOBAL namespace.

        Use this for general financial knowledge that should be shared across
        all users (e.g., company information, market events, financial concepts).

        Args:
            name: Unique name for the episode
            episode_body: Content of the episode (text or dict for JSON)
            source: Type of episode (text, message, json)
            source_description: Description of the content source
            reference_time: Timestamp for the episode (defaults to now)

        Returns:
            AddEpisodeResults containing extracted nodes and edges
        """
        logger.info(f"Adding global episode: {name}")
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        # Convert dict to JSON string if needed
        body = episode_body if isinstance(episode_body, str) else json.dumps(episode_body)

        result = await self._graphiti.add_episode(
            name=name,
            episode_body=body,
            source=source,
            source_description=source_description,
            reference_time=reference_time,
            group_id=self.GLOBAL_NAMESPACE,
            entity_types=GLOBAL_ENTITY_TYPES,
            edge_types=GLOBAL_EDGE_TYPES,
            edge_type_map=GLOBAL_EDGE_TYPE_MAP,
            excluded_entity_types=["Entity"],
            custom_extraction_instructions=GLOBAL_EXTRACTION_PROMPT,
        )

        logger.info(f"Added global episode: {name} with {len(result.nodes)} nodes")
        return result

    async def add_user_episode(
        self,
        user_id: str,
        name: str,
        episode_body: Union[str, dict[str, Any]],
        global_nodes: Optional[list[EntityNode]] = None,
        source: EpisodeType = EpisodeType.text,
        source_description: str = "User interaction",
        reference_time: Optional[datetime] = None,
    ) -> AddEpisodeResults:
        """
        Add an episode to a user-specific namespace.

        Use this for personal user data (portfolios, preferences, interactions).

        Args:
            user_id: Unique user identifier
            name: Unique name for the episode
            episode_body: Content of the episode (text or dict for JSON)
            global_nodes: List of global entities to avoid duplicating (creates edges instead)
            source: Type of episode (text, message, json)
            source_description: Description of the content source
            reference_time: Timestamp for the episode (defaults to now)

        Returns:
            AddEpisodeResults containing extracted nodes and edges
        """
        logger.info(f"Adding user episode for {user_id}: {name}")
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        namespace = self.get_user_namespace(user_id)
        body = episode_body if isinstance(episode_body, str) else json.dumps(episode_body)

        # Build extraction prompt with global entity names to avoid duplication
        global_entity_names = [node.name for node in (global_nodes or [])]
        extraction_instructions = get_user_extraction_prompt(global_entity_names)

        result = await self._graphiti.add_episode(
            name=name,
            episode_body=body,
            source=source,
            source_description=source_description,
            reference_time=reference_time,
            group_id=namespace,
            entity_types=USER_ENTITY_TYPES,
            edge_types=USER_EDGE_TYPES,
            edge_type_map=USER_EDGE_TYPE_MAP,
            excluded_entity_types=["Entity"],
            custom_extraction_instructions=extraction_instructions,
        )

        logger.info(f"Added user episode for {user_id}: {name} with {len(result.nodes)} nodes")
        return result

    async def add_episode(
        self,
        user_id: str,
        name: str,
        episode_body: Union[str, dict[str, Any]],
        source: EpisodeType = EpisodeType.text,
        source_description: str = "Mixed content",
        reference_time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Unified method to add an episode to both global and user namespaces.

        This method implements a dual-namespace pipeline:
        1. Adds episode to GLOBAL namespace first, extracting shared entities
        2. Gets extracted global entities from results
        3. Adds episode to user namespace with global entity context
        4. Creates cross-namespace edges linking user nodes to global entities
        5. Cleans up redundant reference nodes in user namespace

        Args:
            user_id: Unique user identifier
            name: Unique name for the episode
            episode_body: Content of the episode
            source: Type of episode
            source_description: Description of the content source
            reference_time: Timestamp for the episode

        Returns:
            Dict containing global_result, user_result, and shared_episode_uuid
        """
        logger.info(f"Adding episode for {user_id}: {name}")
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        body = episode_body if isinstance(episode_body, str) else json.dumps(episode_body)

        # Generate shared UUID for episode correlation (unused for now, but kept for reference)
        shared_episode_uuid = str(uuid.uuid4())

        # Step 1: Process global namespace first
        global_result = await self.add_global_episode(
            name=f"{name}_global",
            episode_body=body,
            source=source,
            source_description=source_description,
            reference_time=reference_time,
        )

        # Step 2: Extract global nodes for user namespace context
        global_nodes = global_result.nodes

        # Step 3: Process user namespace with global context
        user_result = await self.add_user_episode(
            user_id=user_id,
            name=f"{name}_user",
            episode_body=body,
            global_nodes=global_nodes,
            source=source,
            source_description=source_description,
            reference_time=reference_time,
        )

        # Step 4: Create cross-namespace edges linking user nodes to global entities
        await self._link_user_edges_to_global(
            user_id=user_id,
            user_edges=user_result.edges,
            global_nodes=global_nodes,
        )

        # Step 5: Clean up reference nodes that duplicate global entities
        await self._cleanup_reference_nodes(
            user_id=user_id,
            user_nodes=user_result.nodes,
            global_nodes=global_nodes,
        )

        logger.info(
            f"Added episode to both namespaces: {name} "
            f"(global: {len(global_nodes)} nodes, user: {len(user_result.nodes)} nodes)"
        )

        return {
            "global_result": global_result,
            "user_result": user_result,
            "shared_episode_uuid": shared_episode_uuid,
        }

    async def _link_user_edges_to_global(
        self,
        user_id: str,
        user_edges: list[EntityEdge],
        global_nodes: list[EntityNode],
    ) -> None:
        """
        Create cross-namespace edges linking user nodes to global entities.

        For each user edge whose target matches a global entity by name,
        creates a new edge pointing from the user source node to the global target.

        Args:
            user_id: Unique user identifier
            user_edges: Edges from the user namespace
            global_nodes: Entities extracted to the global namespace
        """
        logger.info(f"Linking user edges to global entities for {user_id}")
        namespace = self.get_user_namespace(user_id)

        # Build global node lookup by name (case-insensitive)
        global_node_map = {node.name.lower(): node for node in global_nodes}

        for edge in user_edges:
            # Get target node to check if it matches a global entity
            try:
                target_node = await self.get_entity_by_uuid(edge.target_node_uuid)
            except Exception:
                logger.warning(
                    f"Failed to get target node {edge.target_node_uuid}: {e}"
                )
                continue

            if target_node and target_node.name.lower() in global_node_map:
                global_node = global_node_map[target_node.name.lower()]

                # Get source node (should be in user namespace)
                source_node = await self.get_entity_by_uuid(edge.source_node_uuid)
                if not source_node:
                    continue

                # Create cross-namespace edge
                cross_edge = EntityEdge(
                    uuid=str(uuid.uuid4()),
                    group_id=namespace,
                    source_node_uuid=source_node.uuid,
                    target_node_uuid=global_node.uuid,  # Link to global node
                    created_at=datetime.now(timezone.utc),
                    name=edge.name,
                    fact=edge.fact,
                )

                try:
                    await self._graphiti.add_triplet(source_node, cross_edge, global_node)
                    logger.info(
                        f"Created cross-namespace edge: {source_node.name} -> {global_node.name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create cross-namespace edge {source_node.name} -> {global_node.name}: {e}"
                    )
            else:
                logger.warning(
                    f"Target node {edge.target_node_uuid} not found in global namespace"
                )

    async def _cleanup_reference_nodes(
        self,
        user_id: str,
        user_nodes: list[EntityNode],
        global_nodes: list[EntityNode],
    ) -> None:
        """
        Delete redundant reference nodes that duplicate global entities.

        If a user node has the same name as a global entity, it's considered
        a reference node that should be removed (edges now point directly to global).

        Args:
            user_id: Unique user identifier
            user_nodes: Nodes extracted to the user namespace
            global_nodes: Nodes extracted to the global namespace
        """
        logger.info(f"Cleaning up reference nodes for {user_id}")
        namespace = self.get_user_namespace(user_id)

        # Build global node name set (case-insensitive)
        global_node_names = {node.name.lower() for node in global_nodes}

        for node in user_nodes:
            # Check if this user node duplicates a global entity
            if node.name.lower() in global_node_names and node.group_id == namespace:
                # Delete the redundant reference node
                try:
                    await node.delete(self._graphiti.driver)
                    logger.info(f"Deleted redundant reference node: {node.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete reference node {node.name}: {e}")

    # ==========================================================================
    # SEARCH METHODS
    # ==========================================================================

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        include_global: bool = True,
        include_user: bool = True,
        limit: int = 10,
        center_node_uuid: Optional[str] = None,
    ) -> list:
        """
        Search the knowledge graph across namespaces.

        Args:
            query: Search query string
            user_id: User ID for user-specific search (required if include_user=True)
            include_global: Whether to search the GLOBAL namespace
            include_user: Whether to search the user's namespace
            limit: Maximum number of results to return
            center_node_uuid: Optional node UUID for distance-based reranking

        Returns:
            List of matching edges (facts)
        """
        # Build list of group_ids to search
        group_ids: list[str] = []
        if include_global:
            group_ids.append(self.GLOBAL_NAMESPACE)
        if include_user and user_id:
            group_ids.append(self.get_user_namespace(user_id))

        if not group_ids:
            raise ValueError("At least one namespace must be included in search")

        # Perform search with multiple group_ids
        # Note: Graphiti's search method supports searching multiple group_ids
        results = await self._graphiti.search(
            query=query,
            group_ids=group_ids,
            center_node_uuid=center_node_uuid,
            num_results=limit,
        )

        logger.info(
            f"Search for '{query}' returned {len(results) if results else 0} edges"
        )
        return results

    async def search_nodes(
        self,
        query: str,
        user_id: Optional[str] = None,
        include_global: bool = True,
        include_user: bool = True,
        limit: int = 10,
    ) -> SearchResults:
        """
        Search for nodes (entities) in the knowledge graph.

        Args:
            query: Search query string
            user_id: User ID for user-specific search
            include_global: Whether to search the GLOBAL namespace
            include_user: Whether to search the user's namespace
            limit: Maximum number of results

        Returns:
            SearchResults containing matching nodes
        """
        group_ids: list[str] = []
        if include_global:
            group_ids.append(self.GLOBAL_NAMESPACE)
        if include_user and user_id:
            group_ids.append(self.get_user_namespace(user_id))

        if not group_ids:
            raise ValueError("At least one namespace must be included in search")

        # Use node-specific search config
        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit

        results = await self._graphiti._search(
            query=query,
            group_ids=group_ids,
            config=config,
        )

        logger.info(f"Node search for '{query}' returned {len(results.nodes)} nodes")
        return results

    async def search_edges(
        self,
        query: str,
        user_id: Optional[str] = None,
        include_global: bool = True,
        include_user: bool = True,
        limit: int = 10,
    ) -> SearchResults:
        """
        Search for edges (relationships) in the knowledge graph.

        Args:
            query: Search query string
            user_id: User ID for user-specific search
            include_global: Whether to search the GLOBAL namespace
            include_user: Whether to search the user's namespace
            limit: Maximum number of results

        Returns:
            SearchResults containing matching edges
        """
        group_ids: list[str] = []
        if include_global:
            group_ids.append(self.GLOBAL_NAMESPACE)
        if include_user and user_id:
            group_ids.append(self.get_user_namespace(user_id))

        if not group_ids:
            raise ValueError("At least one namespace must be included in search")

        # Use edge-specific search config
        config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit

        results = await self._graphiti._search(
            query=query,
            group_ids=group_ids,
            config=config,
        )

        logger.info(f"Edge search for '{query}' returned {len(results.edges)} edges")
        return results

    async def get_user_global_links(
        self,
        user_id: str,
        global_entity_uuid: Optional[str] = None,
    ) -> list[EntityEdge]:
        """
        Get all cross-namespace edges from user entities to global entities.

        Args:
            user_id: Unique user identifier
            global_entity_uuid: Optional filter by referenced global entity

        Returns:
            List of EntityEdge linking user entities to global entities
        """
        namespace = self.get_user_namespace(user_id)

        # Query for edges from user namespace to global namespace
        query = """
        MATCH (u:Entity)-[e]->(g:Entity {group_id: $global_namespace})
        WHERE e.group_id = $user_namespace
        """
        params = {
            "user_namespace": namespace,
            "global_namespace": self.GLOBAL_NAMESPACE,
        }

        if global_entity_uuid:
            query += " AND g.uuid = $global_uuid"
            params["global_uuid"] = global_entity_uuid

        query += " RETURN e, u.uuid AS source_uuid, g.uuid AS target_uuid"

        records, _, _ = await self._graphiti.driver.execute_query(query, **params)

        edges = []
        for record in records:
            edge_data = record["e"]
            edges.append(
                EntityEdge(
                    uuid=edge_data.get("uuid", str(uuid.uuid4())),
                    group_id=namespace,
                    source_node_uuid=record["source_uuid"],
                    target_node_uuid=record["target_uuid"],
                    created_at=edge_data.get("created_at"),
                    name=edge_data.get("name", "REFERENCES"),
                    fact=edge_data.get("fact", ""),
                )
            )

        return edges

    # ==========================================================================
    # CRUD HELPERS
    # ==========================================================================

    async def get_entity_by_uuid(
        self,
        entity_uuid: str,
        user_id: Optional[str] = None,
    ) -> Optional[EntityNode]:
        """
        Get an entity by its UUID.

        Args:
            entity_uuid: UUID of the entity
            user_id: Optional user ID (not used for lookup, but for logging)

        Returns:
            EntityNode if found, None otherwise
        """
        try:
            entity = await EntityNode.get_by_uuid(self._graphiti.driver, entity_uuid)
            return entity
        except Exception as e:
            logger.warning(f"Failed to get entity {entity_uuid}: {e}")
            return None

    async def delete_user_data(self, user_id: str) -> int:
        """
        Delete all data in a user's namespace.

        WARNING: This is a destructive operation that cannot be undone.

        Args:
            user_id: Unique user identifier

        Returns:
            Number of nodes deleted
        """
        namespace = self.get_user_namespace(user_id)

        # Delete all nodes and relationships in the user's namespace
        query = """
        MATCH (n {group_id: $namespace})
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """

        records, _, _ = await self._graphiti.driver.execute_query(
            query, namespace=namespace
        )

        deleted_count = records[0]["deleted_count"] if records else 0
        logger.warning(f"Deleted {deleted_count} nodes for user {user_id}")

        return deleted_count

    # ==========================================================================
    # LIFECYCLE
    # ==========================================================================

    async def build_indices(self) -> None:
        """Build database indices and constraints for optimal performance."""
        await self._graphiti.build_indices_and_constraints()
        logger.info("Built database indices and constraints")

    async def close(self) -> None:
        """Close the connection to Neo4j and release resources."""
        await self._graphiti.close()
        logger.info("FinancialKnowledgeMemory connection closed")

    @property
    def graphiti(self) -> Graphiti:
        """Access the underlying Graphiti instance for advanced operations."""
        return self._graphiti
