import logging
from typing import Any, Dict, List

from .graph_rag import GraphStoreManager
from .ontology import NodeLabel, QueryPlan
from .vector_rag import VectorStoreManager

logger = logging.getLogger(__name__)


class HybridRAGManager:
    def __init__(
        self, graph_manager: GraphStoreManager, vector_manager: VectorStoreManager
    ):
        self.graph_manager = graph_manager
        self.vector_manager = vector_manager

    async def retrieve_personalized_context(
        self, user_id: str, query: str, target_entities: List[str]
    ) -> Dict[str, Any]:
        """
        The Pincer Strategy:
        1. Graph: Get user's profile and relation to target entities.
        2. Vector: Retrieve news, hard-filtered by the assets found in the graph.
        """
        # Arm A: Structural Personalization
        user_plan = QueryPlan(
            start_node_label=NodeLabel.USER,
            start_node_identifier=user_id,
            traversal_path=[
                {"relationship_hint": "holdings", "target_label": NodeLabel.ASSET}
            ],
        )
        user_assets = await self.graph_manager.execute_query_plan(user_plan)
        asset_tickers = [a.get("ticker") for a in user_assets]

        # Arm B: Semantic Narrative
        # Combine user's assets + query entities for a broad filter
        combined_filters = list(set(asset_tickers + target_entities))

        vector_docs = self.vector_manager.retrieve(
            query=query,
            filter_dict={"ticker": combined_filters} if combined_filters else None,
            k=5,
        )

        return {
            "user_context": user_assets,
            "market_narrative": [d.page_content for d in vector_docs],
            "metadata": [d.metadata for d in vector_docs],
        }

    async def ingest_knowledge_item(self, text: str, metadata: Dict[str, Any]):
        """
        Hybrid Ingestion:
        1. Extract entities (simulated here, usually via an LLM or SpaCy).
        2. Update Neo4j skeleton.
        3. Store in Chroma with links.
        """
        # This would typically call an LLM to extract entities from 'text'
        # For this example, we assume metadata contains extracted tickers
        tickers = metadata.get("tickers", [])

        # 1. Update Graph
        for ticker in tickers:
            cypher = """
            MERGE (a:Asset {ticker: $ticker})
            MERGE (d:Document {url: $url})
            MERGE (d)-[:MENTIONS]->(a)
            SET d.timestamp = timestamp()
            """
            self.graph_manager.graph.query(
                cypher, {"ticker": ticker, "url": metadata.get("url")}
            )

        # 2. Update Vector Store
        await self.vector_manager.ingest_article(text, metadata)

    def update_user_state(
        self, user_id: str, asset_ticker: str, relation: str = "WATCHING"
    ):
        """Updates the user's personal graph nodes."""
        try:
            query = f"""
            MERGE (u:User {{user_id: $uid}})
            MERGE (a:Asset {{ticker: $ticker}})
            MERGE (u)-[r:{relation}]->(a)
            """
            self.graph_manager.graph.query(
                query, {"uid": user_id, "ticker": asset_ticker}
            )
        except Exception as e:
            logger.error(f"Failed to update user state: {e}")
