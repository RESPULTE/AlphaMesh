import logging
from typing import Dict, List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph

from .ontology import RELATIONSHIP_DESCRIPTIONS, NodeLabel, QueryPlan

logger = logging.getLogger(__name__)


class GraphStoreManager:
    def __init__(self, graph: Neo4jGraph, embeddings: Embeddings):
        self.graph = graph
        self.embeddings = embeddings
        self._rel_vectors = None

    async def _get_rel_vectors(self):
        """Lazy load embeddings for canonical relationships for semantic matching."""
        if self._rel_vectors is None:
            tasks = [
                self.embeddings.aembed_query(desc)
                for desc in RELATIONSHIP_DESCRIPTIONS.values()
            ]
            vectors = await asyncio.gather(*tasks)
            self._rel_vectors = dict(zip(RELATIONSHIP_DESCRIPTIONS.keys(), vectors))
        return self._rel_vectors

    async def map_relationship(self, hint: str) -> str:
        """Dynamically maps a natural language hint to a canonical Neo4j Relationship."""
        hint_vec = await self.embeddings.aembed_query(hint)
        rel_vectors = await self._get_rel_vectors()

        best_rel = None
        max_sim = -1.0

        for rel_type, rel_vec in rel_vectors.items():
            sim = np.dot(hint_vec, rel_vec) / (
                np.linalg.norm(hint_vec) * np.linalg.norm(rel_vec)
            )
            if sim > max_sim:
                max_sim = sim
                best_rel = rel_type.value

        return best_rel if max_sim > 0.7 else "RELATED_TO"

    async def resolve_entity(self, label: NodeLabel, identifier: str) -> str:
        """Resolves fuzzy names (Apple) to canonical IDs (AAPL) via Neo4j full-text or exact match."""
        # Simple implementation: Exact match or property search
        query = f"MATCH (n:{label.value}) WHERE n.name =~ $id OR n.ticker =~ $id OR n.user_id = $id RETURN n LIMIT 1"
        res = self.graph.query(query, {"id": identifier})
        if res:
            return res[0]["n"].get("ticker") or res[0]["n"].get("user_id") or identifier
        return identifier

    async def execute_query_plan(self, plan: QueryPlan) -> List[Dict]:
        """Compiles the DSL Plan into Cypher and executes it."""
        try:
            canonical_id = await self.resolve_entity(
                plan.start_node_label, plan.start_node_identifier
            )

            cypher = f"MATCH (start:{plan.start_node_label.value} {{ticker: $id}})"
            if plan.start_node_label == NodeLabel.USER:
                cypher = f"MATCH (start:{plan.start_node_label.value} {{user_id: $id}})"

            for i, step in enumerate(plan.traversal_path):
                rel_type = await self.map_relationship(step.relationship_hint)
                dir_in = "<" if step.direction == "IN" else ""
                dir_out = ">" if step.direction == "OUT" else ""
                cypher += f"{dir_in}-[:{rel_type}]-{dir_out}(step{i}:{step.target_label.value})"

            cypher += f" RETURN DISTINCT step{len(plan.traversal_path)-1} AS result LIMIT {plan.limit}"

            results = self.graph.query(cypher, {"id": canonical_id})
            return [r["result"] for r in results]
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []
