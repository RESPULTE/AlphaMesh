from typing import Optional, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings
from models import QueryPlan
from ontology import NodeLabel, RelationshipType


class QueryCompiler:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        # Pre-cache embeddings for relationship types for dynamic mapping
        self._rel_types = [r.value for r in RelationshipType]
        self._rel_embeddings = self.embeddings.embed_documents(self._rel_types)

    def _get_best_relationship(self, hint: str) -> str:
        """Dynamically maps a natural language hint to a canonical relationship type."""
        hint_embedding = self.embeddings.embed_query(hint)
        # Cosine similarity to find the closest enum match
        similarities = [
            np.dot(hint_embedding, rel_emb)
            / (np.linalg.norm(hint_embedding) * np.linalg.norm(rel_emb))
            for rel_emb in self._rel_embeddings
        ]
        best_idx = np.argmax(similarities)
        return self._rel_types[best_idx]

    def compile(
        self, plan: QueryPlan, user_id: Optional[str] = None
    ) -> Tuple[str, dict]:
        """Compiles the QueryPlan into a parameterized Cypher string."""
        params = {"limit": plan.limit}

        # 1. Resolve Start Node Logic
        # We use a case-insensitive search or ticker match as a starting point
        id_key = "ticker" if plan.start_node_label == NodeLabel.ASSET else "name"
        if plan.start_node_label == NodeLabel.USER:
            id_key = "user_id"
            start_val = user_id
        else:
            start_val = plan.start_node_identifier

        cypher = f"MATCH (n:{plan.start_node_label.value} {{{id_key}: $start_val}})\n"
        params["start_val"] = start_val

        # 2. Build Traversal
        for i, step in enumerate(plan.traversal_path):
            rel_type = self._get_best_relationship(step.relationship_hint)
            target_alias = f"m{i}"

            if step.direction == "OUT":
                cypher += (
                    f"-[:{rel_type}]->({target_alias}:{step.target_label.value})\n"
                )
            elif step.direction == "IN":
                cypher += (
                    f"<-[:{rel_type}]-({target_alias}:{step.target_label.value})\n"
                )
            else:
                cypher += f"-[:{rel_type}]-({target_alias}:{step.target_label.value})\n"

            # Update return reference
            last_alias = target_alias

        # 3. Apply Filters
        if plan.filters:
            cypher += "WHERE "
            filter_clauses = []
            for f_idx, f in enumerate(plan.filters):
                p_key = f"f_val_{f_idx}"
                op = (
                    "=" if f.operator == "EQ" else ">" if f.operator == "GT" else "<"
                )  # simplified
                filter_clauses.append(f"{last_alias}.{f.property} {op} ${p_key}")
                params[p_key] = f.value
            cypher += " AND ".join(filter_clauses) + "\n"

        # 4. Return Clause
        if plan.return_target == "node_properties":
            cypher += f"RETURN {last_alias} as result LIMIT $limit"
        elif plan.return_target == "neighbors":
            cypher += f"RETURN collect({last_alias}.name) as result LIMIT $limit"
        else:
            cypher += "RETURN * LIMIT $limit"

        return cypher, params
