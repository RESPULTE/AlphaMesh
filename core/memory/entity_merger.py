"""
core/memory/entity_merger.py

Implementation for merging global entities with identical or highly similar names.
Operates directly on the Neo4j graph using apoc.refactor.mergeNodes.
"""

import logging
from typing import Any, List, Dict
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.infrastructure.databases.relational import get_relational_engine
from sqlalchemy import text, bindparam

logger = logging.getLogger(__name__)


async def run_entity_merging_neo4j(
    graph_client: Any, similarity_threshold: float = 0.85
) -> None:
    """
    Executes a standalone graph-maintenance query to merge similar global entities.

    1. Fetches candidate GlobalEntity nodes (Company, Sector, GlobalEvent, MacroTrend, FinancialConcept).
    2. Groups them by normalized name ID (via generate_node_id).
    3. Executes apoc.refactor.mergeNodes for each group of duplicates.

    Args:
        graph_client: The initialized Neo4j graph client (e.g., from get_graph_engine()).
        similarity_threshold: Future-proofing for semantic matching (currently maps exact IDs).
    """
    if not graph_client:
        logger.warning(
            "run_entity_merging_neo4j called with None graph_client. Skipping."
        )
        return

    logger.info("Starting Neo4j entity merging routine...")

    try:
        # Step 1: Fetch canditate entities.
        # We target labels that inherit from GlobalEntity or are standalone globals.
        # Neo4j adapter forces `__Node__` on all nodes, and dynamically adds `name` etc.
        # We look for nodes that have a 'name' property and are NOT InvestmentThesis.

        fetch_query = """
        MATCH (n:`__Node__`)
        WHERE (n:Company OR n:Sector OR n:GlobalEvent OR n:MacroTrend OR n:FinancialConcept)
          AND n.name IS NOT NULL
        RETURN id(n) AS neo4j_id, n.id AS cognee_id, n.name AS name, labels(n) AS labels
        """
        # Await execution. Typically neo4j execute() returns a list of Neo4j Records.
        results = await graph_client.query(fetch_query)

        if not results:
            logger.info("No mergeable global entities found.")
            return

        # Step 2: Group by normalized ID
        grouped_nodes: Dict[str, List[Dict[str, Any]]] = {}
        for record in results:
            # Safely extract from Neo4j records, dicts, or tuples
            data_dict = {}
            if hasattr(record, "data"):
                # Neo4j python driver Record object
                data_dict = record.data()
            elif isinstance(record, dict):
                data_dict = record
            else:
                try:
                    # Fallback for tuples if mapped as dict-like
                    data_dict = dict(record)
                except Exception:
                    pass

            neo4j_id = data_dict.get("neo4j_id")
            cognee_id = data_dict.get("cognee_id")
            name = data_dict.get("name")

            if neo4j_id is None or name is None or cognee_id is None:
                continue

            norm_id = str(generate_node_id(str(name).strip()))

            if norm_id not in grouped_nodes:
                grouped_nodes[norm_id] = []
            grouped_nodes[norm_id].append(
                {"neo4j_id": int(neo4j_id), "cognee_id": cognee_id}
            )

        # Filter down to only groups that have more than 1 node (duplicates exist)
        merge_groups = {k: v for k, v in grouped_nodes.items() if len(v) > 1}

        if not merge_groups:
            logger.info("No duplicate entities found to merge.")
            return

        logger.info(f"Found {len(merge_groups)} groups of duplicate entities to merge.")

        relational_engine = get_relational_engine()

        # Step 3: Execute APOC merge for each group
        success_count = 0
        for norm_id, nodes in merge_groups.items():

            canonical_cognee_id = nodes[0]["cognee_id"]
            old_cognee_ids = tuple(n["cognee_id"] for n in nodes[1:])
            node_ids = [n["neo4j_id"] for n in nodes]

            logger.debug(
                f"Merging group {norm_id} into canonical uuid {canonical_cognee_id}"
            )

            try:
                # Update orphaned edge references in Cognee's internal relational DB
                if hasattr(relational_engine, "engine"):
                    async with relational_engine.engine.begin() as conn:
                        await conn.execute(
                            text(
                                "UPDATE edges SET source_node_id = :canonical_id WHERE source_node_id IN :old_ids"
                            ).bindparams(bindparam("old_ids", expanding=True)),
                            {
                                "canonical_id": canonical_cognee_id,
                                "old_ids": list(old_cognee_ids),
                            },
                        )
                        await conn.execute(
                            text(
                                "UPDATE edges SET destination_node_id = :canonical_id WHERE destination_node_id IN :old_ids"
                            ).bindparams(bindparam("old_ids", expanding=True)),
                            {
                                "canonical_id": canonical_cognee_id,
                                "old_ids": list(old_cognee_ids),
                            },
                        )
            except Exception as e:
                logger.error(f"Failed to rewire Cognee edge UUIDs for {norm_id}: {e}")
                continue

            merge_query = """
            MATCH (n:`__Node__`)
            WHERE id(n) IN $node_ids
            WITH collect(n) as neo4j_nodes
            CALL apoc.refactor.mergeNodes(neo4j_nodes, {
                properties: "overwrite",
                mergeRels: true,
                preserveExistingSelfRels: false
            })
            YIELD node
            SET node.id = $canonical_id
            WITH node
            OPTIONAL MATCH (node)-[r]-()
            FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
                SET r.source_node_id = startNode(r).id,
                    r.target_node_id = endNode(r).id
            )
            RETURN DISTINCT id(node) as merged_neo4j_id, node.id as cognee_id, node.name as name
            """

            try:
                # Must cast to ensure native types (list of ints usually ok)
                await graph_client.query(
                    merge_query,
                    {"node_ids": node_ids, "canonical_id": canonical_cognee_id},
                )
                success_count += 1
                logger.info(
                    f"Successfully merged group {norm_id} with neo4j IDs {node_ids}."
                )
            except Exception as e:
                logger.error(f"Failed to merge group {norm_id}: {e}")

        logger.info(
            f"Successfully merged {success_count} out of {len(merge_groups)} entity groups."
        )

    except Exception as e:
        logger.error(f"Error during entity merging: {e}")
