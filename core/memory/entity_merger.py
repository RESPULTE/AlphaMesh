"""
core/memory/entity_merger.py

Implementation for merging global entities with identical or highly similar names.
Operates directly on the Neo4j graph using apoc.refactor.mergeNodes.
"""

import logging
import difflib
import networkx as nx
from typing import Any, List, Dict, cast as typing_cast
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.engine import DataPoint
from sqlalchemy import text, bindparam
from cognee.api.v1.search import search
from cognee.modules.search.types import SearchType
from cognee.tasks.storage import index_data_points, index_graph_edges
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.databases.vector import get_vector_engine
from ast import literal_eval

logger = logging.getLogger(__name__)

# Configurable similarity thresholds
AUTO_MERGE_THRESHOLD = 0.85
SEMANTIC_CHECK_THRESHOLD = 0.50


async def run_entity_merging_neo4j(
    graph_client: Any, similarity_threshold: float = AUTO_MERGE_THRESHOLD
) -> None:
    """
    Executes a standalone graph-maintenance query to merge similar global entities.

    1. Fetches candidate GlobalEntity nodes (Company, Sector, GlobalEvent, MacroTrend, FinancialConcept).
    2. Groups them by labels and evaluates pairs using difflib fuzzy matching.
    3. Auto-merges nodes with similarity >= AUTO_MERGE_THRESHOLD.
    4. Semantically checks nodes with similarity >= SEMANTIC_CHECK_THRESHOLD or subsets using SearchType.CHUNKS.
    5. Resolves connected components and merges each cluster using apoc.refactor.mergeNodes.
    """
    if not graph_client:
        logger.warning(
            "run_entity_merging_neo4j called with None graph_client. Skipping."
        )
        return

    logger.info("Starting Neo4j entity merging routine...")

    try:
        fetch_query = """
        MATCH (n:`__Node__`)
        WHERE (n:Company OR n:Sector OR n:GlobalEvent OR n:MacroTrend OR n:FinancialConcept)
          AND n.name IS NOT NULL
        RETURN id(n) AS neo4j_id, n.id AS cognee_id, n.name AS name, labels(n) AS labels
        """
        results = await graph_client.query(fetch_query)

        if not results:
            logger.info("No mergeable global entities found.")
            return

        # Group nodes by their primary global label
        label_groups: Dict[str, List[Dict[str, Any]]] = {}
        target_labels = {
            "Company",
            "Sector",
            "GlobalEvent",
            "MacroTrend",
            "FinancialConcept",
        }

        for record in results:
            data_dict = {}
            if hasattr(record, "data"):
                data_dict = record.data()
            elif isinstance(record, dict):
                data_dict = record
            else:
                try:
                    data_dict = dict(record)
                except Exception:
                    pass

            neo4j_id = data_dict.get("neo4j_id")
            cognee_id = data_dict.get("cognee_id")
            name = data_dict.get("name")
            labels = data_dict.get("labels", [])

            if neo4j_id is None or name is None or cognee_id is None:
                continue

            # Find the primary label to group by
            primary_label = None
            for lbl in labels:
                if lbl in target_labels:
                    primary_label = lbl
                    break

            if not primary_label:
                continue

            if primary_label not in label_groups:
                label_groups[primary_label] = []

            label_groups[primary_label].append(
                {
                    "neo4j_id": int(neo4j_id),
                    "cognee_id": cognee_id,
                    "name": str(name).strip(),
                }
            )

        # Build a graph of equivalences
        G = nx.Graph()

        # Exact ID matches (the original logic)
        for label, nodes in label_groups.items():
            for node in nodes:
                G.add_node(node["neo4j_id"], **node)

            # Compare all pairs within the same label group
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_a = nodes[i]
                    node_b = nodes[j]

                    name_a = node_a["name"].lower()
                    name_b = node_b["name"].lower()

                    # 1. Exact canonical ID match (original logic)
                    if generate_node_id(name_a) == generate_node_id(name_b):
                        G.add_edge(node_a["neo4j_id"], node_b["neo4j_id"])
                        continue

                    # 2. Fuzzy Matching
                    ratio = difflib.SequenceMatcher(None, name_a, name_b).ratio()

                    # Condition 1: Auto-merge
                    if ratio >= AUTO_MERGE_THRESHOLD:
                        logger.debug(
                            f"Auto-merging '{name_a}' and '{name_b}' (ratio: {ratio:.2f})"
                        )
                        G.add_edge(node_a["neo4j_id"], node_b["neo4j_id"])
                        continue

                    # Condition 2: Semantic check via Vector search chunks
                    is_subset = (name_a in name_b) or (name_b in name_a)
                    if ratio >= SEMANTIC_CHECK_THRESHOLD or is_subset:
                        try:
                            # Use CHUNKS to see if they align semantically without LLM overhead
                            search_results = await search(
                                query_text=name_a,
                                query_type=SearchType.CHUNKS,
                                top_k=10,
                            )

                            # check if the other name appears in the top retrieved chunks
                            found_semantic_match = False
                            for res in search_results:
                                chunk_text = getattr(res, "text", "")
                                if isinstance(res, dict):
                                    chunk_text = res.get("text", "")

                                if chunk_text and name_b in chunk_text.lower():
                                    found_semantic_match = True
                                    break

                            if found_semantic_match:
                                logger.debug(
                                    f"Semantic match found for '{name_a}' and '{name_b}'"
                                )
                                G.add_edge(node_a["neo4j_id"], node_b["neo4j_id"])
                        except Exception as e:
                            logger.warning(
                                f"Semantic check failed for {name_a} and {name_b}: {e}"
                            )

        # Extract connected components (clusters of duplicates)
        merge_groups = []
        for component in nx.connected_components(G):
            if len(component) > 1:
                # Get the actual node dicts
                cluster_nodes = [G.nodes[n_id] for n_id in component]

                # Sort to ensure deterministic canonical selection (shortest name, then lowest ID)
                cluster_nodes.sort(key=lambda x: (len(x["name"]), x["neo4j_id"]))
                merge_groups.append(cluster_nodes)

        if not merge_groups:
            logger.info("No duplicate entities found to merge.")
            return

        logger.info(f"Found {len(merge_groups)} groups of duplicate entities to merge.")

        relational_engine = get_relational_engine()

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

        # Execute APOC merge for each group; update relational DB only on success
        success_count = 0
        affected_node_ids = set()
        for nodes in merge_groups:
            canonical_cognee_id = nodes[0]["cognee_id"]
            canonical_name = nodes[0]["name"]

            old_cognee_ids = [n["cognee_id"] for n in nodes[1:]]
            node_ids = [n["neo4j_id"] for n in nodes]

            affected_node_ids.update([canonical_cognee_id] + old_cognee_ids)

            logger.debug(
                f"Merging {len(node_ids)} nodes into canonical uuid {canonical_cognee_id} ('{canonical_name}')"
            )

            # 1. Execute the APOC merge in Neo4j first
            try:
                await graph_client.query(
                    merge_query,
                    {"node_ids": node_ids, "canonical_id": canonical_cognee_id},
                )
                success_count += 1
                logger.info(
                    f"Successfully merged group '{canonical_name}' with neo4j IDs {node_ids}."
                )
            except Exception as e:
                logger.error(f"Failed to merge group '{canonical_name}': {e}")
                continue  # Skip relational update if Neo4j merge failed

            # 2. Only after a confirmed Neo4j merge: rewire stale UUIDs in Cognee's
            #    relational `edges` table (source_node_id / destination_node_id).
            #    There is no cognee built-in for this — index_data_points / index_graph_edges
            #    only touch the vector index, not the relational DB.
            if not old_cognee_ids:
                continue
            try:
                if hasattr(relational_engine, "engine"):
                    async with relational_engine.engine.begin() as conn:
                        await conn.execute(
                            text(
                                "UPDATE edges SET source_node_id = :canonical_id"
                                " WHERE source_node_id IN :old_ids"
                            ).bindparams(bindparam("old_ids", expanding=True)),
                            {
                                "canonical_id": canonical_cognee_id,
                                "old_ids": old_cognee_ids,
                            },
                        )
                        await conn.execute(
                            text(
                                "UPDATE edges SET destination_node_id = :canonical_id"
                                " WHERE destination_node_id IN :old_ids"
                            ).bindparams(bindparam("old_ids", expanding=True)),
                            {
                                "canonical_id": canonical_cognee_id,
                                "old_ids": old_cognee_ids,
                            },
                        )
            except Exception as e:
                logger.error(
                    f"Failed to rewire Cognee edge UUIDs for '{canonical_name}': {e}"
                )

        logger.info(
            f"Successfully merged {success_count} out of {len(merge_groups)} entity groups."
        )

        if success_count == 0:
            return

        # 3. Reindex the vector store from the current graph state so that stale
        #    entries left by deleted nodes are replaced with live data.
        #    This keeps ChunksRetriever / JaccardChunksRetriever aligned with Neo4j.
        try:
            logger.info("Reindexing vector store after entity merges...")
            graph_engine: Neo4jAdapter = await get_graph_engine()
            vector_engine = get_vector_engine()

            nodes_data, edges_data = await graph_engine.get_id_filtered_graph_data(
                list(affected_node_ids)
            )

            collections_to_clean = [
                "Entity_name",
                "EntityType_name",
                "DocumentChunk_text",
                "EdgeType_relationship_name",
                "TextDocument_name",
                "TextSummary_text",
            ]
            for coll in collections_to_clean:
                if await vector_engine.has_collection(coll):
                    logger.info(f"Deleting data points from collection {coll}")
                    await vector_engine.delete_data_points(
                        coll, list(affected_node_ids)
                    )

            # index_data_points expects DataPoint instances; nodes from get_graph_data
            # are (node_id, properties_dict) tuples — build a lightweight wrapper list
            # that only passes actual DataPoint objects through (graph backends may
            # return either raw dicts or DataPoint subclasses).
            indexable_nodes = []
            for _, props in nodes_data:
                props = {
                    k: literal_eval(v) if k == "metadata" else v
                    for k, v in props.items()
                }
                if "metadata" in props and isinstance(props["metadata"], dict):
                    props["metadata"].setdefault("type", props.get("type"))
                indexable_nodes.append(DataPoint(**props))
            logger.info(f"Reindexing {len(indexable_nodes)} nodes.")
            if indexable_nodes:
                await index_data_points(indexable_nodes)

            logger.info(f"Reindexing {len(edges_data)} edges.")
            await index_graph_edges(edges_data)

            logger.info("Vector store reindex complete.")
        except Exception as e:
            logger.error(f"Vector store reindex failed after entity merges: {e}")

    except Exception as e:
        logger.error(f"Error during entity merging: {e}")


def cleanup():
    pass
