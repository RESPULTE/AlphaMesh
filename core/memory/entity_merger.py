"""
core/memory/entity_merger.py

Selective entity merging for freshly added graph nodes.

Algorithm:
  1. Receive graph-level DataPoint nodes that were just written by add_data_points.
  2. Filter to global entity types (Company, Sector, FinancialEvent, MacroTrend, FinancialConcept).
  3. For each node, query Neo4j with APOC sorensenDiceSimilarity to find fuzzy candidates.
  4. For each candidate above the fuzzy threshold, confirm via vector engine search (ScoredResult).
  5. If confirmed above semantic threshold, merge with apoc.refactor.mergeNodes.
  6. Update relational edges table and reindex vector store for affected nodes.
"""

import logging
from ast import literal_eval
from typing import Any, Dict, List, Set, Tuple
from typing import cast as typing_cast

import networkx as nx
from cognee.api.v1.datasets.datasets import datasets
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage import index_data_points, index_graph_edges
from sqlalchemy import bindparam, text

from core.memory.graph_models import ALL_ENTITIES, DATASET_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# APOC fuzzy gate — casts a wide net; cheap, runs on Neo4j side
FUZZY_CANDIDATE_THRESHOLD = 0.50

# Vector engine gate — expensive embedding comparison; final merge decision
SEMANTIC_MERGE_THRESHOLD = 0.85

# Map entity type name → vector collection name used by index_data_points
# Collection naming follows the pattern: {ClassName}_{first_index_field}
_COLLECTION_MAP: Dict[str, str] = {k: f"{k}_name" for k in ALL_ENTITIES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_collection_for(data_point: DataPoint) -> str | None:
    """Return the vector collection name for the given data point, or None."""
    type_name = type(data_point).__name__
    if type_name in _COLLECTION_MAP:
        return _COLLECTION_MAP[type_name]
    # Try MRO to find a known parent
    for cls in type(data_point).__mro__:
        if cls.__name__ in _COLLECTION_MAP:
            return _COLLECTION_MAP[cls.__name__]
    return None


def _embeddable_text(data_point: DataPoint) -> str | None:
    """Extract the primary embeddable text via DataPoint.get_embeddable_data()."""
    result = DataPoint.get_embeddable_data(data_point)
    if result and isinstance(result, str):
        return result.strip() or None
    return None


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


async def find_and_merge_candidates(
    graph_client: Any,
    graph_nodes: List[DataPoint],
) -> None:
    """
    Given a list of DataPoint graph nodes just added to Neo4j, search for fuzzy
    name-duplicates within the same label group and merge them if confirmed
    semantically via the vector engine.

    Args:
        graph_client: Active graph engine instance (Neo4jAdapter).
        graph_nodes:  DataPoint objects at the graph-node level (entities extracted
                      from DocumentChunks), NOT the DocumentChunk wrappers.
    """
    if not graph_client or not graph_nodes:
        return

    candidates = _filter_mergeable_candidates(graph_nodes)
    if not candidates:
        logger.debug("merge_entities: no mergeable global entities in batch.")
        return

    vector_engine = get_vector_engine()

    # 1. Build an equivalence graph
    G, neo4j_id_map, name_map = await _build_equivalence_graph(
        graph_client, vector_engine, candidates
    )

    # 2. Resolve connected components → merge groups
    merge_groups = _resolve_merge_groups(G)
    if not merge_groups:
        logger.debug("merge_entities: no duplicate entities detected.")
        return

    logger.debug("merge_entities: %d merge group(s) identified.", len(merge_groups))

    # 3. Fetch missing neo4j internal IDs for nodes discovered only via APOC
    await _fetch_missing_neo4j_ids(graph_client, merge_groups, neo4j_id_map)

    # 4. Execute APOC merges + relational rewires
    affected_node_ids, node_ids_to_keep, success_count = await _execute_apoc_merges(
        graph_client, merge_groups, neo4j_id_map
    )

    if success_count == 0 or not affected_node_ids:
        return

    # 5. Reindex vector store for affected nodes
    await _reindex_vector_store(node_ids_to_keep)

    await _delete_merged_orphaned_nodes(affected_node_ids)


def _filter_mergeable_candidates(graph_nodes: List[DataPoint]) -> List[DataPoint]:
    """Filter to mergeable global entity types with a name attribute."""
    return [
        dp
        for dp in graph_nodes
        if type(dp).__name__ in ALL_ENTITIES and getattr(dp, "name", None)
    ]


async def _build_equivalence_graph(
    graph_client: Any, vector_engine: Any, candidates: List[DataPoint]
) -> Tuple[nx.Graph, Dict[str, int], Dict[str, str]]:
    """Build an equivalence graph of fuzzy name-duplicates."""
    G: nx.Graph = nx.Graph()
    neo4j_id_map: Dict[str, int] = {}
    name_map: Dict[str, str] = {}

    for dp in candidates:
        canonical_id = str(dp.id)
        label = type(dp).__name__
        name = str(getattr(dp, "name", "") or "").strip()

        if not name:
            continue

        G.add_node(canonical_id, cognee_id=canonical_id, label=label, name=name)
        name_map[canonical_id] = name

        await _find_and_add_fuzzy_matches(
            graph_client,
            vector_engine,
            dp,
            G,
            neo4j_id_map,
            name_map,
            canonical_id,
            label,
            name,
        )

    return G, neo4j_id_map, name_map


async def _find_and_add_fuzzy_matches(
    graph_client: Any,
    vector_engine: Any,
    dp: DataPoint,
    G: nx.Graph,
    neo4j_id_map: Dict[str, int],
    name_map: Dict[str, str],
    canonical_id: str,
    label: str,
    name: str,
) -> None:
    """Find fuzzy candidates on Neo4j and confirm them via the vector engine."""
    fuzzy_query = """
    MATCH (n:`__Node__`)
    WHERE n.name IS NOT NULL AND n.id <> $cognee_id
    WITH n, apoc.text.sorensenDiceSimilarity(
            toLower(n.name), toLower($name)
         ) AS sim
    WHERE sim >= $threshold
    WITH n, id(n) AS neo4j_id, sim, n.type AS type
    ORDER BY sim DESC
    LIMIT 10
    RETURN neo4j_id, n.id AS cognee_id, n.name AS name, sim, type
    """

    try:
        rows = await graph_client.query(
            fuzzy_query,
            {
                "cognee_id": canonical_id,
                "name": name,
                "threshold": FUZZY_CANDIDATE_THRESHOLD,
            },
        )
    except Exception as exc:
        logger.warning(
            "merge_entities: APOC fuzzy query failed for '%s': %s", name, exc
        )
        return

    if not rows:
        return

    for row in rows:
        data = (
            row.data()
            if hasattr(row, "data")
            else (dict(row) if isinstance(row, dict) else {})
        )
        match_cognee_id = str(data.get("cognee_id", ""))
        match_name = str(data.get("name", "")).strip()
        match_neo4j_id = data.get("neo4j_id")
        fuzzy_score = float(data.get("sim", 0.0))

        if not match_cognee_id or match_cognee_id == canonical_id:
            continue

        candidate_label = str(data.get("type", ""))
        # entities must be superseded by the sector
        if candidate_label != label and candidate_label.lower() != "sector":
            logger.debug(
                "merge_entities: skipping cross-type merge '%s' (%s) ↔ '%s' (%s)",
                name,
                label,
                match_name,
                candidate_label,
            )
            continue

        if match_neo4j_id is not None:
            neo4j_id_map[match_cognee_id] = int(match_neo4j_id)

        logger.debug(
            "merge_entities: fuzzy candidate '%s' ↔ '%s' (dice=%.3f)",
            name,
            match_name,
            fuzzy_score,
        )

        match_confirmed = False
        if fuzzy_score >= SEMANTIC_MERGE_THRESHOLD:
            logger.debug(
                "merge_entities: confirmed match '%s' ↔ '%s' (dice=%.3f)",
                name,
                match_name,
                fuzzy_score,
            )
            match_confirmed = True
        else:
            match_confirmed = await _confirm_semantic_match(
                vector_engine, dp, name, label, match_cognee_id, match_name, fuzzy_score
            )

        if match_confirmed:
            # entities must be superseded by the sector
            final_label = (
                candidate_label if candidate_label.lower() == "sector" else label
            )
            G.add_node(
                match_cognee_id,
                cognee_id=match_cognee_id,
                label=final_label,
                name=match_name,
            )
            G.add_edge(canonical_id, match_cognee_id)
            name_map[match_cognee_id] = match_name


async def _confirm_semantic_match(
    vector_engine: Any,
    dp: DataPoint,
    name: str,
    label: str,
    match_cognee_id: str,
    match_name: str,
    fuzzy_score: float,
) -> bool:
    """Confirm a fuzzy match using the vector engine."""
    embeddable = _embeddable_text(dp)
    if not embeddable:
        embeddable = name

    collection = _get_collection_for(dp)
    if not collection:
        logger.debug(
            "merge_entities: no collection for type %s, skipping semantic gate.",
            label,
        )
        return False

    try:
        scored_results = await vector_engine.search(
            collection_name=collection,
            query_text=embeddable,
            limit=10,
        )
        logger.debug(
            "merge_entities: vector search results for '%s': %s",
            name,
            scored_results,
        )
    except Exception as exc:
        logger.warning("merge_entities: vector search failed for '%s': %s", name, exc)
        return False

    # Check if the candidate node appears in the top results above threshold
    # Cognee's normalisation for the score is opposite, lower = closer in similarity fkc
    for sr in scored_results:
        if str(sr.id) == match_cognee_id and sr.score <= (1 - SEMANTIC_MERGE_THRESHOLD):
            logger.debug(
                "merge_entities: confirmed match '%s' ↔ '%s' "
                "(dice=%.3f, vector=%.3f)",
                name,
                match_name,
                fuzzy_score,
                sr.score,
            )
            return True

    return False


def _resolve_merge_groups(G: nx.Graph) -> List[List[Dict[str, Any]]]:
    """Resolve connected components into a list of merge groups."""
    merge_groups: List[List[Dict[str, Any]]] = []
    for component in nx.connected_components(G):
        if len(component) < 2:
            continue
        cluster = [G.nodes[cid] for cid in component]
        # Canonical = shortest name, then lexicographic — deterministic
        cluster.sort(key=lambda x: (len(x["name"]), x["name"]))
        merge_groups.append(cluster)
    return merge_groups


async def _fetch_missing_neo4j_ids(
    graph_client: Any,
    merge_groups: List[List[Dict[str, Any]]],
    neo4j_id_map: Dict[str, int],
) -> None:
    """Fetch missing neo4j internal IDs for nodes discovered only via APOC."""
    all_cognee_ids_needed: Set[str] = set()
    for group in merge_groups:
        for node in group:
            cid = node["cognee_id"]
            if cid not in neo4j_id_map:
                all_cognee_ids_needed.add(cid)

    if not all_cognee_ids_needed:
        return

    id_fetch_query = """
    MATCH (n:`__Node__`)
    WHERE n.id IN $ids
    RETURN n.id AS cognee_id, id(n) AS neo4j_id
    """
    try:
        id_rows = await graph_client.query(
            id_fetch_query, {"ids": list(all_cognee_ids_needed)}
        )
        for row in id_rows:
            data = (
                row.data()
                if hasattr(row, "data")
                else (dict(row) if isinstance(row, dict) else {})
            )
            if data.get("cognee_id") and data.get("neo4j_id") is not None:
                neo4j_id_map[str(data["cognee_id"])] = int(data["neo4j_id"])
    except Exception as exc:
        logger.warning("merge_entities: failed to fetch neo4j IDs: %s", exc)


async def _execute_apoc_merges(
    graph_client: Any,
    merge_groups: List[List[Dict[str, Any]]],
    neo4j_id_map: Dict[str, int],
) -> Tuple[Set[str], List[str], int]:
    """Execute APOC merges and update relational edges table."""

    async def _update_relational_database(canonical_cid: str, old_cids: List[str]):
        try:
            relational_engine = get_relational_engine()
            if hasattr(relational_engine, "engine"):
                async with relational_engine.engine.begin() as conn:
                    await conn.execute(
                        text(
                            "UPDATE edges SET source_node_id = :canonical_id"
                            " WHERE source_node_id IN :old_ids"
                        ).bindparams(bindparam("old_ids", expanding=True)),
                        {"canonical_id": canonical_cid, "old_ids": old_cids},
                    )
                    await conn.execute(
                        text(
                            "UPDATE edges SET destination_node_id = :canonical_id"
                            " WHERE destination_node_id IN :old_ids"
                        ).bindparams(bindparam("old_ids", expanding=True)),
                        {"canonical_id": canonical_cid, "old_ids": old_cids},
                    )
        except Exception as exc:
            logger.error(
                "merge_entities: relational rewire failed for '%s': %s",
                canonical_name,
                exc,
            )

    merge_query = """
    MATCH (n:`__Node__`)
    WHERE id(n) IN $node_ids
    WITH collect(n) AS neo4j_nodes
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
    RETURN DISTINCT id(node) AS merged_neo4j_id, node.id AS cognee_id, node.name AS name
    """

    affected_node_ids: Set[str] = set()
    node_ids_to_keep = []
    success_count = 0

    for group in merge_groups:
        canonical = group[0]
        canonical_cid = canonical["cognee_id"]
        canonical_name = canonical["name"]
        old_cids = [n["cognee_id"] for n in group[1:]]

        node_ids = [
            neo4j_id_map[n["cognee_id"]]
            for n in group
            if n["cognee_id"] in neo4j_id_map
        ]

        if len(node_ids) < 2:
            logger.warning(
                "merge_entities: cannot merge '%s' — missing neo4j IDs.", canonical_name
            )
            continue

        affected_node_ids.update(old_cids)
        node_ids_to_keep.append(canonical_cid)

        try:
            await graph_client.query(
                merge_query,
                {"node_ids": node_ids, "canonical_id": canonical_cid},
            )
            success_count += 1
            logger.debug(
                "merge_entities: merged %d nodes → '%s'.", len(node_ids), canonical_name
            )
        except Exception as exc:
            logger.error(
                "merge_entities: APOC merge failed for '%s': %s", canonical_name, exc
            )
            continue

        # Rewire stale UUIDs in relational edges table
        if not old_cids:
            continue

        await _update_relational_database(canonical_cid, old_cids)

    logger.debug(
        "merge_entities: %d/%d group(s) merged successfully.",
        success_count,
        len(merge_groups),
    )

    return affected_node_ids, node_ids_to_keep, success_count


async def _reindex_vector_store(node_ids_to_keep: List[str]) -> None:
    """Reindex vector store for affected nodes."""

    graph_engine = await get_graph_engine()
    # get_id_filtered_graph_data is a Neo4j-specific method
    neo4j_engine = typing_cast(Neo4jAdapter, graph_engine)
    nodes_data, edges_data = await neo4j_engine.get_id_filtered_graph_data(
        node_ids_to_keep
    )

    indexable_nodes = []
    for _, props in nodes_data:
        # logger.debug("Keeping node with id: %s", _)
        props = {k: literal_eval(v) if k == "metadata" else v for k, v in props.items()}
        if "metadata" in props and isinstance(props["metadata"], dict):
            props["metadata"].setdefault("type", props.get("type"))
        indexable_nodes.append(DataPoint(**props))

    if indexable_nodes:
        await index_data_points(indexable_nodes)

    if edges_data:
        await index_graph_edges(edges_data)


async def _delete_merged_orphaned_nodes(affected_node_ids: Set[str]):
    import uuid

    from cognee.modules.data.methods import get_unique_dataset_id
    from cognee.modules.users.methods import get_default_user

    dataset_id = await get_unique_dataset_id(DATASET_NAME, await get_default_user())
    for id_to_del in affected_node_ids:
        logger.debug("Deleting node with id: %s", id_to_del)
        await datasets.delete_data(dataset_id=dataset_id, data_id=uuid.UUID(id_to_del))
