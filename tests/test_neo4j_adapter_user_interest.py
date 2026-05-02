from __future__ import annotations

import asyncio

from core.memory.stores.neo4j_adapter import Neo4jAdapter


def test_merge_user_interest_domain_persists_nodeset_link() -> None:
    adapter = Neo4jAdapter(
        uri="bolt://unused",
        username="unused",
        password="unused",
    )

    captured: dict = {}

    async def _fake_execute_write(cypher: str, params: dict) -> None:
        captured["cypher"] = cypher
        captured["params"] = params

    adapter._execute_write = _fake_execute_write  # type: ignore[method-assign]

    async def _run() -> None:
        await adapter.merge_user_interest_domain(
            "domain-1",
            {
                "id": "domain-1",
                "user_email": "demo@alphamesh.local",
                "domain_type": "investment",
                "category": "Technology",
                "nodeset_id": "nodeset-1",
            },
        )

    asyncio.run(_run())

    assert "BELONGS_TO_NODESET" in captured["cypher"]
    assert captured["params"]["nodeset_id"] == "nodeset-1"
    assert "nodeset_id" not in captured["params"]["props"]


def test_merge_session_node_persists_nodeset_link() -> None:
    adapter = Neo4jAdapter(
        uri="bolt://unused",
        username="unused",
        password="unused",
    )

    captured: dict = {}

    async def _fake_execute_write(cypher: str, params: dict) -> None:
        captured["cypher"] = cypher
        captured["params"] = params

    adapter._execute_write = _fake_execute_write  # type: ignore[method-assign]

    async def _run() -> None:
        await adapter.merge_session_node(
            "session-1",
            {
                "id": "session-1",
                "user_email": "demo@alphamesh.local",
                "conversation_id": "conv-1",
                "started_at": "2026-01-01T00:00:00+00:00",
                "nodeset_id": "nodeset-1",
            },
        )

    asyncio.run(_run())

    assert "BELONGS_TO_NODESET" in captured["cypher"]
    assert captured["params"]["nodeset_id"] == "nodeset-1"
    assert "nodeset_id" not in captured["params"]["props"]


def test_get_user_interest_domain_summary_clamps_limit_and_uses_ranked_query() -> None:
    adapter = Neo4jAdapter(
        uri="bolt://unused",
        username="unused",
        password="unused",
    )
    captured: dict = {}

    async def _fake_execute_read(cypher: str, params: dict):
        captured["cypher"] = cypher
        captured["params"] = params
        return []

    adapter._execute_read = _fake_execute_read  # type: ignore[method-assign]

    async def _run() -> None:
        await adapter.get_user_interest_domain_summary(
            user_email="demo@alphamesh.local",
            nodeset_id="nodeset-1",
            limit=99,
        )

    asyncio.run(_run())

    assert "positive_weight" in captured["cypher"]
    assert "ORDER BY positive_weight DESC" in captured["cypher"]
    assert captured["params"]["limit"] == 10
    assert captured["params"]["user_email"] == "demo@alphamesh.local"


def test_query_user_interest_context_applies_filters_and_hop_clamp() -> None:
    adapter = Neo4jAdapter(
        uri="bolt://unused",
        username="unused",
        password="unused",
    )
    captured: dict = {}

    async def _fake_execute_read(cypher: str, params: dict):
        captured["cypher"] = cypher
        captured["params"] = params
        return []

    adapter._execute_read = _fake_execute_read  # type: ignore[method-assign]

    async def _run() -> None:
        await adapter.query_user_interest_context(
            user_email="demo@alphamesh.local",
            nodeset_id="nodeset-1",
            domain_type="investment",
            category="Technology",
            target_entities=[
                {"entity_name": "Apple Inc.", "entity_type": "Company"},
                {"entity_name": "", "entity_type": "Company"},
            ],
            hops=99,
            risk_or_avoidance_intent=False,
            domain_limit=4,
            edge_limit=6,
            expanded_entity_limit=9,
        )

    asyncio.run(_run())

    assert "size($target_filters) = 0" in captured["cypher"]
    assert "edge_limit" in captured["cypher"]
    assert "WITH collect(DISTINCT {id: neighbor.id, name: neighbor.name, entity_type: neighbor.entity_type}) AS all_expanded_neighbors" in captured["cypher"]
    assert "RETURN all_expanded_neighbors[..expanded_entity_limit] AS expanded_neighbors" in captured["cypher"]
    assert "all(rel IN relationships(path) WHERE NOT type(rel) IN blocked_relationship_types)" in captured["cypher"]
    assert "none(label IN labels(path_node) WHERE label IN blocked_node_labels)" in captured["cypher"]
    assert "coalesce(trim(path_node.user_email), '') IN ['', user_email]" in captured["cypher"]
    assert captured["params"]["hops"] == 2
    assert captured["params"]["include_negative"] is False
    assert captured["params"]["domain_type"] == "investment"
    assert captured["params"]["category"] == "Technology"
    assert captured["params"]["blocked_relationship_types"] == [
        "BELONGS_TO_NODESET",
        "HAS_EVENT",
        "HAS_INTEREST_IN",
        "OBSERVED_IN",
        "TARGETS",
    ]
    assert captured["params"]["blocked_node_labels"] == [
        "SessionNode",
        "UserInterestDomain",
        "UserInterestEdge",
        "UserInterestEvent",
    ]
    assert captured["params"]["target_filters"] == [
        {"name_lower": "apple inc.", "entity_type": "Company"}
    ]


def test_query_user_interest_context_enables_negative_stance_for_risk_intent() -> None:
    adapter = Neo4jAdapter(
        uri="bolt://unused",
        username="unused",
        password="unused",
    )
    captured: dict = {}

    async def _fake_execute_read(cypher: str, params: dict):
        captured["params"] = params
        return []

    adapter._execute_read = _fake_execute_read  # type: ignore[method-assign]

    async def _run() -> None:
        await adapter.query_user_interest_context(
            user_email="demo@alphamesh.local",
            nodeset_id="nodeset-1",
            risk_or_avoidance_intent=True,
        )

    asyncio.run(_run())
    assert captured["params"]["include_negative"] is True


def test_get_entity_neighbors_excludes_user_scoped_edges_and_nodes() -> None:
    adapter = Neo4jAdapter(
        uri="bolt://unused",
        username="unused",
        password="unused",
    )
    captured: dict = {}

    async def _fake_execute_read(cypher: str, params: dict):
        captured["cypher"] = cypher
        captured["params"] = params
        return []

    adapter._execute_read = _fake_execute_read  # type: ignore[method-assign]

    async def _run() -> None:
        await adapter.get_entity_neighbors(["entity-1"], ["entity-2"])

    asyncio.run(_run())

    assert "NOT type(r) IN $blocked_relationship_types" in captured["cypher"]
    assert "coalesce(trim(e.user_email), '') = ''" in captured["cypher"]
    assert "coalesce(trim(neighbor.user_email), '') = ''" in captured["cypher"]
    assert captured["params"]["blocked_relationship_types"] == [
        "BELONGS_TO_NODESET",
        "HAS_EVENT",
        "HAS_INTEREST_IN",
        "OBSERVED_IN",
        "TARGETS",
    ]


def test_get_chunks_for_entities_excludes_user_owned_entities() -> None:
    adapter = Neo4jAdapter(
        uri="bolt://unused",
        username="unused",
        password="unused",
    )
    captured: dict = {}

    async def _fake_execute_read(cypher: str, params: dict):
        captured["cypher"] = cypher
        captured["params"] = params
        return []

    adapter._execute_read = _fake_execute_read  # type: ignore[method-assign]

    async def _run() -> None:
        await adapter.get_chunks_for_entities(["entity-1"], ["chunk-1"])

    asyncio.run(_run())

    assert "coalesce(trim(e.user_email), '') = ''" in captured["cypher"]
