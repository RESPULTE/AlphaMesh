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
    assert captured["params"]["hops"] == 2
    assert captured["params"]["include_negative"] is False
    assert captured["params"]["domain_type"] == "investment"
    assert captured["params"]["category"] == "Technology"
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
