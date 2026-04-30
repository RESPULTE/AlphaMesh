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
