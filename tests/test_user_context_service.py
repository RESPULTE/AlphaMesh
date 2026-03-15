from datetime import datetime, timezone

import pytest

from core.memory.graph.models import ENTITY_NAMESPACE, UserInvestmentInterestNode
from core.memory.user_context_service import UserContextService


class FakeAdapter:
    def __init__(self):
        self.investment_calls = 0
        self.learning_calls = 0
        self.upserted_nodes = []

    async def get_user_investment_interests(self, user_email: str):
        self.investment_calls += 1
        return []

    async def get_user_learning_interests(self, user_email: str):
        self.learning_calls += 1
        return []

    async def upsert_user_connected_nodes(self, node, nodeset_id: str):
        self.upserted_nodes.append((node, nodeset_id))


class FakeNodeSetManager:
    async def get_or_create_user_nodeset(self, user_email: str):
        return ("USER_test", "nodeset-1")


@pytest.mark.asyncio
async def test_load_for_user_caches_results():
    adapter = FakeAdapter()
    nodeset_manager = FakeNodeSetManager()
    svc = UserContextService(adapter, nodeset_manager)

    ctx1 = await svc.load_for_user("User@Example.com")
    ctx2 = await svc.load_for_user("user@example.com")

    assert ctx1 == ctx2
    assert adapter.investment_calls == 1
    assert adapter.learning_calls == 1


@pytest.mark.asyncio
async def test_get_formatted_context_empty():
    adapter = FakeAdapter()
    nodeset_manager = FakeNodeSetManager()
    svc = UserContextService(adapter, nodeset_manager)

    await svc.load_for_user("user@example.com")
    result = svc.get_formatted_context("user@example.com")
    assert result == "USER CONTEXT: None"


@pytest.mark.asyncio
async def test_schedule_upsert_deterministic_id():
    adapter = FakeAdapter()
    nodeset_manager = FakeNodeSetManager()
    svc = UserContextService(adapter, nodeset_manager)

    node = UserInvestmentInterestNode(
        id="",
        user_email="User@Example.com",
        status="Interested",
        reason="testing",
        confidence="high",
        updated_at=datetime.now(timezone.utc),
        target_entity_ids=["b", "a"],
    )

    await svc.schedule_upsert(node, "User@Example.com")

    assert len(adapter.upserted_nodes) == 1
    stored_node, nodeset_id = adapter.upserted_nodes[0]
    assert nodeset_id == "nodeset-1"
    expected_key = "user@example.com|Interested|UserInvestmentInterestNode|a,b"
    expected_id = str(__import__("uuid").uuid5(ENTITY_NAMESPACE, expected_key))
    assert stored_node.id == expected_id
