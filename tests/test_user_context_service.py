from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from core.memory.graph.models import UserInvestmentInterestNode
from core.memory.graph.utils import generate_uuid5
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
    expected_id = generate_uuid5(expected_key)
    assert stored_node.id == expected_id


@pytest.mark.asyncio
async def test_entries_have_cached_at():
    """Every entry in the cache has a non-epoch cached_at."""
    adapter = FakeAdapter()
    adapter.get_user_investment_interests = AsyncMock(
        return_value=[
            {
                "node": {
                    "id": "n1",
                    "user_email": "a@b.com",
                    "status": "Interested",
                    "reason": "r",
                    "confidence": "low",
                    "updated_at": None,
                },
                "targets": [],
            }
        ]
    )
    svc = UserContextService(adapter, FakeNodeSetManager())
    await svc.load_for_user("a@b.com")
    entries = svc._cache["a@b.com"]["entries"]
    assert all(e.cached_at.year >= 2024 for e in entries)


@pytest.mark.asyncio
async def test_cache_capped_at_10():
    """Loading 15 investment rows stores at most CACHE_MAX_INTERESTS=10 entries."""
    from core.memory.user_context_service import CACHE_MAX_INTERESTS

    adapter = FakeAdapter()
    rows = [
        {
            "node": {
                "id": f"n{i}",
                "user_email": "a@b.com",
                "status": "Interested",
                "reason": "",
                "confidence": "low",
                "updated_at": None,
            },
            "targets": [],
        }
        for i in range(15)
    ]
    adapter.get_user_investment_interests = AsyncMock(return_value=rows)
    svc = UserContextService(adapter, FakeNodeSetManager())
    await svc.load_for_user("a@b.com")
    assert len(svc._cache["a@b.com"]["entries"]) == CACHE_MAX_INTERESTS


@pytest.mark.asyncio
async def test_entries_ranked_newest_first():
    """Entries with later cached_at appear first in get_formatted_context output."""
    adapter = FakeAdapter()
    now = datetime.now(timezone.utc)
    rows = [
        {
            "node": {
                "id": "old",
                "user_email": "a@b.com",
                "status": "Interested",
                "reason": "old entry",
                "confidence": "low",
                "updated_at": (now - timedelta(days=5)).isoformat(),
            },
            "targets": [{"id": "e1", "name": "OldCo"}],
        },
        {
            "node": {
                "id": "new",
                "user_email": "a@b.com",
                "status": "Interested",
                "reason": "new entry",
                "confidence": "low",
                "updated_at": now.isoformat(),
            },
            "targets": [{"id": "e2", "name": "NewCo"}],
        },
    ]
    adapter.get_user_investment_interests = AsyncMock(return_value=rows)
    svc = UserContextService(adapter, FakeNodeSetManager())
    await svc.load_for_user("a@b.com")
    text = svc.get_formatted_context("a@b.com")
    assert text.index("NewCo") < text.index("OldCo")


@pytest.mark.asyncio
async def test_merge_insert_updates_cache_without_full_reload():
    """schedule_upsert merges the new node into the cache; adapter.investment_calls stays 1."""
    adapter = FakeAdapter()
    svc = UserContextService(adapter, FakeNodeSetManager())
    await svc.load_for_user("a@b.com")
    assert adapter.investment_calls == 1

    node = UserInvestmentInterestNode(
        id="",
        user_email="a@b.com",
        status="Bought",
        reason="new",
        confidence="high",
        updated_at=datetime.now(timezone.utc),
        target_entity_ids=["x"],
    )
    await svc.schedule_upsert(node, "a@b.com")
    assert adapter.investment_calls == 1
    entries = svc._cache["a@b.com"]["entries"]
    assert any(e.node.status == "Bought" for e in entries)


@pytest.mark.asyncio
async def test_merge_evicts_oldest_when_over_cap():
    """After merge-insert that pushes count to 11, oldest entry is evicted."""
    from core.memory.user_context_service import CACHE_MAX_INTERESTS

    adapter = FakeAdapter()
    rows = [
        {
            "node": {
                "id": f"n{i}",
                "user_email": "a@b.com",
                "status": "Interested",
                "reason": "",
                "confidence": "low",
                "updated_at": None,
            },
            "targets": [],
        }
        for i in range(CACHE_MAX_INTERESTS)
    ]
    adapter.get_user_investment_interests = AsyncMock(return_value=rows)
    svc = UserContextService(adapter, FakeNodeSetManager())
    await svc.load_for_user("a@b.com")

    new_node = UserInvestmentInterestNode(
        id="",
        user_email="a@b.com",
        status="Bought",
        reason="extra",
        confidence="high",
        updated_at=datetime.now(timezone.utc),
        target_entity_ids=["z"],
    )
    await svc.schedule_upsert(new_node, "a@b.com")
    assert len(svc._cache["a@b.com"]["entries"]) == CACHE_MAX_INTERESTS


@pytest.mark.asyncio
async def test_formatted_context_includes_cached_timestamp():
    """get_formatted_context output contains a 'cached' timestamp string."""
    adapter = FakeAdapter()
    rows = [
        {
            "node": {
                "id": "n1",
                "user_email": "a@b.com",
                "status": "Bought",
                "reason": "long hold",
                "confidence": "high",
                "updated_at": None,
            },
            "targets": [{"id": "e1", "name": "AAPL"}],
        },
    ]
    adapter.get_user_investment_interests = AsyncMock(return_value=rows)
    svc = UserContextService(adapter, FakeNodeSetManager())
    await svc.load_for_user("a@b.com")
    text = svc.get_formatted_context("a@b.com")
    assert "cached" in text
    assert "AAPL" in text
