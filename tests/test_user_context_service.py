from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from core.memory.user_context_service import InterestCacheEntry, UserContextService


class FakeAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.rows = []

    async def get_user_interest_data(self, _user_email: str, _nodeset_id: str):
        self.calls += 1
        return list(self.rows)


class FakeNodeSetManager:
    async def get_or_create_user_nodeset(self, _user_email: str):
        return ("USER_test", "nodeset-1")


def test_load_for_user_caches_results() -> None:
    adapter = FakeAdapter()
    now = datetime.now(timezone.utc).isoformat()
    adapter.rows = [
        {
            "d": {"domain_type": "investment", "category": "Technology"},
            "e": {
                "entity_id": "entity_aapl",
                "cumulative_weight": 1.4,
                "reinforcement_count": 2,
                "invalidation_count": 0,
                "current_stance": "positive",
                "last_changed_at": now,
            },
            "entity": {"id": "entity_aapl", "name": "Apple", "entity_type": "Company"},
            "latest_event": {"source_excerpt": "I like Apple", "observed_at": now},
            "previous_event": None,
        }
    ]

    svc = UserContextService(adapter, FakeNodeSetManager())
    ctx1 = asyncio.run(svc.load_for_user("User@Example.com"))
    ctx2 = asyncio.run(svc.load_for_user("user@example.com"))

    assert ctx1 == ctx2
    assert adapter.calls == 1
    assert len(ctx1.investment_entries) == 1


def test_update_cache_keeps_conflict_timeline_and_nudge_json() -> None:
    adapter = FakeAdapter()
    svc = UserContextService(adapter, FakeNodeSetManager())
    now = datetime.now(timezone.utc)

    positive = InterestCacheEntry(
        kind="investment",
        category="Technology",
        entity_id="entity_aapl",
        entity_name="Apple",
        entity_type="Company",
        cumulative_weight=0.8,
        reinforcement_count=1,
        invalidation_count=0,
        current_stance="positive",
        previous_stance=None,
        last_changed_at=now - timedelta(days=5),
        cached_at=now - timedelta(days=5),
        reason="I like Apple",
    )
    negative = InterestCacheEntry(
        kind="investment",
        category="Technology",
        entity_id="entity_aapl",
        entity_name="Apple",
        entity_type="Company",
        cumulative_weight=0.9,
        reinforcement_count=0,
        invalidation_count=1,
        current_stance="negative",
        previous_stance=None,
        last_changed_at=now,
        cached_at=now,
        reason="I do not like Apple anymore",
    )
    svc.update_cache([positive], "user@example.com")
    svc.update_cache([negative], "user@example.com")

    text = svc.get_formatted_context("user@example.com")
    assert "used to be positive on Apple, now negative" in text
    assert "NUDGE_CANDIDATES_JSON" in text
    assert "\"old_stance\":\"positive\"" in text
    assert "\"new_stance\":\"negative\"" in text


def test_load_for_user_uses_previous_event_for_conflict_render() -> None:
    adapter = FakeAdapter()
    now = datetime.now(timezone.utc)
    adapter.rows = [
        {
            "d": {"domain_type": "learning", "category": "Valuation"},
            "e": {
                "entity_id": "entity_dcf",
                "cumulative_weight": 1.9,
                "reinforcement_count": 1,
                "invalidation_count": 1,
                "current_stance": "negative",
                "last_changed_at": now.isoformat(),
            },
            "entity": {
                "id": "entity_dcf",
                "name": "Discounted Cash Flow",
                "entity_type": "FinancialConcept",
            },
            "latest_event": {"observed_at": now.isoformat(), "source_excerpt": "not interested"},
            "previous_event": {
                "observed_at": (now - timedelta(days=3)).isoformat(),
                "stance": "positive",
            },
        }
    ]
    svc = UserContextService(adapter, FakeNodeSetManager())
    asyncio.run(svc.load_for_user("u@example.com"))
    text = svc.get_formatted_context("u@example.com")
    assert "Discounted Cash Flow" in text
    assert "used to be positive" in text
