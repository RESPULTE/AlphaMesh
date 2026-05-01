from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from core.memory.user_context_service import InterestCacheEntry, UserContextService
from core.memory.user_interest_models import UserInterestQuerySpec


class FakeAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.rows = []
        self.summary_rows = []
        self.targeted_rows = []
        self.last_targeted_kwargs = None

    async def get_user_interest_data(self, _user_email: str, _nodeset_id: str):
        self.calls += 1
        return list(self.rows)

    async def get_user_interest_domain_summary(
        self, user_email: str, nodeset_id: str, limit: int = 3
    ):
        return list(self.summary_rows)[:limit]

    async def query_user_interest_context(self, **kwargs):
        self.last_targeted_kwargs = dict(kwargs)
        return list(self.targeted_rows)


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


def test_build_targeted_orchestrator_context_fallback_domains_only() -> None:
    adapter = FakeAdapter()
    adapter.summary_rows = [
        {
            "domain": {"domain_type": "investment", "category": "Technology"},
            "positive_weight": 3.2,
            "edge_count": 4,
            "last_changed_at": "2026-04-30T00:00:00+00:00",
        }
    ]
    svc = UserContextService(adapter, FakeNodeSetManager())

    class FakeStructuredLLM:
        async def ainvoke(self, _messages):
            return UserInterestQuerySpec(broad_fallback=True)

    llm = SimpleNamespace(with_structured_output=lambda _schema: FakeStructuredLLM())

    result = asyncio.run(
        svc.build_targeted_orchestrator_context(
            user_email="demo@example.com",
            latest_user_message="what are my interests",
            baseline_user_context_block="USER CONTEXT: baseline",
            portfolio_block="[]",
            llm=llm,
        )
    )

    assert result.query_spec is not None
    assert result.debug_payload["mode"] == "fallback_domains_only"
    assert "Domain Summary" in result.context_block
    assert "investment:Technology" in result.context_block


def test_build_targeted_orchestrator_context_targeted_and_hop_clamped() -> None:
    adapter = FakeAdapter()
    adapter.targeted_rows = [
        {
            "domain": {"domain_type": "investment", "category": "Technology"},
            "edge": {"cumulative_weight": 2.4, "last_changed_at": "2026-04-30T00:00:00+00:00"},
            "entity": {"name": "Apple Inc.", "entity_type": "Company"},
            "stance": "positive",
            "edge_last_changed": "2026-04-30T00:00:00+00:00",
            "expanded_neighbors": [
                {"name": "Microsoft", "entity_type": "Company"},
                {"name": "Technology", "entity_type": "Sector"},
            ],
        }
    ]
    svc = UserContextService(adapter, FakeNodeSetManager())

    class FakeStructuredLLM:
        async def ainvoke(self, _messages):
            return UserInterestQuerySpec(
                domain_type="investment",
                category="Technology",
                target_entities=[{"entity_name": "Apple Inc.", "entity_type": "Company"}],
                hops=99,
            )

    llm = SimpleNamespace(with_structured_output=lambda _schema: FakeStructuredLLM())

    result = asyncio.run(
        svc.build_targeted_orchestrator_context(
            user_email="demo@example.com",
            latest_user_message="focus on my tech names around apple",
            baseline_user_context_block="USER CONTEXT: baseline",
            portfolio_block='[{"ticker":"AAPL"}]',
            llm=llm,
        )
    )

    assert result.debug_payload["mode"] == "targeted"
    assert result.debug_payload["hops"] == 2
    assert "Matched Domain" in result.context_block
    assert "Top Interest Edges" in result.context_block
    assert "Expanded Context (hops=2)" in result.context_block
    assert adapter.last_targeted_kwargs is not None
    assert adapter.last_targeted_kwargs["hops"] == 2


def test_build_targeted_orchestrator_context_risk_intent_includes_negative() -> None:
    adapter = FakeAdapter()
    svc = UserContextService(adapter, FakeNodeSetManager())

    class FakeStructuredLLM:
        async def ainvoke(self, _messages):
            return UserInterestQuerySpec(
                domain_type="investment",
                category="Technology",
                target_entities=[{"entity_name": "Apple", "entity_type": "Company"}],
                risk_or_avoidance_intent=True,
            )

    llm = SimpleNamespace(with_structured_output=lambda _schema: FakeStructuredLLM())
    _ = asyncio.run(
        svc.build_targeted_orchestrator_context(
            user_email="demo@example.com",
            latest_user_message="i want downside risk checks for apple",
            baseline_user_context_block="USER CONTEXT: baseline",
            portfolio_block="[]",
            llm=llm,
        )
    )
    assert adapter.last_targeted_kwargs is not None
    assert adapter.last_targeted_kwargs["risk_or_avoidance_intent"] is True


def test_build_targeted_orchestrator_context_returns_safe_none_on_failure() -> None:
    adapter = FakeAdapter()
    svc = UserContextService(adapter, FakeNodeSetManager())

    class FakeStructuredLLM:
        async def ainvoke(self, _messages):
            raise RuntimeError("llm failure")

    llm = SimpleNamespace(with_structured_output=lambda _schema: FakeStructuredLLM())
    result = asyncio.run(
        svc.build_targeted_orchestrator_context(
            user_email="demo@example.com",
            latest_user_message="check this",
            baseline_user_context_block="USER CONTEXT: baseline",
            portfolio_block="[]",
            llm=llm,
        )
    )

    assert result.context_block == "(none)"
    assert result.debug_payload["mode"] == "error"
