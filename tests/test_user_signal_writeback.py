from __future__ import annotations

import asyncio

from core.memory.user_signal_writeback import (
    DetectedEntity,
    InvestmentSignal,
    LearningSignal,
    UserSignalPayload,
    _build_interest_relationships,
)


class _FakeResolution:
    def __init__(self, entity_id: str):
        self.entity_id = entity_id


class FakeResolver:
    async def resolve_entity(self, name: str, entity_type: str):
        return _FakeResolution(f"{entity_type}:{name.lower()}")


class FakeNeo4j:
    async def get_entity_category(self, _entity_id: str):
        return "Valuation"


def test_build_interest_relationships_emits_session_and_event_edges() -> None:
    payload = UserSignalPayload(
        user_email="user@example.com",
        conversation_id="conv-1",
        turn_id="turn-1",
        user_message="I like Apple and want to learn DCF.",
        ticker_metadata={"AAPL": {"long_name": "Apple", "sector": "Technology"}},
        investment_signals=[
            InvestmentSignal(
                status="Interested",
                confidence=0.8,
                target_entities=[DetectedEntity(entity_name="AAPL", entity_type="Company")],
            )
        ],
        learning_signals=[
            LearningSignal(
                status="Interested",
                confidence=0.7,
                target_entities=[
                    DetectedEntity(
                        entity_name="Discounted Cash Flow",
                        entity_type="FinancialConcept",
                    )
                ],
            )
        ],
    )

    relationships, cache_entries = asyncio.run(
        _build_interest_relationships(
            payload=payload,
            entity_resolver=FakeResolver(),
            neo4j=FakeNeo4j(),
            nodeset_id="nodeset-1",
        )
    )

    rel_types = {rel["relation"] for rel in relationships}
    assert "HAS_INTEREST_IN" in rel_types
    assert "TARGETS" in rel_types
    assert "HAS_EVENT" in rel_types
    assert "OBSERVED_IN" in rel_types
    assert any(rel.get("to_type") == "SessionNode" for rel in relationships)
    assert any(rel.get("to_type") == "UserInterestEvent" for rel in relationships)
    assert {entry.kind for entry in cache_entries} == {"investment", "learning"}
