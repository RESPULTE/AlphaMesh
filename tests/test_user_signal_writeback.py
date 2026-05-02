from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.memory.graph.utils import generate_uuid5
from core.memory.user_signal_writeback import (
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


@dataclass
class DetectedEntity:
    entity_name: str
    entity_type: str


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
    session_rels = [rel for rel in relationships if rel.get("to_type") == "SessionNode"]
    assert session_rels
    expected_session_id = generate_uuid5("user@example.com::session::conv-1")
    assert all(rel.get("to_name") == expected_session_id for rel in session_rels)
    assert all(
        rel.get("to_node_props", {}).get("conversation_id") == "conv-1"
        for rel in session_rels
    )
    assert all(
        rel.get("to_node_props", {}).get("nodeset_id") == "nodeset-1"
        for rel in session_rels
    )
    assert any(rel.get("to_type") == "UserInterestEvent" for rel in relationships)
    assert {entry.kind for entry in cache_entries} == {"investment", "learning"}


def test_build_interest_relationships_canonicalizes_sector_suffix_name() -> None:
    payload = UserSignalPayload(
        user_email="user@example.com",
        conversation_id="conv-1",
        turn_id="turn-1",
        user_message="I like technology sector.",
        ticker_metadata={},
        investment_signals=[
            InvestmentSignal(
                status="Interested",
                confidence=1.0,
                target_entities=[
                    DetectedEntity(entity_name="Technology Sector", entity_type="Sector")
                ],
            )
        ],
        learning_signals=[],
    )

    relationships, cache_entries = asyncio.run(
        _build_interest_relationships(
            payload=payload,
            entity_resolver=FakeResolver(),
            neo4j=FakeNeo4j(),
            nodeset_id="nodeset-1",
        )
    )

    target_rels = [rel for rel in relationships if rel.get("relation") == "TARGETS"]
    assert target_rels
    assert target_rels[0]["to_type"] == "Sector"
    assert target_rels[0]["to_name"] == "Technology"

    assert cache_entries
    assert cache_entries[0].category == "Technology"
    assert cache_entries[0].entity_name == "Technology"
