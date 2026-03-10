import asyncio
import uuid

print("--- TASK 3.1: CHECK EntityRelationship ---")
try:
    from core.memory.conversation_writeback import EntityRelationship, _relationship_id

    rel = EntityRelationship(
        id=_relationship_id("AAPL", "AFFECTS", "Fed Rate Hike"),
        from_name="AAPL",
        from_type="Company",
        relation_type="AFFECTS",
        to_name="Fed Rate Hike",
        to_type="FinancialEvent",
        confidence="high",
        source_conversation_id="test-conv-001",
    )
    assert rel.from_name == "AAPL"
    assert rel.relation_type == "AFFECTS"
    assert rel.confidence == "high"
    assert isinstance(rel.id, uuid.UUID)
    print("PASS: EntityRelationship instantiates correctly")
except Exception as e:
    print(f"FAIL task 3.1: {e}")

print("--- TASK 3.2: CHECK _relationship_id ---")
try:
    from core.memory.conversation_writeback import _relationship_id

    id1 = _relationship_id("AAPL", "AFFECTS", "Fed Rate Hike")
    id2 = _relationship_id("aapl", "AFFECTS", "fed rate hike")
    id3 = _relationship_id("MSFT", "AFFECTS", "Fed Rate Hike")
    assert id1 == id2, "Should be case-insensitive"
    assert id1 != id3, "Different entities must produce different IDs"
    assert isinstance(id1, uuid.UUID)
    print("PASS: _relationship_id is deterministic and case-insensitive")
except Exception as e:
    print(f"FAIL task 3.2: {e}")

print("--- TASK 3.3: CHECK _resolve_entity_nodeset ---")
try:
    from cognee.modules.engine.operations.setup import setup

    from core.memory.conversation_writeback import _resolve_entity_nodeset
    from core.memory.graph_models import Company, FinancialEvent
    from core.memory.nodeset_manager import get_or_create_global_nodeset

    async def test33():
        await setup()
        await get_or_create_global_nodeset()
        company = Company(
            ticker="AAPL",
            name="Apple Inc.",
            description="Test",
            sector="Information Technology",
        )
        await _resolve_entity_nodeset(company)
        assert (
            len(company.belongs_to_set) >= 1
        ), "Company should have at least global NodeSet"

        event = FinancialEvent(name="Fed Rate Hike", description="Test event")
        await _resolve_entity_nodeset(event)
        assert (
            len(event.belongs_to_set) >= 1
        ), "FinancialEvent should have event NodeSet"
        print("PASS: _resolve_entity_nodeset assigns NodeSets correctly")

    asyncio.run(test33())
except Exception as e:
    print(f"FAIL task 3.3: {e}")

print("--- TASK 3.4: CHECK _should_write_entity ---")
try:
    from core.memory.conversation_writeback import _should_write_entity
    from core.memory.graph_models import Company

    async def test34():
        e = Company(ticker="ZZZZ", name="Fake Co", description="x", sector="Energy")
        e.id = None
        result = await _should_write_entity(e)
        assert result is True, "Entity with no id should always write"
        print("PASS: _should_write_entity returns True for entity with no id")

    asyncio.run(test34())
except Exception as e:
    print(f"FAIL task 3.4: {e}")

print("--- TASK 3.5: CHECK _build_relationship_datapoints ---")
try:
    from core.memory.conversation_writeback import _build_relationship_datapoints

    rels = [
        {
            "from_name": "AAPL",
            "from_type": "Company",
            "relation": "AFFECTS",
            "to_name": "Fed Rate Hike",
            "to_type": "FinancialEvent",
            "confidence": "high",
        },
        {"from_name": "MSFT", "relation": "CORRELATED_WITH", "to_name": "AAPL"},
        {"from_name": "BAD"},
    ]
    result = _build_relationship_datapoints(rels, conversation_id="test-123")
    assert len(result) == 2, f"Expected 2, got {len(result)}"
    assert result[0].relation_type == "AFFECTS"
    assert result[0].confidence == "high"
    assert result[1].relation_type == "CORRELATED_WITH"
    assert result[1].confidence == "low"
    assert result[0].source_conversation_id == "test-123"
    print("PASS: _build_relationship_datapoints filters malformed entries correctly")
except Exception as e:
    print(f"FAIL task 3.5: {e}")

print("--- TASK 3.6: CHECK run_conversation_writeback ---")
try:
    from core.memory.conversation_writeback import run_conversation_writeback

    async def test36():
        await run_conversation_writeback(
            relationships=[], enriched_entities=[], conversation_id="test-empty"
        )
        bad_rels = [{"garbage": True}, None, 42]
        await run_conversation_writeback(
            relationships=bad_rels,
            enriched_entities=[],
            conversation_id="test-bad-rels",
        )
        print("PASS: run_conversation_writeback handles missing/bad inputs correctly")

    asyncio.run(test36())
except Exception as e:
    print(f"FAIL task 3.6: {e}")

print("--- TASK 3.7: CHECK EXPORTS ---")
try:
    from core.memory import EntityRelationship, run_conversation_writeback

    print("PASS: conversation_writeback exports accessible from core.memory")
except Exception as e:
    print(f"FAIL task 3.7: {e}")
