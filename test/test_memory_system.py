"""

test/test_memory_system.py


Unit tests for the AlphaMesh multi-tenant financial memory system.


Tests:

  1. NodeSet hashing determinism

  2-3. assign_nodesets — valid GLOBAL routing

  4-5. assign_nodesets — valid USER routing

  6-8. assign_nodesets — mixed/edge cases

  9-12. Query isolation


No live LLM or DB required — external I/O is mocked.
"""

from __future__ import annotations


from typing import List

from unittest.mock import AsyncMock, MagicMock, patch

from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest


from core.memory.exceptions import (
    NodeSetResolutionError,
    QueryError,
)

from core.memory.graph_models import (
    ListedStock,
    FinancialConcept,
    FinancialKnowledgeGraph,
    InvestmentThesis,
)

from core.memory.nodeset_manager import (
    GLOBAL_NODESET_NAME,
    get_user_nodeset_name,
    hash_user_email,
    get_user_nodeset_names,
    get_or_create_nodeset,
)

from core.memory.pipeline_tasks import decide_assign_nodeset

from core.memory.prompts import FINANCIAL_COGNIFY_SYSTEM_PROMPT

from cognee.modules.engine.models.node_set import NodeSet


# ---------------------------------------------------------------------------

# Helpers

# ---------------------------------------------------------------------------


def make_nodeset(name: str) -> NodeSet:

    return NodeSet(id=uuid4(), name=name)


def make_chunk_with_entity(
    entity, user_ns: NodeSet = None, global_ns: NodeSet = None
) -> MagicMock:

    chunk = MagicMock()

    chunk.contains = [entity]

    # Mock the document's belongs_to_set, which is populated during classify_documents

    doc_mock = MagicMock()

    doc_node_sets = []

    if global_ns:

        doc_node_sets.append(global_ns)

    if user_ns:

        doc_node_sets.append(user_ns)

    doc_mock.belongs_to_set = doc_node_sets

    chunk.is_part_of = doc_mock

    return chunk


# ---------------------------------------------------------------------------

# 1. Hashing determinism

# ---------------------------------------------------------------------------


class TestHashUserEmail:

    def test_same_email_same_hash(self):

        assert hash_user_email("alice@example.com") == hash_user_email(
            "alice@example.com"
        )

    def test_case_insensitive(self):

        assert hash_user_email("Alice@Example.COM") == hash_user_email(
            "alice@example.com"
        )

    def test_whitespace_stripped(self):

        assert hash_user_email("  alice@example.com  ") == hash_user_email(
            "alice@example.com"
        )

    def test_different_emails_different_hashes(self):

        assert hash_user_email("alice@example.com") != hash_user_email(
            "bob@example.com"
        )

    def test_hash_length(self):

        assert len(hash_user_email("alice@example.com")) == 16

    def test_invalid_email_raises(self):

        with pytest.raises(ValueError):

            hash_user_email("")

        with pytest.raises(ValueError):

            hash_user_email(None)  # type: ignore

    def test_nodeset_name_format(self):

        name = get_user_nodeset_name("alice@example.com")

        assert name.startswith("USER_")

        assert len(name) == len("USER_") + 16

    def test_get_user_nodeset_names_returns_two(self):

        names = get_user_nodeset_names("alice@example.com")

        assert len(names) == 2

        assert GLOBAL_NODESET_NAME in names

        assert get_user_nodeset_name("alice@example.com") in names

    @pytest.mark.asyncio
    async def test_get_or_create_nodeset_uses_cognee_compatible_id_and_normalized_name(
        self,
    ):

        captured = {}

        async def fake_add_data_points(*, data_points):

            captured["nodeset"] = data_points[0]

            return None

        expected_id = uuid5(NAMESPACE_OID, "nodeset:global")

        with patch(
            "core.memory.nodeset_manager.cognee_add_dp",
            side_effect=fake_add_data_points,
        ):

            ns = await get_or_create_nodeset("  global  ")

        assert ns.name == GLOBAL_NODESET_NAME

        assert ns.id == expected_id

        assert captured["nodeset"].name == GLOBAL_NODESET_NAME

        assert captured["nodeset"].id == expected_id


# ---------------------------------------------------------------------------

# 2. assign_nodesets — valid GLOBAL routing

# ---------------------------------------------------------------------------


class TestAssignNodesetGlobal:

    @pytest.mark.asyncio
    async def test_global_entity_routes_to_global_nodeset(self):

        entity = ListedStock.model_construct(
            ticker="MSFT", name="Microsoft", belongs_to_set=None
        )

        global_ns = make_nodeset(GLOBAL_NODESET_NAME)

        user_ns = make_nodeset("USER_abc123def456ab")

        chunk = make_chunk_with_entity(entity, user_ns, global_ns)

        result = await decide_assign_nodeset([chunk], global_ns)

        assert entity.belongs_to_set == [global_ns]

        assert result == [chunk]


# ---------------------------------------------------------------------------

# 3. assign_nodesets — valid USER routing

# ---------------------------------------------------------------------------


class TestAssignNodesetUser:

    @pytest.mark.asyncio
    async def test_user_entity_routes_to_user_nodeset(self):
        import datetime

        entity = InvestmentThesis.model_construct(
            thesis_id="T1",
            summary="test",
            status="Active",
            created_at=datetime.datetime.now(),
            targets=[],
            belongs_to_set=None,
        )

        global_ns = make_nodeset(GLOBAL_NODESET_NAME)

        user_ns = make_nodeset("USER_abc123def456ab")

        chunk = make_chunk_with_entity(entity, user_ns)

        await decide_assign_nodeset([chunk], global_ns)

        assert entity.belongs_to_set == [user_ns]

    @pytest.mark.asyncio
    async def test_user_entity_with_no_user_doc(self):
        """User-specific entity in a GLOBAL document triggers warning and drops down to GLOBAL (or raises)."""
        import datetime

        entity = InvestmentThesis.model_construct(
            thesis_id="T2",
            summary="test",
            status="Active",
            created_at=datetime.datetime.now(),
            targets=[],
            belongs_to_set=None,
        )

        global_ns = make_nodeset(GLOBAL_NODESET_NAME)

        # Pass only global_ns to the chunk to simulate a public document

        chunk = make_chunk_with_entity(entity, global_ns=global_ns)

        await decide_assign_nodeset([chunk], global_ns)

        # Should be overridden to GLOBAL since document is GLOBAL

        assert entity.belongs_to_set == [global_ns]


# ---------------------------------------------------------------------------

# 6. Mixed entities / edge cases

# ---------------------------------------------------------------------------


class TestAssignNodesetMixed:

    @pytest.mark.asyncio
    async def test_mixed_entities_routed_correctly(self):

        entity_global = ListedStock.model_construct(
            ticker="GOOG", name="Alphabet", belongs_to_set=None
        )
        import datetime

        entity_user = InvestmentThesis.model_construct(
            thesis_id="T3",
            summary="test",
            status="Active",
            created_at=datetime.datetime.now(),
            targets=[],
            belongs_to_set=None,
        )

        global_ns = make_nodeset(GLOBAL_NODESET_NAME)

        user_ns = make_nodeset("USER_abc123def456ab")

        chunk = make_chunk_with_entity(entity_global, user_ns, global_ns)

        chunk.contains = [entity_global, entity_user]

        await decide_assign_nodeset([chunk], global_ns)

        assert entity_global.belongs_to_set == [global_ns]

        assert entity_user.belongs_to_set == [user_ns]

    @pytest.mark.asyncio
    async def test_empty_contains_returns_unchanged(self):

        global_ns = make_nodeset(GLOBAL_NODESET_NAME)

        chunk = MagicMock()

        chunk.contains = []

        chunk.is_part_of = MagicMock()

        result = await decide_assign_nodeset([chunk], global_ns)

        assert result == [chunk]

    @pytest.mark.asyncio
    async def test_non_financial_entities_skipped(self):
        """Non-FinancialBaseDataPoint items in contains are silently skipped."""

        non_financial = MagicMock()  # not a FinancialBaseDataPoint

        global_ns = make_nodeset(GLOBAL_NODESET_NAME)

        chunk = MagicMock()

        chunk.contains = [non_financial]

        chunk.is_part_of = MagicMock()

        result = await decide_assign_nodeset([chunk], global_ns)

        assert result == [chunk]

    @pytest.mark.asyncio
    async def test_non_list_input_raises(self):

        global_ns = make_nodeset(GLOBAL_NODESET_NAME)

        with pytest.raises(TypeError):

            await decide_assign_nodeset("not a list", global_ns)  # type: ignore


# ---------------------------------------------------------------------------

# 8. Query isolation

# ---------------------------------------------------------------------------


class TestQueryIsolation:

    @pytest.mark.asyncio
    async def test_query_passes_exactly_two_nodeset_names(self):
        """

        search() must receive EXACTLY [GLOBAL, USER_<hash>] via node_name.

        No other user's NodeSet may be included.
        """

        from core.memory.memory_system import FinancialMemorySystem

        system = FinancialMemorySystem()

        system._initialized = True

        captured: dict = {}

        async def mock_search(
            query_text, query_type, datasets, node_type, node_name, top_k
        ):

            captured["node_name"] = node_name

            captured["node_type"] = node_type

            return [{"result": "mock"}]

        with patch("core.memory.memory_system.cognee.search", side_effect=mock_search):

            results = await system.query("alice@example.com", "What is P/E ratio?")

        assert results == [{"result": "mock"}]

        assert GLOBAL_NODESET_NAME in captured["node_name"]

        assert get_user_nodeset_name("alice@example.com") in captured["node_name"]

        assert len(captured["node_name"]) == 2, "Must pass EXACTLY 2 nodeset names"

        assert captured["node_type"] is NodeSet

    @pytest.mark.asyncio
    async def test_different_users_get_different_nodeset_names(self):
        """Alice and Bob's node_name lists must not overlap on the user nodeset."""

        alice_names = get_user_nodeset_names("alice@example.com")

        bob_names = get_user_nodeset_names("bob@example.com")

        alice_user = [n for n in alice_names if n != GLOBAL_NODESET_NAME][0]

        bob_user = [n for n in bob_names if n != GLOBAL_NODESET_NAME][0]

        assert (
            alice_user != bob_user
        ), "Different users must have different user NodeSets"

    @pytest.mark.asyncio
    async def test_query_with_empty_text_raises(self):

        from core.memory.memory_system import FinancialMemorySystem

        system = FinancialMemorySystem()

        system._initialized = True

        with pytest.raises(ValueError, match="query_text must not be empty"):

            await system.query("alice@example.com", "")

    @pytest.mark.asyncio
    async def test_query_before_init_raises(self):

        from core.memory.memory_system import FinancialMemorySystem

        from core.memory.exceptions import MemorySystemError

        system = FinancialMemorySystem()

        with pytest.raises(MemorySystemError, match="initialize"):

            await system.query("alice@example.com", "What is ETF?")
