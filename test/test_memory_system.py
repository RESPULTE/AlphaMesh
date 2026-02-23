"""
test/test_memory_system.py

Unit tests for the AlphaMesh multi-tenant financial memory system.

Tests (24 total):
  1. NodeSet hashing determinism
  2-3. assign_nodeset_from_target — missing target_nodeset
  4-5. assign_nodeset_from_target — invalid target_nodeset
  6-7. assign_nodeset_from_target — valid GLOBAL routing (enum + raw string)
  8-9. assign_nodeset_from_target — valid USER routing (enum + raw string)
  10-12. assign_nodeset_from_target — mixed/edge cases
  13-16. Prompt building
  17-20. Query isolation

No live LLM or DB required — external I/O is mocked.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from core.memory.exceptions import (
    InvalidTargetNodeSetError,
    MissingTargetNodeSetError,
    NodeSetResolutionError,
    QueryError,
)
from core.memory.graph_models import (
    Company,
    FinancialConcept,
    FinancialKnowledgeGraph,
    FinancialReport,
    News,
    NodeSetTarget,
    UserConversation,
)
from core.memory.nodeset_manager import (
    GLOBAL_NODESET_NAME,
    get_user_nodeset_name,
    hash_user_email,
    get_user_nodeset_names,
)
from core.memory.pipeline_tasks import assign_nodeset_from_target
from core.memory.prompts import FINANCIAL_COGNIFY_SYSTEM_PROMPT, build_cognify_prompt
from cognee.modules.engine.models.node_set import NodeSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_nodeset(name: str) -> NodeSet:
    return NodeSet(id=uuid4(), name=name)


def make_chunk_with_entity(entity) -> MagicMock:
    chunk = MagicMock()
    chunk.contains = [entity]
    return chunk


# ---------------------------------------------------------------------------
# 1. Hashing determinism
# ---------------------------------------------------------------------------


class TestHashUserEmail:
    def test_same_email_same_hash(self):
        assert hash_user_email("alice@example.com") == hash_user_email("alice@example.com")

    def test_case_insensitive(self):
        assert hash_user_email("Alice@Example.COM") == hash_user_email("alice@example.com")

    def test_whitespace_stripped(self):
        assert hash_user_email("  alice@example.com  ") == hash_user_email("alice@example.com")

    def test_different_emails_different_hashes(self):
        assert hash_user_email("alice@example.com") != hash_user_email("bob@example.com")

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


# ---------------------------------------------------------------------------
# 2. assign_nodeset_from_target — missing target_nodeset
# ---------------------------------------------------------------------------


class TestAssignNodesetMissing:
    @pytest.mark.asyncio
    async def test_missing_target_nodeset_raises(self):
        entity = Company(ticker="AAPL", name="Apple Inc.", target_nodeset=None)
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")
        chunk = make_chunk_with_entity(entity)

        with pytest.raises(MissingTargetNodeSetError) as exc_info:
            await assign_nodeset_from_target([chunk], user_ns, global_ns)

        assert "Company" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. assign_nodeset_from_target — invalid target_nodeset
# ---------------------------------------------------------------------------


class TestAssignNodesetInvalid:
    @pytest.mark.asyncio
    async def test_invalid_string_raises(self):
        """model_construct bypasses Pydantic so we can inject illegal LLM output."""
        entity = News.model_construct(
            id=uuid4(), headline="Fed hikes", target_nodeset="PUBLIC",
            created_at=0, updated_at=0, ontology_valid=False,
            version=1, topological_rank=0, metadata={"index_fields": []},
            type="News", belongs_to_set=None,
        )
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")

        with pytest.raises(InvalidTargetNodeSetError) as exc_info:
            await assign_nodeset_from_target([make_chunk_with_entity(entity)], user_ns, global_ns)

        assert "PUBLIC" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_string_raises(self):
        entity = FinancialConcept.model_construct(
            id=uuid4(), name="P/E ratio", definition="Price-to-earnings.",
            target_nodeset="",
            created_at=0, updated_at=0, ontology_valid=False,
            version=1, topological_rank=0, metadata={"index_fields": []},
            type="FinancialConcept", belongs_to_set=None,
        )
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")

        with pytest.raises(InvalidTargetNodeSetError):
            await assign_nodeset_from_target([make_chunk_with_entity(entity)], user_ns, global_ns)


# ---------------------------------------------------------------------------
# 4. assign_nodeset_from_target — valid GLOBAL routing
# ---------------------------------------------------------------------------


class TestAssignNodesetGlobal:
    @pytest.mark.asyncio
    async def test_global_enum_routes_to_global_nodeset(self):
        entity = Company(ticker="MSFT", name="Microsoft", target_nodeset=NodeSetTarget.GLOBAL)
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")
        chunk = make_chunk_with_entity(entity)

        result = await assign_nodeset_from_target([chunk], user_ns, global_ns)

        assert entity.belongs_to_set == [global_ns]
        assert result == [chunk]

    @pytest.mark.asyncio
    async def test_global_raw_string_coercion(self):
        """LLM may return raw "GLOBAL" string — must be coerced and accepted."""
        entity = FinancialReport.model_construct(
            id=uuid4(), ticker="TSLA", report_type="10-K",
            content="Annual report.", target_nodeset="GLOBAL",
            created_at=0, updated_at=0, ontology_valid=False,
            version=1, topological_rank=0, metadata={"index_fields": []},
            type="FinancialReport", belongs_to_set=None,
        )
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")

        await assign_nodeset_from_target([make_chunk_with_entity(entity)], user_ns, global_ns)
        assert entity.belongs_to_set == [global_ns]


# ---------------------------------------------------------------------------
# 5. assign_nodeset_from_target — valid USER routing
# ---------------------------------------------------------------------------


class TestAssignNodesetUser:
    @pytest.mark.asyncio
    async def test_user_enum_routes_to_user_nodeset(self):
        entity = UserConversation(
            role="user", content="What is AAPL P/E?", target_nodeset=NodeSetTarget.USER
        )
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")

        await assign_nodeset_from_target([make_chunk_with_entity(entity)], user_ns, global_ns)
        assert entity.belongs_to_set == [user_ns]

    @pytest.mark.asyncio
    async def test_user_lowercase_string_coercion(self):
        """LLM may return lowercase "user" — must be coerced."""
        entity = UserConversation.model_construct(
            id=uuid4(), role="assistant", content="The P/E ratio is 30.",
            target_nodeset="user",
            created_at=0, updated_at=0, ontology_valid=False,
            version=1, topological_rank=0, metadata={"index_fields": []},
            type="UserConversation", belongs_to_set=None,
        )
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")

        await assign_nodeset_from_target([make_chunk_with_entity(entity)], user_ns, global_ns)
        assert entity.belongs_to_set == [user_ns]


# ---------------------------------------------------------------------------
# 6. Mixed entities / edge cases
# ---------------------------------------------------------------------------


class TestAssignNodesetMixed:
    @pytest.mark.asyncio
    async def test_mixed_entities_routed_correctly(self):
        entity_global = Company(ticker="GOOG", name="Alphabet", target_nodeset=NodeSetTarget.GLOBAL)
        entity_user = UserConversation(
            role="user", content="Buy Google?", target_nodeset=NodeSetTarget.USER
        )
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")

        chunk = MagicMock()
        chunk.contains = [entity_global, entity_user]

        await assign_nodeset_from_target([chunk], user_ns, global_ns)
        assert entity_global.belongs_to_set == [global_ns]
        assert entity_user.belongs_to_set == [user_ns]

    @pytest.mark.asyncio
    async def test_empty_contains_returns_unchanged(self):
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")
        chunk = MagicMock()
        chunk.contains = []

        result = await assign_nodeset_from_target([chunk], user_ns, global_ns)
        assert result == [chunk]

    @pytest.mark.asyncio
    async def test_non_financial_entities_skipped(self):
        """Non-FinancialBaseDataPoint items in contains are silently skipped."""
        non_financial = MagicMock()  # not a FinancialBaseDataPoint
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")
        chunk = MagicMock()
        chunk.contains = [non_financial]

        result = await assign_nodeset_from_target([chunk], user_ns, global_ns)
        assert result == [chunk]

    @pytest.mark.asyncio
    async def test_non_list_input_raises(self):
        global_ns = make_nodeset(GLOBAL_NODESET_NAME)
        user_ns = make_nodeset("USER_abc123def456ab")
        with pytest.raises(TypeError):
            await assign_nodeset_from_target("not a list", user_ns, global_ns)  # type: ignore


# ---------------------------------------------------------------------------
# 7. Prompt building
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    def test_base_prompt_contains_rules(self):
        assert "GLOBAL" in FINANCIAL_COGNIFY_SYSTEM_PROMPT
        assert "USER" in FINANCIAL_COGNIFY_SYSTEM_PROMPT
        assert "target_nodeset" in FINANCIAL_COGNIFY_SYSTEM_PROMPT

    def test_user_prompt_includes_email(self):
        prompt = build_cognify_prompt("alice@example.com")
        assert "alice@example.com" in prompt
        assert "GLOBAL" in prompt

    def test_prompt_normalizes_email_case(self):
        assert build_cognify_prompt("Alice@Example.COM") == build_cognify_prompt("alice@example.com")

    def test_invalid_email_raises(self):
        with pytest.raises(ValueError):
            build_cognify_prompt("")
        with pytest.raises(ValueError):
            build_cognify_prompt(None)  # type: ignore


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

        async def mock_search(query_text, query_type, datasets, node_type, node_name, top_k):
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
        assert alice_user != bob_user, "Different users must have different user NodeSets"

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
