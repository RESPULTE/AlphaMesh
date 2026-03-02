"""
test/test_financial_retriever.py

Unit tests for core/memory/financial_retriever.py

Covers:
  - QueryScope enum values
  - rerank_by_recency ordering and missing-timestamp fallback
  - FinancialGraphRetriever fan-out: 3 / 2 / 1 concurrent sub-searches per scope
  - NodeSet isolation — cross-user names never appear in sub-search filters
  - Citation extraction from node attributes
  - Runtime system_prompt_override via memory_system.query() and prompts helper
  - get_search_system_prompt() default vs override

No live LLM or DB required — all external I/O is mocked.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from core.memory.financial_retriever import (
    RERANK_TIMESTAMP_FIELD,
    Citation,
    FinancialGraphRetriever,
    QueryScope,
    RetrievalContext,
    rerank_by_recency,
)
from core.memory.prompts import FINANCIAL_SEARCH_SYSTEM_PROMPT, get_search_system_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(node_id=None, ts: Optional[datetime] = None, url=None, text=None):
    """Build a minimal mock graph node."""
    node = MagicMock()
    node.id = node_id or uuid4()
    node.attributes = {}
    if ts is not None:
        node.attributes[RERANK_TIMESTAMP_FIELD] = ts
    if url:
        node.attributes["url"] = url
    if text:
        node.attributes["text"] = text
    return node


def _make_edge(ts1=None, ts2=None, url=None, text=None, rel="RELATES_TO"):
    """Build a minimal mock Edge with two endpoint nodes."""
    edge = MagicMock()
    edge.node1 = _make_node(ts=ts1, url=url, text=text)
    edge.node2 = _make_node(ts=ts2)
    edge.relationship_name = rel
    edge.attributes = {"relationship_name": rel}
    return edge


def _make_retriever(
    user_nodesets=None,
    scope=QueryScope.MARKET,
    entity_name=None,
    system_prompt=None,
    top_k=10,
):
    return FinancialGraphRetriever(
        user_nodeset_names=user_nodesets or ["Market", "USER_abc123"],
        query_scope=scope,
        entity_name=entity_name,
        system_prompt=system_prompt,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# 1. QueryScope enum
# ---------------------------------------------------------------------------


class TestQueryScope:
    def test_has_company_sector_market(self):
        assert QueryScope.COMPANY == "company"
        assert QueryScope.SECTOR == "sector"
        assert QueryScope.MARKET == "market"

    def test_is_string_enum(self):
        assert isinstance(QueryScope.COMPANY, str)


# ---------------------------------------------------------------------------
# 2. rerank_by_recency
# ---------------------------------------------------------------------------


class TestRerankByRecency:
    def test_orders_by_most_recent_first(self):
        now = datetime(2025, 3, 1, 12, 0, 0)
        old = now - timedelta(days=30)
        newest = now + timedelta(days=1)

        e1 = _make_edge(ts1=old)
        e2 = _make_edge(ts1=now)
        e3 = _make_edge(ts1=newest)

        result = rerank_by_recency([e1, e2, e3], top_k=3)
        assert result == [e3, e2, e1]

    def test_top_k_is_respected(self):
        now = datetime(2025, 3, 1)
        edges = [_make_edge(ts1=now - timedelta(days=i)) for i in range(5)]
        result = rerank_by_recency(edges, top_k=3)
        assert len(result) == 3

    def test_missing_timestamps_sink_to_bottom(self):
        now = datetime(2025, 3, 1)
        with_ts = _make_edge(ts1=now)
        no_ts = _make_edge()  # no RERANK_TIMESTAMP_FIELD in attributes

        result = rerank_by_recency([no_ts, with_ts], top_k=2)
        assert result[0] is with_ts
        assert result[1] is no_ts

    def test_empty_list_returns_empty(self):
        assert rerank_by_recency([], top_k=5) == []

    def test_uses_max_of_both_nodes(self):
        now = datetime(2025, 3, 1)
        recent = now + timedelta(days=5)

        e1 = _make_edge(ts1=now, ts2=None)  # only node1 has ts
        e2 = _make_edge(ts1=None, ts2=recent)  # only node2 has ts, but newer

        result = rerank_by_recency([e1, e2], top_k=2)
        assert result[0] is e2


# ---------------------------------------------------------------------------
# 3. Fan-out concurrency per scope
# ---------------------------------------------------------------------------


class TestFanOut:
    """
    Verify the correct number of sub-searches fire per QueryScope.
    brute_force_triplet_search is mocked to return an empty list.
    We count call count to confirm parallelism shape.
    """

    def _patch_triplet_search(self, return_value=None):
        return patch(
            "core.memory.financial_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            return_value=return_value or [],
        )

    @pytest.mark.asyncio
    async def test_company_scope_fires_3_searches(self):
        with self._patch_triplet_search() as mock_search:
            retriever = _make_retriever(scope=QueryScope.COMPANY, entity_name="AAPL")
            await retriever.get_retrieved_objects(query="Tell me about Apple")
            assert mock_search.call_count == 3

    @pytest.mark.asyncio
    async def test_sector_scope_fires_2_searches(self):
        with self._patch_triplet_search() as mock_search:
            retriever = _make_retriever(scope=QueryScope.SECTOR, entity_name="Energy")
            await retriever.get_retrieved_objects(query="How is Energy performing?")
            assert mock_search.call_count == 2

    @pytest.mark.asyncio
    async def test_market_scope_fires_1_search(self):
        with self._patch_triplet_search() as mock_search:
            retriever = _make_retriever(scope=QueryScope.MARKET)
            await retriever.get_retrieved_objects(query="What happened in the market?")
            assert mock_search.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_results_return_empty_list(self):
        with self._patch_triplet_search(return_value=[]):
            retriever = _make_retriever(scope=QueryScope.COMPANY, entity_name="TSLA")
            result = await retriever.get_retrieved_objects(query="Tesla earnings")
            assert result == []

    @pytest.mark.asyncio
    async def test_results_are_deduped_and_reranked(self):
        now = datetime(2025, 3, 1)
        shared_edge = _make_edge(ts1=now, rel="REL")
        older_edge = _make_edge(ts1=now - timedelta(days=10), rel="REL2")

        # Return same edge from all 3 sub-searches + one unique
        with self._patch_triplet_search(return_value=[shared_edge, older_edge]):
            retriever = _make_retriever(
                scope=QueryScope.COMPANY, entity_name="AAPL", top_k=10
            )
            result = await retriever.get_retrieved_objects(query="Apple stock")
            # Duplicate shared_edge should appear only once
            ids = [(str(e.node1.id), str(e.node2.id)) for e in result]
            assert len(ids) == len(set(ids)) or len(result) <= 2


# ---------------------------------------------------------------------------
# 4. NodeSet isolation
# ---------------------------------------------------------------------------


class TestNodeSetIsolation:
    """
    Verify that node_name arguments passed to brute_force_triplet_search
    always include the user's private nodeset and never include another user's.
    """

    @pytest.mark.asyncio
    async def test_user_nodeset_always_present_market_scope(self):
        captured_node_names: List[List[str]] = []

        async def mock_search(*args, **kwargs):
            captured_node_names.append(kwargs.get("node_name", []))
            return []

        with patch(
            "core.memory.financial_retriever.brute_force_triplet_search",
            side_effect=mock_search,
        ):
            retriever = _make_retriever(
                user_nodesets=["Market", "USER_myuser123"],
                scope=QueryScope.MARKET,
            )
            await retriever.get_retrieved_objects(query="market overview")

        for node_names in captured_node_names:
            assert (
                "USER_myuser123" in node_names
            ), "User nodeset must always be in filter"

    @pytest.mark.asyncio
    async def test_other_user_nodeset_never_present(self):
        captured_node_names: List[List[str]] = []

        async def mock_search(*args, **kwargs):
            captured_node_names.append(kwargs.get("node_name", []))
            return []

        with patch(
            "core.memory.financial_retriever.brute_force_triplet_search",
            side_effect=mock_search,
        ):
            retriever = _make_retriever(
                user_nodesets=["Market", "USER_alice"],
                scope=QueryScope.COMPANY,
                entity_name="TSLA",
            )
            await retriever.get_retrieved_objects(query="Tesla")

        for node_names in captured_node_names:
            assert (
                "USER_bob" not in node_names
            ), "Other user's nodeset must never appear"
            assert "USER_charlie" not in node_names

    @pytest.mark.asyncio
    async def test_sector_name_appended_on_top_of_user_nodeset(self):
        captured_node_names: List[List[str]] = []

        async def mock_search(*args, **kwargs):
            captured_node_names.append(list(kwargs.get("node_name", [])))
            return []

        with patch(
            "core.memory.financial_retriever.brute_force_triplet_search",
            side_effect=mock_search,
        ):
            retriever = _make_retriever(
                user_nodesets=["Market", "USER_alice"],
                scope=QueryScope.SECTOR,
                entity_name="Financials",
            )
            await retriever.get_retrieved_objects(query="Banks")

        # At least one sub-search should include "Financials"
        sector_included = any("Financials" in nn for nn in captured_node_names)
        assert sector_included, "Sector name must be present in at least one sub-search"

        # And user nodeset must still be there in every call
        for nn in captured_node_names:
            assert "USER_alice" in nn


# ---------------------------------------------------------------------------
# 5. Citation extraction
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    @pytest.mark.asyncio
    async def test_citation_attached_when_url_present(self):
        edge = _make_edge(
            url="https://reuters.com/article/123", text="Fed raises rates."
        )
        retriever = _make_retriever()

        with patch.object(
            retriever,
            "resolve_edges_to_text",
            new_callable=AsyncMock,
            return_value="context",
        ):
            context = await retriever.get_context_from_objects(
                query="rates", retrieved_objects=[edge]
            )

        assert isinstance(context, RetrievalContext)
        assert len(context.citations) >= 1
        assert context.citations[0].source_url == "https://reuters.com/article/123"
        assert context.citations[0].chunk_text == "Fed raises rates."

    @pytest.mark.asyncio
    async def test_no_citation_for_nodes_without_url_or_text(self):
        edge = _make_edge()  # no url, no text
        retriever = _make_retriever()

        with patch.object(
            retriever, "resolve_edges_to_text", new_callable=AsyncMock, return_value=""
        ):
            context = await retriever.get_context_from_objects(
                query="test", retrieved_objects=[edge]
            )

        assert context.citations == []

    @pytest.mark.asyncio
    async def test_empty_retrieved_objects_returns_empty_context(self):
        retriever = _make_retriever()
        context = await retriever.get_context_from_objects(
            query="test", retrieved_objects=[]
        )
        assert context.context_text == ""
        assert context.citations == []

    @pytest.mark.asyncio
    async def test_citations_serialised_in_completion_result(self):
        edge = _make_edge(url="https://example.com", text="Some chunk.")
        retriever = _make_retriever()

        retrieval_ctx = RetrievalContext(
            context_text="graph context",
            citations=[
                Citation(
                    source_url="https://example.com",
                    chunk_text="Some chunk.",
                    node_id="n1",
                )
            ],
        )

        with patch.object(
            FinancialGraphRetriever,
            "get_completion_from_context",
            wraps=retriever.get_completion_from_context,
        ):
            # Mock parent completion to return a plain string answer
            with patch(
                "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.get_completion_from_context",
                new_callable=AsyncMock,
                return_value=["The Fed raised rates."],
            ):
                results = await retriever.get_completion_from_context(
                    query="rates",
                    retrieved_objects=[edge],
                    context=retrieval_ctx,
                )

        assert len(results) == 1
        assert "answer" in results[0]
        assert "citations" in results[0]
        assert results[0]["citations"][0]["source_url"] == "https://example.com"


# ---------------------------------------------------------------------------
# 6. Runtime prompt override
# ---------------------------------------------------------------------------


class TestRuntimePromptOverride:
    def test_get_search_system_prompt_returns_default_when_no_override(self):
        prompt = get_search_system_prompt()
        assert prompt == FINANCIAL_SEARCH_SYSTEM_PROMPT

    def test_get_search_system_prompt_returns_override_when_supplied(self):
        custom = "You are a pirate financial analyst."
        result = get_search_system_prompt(override=custom)
        assert result == custom

    def test_get_search_system_prompt_ignores_empty_override(self):
        assert get_search_system_prompt(override="") == FINANCIAL_SEARCH_SYSTEM_PROMPT
        assert (
            get_search_system_prompt(override="   ") == FINANCIAL_SEARCH_SYSTEM_PROMPT
        )

    @pytest.mark.asyncio
    async def test_retriever_passes_system_prompt_to_parent(self):
        """The system_prompt kwarg is forwarded to the parent GraphCompletionRetriever."""
        custom_prompt = "Concise mode only."
        retriever = _make_retriever(system_prompt=custom_prompt)
        assert retriever.system_prompt == custom_prompt

    @pytest.mark.asyncio
    async def test_memory_system_query_passes_override_to_retriever(self):
        """
        FinancialMemorySystem.query() must pass system_prompt_override through
        to get_search_system_prompt() and ultimately to FinancialGraphRetriever.
        """
        from core.memory.memory_system import FinancialMemorySystem

        system = FinancialMemorySystem()
        system._initialized = True

        captured: dict = {}

        class _CapturingRetriever(FinancialGraphRetriever):
            def __init__(self, **kwargs):
                captured.update(kwargs)
                super().__init__(**kwargs)

            async def get_completion(self, query=None, query_batch=None):
                return [{"answer": "ok", "citations": []}]

        with patch(
            "core.memory.memory_system.FinancialGraphRetriever",
            side_effect=lambda **kw: _CapturingRetriever(**kw),
        ):
            await system.query(
                user_email="alice@example.com",
                query_text="What is P/E?",
                system_prompt_override="My custom prompt.",
            )

        # The retriever's system_prompt must be the custom one (after get_search_system_prompt)
        assert captured.get("system_prompt") == "My custom prompt."
