from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.retrieval.models import RetrievedChunk, RewrittenQueries
from core.memory.retrieval.retrieval_service import MemoryRetrievalService


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    # Let's say it returns some RetrievedChunks
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1", text="t1", source="vector", score=0.9, metadata={}
        )
    ]
    return retriever


@pytest.fixture
def mock_reranker():
    reranker = MagicMock()
    reranker.rank.side_effect = lambda chunks: chunks  # just return them
    return reranker


@pytest.mark.asyncio
async def test_memory_retrieval_service_fans_out(mock_retriever, mock_reranker):
    service = MemoryRetrievalService(mock_retriever, mock_reranker)

    queries = RewrittenQueries(
        active_domains=["company", "market"],
        company_query="AAPL",
        sector_query="Tech",  # Not active
        market_query="Market trend",
        knowledge_query=None,
    )

    context = await service.retrieve(queries)

    assert mock_retriever.retrieve.call_count == 2
    mock_retriever.retrieve.assert_any_call("AAPL")
    mock_retriever.retrieve.assert_any_call("Market trend")

    assert len(context.chunks) == 2
    assert context.rewritten_queries == queries

    domains_returned = {c.domain for c in context.chunks}
    assert "company" in domains_returned
    assert "market" in domains_returned


@pytest.mark.asyncio
async def test_memory_retrieval_service_handles_exceptions(
    mock_retriever, mock_reranker
):
    service = MemoryRetrievalService(mock_retriever, mock_reranker)

    queries = RewrittenQueries(
        active_domains=["company", "market"],
        company_query="AAPL",
        sector_query=None,
        market_query="Market trend",
        knowledge_query=None,
    )

    # One succeeds, one fails
    async def side_effect(query):
        if query == "AAPL":
            return [
                RetrievedChunk(
                    chunk_id="c1", text="t1", source="vector", score=0.9, metadata={}
                )
            ]
        else:
            raise Exception("Retriever failed")

    mock_retriever.retrieve.side_effect = side_effect

    context = await service.retrieve(queries)

    # Still succeeds gracefully, returning partial results
    assert len(context.chunks) == 1
    assert context.chunks[0].domain == "company"
