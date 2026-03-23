from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from core.agents.models.base_agent_models import BaseAgentInput
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.memory.retrieval.models import MemoryContext, RetrievedChunk


@pytest.fixture
def mock_service_manager(monkeypatch):
    mock_news_api = MagicMock()
    mock_news_api.get_everything.return_value = {
        "status": "ok",
        "articles": [
            {
                "title": "Apple News",
                "content": "Apple releases new iPad",
                "source": {"name": "TechNews"},
                "url": "http://apple.com",
            }
        ],
    }

    mock_ingestor = AsyncMock()
    mock_ingestor.ingest_articles.return_value = (["c1"], [])

    mock_chroma_adapter = AsyncMock()
    mock_chroma_adapter.get_documents_by_ids.return_value = [
        Document(
            page_content="Apple releases new iPad",
            metadata={"article_title": "Apple News", "source_url": "http://apple.com"},
            id="c1",
        )
    ]

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            text="Apple releases new iPad",
            score=0.9,
            source="vector",
            metadata={"article_title": "Apple News"},
        )
    ]

    mock_reranker = MagicMock()
    mock_reranker.rank.side_effect = lambda chunks: chunks  # Pass-through

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content="Apple has released a new iPad based on the context."
    )

    monkeypatch.setattr(
        "core.agents.news_analysis_agent.service_manager.get_news_api",
        lambda: mock_news_api,
    )
    monkeypatch.setattr(
        "core.agents.news_analysis_agent.service_manager.get_ingestor",
        lambda: mock_ingestor,
    )
    monkeypatch.setattr(
        "core.agents.news_analysis_agent.service_manager.get_chroma_adapter",
        lambda: mock_chroma_adapter,
    )
    monkeypatch.setattr(
        "core.agents.news_analysis_agent.service_manager.get_retriever",
        lambda: mock_retriever,
    )
    monkeypatch.setattr(
        "core.agents.news_analysis_agent.service_manager.get_reranker",
        lambda: mock_reranker,
    )
    monkeypatch.setattr(
        "core.agents.news_analysis_agent.service_manager.get_agent", lambda: mock_llm
    )

    return (
        mock_news_api,
        mock_ingestor,
        mock_chroma_adapter,
        mock_retriever,
        mock_reranker,
        mock_llm,
    )


@pytest.mark.asyncio
async def test_news_agent_mocked_workflow(mock_service_manager):
    agent = NewsAnalysisAgent()

    input_data = BaseAgentInput(
        query="What is new with Apple?",
        vector_query="What is new with Apple?",
        ticker="AAPL",
        start_date=datetime.now(),
        end_date=datetime.now(),
        target_agents=["news_agent"],
        request_requires_agents=True,
    )

    output = await agent.run(input_data)

    assert output.agent_name == "news_agent"
    assert output.analysis == "Apple has released a new iPad based on the context."
    assert len(output.sources) == 1
    assert output.sources[0].title == "Apple News"


@pytest.mark.asyncio
async def test_news_agent_with_memory_task(mock_service_manager):
    agent = NewsAnalysisAgent()

    from core.memory.retrieval.models import RewrittenQueries

    # Pre-resolved memory context
    memory_chunk = RetrievedChunk(
        chunk_id="mc1",
        text="Past memory",
        domain="company",
        source="vector",
        embedding_score=0.9,
        graph_depth=0,
        composite_score=0.9,
        metadata={},
    )

    mock_queries = RewrittenQueries(
        active_domains=["company"],
        company_query="Apple",
        sector_query=None,
        market_query=None,
        knowledge_query=None,
    )
    memory_context = MemoryContext(
        chunks=[memory_chunk], rewritten_queries=mock_queries
    )

    import asyncio

    async def get_memory_context():
        return memory_context

    memory_task = asyncio.create_task(get_memory_context())

    input_data = BaseAgentInput(
        query="What is new with Apple?",
        vector_query="What is new with Apple?",
        ticker="AAPL",
        start_date=datetime.now(),
        end_date=datetime.now(),
        target_agents=["news_agent"],
        request_requires_agents=True,
        memory_task=memory_task,
    )

    output = await agent.run(input_data)

    assert output.agent_name == "news_agent"
    assert "Apple has released a new iPad" in output.analysis
