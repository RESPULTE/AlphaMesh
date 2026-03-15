from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from core.memory.graph.models import DocumentMetadata
from core.memory.retrieval.models import RetrievedChunk
from core.memory.ingestion.ingestor import DualStoreIngestor


@pytest.fixture
def mock_adapters():
    neo4j = AsyncMock()
    neo4j.get_chunk_extraction_status.return_value = {}

    chroma = AsyncMock()
    chroma.get_chunks_with_source_url.return_value = []
    chroma.get_documents_by_ids.return_value = []

    nodeset_manager = AsyncMock()
    nodeset_manager.get_global_financial_events_id.return_value = "global_id"
    nodeset_manager.assign_to_chunk_metadata = MagicMock(side_effect=lambda x, y: x)

    embedding_func = AsyncMock()

    chunker = MagicMock()

    llm = AsyncMock()

    return neo4j, chroma, nodeset_manager, embedding_func, chunker, llm


@pytest.mark.asyncio
async def test_ingest_articles_skips_existing(mock_adapters):
    neo4j, chroma, nodeset_manager, embedding_func, chunker, llm = mock_adapters
    ingestor = DualStoreIngestor(
        neo4j, chroma, chroma, nodeset_manager, embedding_func, chunker, llm=llm
    )

    chroma.get_chunks_with_source_url.return_value = [
        Document(page_content="text", metadata={}, id="c1")
    ]
    chroma.get_documents_by_ids.return_value = [
        Document(page_content="text", metadata={}, id="c1")
    ]

    articles = [{"url": "https://example.com"}]
    chunk_ids, _chunks = await ingestor.ingest_articles(articles)

    assert chunk_ids == ["c1"]
    chunker.chunk_article.assert_not_called()
    neo4j.merge_document_node.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_articles_processes_new(mock_adapters):
    neo4j, chroma, nodeset_manager, embedding_func, chunker, llm = mock_adapters
    ingestor = DualStoreIngestor(
        neo4j, chroma, chroma, nodeset_manager, embedding_func, chunker, llm=llm
    )

    doc_meta = DocumentMetadata(
        document_id="d1",
        title="title",
        source_url="https://example.com/2",
        published_at=datetime.now(timezone.utc),
        companies_involved=["TEST"],
    )
    chunk = RetrievedChunk(
        source="vector",
        document_id="d1",
        chunk_id="c1",
        chunk_index=0,
        text="chunk text",
        article_title="title",
        source_url="https://example.com/2",
        published_at=datetime.now(timezone.utc),
        companies_involved=["TEST"],
    )

    chunker.chunk_article.return_value = (doc_meta, [chunk])
    chroma.get_documents_by_ids.return_value = [
        Document(page_content="chunk text", metadata={}, id="c1")
    ]

    articles = [{"url": "https://example.com/2"}]
    chunk_ids, _chunks = await ingestor.ingest_articles(articles)

    assert chunk_ids == ["c1"]

    neo4j.merge_document_node.assert_called_once()
    neo4j.merge_chunk_node.assert_called_once()
    chroma.upsert_chunks.assert_called_once()
    chroma.get_documents_by_ids.assert_called_once()
    nodeset_manager.get_global_financial_events_id.assert_called()


