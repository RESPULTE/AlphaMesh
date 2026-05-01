import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from core.memory.graph.models import DocumentMetadata
from core.memory.graph.queue.ingestor import ArticleIngestor
from core.memory.retrieval.models import RetrievedChunk


@pytest.fixture
def mock_adapters():
    neo4j = AsyncMock()

    chroma = AsyncMock()
    chroma.get_chunks_with_source_url.return_value = []
    chroma.get_documents_by_ids.return_value = []

    nodeset_manager = AsyncMock()
    nodeset_manager.get_global_financial_events_id.return_value = "global_id"
    nodeset_manager.assign_to_chunk_metadata = MagicMock(side_effect=lambda x, y: x)

    chunker = MagicMock()

    return neo4j, chroma, nodeset_manager, chunker


def test_ingest_articles_skips_existing(mock_adapters):
    neo4j, chroma, nodeset_manager, chunker = mock_adapters
    ingestor = ArticleIngestor(neo4j, chroma, nodeset_manager, chunker)

    chroma.get_chunks_with_source_url.return_value = [
        Document(page_content="text", metadata={}, id="c1")
    ]
    chroma.get_documents_by_ids.return_value = [
        Document(page_content="text", metadata={}, id="c1")
    ]

    articles = [{"url": "https://example.com"}]
    chunk_ids, existing_chunk_ids, involved_chunks = asyncio.run(
        ingestor.ingest_articles(articles)
    )

    assert chunk_ids == []
    assert existing_chunk_ids == ["c1"]
    assert len(involved_chunks) == 1
    chunker.chunk_article.assert_not_called()
    neo4j.merge_document_node.assert_not_called()


def test_ingest_articles_processes_new(mock_adapters):
    neo4j, chroma, nodeset_manager, chunker = mock_adapters
    ingestor = ArticleIngestor(neo4j, chroma, nodeset_manager, chunker)

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
    )

    chunker.chunk_article.return_value = (doc_meta, [chunk])
    chroma.get_documents_by_ids.return_value = [
        Document(page_content="chunk text", metadata={}, id="c1")
    ]

    articles = [{"url": "https://example.com/2"}]
    chunk_ids, existing_chunk_ids, involved_chunks = asyncio.run(
        ingestor.ingest_articles(articles)
    )

    assert chunk_ids == ["c1"]
    assert existing_chunk_ids == []
    assert len(involved_chunks) == 1

    neo4j.merge_document_node.assert_called_once()
    neo4j.merge_chunk_node.assert_called_once()
    chroma.upsert_chunks.assert_called_once()
    nodeset_manager.get_global_financial_events_id.assert_called()
