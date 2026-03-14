from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.ingestion.chunker import ChunkRecord, DocumentMetadata
from core.memory.ingestion.ingestor import DualStoreIngestor


@pytest.fixture
def mock_adapters():
    neo4j = AsyncMock()
    neo4j.get_chunk_extraction_status.return_value = {}

    chroma = AsyncMock()
    chroma.get_chunks_with_source_url.return_value = []

    nodeset_manager = AsyncMock()
    nodeset_manager.get_global_financial_events_id.return_value = "global_id"
    nodeset_manager.assign_to_chunk_metadata = MagicMock(side_effect=lambda x, y: x)

    embedding_func = AsyncMock()
    embedding_func.aembed_documents.return_value = [[0.1, 0.2]]

    chunker = MagicMock()

    llm = AsyncMock()

    return neo4j, chroma, nodeset_manager, embedding_func, chunker, llm


@pytest.mark.asyncio
async def test_ingest_articles_skips_existing(mock_adapters):
    neo4j, chroma, nodeset_manager, embedding_func, chunker, llm = mock_adapters
    ingestor = DualStoreIngestor(
        neo4j, chroma, nodeset_manager, embedding_func, chunker, llm=None
    )

    chroma.get_chunks_with_source_url.return_value = [{"id": "c1"}]

    articles = [{"url": "https://example.com"}]
    chunk_ids = await ingestor.ingest_articles(articles, ["TEST"])

    assert chunk_ids == []
    chunker.chunk_article.assert_not_called()
    neo4j.merge_document_node.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_articles_processes_new(mock_adapters):
    neo4j, chroma, nodeset_manager, embedding_func, chunker, llm = mock_adapters
    ingestor = DualStoreIngestor(
        neo4j, chroma, nodeset_manager, embedding_func, chunker, llm=None
    )

    doc_meta = DocumentMetadata(
        document_id="d1",
        title="title",
        source_url="https://example.com/2",
        published_at=datetime.now(timezone.utc),
        companies_involved=["TEST"],
    )
    chunk = ChunkRecord(
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

    articles = [{"url": "https://example.com/2"}]
    chunk_ids = await ingestor.ingest_articles(articles, ["TEST"])

    assert chunk_ids == ["c1"]

    neo4j.merge_document_node.assert_called_once()
    neo4j.merge_chunk_node.assert_called_once()
    chroma.upsert_chunks.assert_called_once()
    embedding_func.aembed_documents.assert_called_once()
    nodeset_manager.get_global_financial_events_id.assert_called()


@pytest.mark.asyncio
async def test_ingestor_schedules_extraction(mock_adapters):
    neo4j, chroma, nodeset_manager, embedding_func, chunker, llm = mock_adapters
    ingestor = DualStoreIngestor(
        neo4j, chroma, nodeset_manager, embedding_func, chunker, llm
    )

    # We will test _extract_entities_for_chunks directly to avoid background task complexities
    neo4j.get_chunk_extraction_status.return_value = {"c1": "PENDING"}
    chroma.get_by_ids.return_value = {
        "ids": ["c1"],
        "documents": ["chunk text"],
        "metadatas": [{"extraction_status": "PENDING"}],
    }

    from langchain_core.runnables import RunnableLambda

    from core.memory.graph.models import BatchExtractionResult, ChunkExtractionResult

    async def mock_ainvoke(*args, **kwargs):
        return BatchExtractionResult(
            results=[
                ChunkExtractionResult(chunk_id="c1", entities=[], relationships=[])
            ]
        )

    mock_chain = RunnableLambda(func=lambda x: None, afunc=mock_ainvoke)
    from unittest.mock import MagicMock

    llm.with_structured_output = MagicMock(return_value=mock_chain)

    # Run the background extraction coroutine explicitly
    await ingestor._extract_entities_for_chunks(["c1"])

    # Should update extraction status to EXTRACTED
    neo4j.update_chunk_extraction_status.assert_called_with("c1", "EXTRACTED")
    chroma.update_metadata.assert_called_once()
    updated_meta = chroma.update_metadata.call_args[0][1][0]
    assert updated_meta["extraction_status"] == "EXTRACTED"
