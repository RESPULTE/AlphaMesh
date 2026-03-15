"""Unit tests for the ChromaDBAdapter."""

import pytest
from langchain_core.documents import Document

from core.memory.stores.chroma_adapter import ChromaDBAdapter


@pytest.mark.asyncio
async def test_upsert_serializes_lists(chroma_adapter_stub):
    adapter: ChromaDBAdapter
    adapter, vectorstore = chroma_adapter_stub
    await adapter.upsert_chunks(
        chunk_ids=["c1"],
        texts=["text"],
        metadatas=[{"companies_involved": ["A", "B"], "nodeset_ids": ["n1"]}],
    )
    metadata = vectorstore.last_payload["documents"][0].metadata
    assert metadata["companies_involved"] == "A,B"
    assert metadata["nodeset_ids"] == "n1"


@pytest.mark.asyncio
async def test_query_deserializes_lists(chroma_adapter_stub):
    adapter, vectorstore = chroma_adapter_stub
    vectorstore.similarity_scores = [
        (
            Document(
                page_content="text",
                metadata={"companies_involved": "A,B", "nodeset_ids": "n1"},
                id="c1",
            ),
            0.1,
        )
    ]
    result = await adapter.query(query_text="hello", n_results=1)
    metadata = result[0][0].metadata
    assert metadata["companies_involved"] == ["A", "B"]
    assert metadata["nodeset_ids"] == ["n1"]


@pytest.mark.asyncio
async def test_update_metadata_serializes(chroma_adapter_stub):
    adapter, vectorstore = chroma_adapter_stub
    await adapter.update_metadata(
        ids=["c1"],
        metadatas=[{"companies_involved": ["A"], "nodeset_ids": ["n1"]}],
    )
    updated = vectorstore.last_payload["documents"][0].metadata
    assert updated["companies_involved"] == "A"
    assert updated["nodeset_ids"] == "n1"


@pytest.mark.asyncio
async def test_query_mmr_returns_scores(chroma_adapter_stub):
    adapter, vectorstore = chroma_adapter_stub
    vectorstore.mmr_docs = [
        Document(
            page_content="text",
            metadata={"companies_involved": "A,B", "nodeset_ids": "n1"},
            id="c1",
        )
    ]
    vectorstore.similarity_scores = [
        (
            Document(
                page_content="text",
                metadata={"companies_involved": "A,B", "nodeset_ids": "n1"},
                id="c1",
            ),
            0.25,
        )
    ]
    result = await adapter.query(query_text="hello", n_results=1, search_type="mmr")
    assert result[0][1] == 0.25
