"""Unit tests for the ChromaDBAdapter."""

import pytest

from core.memory.stores.chroma_adapter import ChromaDBAdapter


@pytest.mark.asyncio
async def test_upsert_serializes_lists(chroma_adapter_stub):
    adapter: ChromaDBAdapter
    adapter, collection = chroma_adapter_stub
    await adapter.upsert_chunks(
        chunk_ids=["c1"],
        texts=["text"],
        embeddings=[[0.1]],
        metadatas=[{"companies_involved": ["A", "B"], "nodeset_ids": ["n1"]}],
    )
    metadata = collection.last_payload["metadatas"][0]
    assert metadata["companies_involved"] == "A,B"
    assert metadata["nodeset_ids"] == "n1"


@pytest.mark.asyncio
async def test_query_deserializes_lists(chroma_adapter_stub):
    adapter, collection = chroma_adapter_stub
    collection.query_result = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[{"companies_involved": "A,B", "nodeset_ids": "n1"}]],
        "distances": [[0.1]],
    }
    result = await adapter.query([0.1, 0.2], n_results=1)
    metadata = result["metadatas"][0][0]
    assert metadata["companies_involved"] == ["A", "B"]
    assert metadata["nodeset_ids"] == ["n1"]


@pytest.mark.asyncio
async def test_update_metadata_serializes(chroma_adapter_stub):
    adapter, collection = chroma_adapter_stub
    await adapter.update_metadata(
        ids=["c1"],
        metadatas=[{"companies_involved": ["A"], "nodeset_ids": ["n1"]}],
    )
    updated = collection.last_payload["metadatas"][0]
    assert updated["companies_involved"] == "A"
    assert updated["nodeset_ids"] == "n1"
