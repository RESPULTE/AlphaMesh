"""Pytest fixtures and stubs for AlphaMesh unit tests."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from core.stores.chroma_adapter import ChromaDBAdapter


class DummyCollection:
    def __init__(self) -> None:
        self.last_payload: Dict[str, Any] = {}
        self.query_result: Dict[str, Any] = {
            "ids": [["chunk-1"]],
            "documents": [["doc text"]],
            "metadatas": [[{"companies_involved": "A,B", "nodeset_ids": "n1"}]],
            "distances": [[0.1]],
        }

    def upsert(self, **payload: Any) -> None:
        self.last_payload = payload

    def query(self, **payload: Any) -> Dict[str, Any]:
        return self.query_result

    def get(self, ids: List[str]) -> Dict[str, Any]:
        return {
            "ids": ids,
            "documents": ["dummy"] * len(ids),
            "metadatas": [{"companies_involved": "A,B", "nodeset_ids": "n1"} for _ in ids],
        }

    def delete(self, ids: List[str]) -> None:
        self.last_payload = {"deleted": ids}

    def update(self, ids: List[str], metadatas: List[dict]) -> None:
        self.last_payload = {"ids": ids, "metadatas": metadatas}


@pytest.fixture
def chroma_adapter_stub(monkeypatch: pytest.MonkeyPatch) -> Tuple[ChromaDBAdapter, DummyCollection]:
    """Provide a ChromaDBAdapter backed by a dummy collection."""
    adapter = ChromaDBAdapter(collection_name="news_chunks", persist_directory=".chroma_test")
    collection = DummyCollection()

    async def fake_get_collection() -> DummyCollection:
        return collection

    monkeypatch.setattr(adapter, "_get_collection", fake_get_collection)
    return adapter, collection
