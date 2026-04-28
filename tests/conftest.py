"""Pytest fixtures and stubs for AlphaMesh unit tests."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
from langchain_core.documents import Document

from core.memory.stores.chroma_adapter import ChromaDBAdapter


class DummyVectorStore:
    def __init__(self) -> None:
        self.last_payload: Dict[str, Any] = {}
        self.similarity_scores: List[Tuple[Document, float]] = [
            (
                Document(
                    page_content="doc text",
                    metadata={"companies_involved": "A,B", "nodeset_ids": "n1"},
                    id="chunk-1",
                ),
                0.1,
            )
        ]
        self.mmr_docs: List[Document] = [
            Document(
                page_content="doc text",
                metadata={"companies_involved": "A,B", "nodeset_ids": "n1"},
                id="chunk-1",
            )
        ]

    def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        self.last_payload = {"documents": documents, "ids": ids}

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any):
        _ = (query, k, kwargs)
        return self.similarity_scores

    def similarity_search_by_vector_with_relevance_scores(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ):
        _ = (embedding, k, kwargs)
        return self.similarity_scores

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        _ = (query, k, fetch_k, lambda_mult, kwargs)
        return self.mmr_docs

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        _ = (embedding, k, fetch_k, lambda_mult, kwargs)
        return self.mmr_docs

    def get(
        self,
        ids: List[str] | None = None,
        where: Dict[str, Any] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = (where, limit, kwargs)
        if ids is None:
            ids = ["chunk-1"]
        return {
            "ids": ids,
            "documents": ["dummy"] * len(ids),
            "metadatas": [
                {"companies_involved": "A,B", "nodeset_ids": "n1"} for _ in ids
            ],
        }

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        return [
            Document(
                page_content="dummy",
                metadata={"companies_involved": "A,B", "nodeset_ids": "n1"},
                id=id_,
            )
            for id_ in ids
        ]

    def update_documents(self, ids: List[str], documents: List[Document]) -> None:
        self.last_payload = {"ids": ids, "documents": documents}

    def delete(self, ids: List[str] | None = None) -> None:
        self.last_payload = {"deleted": ids or []}


@pytest.fixture
def chroma_adapter_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[ChromaDBAdapter, DummyVectorStore]:
    """Provide a ChromaDBAdapter backed by a dummy vector store."""
    adapter = ChromaDBAdapter(
        collection_name="news_chunks",
        persist_directory=".chroma_test",
        embedding_function=object(),
    )
    vectorstore = DummyVectorStore()

    async def fake_get_vectorstore(*, collection_name=None) -> DummyVectorStore:
        _ = collection_name
        return vectorstore

    monkeypatch.setattr(adapter, "_get_vectorstore", fake_get_vectorstore)
    return adapter, vectorstore
