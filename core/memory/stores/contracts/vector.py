"""Vector-store contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document


class VectorStoreAdapter(ABC):
    """Abstract adapter for vector database operations."""

    @abstractmethod
    async def upsert_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[dict],
    ) -> None:
        pass

    @abstractmethod
    async def upsert_entity_embedding(
        self,
        entity_id: str,
        name: str,
        description: str,
        entity_type: str,
        embedding_func=None,
    ) -> None:
        pass

    @abstractmethod
    async def upsert_entity_embeddings_batch(
        self,
        entities: List[Any],
        embedding_func=None,
    ) -> None:
        pass

    @abstractmethod
    async def query(
        self,
        *,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 4,
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    async def query_entity_similar(
        self,
        text: str,
        entity_type: str,
        n_results: int,
        embedding_func=None,
    ) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    async def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        pass

    @abstractmethod
    async def update_metadata(self, ids: List[str], metadatas: List[dict]) -> None:
        pass

    @abstractmethod
    async def get_chunks_with_source_url(self, source_url: str) -> List[Document]:
        pass

