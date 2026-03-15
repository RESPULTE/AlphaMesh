"""Async adapter for local Chroma via LangChain integration."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.logger import get_logger

_LIST_FIELDS = {"companies_involved", "nodeset_ids"}


class ChromaDBAdapter:
    """Encapsulates local Chroma operations via LangChain's Chroma wrapper."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_function=None,
    ) -> None:
        """Initialize the adapter with local persistence settings."""
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._vectorstore: Optional[Chroma] = None
        self._logger = get_logger(__name__)

    async def _get_vectorstore(self) -> Chroma:
        """Lazily initialize and return the underlying Chroma vector store."""
        if self._vectorstore is not None:
            return self._vectorstore

        try:
            self._vectorstore = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_function,
                persist_directory=self._persist_directory,
            )
            return self._vectorstore
        except Exception:
            self._logger.exception("Failed to initialize local Chroma vector store.")
            raise

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize list fields and datetimes for ChromaDB metadata."""
        serialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key in _LIST_FIELDS and isinstance(value, list):
                serialized[key] = ",".join(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value
        return serialized

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize list fields from ChromaDB metadata."""
        deserialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key in _LIST_FIELDS and isinstance(value, str):
                if value.strip() == "":
                    deserialized[key] = []
                else:
                    deserialized[key] = [item.strip() for item in value.split(",")]
            else:
                deserialized[key] = value
        return deserialized

    @staticmethod
    def _doc_id(doc: Document) -> str:
        return doc.id or doc.metadata.get("chunk_id") or ""

    def _with_deserialized_metadata(self, doc: Document) -> Document:
        return Document(
            page_content=doc.page_content,
            metadata=self._deserialize_metadata(doc.metadata or {}),
            id=doc.id,
        )

    async def get_or_create_collection(self, collection_name: str) -> Chroma:
        """Return a Chroma vector store, creating it if needed."""
        _ = collection_name
        return await self._get_vectorstore()

    async def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        """Add documents to Chroma using the vector store API."""
        if not documents:
            return
        try:
            vectorstore = await self._get_vectorstore()
            await asyncio.to_thread(
                vectorstore.add_documents, documents=documents, ids=ids
            )
        except Exception:
            self._logger.exception("Failed to add documents to ChromaDB.")
            raise

    async def upsert_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[dict],
    ) -> None:
        """Batch upsert chunk vectors with metadata."""
        if not self._embedding_function:
            raise ValueError("Embedding function is required for chunk upsert.")
        try:
            documents = [
                Document(page_content=text, metadata=self._serialize_metadata(meta), id=id_)
                for id_, text, meta in zip(chunk_ids, texts, metadatas, strict=False)
            ]
            await self.add_documents(documents, chunk_ids)
        except Exception:
            self._logger.exception("Failed to upsert chunks to ChromaDB.")
            raise

    async def upsert_entity_embedding(
        self,
        entity_id: str,
        name: str,
        description: str,
        entity_type: str,
        embedding_func=None,
    ) -> None:
        """Upsert a single entity embedding into this collection."""
        if not entity_id:
            return
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            return
        cleaned_description = (description or "").strip() or cleaned_name
        text = f"{cleaned_name}. {cleaned_description}"
        embedder = embedding_func or self._embedding_function
        if embedder is None:
            raise ValueError("Embedding function is required for entity upsert.")
        metadata = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "name": cleaned_name,
        }
        await self.upsert_chunks(
            chunk_ids=[entity_id],
            texts=[text],
            metadatas=[metadata],
        )

    async def query_entity_similar(
        self,
        text: str,
        entity_type: str,
        n_results: int,
        embedding_func=None,
    ) -> List[Tuple[Document, float]]:
        """Query entity embeddings by text within a type filter."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return []
        _ = embedding_func
        return await self.query(
            query_text=cleaned_text,
            n_results=n_results,
            search_type="similarity",
            where={"entity_type": entity_type},
        )

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
        """Query ChromaDB for nearest neighbors with scores."""
        if query_text is None and query_embedding is None:
            return []

        vectorstore = await self._get_vectorstore()

        try:
            if search_type == "mmr":
                if query_embedding is not None:
                    mmr_docs = await asyncio.to_thread(
                        vectorstore.max_marginal_relevance_search_by_vector,
                        query_embedding,
                        k=n_results,
                        fetch_k=fetch_k,
                        lambda_mult=lambda_mult,
                        filter=where,
                        where_document=where_document,
                    )
                    scored = await asyncio.to_thread(
                        vectorstore.similarity_search_by_vector_with_relevance_scores,
                        query_embedding,
                        k=fetch_k,
                        filter=where,
                        where_document=where_document,
                    )
                else:
                    mmr_docs = await asyncio.to_thread(
                        vectorstore.max_marginal_relevance_search,
                        query_text,
                        k=n_results,
                        fetch_k=fetch_k,
                        lambda_mult=lambda_mult,
                        filter=where,
                        where_document=where_document,
                    )
                    scored = await asyncio.to_thread(
                        vectorstore.similarity_search_with_score,
                        query_text,
                        k=fetch_k,
                        filter=where,
                        where_document=where_document,
                    )

                score_map = {
                    self._doc_id(doc): score
                    for doc, score in (
                        (self._with_deserialized_metadata(doc), score)
                        for doc, score in scored
                    )
                }

                results: List[Tuple[Document, float]] = []
                for doc in mmr_docs:
                    deserialized = self._with_deserialized_metadata(doc)
                    score = score_map.get(self._doc_id(deserialized))
                    results.append((deserialized, score))
                return results

            if query_embedding is not None:
                scored = await asyncio.to_thread(
                    vectorstore.similarity_search_by_vector_with_relevance_scores,
                    query_embedding,
                    k=n_results,
                    filter=where,
                    where_document=where_document,
                )
            else:
                scored = await asyncio.to_thread(
                    vectorstore.similarity_search_with_score,
                    query_text,
                    k=n_results,
                    filter=where,
                    where_document=where_document,
                )

            return [
                (self._with_deserialized_metadata(doc), score) for doc, score in scored
            ]
        except Exception:
            self._logger.exception("Failed to query ChromaDB.")
            raise

    async def get_by_ids(self, ids: List[str]) -> dict:
        """Fetch documents by their IDs."""
        try:
            vectorstore = await self._get_vectorstore()
            result = await asyncio.to_thread(
                vectorstore.get, ids=ids, include=["documents", "metadatas"]
            )
            metadatas = result.get("metadatas") or []
            result["metadatas"] = [self._deserialize_metadata(m) for m in metadatas]
            return result
        except Exception:
            self._logger.exception("Failed to fetch documents from ChromaDB.")
            raise

    async def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """Fetch LangChain Documents by their IDs."""
        try:
            vectorstore = await self._get_vectorstore()
            docs = await asyncio.to_thread(vectorstore.get_by_ids, ids)
            return [self._with_deserialized_metadata(doc) for doc in docs]
        except Exception:
            self._logger.exception("Failed to fetch documents from ChromaDB.")
            raise

    async def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        try:
            vectorstore = await self._get_vectorstore()
            await asyncio.to_thread(vectorstore.delete, ids=ids)
        except Exception:
            self._logger.error("Failed to delete documents from ChromaDB.")

    async def update_metadata(self, ids: List[str], metadatas: List[dict]) -> None:
        """Update metadata for existing documents."""
        try:
            vectorstore = await self._get_vectorstore()
            existing = await self.get_documents_by_ids(ids)
            metadata_map = {id_: meta for id_, meta in zip(ids, metadatas, strict=False)}
            updated_docs: List[Document] = []
            for doc in existing:
                doc_id = self._doc_id(doc)
                if not doc_id:
                    continue
                update = metadata_map.get(doc_id, {})
                merged = {**(doc.metadata or {}), **(update or {})}
                updated_docs.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=self._serialize_metadata(merged),
                        id=doc_id,
                    )
                )
            if updated_docs:
                await asyncio.to_thread(
                    vectorstore.update_documents,
                    ids=[self._doc_id(doc) for doc in updated_docs],
                    documents=updated_docs,
                )
        except Exception:
            self._logger.exception("Failed to update ChromaDB metadata.")
            raise

    async def get_chunks_with_source_url(self, source_url: str) -> List[Document]:
        """Check whether any chunks exist with the given source URL."""
        if not source_url:
            return []
        try:
            vectorstore = await self._get_vectorstore()
            result = await asyncio.to_thread(
                vectorstore.get,
                where={"source_url": source_url},
                limit=1,
                include=["documents", "metadatas"],
            )
            ids = result.get("ids") or []
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            chunks = []
            for doc_id, content, metadata in zip(ids, docs, metas, strict=False):
                chunks.append(
                    Document(
                        page_content=content or "",
                        metadata=self._deserialize_metadata(metadata or {}),
                        id=doc_id,
                    )
                )
            return chunks

        except Exception as exec:
            self._logger.exception("Failed to check ChromaDB for source URL.: %s", exec)
            raise
