"""Async adapter for local Chroma via LangChain integration."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma

from core.logger import get_logger
from core.memory.graph.models import ChunkNode

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
        self._collection = None
        self._logger = get_logger(__name__)

    async def _get_collection(self):
        """Lazily initialize and return the underlying Chroma collection."""
        if self._collection is not None:
            return self._collection

        try:
            self._vectorstore = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_function,
                persist_directory=self._persist_directory,
            )
            self._collection = self._vectorstore._collection
            return self._collection
        except Exception:
            self._logger.exception("Failed to initialize local Chroma collection.")
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

    async def get_or_create_collection(self, collection_name: str):
        """Return a ChromaDB collection, creating it if needed."""
        _ = collection_name
        return await self._get_collection()

    async def upsert_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[dict],
    ) -> None:
        """Batch upsert chunk vectors with metadata."""
        try:
            collection = await self.get_or_create_collection(self._collection_name)
            embeddings = await self._embedding_function.aembed_documents(texts)
            serialized = [self._serialize_metadata(m) for m in metadatas]
            payload = {
                "ids": chunk_ids,
                "documents": texts,
                "embeddings": embeddings,
                "metadatas": serialized,
            }
            await asyncio.to_thread(collection.upsert, **payload)
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
        try:
            embeddings = await embedder.aembed_documents([text])
            metadata = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "name": cleaned_name,
            }
            await self.upsert_chunks(
                chunk_ids=[entity_id],
                texts=[text],
                embeddings=embeddings,
                metadatas=[metadata],
            )
        except Exception:
            self._logger.exception("Failed to upsert entity embedding.")
            raise

    async def query_entity_similar(
        self,
        text: str,
        entity_type: str,
        n_results: int,
        embedding_func=None,
    ):
        """Query entity embeddings by text within a type filter."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        embedder = embedding_func or self._embedding_function
        if embedder is None:
            raise ValueError("Embedding function is required for entity query.")
        embeddings = await embedder.aembed_documents([cleaned_text])
        return await self.query(
            embeddings[0],
            n_results=n_results,
            where={"entity_type": entity_type},
        )

    async def query(
        self, query_embedding: List[float], n_results: int, where: Optional[dict] = None
    ):
        """Query ChromaDB for nearest neighbors."""
        try:
            collection = await self.get_or_create_collection(self._collection_name)
            payload = {"query_embeddings": [query_embedding], "n_results": n_results}
            if where is not None:
                payload["where"] = where
            result = await asyncio.to_thread(collection.query, **payload)
            if "metadatas" in result and result["metadatas"]:
                result["metadatas"] = [
                    [self._deserialize_metadata(m) for m in batch]
                    for batch in result["metadatas"]
                ]
            return result
        except Exception:
            self._logger.exception("Failed to query ChromaDB.")
            raise

    async def get_by_ids(self, ids: List[str]):
        """Fetch documents by their IDs."""
        try:
            collection = await self.get_or_create_collection(self._collection_name)
            result = await asyncio.to_thread(collection.get, ids=ids)
            if "metadatas" in result and result["metadatas"]:
                result["metadatas"] = [
                    self._deserialize_metadata(m) for m in result["metadatas"]
                ]
            return result
        except Exception:
            self._logger.exception("Failed to fetch documents from ChromaDB.")
            raise

    async def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        try:
            collection = await self.get_or_create_collection(self._collection_name)
            await asyncio.to_thread(collection.delete, ids=ids)
        except Exception:
            self._logger.error("Failed to delete documents from ChromaDB.")

    async def update_metadata(self, ids: List[str], metadatas: List[dict]) -> None:
        """Update metadata for existing documents."""
        try:
            collection = await self.get_or_create_collection(self._collection_name)
            serialized = [self._serialize_metadata(m) for m in metadatas]
            await asyncio.to_thread(collection.update, ids=ids, metadatas=serialized)
        except Exception:
            self._logger.exception("Failed to update ChromaDB metadata.")
            raise

    async def get_chunks_with_source_url(self, source_url: str) -> dict:
        """Check whether any chunks exist with the given source URL."""
        if not source_url:
            return []
        try:
            collection = await self.get_or_create_collection(self._collection_name)
            result = await asyncio.to_thread(
                collection, where={"source_url": source_url}, limit=1
            )
            if not result["ids"] or not result["documents"] or not result["metadatas"]:
                return None

            records = zip(result["ids"], result["documents"], result["metadatas"])
            chunks = []
            for id, document, metadata in records:

                nodeset_ids = metadata.get("nodeset_ids", [])
                if nodeset_ids:
                    if isinstance(nodeset_ids, str):
                        nodeset_ids = [nodeset_ids]
                    del metadata["nodeset_ids"]

                chunk = ChunkNode(
                    id=id, text=document, nodeset_ids=nodeset_ids, **metadata
                )
                chunks.append(chunk)
            return chunks

        except Exception as exec:
            self._logger.exception("Failed to check ChromaDB for source URL.: %s", exec)
            raise
