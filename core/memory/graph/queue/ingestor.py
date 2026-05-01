"""
core/memory/graph/queue/ingestor.py

Dual-store ingestion pipeline for news articles.

ArticleIngestor's single responsibility: ingest articles into Neo4j + ChromaDB.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from core.logger import get_logger
from core.memory.graph.models import DocumentMetadata, DocumentNode
from core.memory.graph.nodeset_manager import NodeSetManager
from core.memory.retrieval.models import RetrievedChunk
from core.memory.stores.chroma_adapter import ChromaDBAdapter
from core.memory.stores.neo4j_adapter import Neo4jAdapter

from .chunker import ArticleChunker

logger = get_logger(__name__)


class ArticleIngestor:
    """Orchestrates article ingestion into Neo4j and ChromaDB."""

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        chroma_adapter: ChromaDBAdapter,
        nodeset_manager: NodeSetManager,
        chunker: ArticleChunker,
    ) -> None:
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._nodeset_manager = nodeset_manager
        self._chunker = chunker
        self._logger = get_logger(__name__)

    async def ingest_articles(
        self, articles: List[dict]
    ) -> Tuple[List[str], List[str], List[RetrievedChunk]]:
        """
        Ingest a batch of articles into both stores.

        Returns (new_chunk_ids, existing_chunk_ids, all_involved_chunks).
        """
        try:
            global_anchor_id = (
                await self._nodeset_manager.get_global_financial_events_id()
            )

            classifications = await asyncio.gather(
                *[self._classify_article(article) for article in articles]
            )

            documents_to_ingest: List[DocumentMetadata] = []
            chunks_to_ingest: List[RetrievedChunk] = []
            existing_chunks: List[RetrievedChunk] = []

            for doc_meta, chunks in classifications:
                if doc_meta is None:
                    existing_chunks.extend(chunks)
                else:
                    documents_to_ingest.append(doc_meta)
                    chunks_to_ingest.extend(chunks)

            if documents_to_ingest:
                await self._write_document_nodes(documents_to_ingest, global_anchor_id)
                await self._write_chunk_nodes(chunks_to_ingest, global_anchor_id)
                await self._write_vector_chunks(chunks_to_ingest, global_anchor_id)

            new_chunk_ids = [c.chunk_id for c in chunks_to_ingest if c.chunk_id]
            existing_chunk_ids = [c.chunk_id for c in existing_chunks if c.chunk_id]
            involved_chunks = chunks_to_ingest + existing_chunks

            return new_chunk_ids, existing_chunk_ids, involved_chunks

        except Exception:
            self._logger.exception("Failed to ingest articles.")
            raise

    async def _classify_article(
        self, article: dict
    ) -> Tuple[Optional[DocumentMetadata], List[RetrievedChunk]]:
        source_url = (article.get("url") or "").strip()
        if source_url:
            existing = await self._chroma_adapter.get_chunks_with_source_url(source_url)
            if existing:
                self._logger.info("Skipping duplicate URL: %s", source_url)
                return None, [
                    RetrievedChunk.from_document(doc, source="vector")
                    for doc in existing
                ]
        doc_meta, chunk_records = self._chunker.chunk_article(article)
        return doc_meta, chunk_records

    async def _write_document_nodes(
        self, docs: List[DocumentMetadata], global_anchor_id: str
    ) -> None:
        ingested_at = datetime.now(timezone.utc)

        async def _write_one(doc: DocumentMetadata) -> None:
            node = DocumentNode(
                id=doc.document_id,
                title=doc.title,
                source_url=doc.source_url,
                published_at=doc.published_at,
                ingested_at=ingested_at,
                nodeset_ids=[global_anchor_id],
            )
            await self._neo4j_adapter.merge_document_node(node)
            await self._nodeset_manager.assign_to_node(
                doc.document_id, "Document", global_anchor_id
            )

        try:
            await asyncio.gather(*[_write_one(doc) for doc in docs])
        except Exception:
            self._logger.exception("Failed to write document nodes.")
            raise

    async def _write_chunk_nodes(
        self, chunks: List[RetrievedChunk], global_anchor_id: str
    ) -> None:
        async def _write_one(chunk: RetrievedChunk) -> None:
            node = chunk.model_copy(
                update={
                    "nodeset_ids": [global_anchor_id],
                    "extraction_status": "PENDING",
                }
            )
            await self._neo4j_adapter.merge_chunk_node(node)

        try:
            await asyncio.gather(*[_write_one(chunk) for chunk in chunks])
        except Exception:
            self._logger.exception("Failed to write chunk nodes.")
            raise

    async def _write_vector_chunks(
        self, chunks: List[RetrievedChunk], global_anchor_id: str
    ) -> None:
        try:
            metadatas = []
            for chunk in chunks:
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "article_title": chunk.article_title,
                    "source_url": chunk.source_url,
                    "published_at": chunk.published_at.isoformat(),
                    "chunk_index": chunk.chunk_index,
                    "nodeset_ids": [global_anchor_id],
                    "extraction_status": "PENDING",
                }
                metadatas.append(
                    self._nodeset_manager.assign_to_chunk_metadata(
                        metadata, global_anchor_id
                    )
                )
            await self._chroma_adapter.upsert_chunks(
                chunk_ids=[c.chunk_id for c in chunks],
                texts=[c.text for c in chunks],
                metadatas=metadatas,
            )
        except Exception:
            self._logger.exception("Failed to write vector chunks.")
            raise
