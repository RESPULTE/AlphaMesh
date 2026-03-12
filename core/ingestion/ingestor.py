"""Dual-store ingestion pipeline for news articles."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from core.graph.models import ChunkNode, DocumentNode
from core.graph.nodeset_manager import NodeSetManager
from core.ingestion.chunker import ArticleChunker, ChunkRecord, DocumentMetadata
from core.logger import get_logger
from core.stores.chroma_adapter import ChromaDBAdapter
from core.stores.neo4j_adapter import Neo4jAdapter


class DualStoreIngestor:
    """Orchestrates ingestion into Neo4j and ChromaDB."""

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        chroma_adapter: ChromaDBAdapter,
        nodeset_manager: NodeSetManager,
        embedding_func,
        chunker: ArticleChunker,
    ) -> None:
        """Initialize the ingestor with adapters and utilities."""
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._nodeset_manager = nodeset_manager
        self._embedding_func = embedding_func
        self._chunker = chunker
        self._logger = get_logger(__name__)

    async def ingest_articles(self, articles: List[dict], companies_involved: List[str]) -> List[str]:
        """Ingest a batch of articles into both stores."""
        try:
            global_anchor_id = await self._nodeset_manager.get_global_financial_events_id()
            documents: List[DocumentMetadata] = []
            chunks: List[ChunkRecord] = []
            for article in articles:
                doc_meta, chunk_records = self._chunker.chunk_article(
                    article, companies_involved
                )
                documents.append(doc_meta)
                chunks.extend(chunk_records)

            await self._write_document_nodes(documents, global_anchor_id)
            await self._write_chunk_nodes(chunks, global_anchor_id)
            await self._write_vector_chunks(chunks, global_anchor_id)
            return [chunk.chunk_id for chunk in chunks]
        except Exception:
            self._logger.exception("Failed to ingest articles.")
            raise

    async def _write_document_nodes(
        self, docs: List[DocumentMetadata], global_anchor_id: str
    ) -> None:
        """Write document nodes and anchor them to the global node."""
        ingested_at = datetime.now(timezone.utc)
        try:
            for doc in docs:
                node = DocumentNode(
                    id=doc.document_id,
                    title=doc.title,
                    source_url=doc.source_url,
                    published_at=doc.published_at,
                    ingested_at=ingested_at,
                    companies_involved=doc.companies_involved,
                    nodeset_ids=[global_anchor_id],
                )
                await self._neo4j_adapter.merge_document_node(node)
                await self._neo4j_adapter.anchor_document_to_global(
                    doc.document_id, global_anchor_id
                )
                await self._nodeset_manager.assign_to_node(
                    doc.document_id, "Document", global_anchor_id
                )
        except Exception:
            self._logger.exception("Failed to write document nodes.")
            raise

    async def _write_chunk_nodes(
        self, chunks: List[ChunkRecord], global_anchor_id: str
    ) -> None:
        """Write chunk nodes to Neo4j."""
        try:
            for chunk in chunks:
                node = ChunkNode(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    document_id=chunk.document_id,
                    companies_involved=chunk.companies_involved,
                    nodeset_ids=[global_anchor_id],
                    extraction_status="PENDING",
                )
                await self._neo4j_adapter.merge_chunk_node(node)
        except Exception:
            self._logger.exception("Failed to write chunk nodes.")
            raise

    async def _write_vector_chunks(
        self, chunks: List[ChunkRecord], global_anchor_id: str
    ) -> None:
        """Write chunk vectors and metadata to ChromaDB."""
        try:
            embeddings = await self._embedding_func.aembed_documents(
                [chunk.text for chunk in chunks]
            )

            metadatas = []
            for chunk in chunks:
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "article_title": chunk.article_title,
                    "source_url": chunk.source_url,
                    "published_at": chunk.published_at.isoformat(),
                    "chunk_index": chunk.chunk_index,
                    "companies_involved": chunk.companies_involved,
                    "nodeset_ids": [global_anchor_id],
                    "extraction_status": "PENDING",
                }
                metadatas.append(
                    self._nodeset_manager.assign_to_chunk_metadata(
                        metadata, global_anchor_id
                    )
                )

            await self._chroma_adapter.upsert_chunks(
                chunk_ids=[chunk.chunk_id for chunk in chunks],
                texts=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception:
            self._logger.exception("Failed to write vector chunks.")
            raise
