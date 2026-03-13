"""Dual-store ingestion pipeline for news articles."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import List

from core.config import settings
from core.graph.extraction_prompts import build_extraction_prompt
from core.graph.models import (
    ENTITY_NAMESPACE,
    BatchExtractionResult,
    ChunkExtractionResult,
    ChunkNode,
    DocumentNode,
)
from core.graph.nodeset_manager import NodeSetManager
from core.ingestion.chunker import ArticleChunker, ChunkRecord, DocumentMetadata
from core.logger import get_logger
from core.stores.chroma_adapter import ChromaDBAdapter
from core.stores.neo4j_adapter import Neo4jAdapter

EXTRACTION_SEMAPHORE = asyncio.Semaphore(settings.EXTRACTION_MAX_CONCURRENCY)


class DualStoreIngestor:
    """Orchestrates ingestion into Neo4j and ChromaDB."""

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        chroma_adapter: ChromaDBAdapter,
        nodeset_manager: NodeSetManager,
        embedding_func,
        chunker: ArticleChunker,
        llm=None,
    ) -> None:
        """Initialize the ingestor with adapters and utilities."""
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._nodeset_manager = nodeset_manager
        self._embedding_func = embedding_func
        self._chunker = chunker
        self._llm = llm
        self._logger = get_logger(__name__)

    async def ingest_articles(
        self, articles: List[dict], companies_involved: List[str]
    ) -> List[str]:
        """Ingest a batch of articles into both stores."""
        try:
            global_anchor_id = (
                await self._nodeset_manager.get_global_financial_events_id()
            )
            documents: List[DocumentMetadata] = []
            chunks: List[ChunkRecord] = []
            for article in articles:
                source_url = (article.get("url") or "").strip()

                existing_chunks = await self._chroma_adapter.get_chunks_with_source_url(
                    source_url
                )
                if source_url and len(existing_chunks) > 0:
                    self._logger.info(
                        "Skipping article with existing source URL: %s", source_url
                    )
                    continue
                doc_meta, chunk_records = self._chunker.chunk_article(
                    article, companies_involved
                )
                documents.append(doc_meta)
                chunks.extend(chunk_records)

            if not documents:
                self._logger.info(
                    "No new articles to ingest after filtering by source URL."
                )
                return []

            await self._write_document_nodes(documents, global_anchor_id)
            await self._write_chunk_nodes(chunks, global_anchor_id)
            await self._write_vector_chunks(chunks, global_anchor_id)
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            self._schedule_extraction(chunk_ids)
            return chunk_ids
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

    def _schedule_extraction(self, chunk_ids: List[str]) -> None:
        """Schedule non-blocking entity extraction for chunk IDs."""
        if not chunk_ids or self._llm is None:
            return
        task = asyncio.create_task(self._extract_entities_for_chunks(chunk_ids))
        task.add_done_callback(self._log_extraction_task_result)

    def _log_extraction_task_result(self, task: asyncio.Task) -> None:
        """Log any exception raised by the background extraction task."""
        try:
            task.result()
        except asyncio.CancelledError:
            self._logger.warning("Background extraction task was cancelled.")
        except Exception:
            self._logger.exception("Background extraction task failed.")

    async def _extract_entities_for_chunks(self, chunk_ids: List[str]) -> None:
        """Extract entities/relationships for pending chunks in the background."""
        if not chunk_ids:
            return

        status_map = await self._neo4j_adapter.get_chunk_extraction_status(chunk_ids)
        pending_chunk_ids = [
            chunk_id
            for chunk_id in chunk_ids
            if status_map.get(chunk_id) == "PENDING"
        ]
        if not pending_chunk_ids:
            return

        chunk_payload = await self._chroma_adapter.get_by_ids(pending_chunk_ids)
        ids = chunk_payload.get("ids") or []
        documents = chunk_payload.get("documents") or []
        metadatas = chunk_payload.get("metadatas") or []

        chunk_lookup = {}
        for idx, chunk_id in enumerate(ids):
            chunk_lookup[chunk_id] = {
                "text": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            }

        chunks_to_process = [
            {"chunk_id": chunk_id, **chunk_lookup[chunk_id]}
            for chunk_id in pending_chunk_ids
            if chunk_id in chunk_lookup
        ]
        if not chunks_to_process:
            return

        prompt = build_extraction_prompt()
        extraction_chain = prompt | self._llm.with_structured_output(
            BatchExtractionResult
        )

        async def _extract_batch(chunks: List[dict]) -> BatchExtractionResult:
            chunk_blocks = "\n\n".join(
                [f"[CHUNK_ID: {chunk['chunk_id']}]\n{chunk['text']}" for chunk in chunks]
            )
            async with EXTRACTION_SEMAPHORE:
                return await extraction_chain.ainvoke({"chunk_blocks": chunk_blocks})

        batch_size = max(settings.EXTRACTION_BATCH_SIZE, 1)
        batches = [
            chunks_to_process[i : i + batch_size]
            for i in range(0, len(chunks_to_process), batch_size)
        ]
        batch_results = await asyncio.gather(
            *[_extract_batch(batch) for batch in batches]
        )

        results: List[ChunkExtractionResult] = []
        for batch in batch_results:
            results.extend(batch.results)

        validated_results: List[ChunkExtractionResult] = []
        for result in results:
            if not result.chunk_id:
                self._logger.warning("Skipping extraction result with missing chunk_id.")
                continue
            if result.chunk_id not in chunk_lookup:
                self._logger.warning(
                    "Skipping extraction result for unknown chunk_id: %s",
                    result.chunk_id,
                )
                continue
            validated_results.append(result)

        for result in validated_results:
            local_id_map = {}
            for entity in result.entities:
                canonical_id = str(
                    uuid.uuid5(
                        ENTITY_NAMESPACE,
                        f"{entity.name.lower()}::{entity.entity_type.lower()}",
                    )
                )
                local_key = entity.local_id or canonical_id
                entity.id = canonical_id
                local_id_map[local_key] = entity
                await self._neo4j_adapter.merge_entity_node(entity)
                await self._neo4j_adapter.merge_relationship(
                    result.chunk_id,
                    entity.id,
                    "MENTIONS_ENTITY",
                    {"confidence": 1.0},
                )

            for rel in result.relationships:
                source_entity = local_id_map.get(rel.source_entity_local_id)
                target_entity = local_id_map.get(rel.target_entity_local_id)
                if not source_entity or not target_entity:
                    continue
                await self._neo4j_adapter.merge_relationship(
                    source_entity.id,
                    target_entity.id,
                    "RELATED_TO",
                    {
                        "relationship_type": rel.relationship_type,
                        "source_chunk_id": result.chunk_id,
                        "confidence": rel.confidence,
                    },
                )

            await self._neo4j_adapter.update_chunk_extraction_status(
                result.chunk_id, "EXTRACTED"
            )

            chunk_meta = chunk_lookup.get(result.chunk_id)
            if chunk_meta:
                updated_metadata = dict(chunk_meta.get("metadata") or {})
                updated_metadata["extraction_status"] = "EXTRACTED"
                await self._chroma_adapter.update_metadata(
                    [result.chunk_id], [updated_metadata]
                )
