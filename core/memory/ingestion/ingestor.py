"""
core/memory/ingestion/ingestor.py

Dual-store ingestion pipeline for news articles.

Changes from previous version
----------------------------------------------------------------------------
- _upsert_graph_to_neo4j()  ? REMOVED (moved to Neo4jAdapter + ConversationQueue)
- _resolve_entity()         ? REMOVED (moved to EntityResolver)
- _resolve_user_node()      ? REMOVED (moved to EntityResolver)
- find_similar_entities()   ? REMOVED (moved to EntityResolver)
- _persist_entity()         ? REMOVED (moved to EntityResolver)
- resolve_entity_id()       ? delegates to injected EntityResolver
- __init__() now receives entity_resolver: EntityResolver instead of embedding_func

DualStoreIngestor's single responsibility: ingest articles into Neo4j + ChromaDB.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from core.logger import get_logger
from core.memory.graph.entity_resolver import EntityResolver
from core.memory.graph.extraction_prompts import CHUNK_EXTRACTION_PROMPT
from core.memory.graph.models import (
    FINANCIAL_CONCEPT_CATEGORIES,
    BatchExtractionResult,
    ChunkExtractionResult,
    DocumentMetadata,
    DocumentNode,
    EntityNode,
)
from core.memory.graph.nodeset_manager import NodeSetManager
from core.memory.graph.utils import canonical_entity_id, normalize_relationship_type
from core.memory.ingestion.chunker import ArticleChunker
from core.memory.retrieval.models import RetrievedChunk
from core.memory.stores.chroma_adapter import ChromaDBAdapter
from core.memory.stores.neo4j_adapter import Neo4jAdapter

EXTRACTION_SEMAPHORE = asyncio.Semaphore(settings.EXTRACTION_MAX_CONCURRENCY)

logger = get_logger(__name__)

CHUNK_EXTRACTION_USER_TEMPLATE = (
    "Extract entities and relationships from the following news chunks. "
    "Each chunk is labeled with [CHUNK_ID: ...].\n\n"
    "{chunk_blocks}\n\n"
)


def build_extraction_prompt() -> ChatPromptTemplate:
    """Build the chat prompt template for chunk extraction."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", CHUNK_EXTRACTION_PROMPT),
            ("user", CHUNK_EXTRACTION_USER_TEMPLATE),
        ]
    )


class DualStoreIngestor:
    """Orchestrates article ingestion into Neo4j and ChromaDB."""

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        chroma_adapter: ChromaDBAdapter,
        entity_chroma_adapter: Optional[ChromaDBAdapter],
        nodeset_manager: NodeSetManager,
        entity_resolver: EntityResolver,  # replaces embedding_func
        chunker: ArticleChunker,
        llm=None,
    ) -> None:
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._entity_chroma_adapter = entity_chroma_adapter
        self._nodeset_manager = nodeset_manager
        self._entity_resolver = entity_resolver
        self._chunker = chunker
        self._llm = llm
        self._logger = get_logger(__name__)

    # Public: article ingestion

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

    async def extract_entities_for_chunks(
        self, chunk_ids: List[str], force: bool = False
    ) -> List[EntityNode]:
        """Extract entities for the given chunks. Already-EXTRACTED chunks skipped."""
        if not chunk_ids:
            return []
        if self._llm is None:
            self._logger.warning("Entity extraction requested but no LLM configured.")
            return []
        try:
            return await self._extract_entities_for_chunks(chunk_ids, force=force)
        except Exception:
            self._logger.exception("Entity extraction failed for chunks.")
            return []

    # Private: write helpers

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

    async def _upsert_with_retry(
        self,
        coro_factory,
        chunk_id: Optional[str] = None,
        max_attempts: int = settings.EXTRACTION_NEO4J_RETRY_ATTEMPTS,
    ) -> bool:
        for attempt in range(max_attempts):
            try:
                await coro_factory()
                return True
            except Exception:
                if attempt == max_attempts - 1:
                    self._logger.error(
                        "Neo4j upsert failed after %d attempts for chunk %s",
                        max_attempts,
                        chunk_id,
                    )
                    await self._mark_chunk_pending(chunk_id)
                    return False
                await asyncio.sleep(2**attempt)
        return False

    async def _mark_chunk_pending(self, chunk_id: Optional[str]) -> None:
        if not chunk_id:
            return
        try:
            await self._neo4j_adapter.update_chunk_extraction_status(
                chunk_id, "PENDING"
            )
        except Exception:
            self._logger.exception(
                "Failed to mark chunk %s pending in Neo4j.", chunk_id
            )
        try:
            docs = await self._chroma_adapter.get_documents_by_ids([chunk_id])
            if not docs:
                return
            meta = dict(docs[0].metadata or {})
            meta["extraction_status"] = "PENDING"
            await self._chroma_adapter.update_metadata([chunk_id], [meta])
        except Exception:
            self._logger.exception(
                "Failed to mark chunk %s pending in ChromaDB.", chunk_id
            )

    async def _mark_chunk_extracted_in_chroma(self, chunk_id: Optional[str]) -> None:
        if not chunk_id:
            return
        try:
            docs = await self._chroma_adapter.get_documents_by_ids([chunk_id])
            if not docs:
                return
            meta = dict(docs[0].metadata or {})
            meta["extraction_status"] = "EXTRACTED"
            await self._chroma_adapter.update_metadata([chunk_id], [meta])
        except Exception:
            self._logger.exception(
                "Failed to mark chunk %s extracted in ChromaDB.", chunk_id
            )

    async def _link_financial_concept_categories(self, entity: EntityNode) -> None:
        """Link FinancialConcept entities to 1-3 category nodes."""
        if entity.entity_type != "FinancialConcept":
            return

        raw_categories = list(entity.concept_categories or [])
        if not raw_categories:
            self._logger.warning(
                "FinancialConcept '%s' missing concept_categories from LLM output",
                entity.name,
            )
            return

        allowed = set(FINANCIAL_CONCEPT_CATEGORIES)
        categories = [c for c in raw_categories if c in allowed]
        if not categories:
            self._logger.warning(
                "FinancialConcept '%s' returned unknown categories: %s",
                entity.name,
                raw_categories,
            )
            return

        if len(categories) > 3:
            categories = categories[:3]

        for category in categories:
            category_id = canonical_entity_id(category, "FinancialConceptCategory")
            try:
                await self._neo4j_adapter.merge_relationship(
                    entity.id,
                    category_id,
                    "BELONGS_TO",
                    {
                        "relationship_type": "BELONGS_TO",
                        "source_agent": "concept_taxonomy",
                        "confidence": 1.0,
                    },
                )
            except Exception:
                self._logger.exception(
                    "Failed to link FinancialConcept '%s' to category '%s'",
                    entity.name,
                    category,
                )

    # Private: entity extraction pipeline

    async def _extract_entities_for_chunks(
        self, chunk_ids: List[str], force: bool = False
    ) -> List[EntityNode]:
        if not chunk_ids:
            return []

        if force:
            pending_chunk_ids = list(chunk_ids)
        else:
            status_map = await self._neo4j_adapter.get_chunk_extraction_status(
                chunk_ids
            )
            pending_chunk_ids = [
                cid for cid in chunk_ids if status_map.get(cid) == "PENDING"
            ]

        if not pending_chunk_ids:
            return []

        chunk_docs = await self._chroma_adapter.get_documents_by_ids(pending_chunk_ids)
        chunk_lookup: Dict[str, Dict] = {}
        for doc in chunk_docs:
            cid = doc.id or (doc.metadata or {}).get("chunk_id")
            if cid:
                chunk_lookup[cid] = {
                    "text": doc.page_content or "",
                    "metadata": doc.metadata or {},
                }

        chunks_to_process = [
            {"chunk_id": cid, **chunk_lookup[cid]}
            for cid in pending_chunk_ids
            if cid in chunk_lookup
        ]
        if not chunks_to_process:
            return []

        prompt = build_extraction_prompt()
        extraction_chain = prompt | self._llm.with_structured_output(
            BatchExtractionResult
        )

        async def _extract_batch(chunks: List[dict]) -> BatchExtractionResult:
            chunk_blocks = "\n\n".join(
                f"[CHUNK_ID:{c['chunk_id']}|{c['text']}]" for c in chunks
            )
            async with EXTRACTION_SEMAPHORE:
                return await extraction_chain.ainvoke({"chunk_blocks": chunk_blocks})

        batch_size = max(settings.EXTRACTION_BATCH_SIZE, 1)
        batches = [
            chunks_to_process[i : i + batch_size]
            for i in range(0, len(chunks_to_process), batch_size)
        ]
        batch_results = await asyncio.gather(*[_extract_batch(b) for b in batches])

        results: List[ChunkExtractionResult] = [
            r
            for batch in batch_results
            for r in batch.results
            if r.chunk_id and r.chunk_id in chunk_lookup
        ]

        deduped_entities: Dict[str, EntityNode] = {}

        for result in results:
            chunk_failed = False
            local_id_map: Dict[str, EntityNode] = {}
            chunk_entities: Dict[str, EntityNode] = {}

            for entity in result.entities:
                if not entity.description:
                    entity.description = entity.name
                canonical_id = canonical_entity_id(entity.name, entity.entity_type)
                local_key = entity.local_id or canonical_id

                # Use EntityResolver for persistence (replaces direct _resolve_entity calls)
                resolution = await self._entity_resolver.resolve_entity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    props=entity,
                )
                if not resolution.entity_id:
                    chunk_failed = True
                    break
                resolved_id = resolution.entity_id

                entity.id = resolved_id
                local_id_map[local_key] = entity
                chunk_entities[resolved_id] = entity

                if not await self._upsert_with_retry(
                    lambda cid=result.chunk_id, eid=resolved_id: (
                        self._neo4j_adapter.merge_relationship(
                            cid, eid, "MENTIONS_ENTITY", {"confidence": 1.0}
                        )
                    ),
                    result.chunk_id,
                ):
                    chunk_failed = True
                    break
                await self._link_financial_concept_categories(entity)

            if chunk_failed:
                continue

            for rel in result.relationships:
                src = local_id_map.get(rel.source_entity_local_id)
                tgt = local_id_map.get(rel.target_entity_local_id)
                if not src or not tgt:
                    self._logger.warning(
                        "Skipping relationship with unresolved entities: %s -> %s",
                        rel.source_entity_local_id,
                        rel.target_entity_local_id,
                    )
                    continue
                rel_type = normalize_relationship_type(rel.relationship_type)
                if not await self._upsert_with_retry(
                    lambda s=src, t=tgt, rt=rel_type, c=rel.confidence: (
                        self._neo4j_adapter.merge_relationship(
                            s.id,
                            t.id,
                            rt,
                            {
                                "relationship_type": rt,
                                "source_chunk_id": result.chunk_id,
                                "confidence": c,
                            },
                        )
                    ),
                    result.chunk_id,
                ):
                    chunk_failed = True
                    break

            if chunk_failed:
                continue

            if not await self._upsert_with_retry(
                lambda cid=result.chunk_id: (
                    self._neo4j_adapter.update_chunk_extraction_status(cid, "EXTRACTED")
                ),
                result.chunk_id,
            ):
                continue
            await self._mark_chunk_extracted_in_chroma(result.chunk_id)

            for eid, entity in chunk_entities.items():
                deduped_entities[eid] = entity

        return list(deduped_entities.values())
