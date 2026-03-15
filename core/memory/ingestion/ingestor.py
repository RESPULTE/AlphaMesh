"""Dual-store ingestion pipeline for news articles."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from core.logger import get_logger
from core.memory.graph.extraction_prompts import build_extraction_prompt
from core.memory.graph.models import (
    ENTITY_NAMESPACE,
    BatchExtractionResult,
    ChunkExtractionResult,
    ChunkNode,
    DocumentNode,
    EntityNode,
)
from core.memory.graph.nodeset_manager import NodeSetManager
from core.memory.ingestion.chunker import ArticleChunker, DocumentMetadata
from core.memory.stores.chroma_adapter import ChromaDBAdapter
from core.memory.stores.neo4j_adapter import Neo4jAdapter
from core.services import service_manager

logger = get_logger(__name__)

EXTRACTION_SEMAPHORE = asyncio.Semaphore(settings.EXTRACTION_MAX_CONCURRENCY)
FUZZY_CANDIDATE_THRESHOLD = 0.50
SEMANTIC_MERGE_THRESHOLD = 0.85
VECTOR_TOP_K = 10

_ALLOWED_ENTITY_TYPES = {"Company", "FinancialEvent", "FinancialConcept", "Sector"}


def _canonical_entity_id(name: str, entity_type: str) -> str:
    key = f"{name.lower()}::{entity_type.lower()}"
    return str(uuid.uuid5(ENTITY_NAMESPACE, key))


def _normalize_entity_type(value: Any) -> Optional[str]:
    if not value:
        return None
    entity_type = str(value).strip()
    if entity_type in _ALLOWED_ENTITY_TYPES:
        return entity_type
    return None


def _normalize_entity_name(value: Any) -> str:
    return str(value or "").strip()


def _normalize_entity_description(value: Any, fallback: str) -> str:
    description = str(value or "").strip()
    return description or fallback


def _entity_key(name: str, entity_type: str) -> Tuple[str, str]:
    return (name.lower(), entity_type)


class DualStoreIngestor:
    """Orchestrates ingestion into Neo4j and ChromaDB."""

    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        chroma_adapter: ChromaDBAdapter,
        entity_chroma_adapter: Optional[ChromaDBAdapter],
        nodeset_manager: NodeSetManager,
        embedding_func,
        chunker: ArticleChunker,
        llm=None,
    ) -> None:
        """Initialize the ingestor with adapters and utilities."""
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._entity_chroma_adapter = entity_chroma_adapter
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
            chunks: List[ChunkNode] = []
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
            chunk_ids = [chunk.id for chunk in chunks]
            # self._schedule_extraction(chunk_ids)
            return (chunk_ids, chunks)
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
        self, chunks: List[ChunkNode], global_anchor_id: str
    ) -> None:
        """Write chunk nodes to Neo4j."""
        try:
            for chunk in chunks:
                node = chunk.model_copy(
                    update={
                        "nodeset_ids": [global_anchor_id],
                        "extraction_status": "PENDING",
                    }
                )
                await self._neo4j_adapter.merge_chunk_node(node)
        except Exception:
            self._logger.exception("Failed to write chunk nodes.")
            raise

    async def _write_vector_chunks(
        self, chunks: List[ChunkNode], global_anchor_id: str
    ) -> None:
        """Write chunk vectors and metadata to ChromaDB."""
        try:
            embeddings = await self._embedding_func.aembed_documents(
                [chunk.text for chunk in chunks]
            )

            metadatas = []
            for chunk in chunks:
                metadata = {
                    "chunk_id": chunk.id,
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
                chunk_ids=[chunk.id for chunk in chunks],
                texts=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception:
            self._logger.exception("Failed to write vector chunks.")
            raise

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
            chunk_id for chunk_id in chunk_ids if status_map.get(chunk_id) == "PENDING"
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
                [
                    f"[CHUNK_ID: {chunk['chunk_id']}]\n{chunk['text']}"
                    for chunk in chunks
                ]
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
                self._logger.warning(
                    "Skipping extraction result with missing chunk_id."
                )
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
                if not getattr(entity, "description", None):
                    entity.description = entity.name
                canonical_id = str(
                    uuid.uuid5(
                        ENTITY_NAMESPACE,
                        f"{entity.name.lower()}::{entity.entity_type.lower()}",
                    )
                )
                local_key = entity.local_id or canonical_id
                entity.id = canonical_id
                local_id_map[local_key] = entity
                await self._persist_entity(entity)
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
                    logger.warning(
                        "Skipping relationship with unresolved entities: %s -> %s",
                        rel.source_entity_local_id,
                        rel.target_entity_local_id,
                    )
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

    async def run_conversation_writeback(
        self,
        relationships: List[dict],
        enriched_entities: List[Any],
        conversation_id: str,
        user_email: Optional[str] = None,
    ) -> None:
        """
        Write enriched entities and synthesiser-derived relationships to the graph.

        Called fire-and-forget from OrchestratorAgent._synthesize_node.
        All exceptions are caught and logged -- this function NEVER raises.
        """
        _ = user_email
        try:
            if not enriched_entities and not relationships:
                logger.debug("write_back [%s]: nothing to write.", conversation_id)
                return

            embedding_func = service_manager.get_embedding_func()

            entity_cache: Dict[tuple[str, str], str] = {}

            # --- Step 1: Resolve + dedup enriched entities ---
            for raw_entity in enriched_entities:
                name = _normalize_entity_name(getattr(raw_entity, "name", None))
                entity_type = _normalize_entity_type(
                    getattr(raw_entity, "entity_type", None)
                )
                if not name or not entity_type:
                    continue
                await self._resolve_entity(
                    name,
                    entity_type,
                    entity_cache,
                    raw=raw_entity,
                )
            # --- Step 2: Write relationships ---
            for rel in relationships or []:
                from_name = _normalize_entity_name(rel.get("from_name"))
                to_name = _normalize_entity_name(rel.get("to_name"))
                relation_type = str(rel.get("relation", "")).strip()
                confidence = str(rel.get("confidence", "low")).strip() or "low"

                from_type = _normalize_entity_type(rel.get("from_type"))
                to_type = _normalize_entity_type(rel.get("to_type"))

                if not from_name or not to_name:
                    continue
                if not from_type or not to_type:
                    continue
                if not relation_type or not relation_type.isidentifier():
                    continue

                source_id = await self._resolve_entity(
                    from_name,
                    from_type,
                    entity_cache,
                )
                target_id = await self._resolve_entity(
                    to_name,
                    to_type,
                    entity_cache,
                )
                if not source_id or not target_id:
                    continue

                props = self._build_relationship_props(
                    relation_type,
                    confidence,
                    conversation_id,
                    from_type,
                    to_type,
                )
                await self._neo4j_adapter.merge_relationship(
                    source_id, target_id, relation_type, props
                )
                await self._neo4j_adapter.merge_relationship(
                    source_id, target_id, "RELATED_TO", props
                )

        except Exception as exc:
            logger.error(
                "write_back [%s]: unhandled error (user response unaffected): %s",
                conversation_id,
                exc,
                exc_info=True,
            )

    async def _persist_entity(self, node: EntityNode) -> None:
        await self._neo4j_adapter.merge_entity_node(node)
        await self._entity_chroma_adapter.upsert_entity_embedding(
            entity_id=node.id,
            name=node.name,
            description=node.description,
            entity_type=node.entity_type,
        )

    async def _find_existing_entity(self, node: EntityNode) -> Optional[str]:
        if await self._neo4j_adapter.entity_exists(node.id):
            return node.id
        return await self.find_similar_entities(node)

    async def _resolve_entity(
        self,
        name: str,
        entity_type: str,
        entity_cache: Dict[Tuple[str, str], str],
        *,
        raw: Any = None,
    ) -> Optional[str]:
        key = _entity_key(name, entity_type)
        if key in entity_cache:
            return entity_cache[key]

        if not entity_type or not name:
            return None

        entity_id = _canonical_entity_id(name, entity_type)
        node = EntityNode(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            description=_normalize_entity_description(
                getattr(raw, "description", None) if raw else None,
                name,
            ),
            aliases=list(getattr(raw, "aliases", []) or []) if raw else [],
            nodeset_ids=list(getattr(raw, "nodeset_ids", []) or []) if raw else [],
        )

        existing_id = await self._find_existing_entity(node)
        if existing_id:
            entity_cache[key] = existing_id
            return existing_id

        await self._persist_entity(node)
        if raw is not None:
            await self._nodeset_manager.assign_to_node(
                node.id,
                node.entity_type,
                node.nodeset_ids or [],
            )

        entity_cache[key] = node.id
        return node.id

    def _build_relationship_props(
        self,
        relation_type: str,
        confidence: str,
        conversation_id: str,
        from_type: str,
        to_type: str,
    ) -> Dict[str, object]:
        return {
            "relationship_type": relation_type,
            "confidence": confidence,
            "source_conversation_id": conversation_id,
            "from_type": from_type,
            "to_type": to_type,
        }

    async def find_similar_entities(
        self,
        entity: EntityNode,
    ) -> Optional[str]:
        if self._chroma_adapter is None:
            return None

        name = _normalize_entity_name(entity.name)
        if not name:
            return None

        description = _normalize_entity_description(entity.description, name)
        text = f"{name}. {description}"

        candidate_ids = None
        try:
            fuzzy_candidates = await self._neo4j_adapter.find_fuzzy_entity_candidates(
                entity_type=entity.entity_type,
                name=name,
                exclude_id=entity.id,
                threshold=FUZZY_CANDIDATE_THRESHOLD,
                limit=VECTOR_TOP_K,
            )
            candidate_ids = set(fuzzy_candidates) if fuzzy_candidates else None
        except Exception:
            logger.exception("write_back: fuzzy candidate query failed.")

        try:
            result = await self._chroma_adapter.query_entity_similar(
                text=text,
                entity_type=entity.entity_type,
                n_results=VECTOR_TOP_K,
            )
        except Exception:
            logger.exception("write_back: entity embedding search failed.")
            return None

        ids_batch = (result.get("ids") or [[]])[0]
        distances_batch = (result.get("distances") or [[]])[0]

        for idx, candidate_id in enumerate(ids_batch):
            if not candidate_id:
                continue
            distance = distances_batch[idx] if idx < len(distances_batch) else None
            if distance is None:
                continue
            if candidate_ids is not None and candidate_id not in candidate_ids:
                continue
            similarity = 1.0 - float(distance)
            if similarity >= SEMANTIC_MERGE_THRESHOLD:
                return str(candidate_id)

        return None
