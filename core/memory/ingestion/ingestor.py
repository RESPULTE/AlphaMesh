"""Dual-store ingestion pipeline for news articles."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from core.config import settings
from core.logger import get_logger
from core.memory.graph.extraction_prompts import build_extraction_prompt
from core.memory.graph.models import (
    BatchExtractionResult,
    ChunkExtractionResult,
    DocumentMetadata,
    DocumentNode,
    EntityNode,
)
from core.memory.graph.nodeset_manager import NodeSetManager
from core.memory.graph.utils import (
    canonical_entity_id,
    entity_key,
    normalize_entity_description,
    normalize_entity_name,
    normalize_entity_type,
    normalize_relationship_type,
)
from core.memory.ingestion.chunker import ArticleChunker
from core.memory.retrieval.models import RetrievedChunk
from core.memory.stores.chroma_adapter import ChromaDBAdapter
from core.memory.stores.neo4j_adapter import Neo4jAdapter
from core.memory.stores.subgraph_store import SubgraphStore

logger = get_logger(__name__)

EXTRACTION_SEMAPHORE = asyncio.Semaphore(settings.EXTRACTION_MAX_CONCURRENCY)
FUZZY_CANDIDATE_THRESHOLD = 0.50
SEMANTIC_MERGE_THRESHOLD = 0.85
VECTOR_TOP_K = 10


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
        subgraph_store=None,
    ) -> None:
        """Initialize the ingestor with adapters and utilities."""
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._entity_chroma_adapter = entity_chroma_adapter
        self._nodeset_manager = nodeset_manager
        self._embedding_func = embedding_func
        self._chunker = chunker
        self._llm = llm
        self._subgraph_store = subgraph_store
        self._logger = get_logger(__name__)

    async def ingest_articles(
        self, articles: List[dict]
    ) -> Tuple[List[str], List[RetrievedChunk]]:
        """Ingest a batch of articles into both stores."""
        try:
            global_anchor_id = (
                await self._nodeset_manager.get_global_financial_events_id()
            )
            documents_to_ingest: List[DocumentMetadata] = []
            chunks_to_ingest: List[RetrievedChunk] = []

            existing_chunks_to_return: List[RetrievedChunk] = []
            for article in articles:
                source_url = (article.get("url") or "").strip()

                existing_chunks = await self._chroma_adapter.get_chunks_with_source_url(
                    source_url
                )
                if source_url and len(existing_chunks) > 0:
                    self._logger.info(
                        "Skipping article with existing source URL: %s", source_url
                    )
                    existing_chunks = [
                        RetrievedChunk.from_document(doc, source="vector")
                        for doc in existing_chunks
                    ]
                    existing_chunks_to_return.extend(existing_chunks)
                    continue
                doc_meta, chunk_records = self._chunker.chunk_article(article)
                documents_to_ingest.append(doc_meta)
                chunks_to_ingest.extend(chunk_records)

            if documents_to_ingest:
                await self._write_document_nodes(documents_to_ingest, global_anchor_id)
                await self._write_chunk_nodes(chunks_to_ingest, global_anchor_id)
                await self._write_vector_chunks(chunks_to_ingest, global_anchor_id)

            new_chunk_ids = [chunk.chunk_id for chunk in chunks_to_ingest]
            existing_chunk_ids = [chunk.chunk_id for chunk in existing_chunks_to_return]
            existing_chunk_ids = [cid for cid in existing_chunk_ids if cid]
            chunk_ids = new_chunk_ids + existing_chunk_ids

            involved_chunks: List[RetrievedChunk] = []
            if chunk_ids:
                docs = await self._chroma_adapter.get_documents_by_ids(chunk_ids)
                involved_chunks = [
                    RetrievedChunk.from_document(doc, source="vector") for doc in docs
                ]
            # self._schedule_extraction(chunk_ids)
            return (chunk_ids, involved_chunks)
            return (chunk_ids, involved_chunks)
        except Exception as exec:
            self._logger.exception("Failed to ingest articles. %s", str(exec))
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
        self, chunks: List[RetrievedChunk], global_anchor_id: str
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
        self, chunks: List[RetrievedChunk], global_anchor_id: str
    ) -> None:
        """Write chunk vectors and metadata to ChromaDB."""
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
                chunk_ids=[chunk.chunk_id for chunk in chunks],
                texts=[chunk.text for chunk in chunks],
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

    async def extract_entities_for_chunks(
        self, chunk_ids: List[str], force: bool = False
    ) -> List[EntityNode]:
        """Extract entities for the given chunks and return a deduped list."""
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

    async def _extract_entities_for_chunks(
        self, chunk_ids: List[str], force: bool = False
    ) -> List[EntityNode]:
        """Extract entities/relationships for pending chunks in the background."""
        if not chunk_ids:
            return []

        if force:
            pending_chunk_ids = list(chunk_ids)
        else:
            status_map = await self._neo4j_adapter.get_chunk_extraction_status(
                chunk_ids
            )
            pending_chunk_ids = [
                chunk_id
                for chunk_id in chunk_ids
                if status_map.get(chunk_id) == "PENDING"
            ]
        if not pending_chunk_ids:
            return await self._neo4j_adapter.get_entities_for_chunks(chunk_ids)

        chunk_docs = await self._chroma_adapter.get_documents_by_ids(pending_chunk_ids)

        chunk_lookup = {}
        for doc in chunk_docs:
            chunk_id = doc.id or (doc.metadata or {}).get("chunk_id")
            if not chunk_id:
                continue
            chunk_lookup[chunk_id] = {
                "text": doc.page_content or "",
                "metadata": doc.metadata or {},
            }
        chunks_to_process = [
            {"chunk_id": chunk_id, **chunk_lookup[chunk_id]}
            for chunk_id in pending_chunk_ids
            if chunk_id in chunk_lookup
        ]
        if not chunks_to_process:
            return []

        prompt = build_extraction_prompt()
        extraction_chain = prompt | self._llm.with_structured_output(
            BatchExtractionResult
        )

        async def _extract_batch(chunks: List[dict]) -> BatchExtractionResult:
            chunk_blocks = "\n\n".join(
                [
                    f"[CHUNK_ID:{chunk['chunk_id']}|{chunk['text']}]"
                    for idx, chunk in enumerate(chunks, 1)
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

        deduped_entities: Dict[str, EntityNode] = {}

        for result in validated_results:
            local_id_map = {}
            chunk_failed = False
            chunk_entities: Dict[str, EntityNode] = {}
            for entity in result.entities:
                if not getattr(entity, "description", None):
                    entity.description = entity.name
                canonical_id = canonical_entity_id(entity.name, entity.entity_type)
                local_key = entity.local_id or canonical_id
                entity.id = canonical_id
                local_id_map[local_key] = entity
                chunk_entities[entity.id] = entity

                if not await self._upsert_with_retry(
                    lambda: self._neo4j_adapter.merge_entity_node(entity),
                    result.chunk_id,
                ):
                    chunk_failed = True
                    break

                if self._entity_chroma_adapter is not None:
                    try:
                        await self._entity_chroma_adapter.upsert_entity_embedding(
                            entity_id=entity.id,
                            name=entity.name,
                            description=entity.description,
                            entity_type=entity.entity_type,
                        )
                    except Exception:
                        self._logger.exception(
                            "Failed to upsert entity embedding for %s", entity.id
                        )

                if not await self._upsert_with_retry(
                    lambda: self._neo4j_adapter.merge_relationship(
                        result.chunk_id,
                        entity.id,
                        "MENTIONS_ENTITY",
                        {"confidence": 1.0},
                    ),
                    result.chunk_id,
                ):
                    chunk_failed = True
                    break

            if chunk_failed:
                continue

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
                rel_type = normalize_relationship_type(rel.relationship_type)
                if not await self._upsert_with_retry(
                    lambda: self._neo4j_adapter.merge_relationship(
                        source_entity.id,
                        target_entity.id,
                        "RELATED_TO",
                        {
                            "relationship_type": rel_type,
                            "source_chunk_id": result.chunk_id,
                            "confidence": rel.confidence,
                        },
                    ),
                    result.chunk_id,
                ):
                    chunk_failed = True
                    break

            if chunk_failed:
                continue

            if not await self._upsert_with_retry(
                lambda: self._neo4j_adapter.update_chunk_extraction_status(
                    result.chunk_id, "EXTRACTED"
                ),
                result.chunk_id,
            ):
                continue

            chunk_meta = chunk_lookup.get(result.chunk_id)
            if chunk_meta:
                updated_metadata = dict(chunk_meta.get("metadata") or {})
                updated_metadata["extraction_status"] = "EXTRACTED"
                try:
                    await self._chroma_adapter.update_metadata(
                        [result.chunk_id], [updated_metadata]
                    )
                except Exception:
                    self._logger.exception(
                        "Failed to update chunk metadata for %s", result.chunk_id
                    )

            for entity_id, entity in chunk_entities.items():
                deduped_entities[entity_id] = entity

        return list(deduped_entities.values())

    async def _upsert_graph_to_neo4j(
        self, graph: nx.DiGraph, conversation_id: str
    ) -> None:
        if graph is None:
            return

        if self._subgraph_store is not None and conversation_id:
            try:
                key = SubgraphStore.make_key("orchestrator", conversation_id)
                await self._subgraph_store.save(key, graph)
            except Exception:
                self._logger.exception(
                    "Failed to persist merged subgraph for %s", conversation_id
                )

        entity_cache: Dict[tuple[str, str], str] = {}
        for source_id, target_id, attrs in graph.edges(data=True):
            source_node = graph.nodes.get(source_id, {})
            target_node = graph.nodes.get(target_id, {})
            from_name = normalize_entity_name(source_node.get("name"))
            to_name = normalize_entity_name(target_node.get("name"))
            from_type = normalize_entity_type(source_node.get("entity_type"))
            to_type = normalize_entity_type(target_node.get("entity_type"))
            if not from_name or not to_name or not from_type or not to_type:
                continue

            resolved_source = await self._resolve_entity(
                from_name,
                from_type,
                entity_cache,
            )
            resolved_target = await self._resolve_entity(
                to_name,
                to_type,
                entity_cache,
            )
            if not resolved_source or not resolved_target:
                continue

            relation_type = str(attrs.get("relation_type", "RELATED_TO")).strip()
            confidence = str(attrs.get("confidence", "low")).strip() or "low"
            reason = str(attrs.get("reason", "")).strip()
            derived_for_user_email = attrs.get("derived_for_user_email")
            source_agent = attrs.get("source_agent")

            props = self._build_relationship_props(
                relation_type,
                confidence,
                conversation_id,
                from_type,
                to_type,
                reason=reason,
                derived_for_user_email=derived_for_user_email,
                source_agent=source_agent,
            )

            await self._upsert_with_retry(
                lambda: self._neo4j_adapter.merge_relationship(
                    resolved_source, resolved_target, relation_type, props
                ),
                None,
            )
            await self._upsert_with_retry(
                lambda: self._neo4j_adapter.merge_relationship(
                    resolved_source, resolved_target, "RELATED_TO", props
                ),
                None,
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

    async def resolve_entity_id(
        self,
        name: str,
        entity_type: str,
        *,
        raw: Any = None,
        entity_cache: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> Optional[str]:
        """Public wrapper for entity resolution to avoid duplicate logic."""
        cache = entity_cache if entity_cache is not None else {}
        return await self._resolve_entity(name, entity_type, cache, raw=raw)

    async def _resolve_entity(
        self,
        name: str,
        entity_type: str,
        entity_cache: Dict[Tuple[str, str], str],
        *,
        raw: Any = None,
    ) -> Optional[str]:
        key = entity_key(name, entity_type)
        if key in entity_cache:
            return entity_cache[key]

        if not entity_type or not name:
            return None

        entity_id = canonical_entity_id(name, entity_type)
        node = EntityNode(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            description=normalize_entity_description(
                getattr(raw, "description", None) if raw else None,
                name,
            ),
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
        reason: Optional[str] = None,
        derived_for_user_email: Optional[str] = None,
        source_agent: Optional[str] = None,
    ) -> Dict[str, object]:
        props = {
            "relationship_type": relation_type,
            "confidence": confidence,
            "source_conversation_id": conversation_id,
            "from_type": from_type,
            "to_type": to_type,
        }
        if reason:
            props["reason"] = reason
        if derived_for_user_email:
            props["derived_for_user_email"] = derived_for_user_email
        if source_agent:
            props["source_agent"] = source_agent
        return props

    async def find_similar_entities(
        self,
        entity: EntityNode,
    ) -> Optional[str]:
        if self._chroma_adapter is None:
            return None

        name = normalize_entity_name(entity.name)
        if not name:
            return None

        description = normalize_entity_description(entity.description, name)
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

        for doc, distance in result:
            candidate_id = doc.id or (doc.metadata or {}).get("entity_id")
            if not candidate_id:
                continue
            if distance is None:
                continue
            if candidate_ids is not None and candidate_id not in candidate_ids:
                continue
            similarity = 1.0 - float(distance)
            if similarity >= SEMANTIC_MERGE_THRESHOLD:
                return str(candidate_id)
        return None
