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
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._entity_chroma_adapter = entity_chroma_adapter
        self._nodeset_manager = nodeset_manager
        self._embedding_func = embedding_func
        self._chunker = chunker
        self._llm = llm
        self._subgraph_store = subgraph_store
        self._logger = get_logger(__name__)

    # ─────────────────────────────────────────────────────────────────────────
    # Public: article ingestion
    # ─────────────────────────────────────────────────────────────────────────

    async def ingest_articles(
        self, articles: List[dict]
    ) -> Tuple[List[str], List[str], List[RetrievedChunk]]:
        """
        Ingest a batch of articles into both stores.

        Returns (new_chunk_ids, existing_chunk_ids, all_involved_chunks).
        Duplicate-URL articles are detected in parallel before writing begins.
        New chunks are already in memory after writing — no round-trip read.
        """
        try:
            global_anchor_id = (
                await self._nodeset_manager.get_global_financial_events_id()
            )

            # Parallel duplicate detection + chunking
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
            # Chunks are already in memory — no ChromaDB round-trip needed
            involved_chunks = chunks_to_ingest + existing_chunks

            return new_chunk_ids, existing_chunk_ids, involved_chunks

        except Exception:
            self._logger.exception("Failed to ingest articles.")
            raise

    async def _classify_article(
        self, article: dict
    ) -> Tuple[Optional[DocumentMetadata], List[RetrievedChunk]]:
        """
        Check for duplicate URL then chunk.
        Returns (None, existing_chunks) for duplicates,
                (doc_meta, new_chunks)  for new articles.
        """
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

    # ─────────────────────────────────────────────────────────────────────────
    # Public: entity extraction
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Public: graph write-back
    # ─────────────────────────────────────────────────────────────────────────
    async def _upsert_graph_to_neo4j(
        self, graph: nx.DiGraph, conversation_id: str
    ) -> None:
        """
        Persist a relationship subgraph to Neo4j.

        Phase 1: resolve all unique entities concurrently.
                Domain entities → _resolve_entity (fuzzy+semantic dedup).
                User-scoped nodes → _resolve_user_node (deterministic, no dedup).
        Phase 2: write all relationship edges concurrently.
        """
        from core.memory.graph.models import _USER_SCOPED_TYPES

        if graph is None or graph.number_of_edges() == 0:
            return

        if self._subgraph_store is not None and conversation_id:
            try:
                key = SubgraphStore.make_key("orchestrator", conversation_id)
                await self._subgraph_store.save(key, graph)
            except Exception:
                self._logger.exception(
                    "Failed to persist subgraph to SubgraphStore for %s",
                    conversation_id,
                )

        # ── Phase 1: resolve all unique entities ──────────────────────────────────
        entity_cache: Dict[Tuple[str, str], str] = {}

        unique_entities: Dict[Tuple[str, str], dict] = {}
        for node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            name = (node_data.get("name") or "").strip()
            raw_etype = (node_data.get("entity_type") or "").strip()
            if not name or not raw_etype:
                continue

            if raw_etype in _USER_SCOPED_TYPES:
                etype = raw_etype  # no normalization for user-scoped types
            else:
                etype = normalize_entity_type(raw_etype)
                if not etype:
                    continue

            key = (name, etype)
            incoming_props = node_data.get("node_props") or {}
            existing_props = unique_entities.get(key, {})
            unique_entities[key] = incoming_props if incoming_props else existing_props

        async def _resolve_safe(name: str, etype: str, props: dict) -> None:
            try:
                if etype in _USER_SCOPED_TYPES:
                    await self._resolve_user_node(
                        name, etype, entity_cache, node_props=props
                    )
                else:
                    await self._resolve_entity(
                        name, etype, entity_cache, raw=props or None
                    )
            except Exception:
                self._logger.exception(
                    "_upsert_graph_to_neo4j: resolution failed for '%s' (%s)",
                    name,
                    etype,
                )

        await asyncio.gather(
            *[
                _resolve_safe(name, etype, props)
                for (name, etype), props in unique_entities.items()
            ]
        )

        # ── Phase 2: write all edges ──────────────────────────────────────────────
        async def _write_edge(source_id: str, target_id: str, attrs: dict) -> None:
            source_node = graph.nodes.get(source_id, {})
            target_node = graph.nodes.get(target_id, {})

            from_name = normalize_entity_name(source_node.get("name"))
            to_name = normalize_entity_name(target_node.get("name"))

            raw_from_type = (source_node.get("entity_type") or "").strip()
            raw_to_type = (target_node.get("entity_type") or "").strip()

            # User-scoped types bypass normalize_entity_type validation
            from_type = (
                raw_from_type
                if raw_from_type in _USER_SCOPED_TYPES
                else normalize_entity_type(raw_from_type)
            )
            to_type = (
                raw_to_type
                if raw_to_type in _USER_SCOPED_TYPES
                else normalize_entity_type(raw_to_type)
            )

            if not from_name or not to_name or not from_type or not to_type:
                return

            resolved_source = entity_cache.get(entity_key(from_name, from_type))
            resolved_target = entity_cache.get(entity_key(to_name, to_type))
            if not resolved_source or not resolved_target:
                return

            relation_type = str(attrs.get("relation_type", "RELATED_TO")).strip()
            confidence = str(attrs.get("confidence", "low")).strip() or "low"
            props = self._build_relationship_props(
                relation_type=relation_type,
                confidence=confidence,
                conversation_id=conversation_id,
                from_type=from_type,
                to_type=to_type,
                reason=str(attrs.get("reason", "")).strip() or None,
                derived_for_user_email=attrs.get("derived_for_user_email"),
                source_agent=attrs.get("source_agent"),
            )
            await self._upsert_with_retry(
                lambda s=resolved_source, t=resolved_target, rt=relation_type, p=props: (
                    self._neo4j_adapter.merge_relationship(s, t, rt, p)
                ),
                None,
            )

        await asyncio.gather(
            *[_write_edge(s, t, a) for s, t, a in graph.edges(data=True)]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public: entity resolution
    # ─────────────────────────────────────────────────────────────────────────

    async def resolve_entity_id(
        self,
        name: str,
        entity_type: str,
        *,
        raw: Any = None,
        entity_cache: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> Optional[str]:
        """Public wrapper: resolve or create an entity and return its ID."""
        cache = entity_cache if entity_cache is not None else {}
        return await self._resolve_entity(name, entity_type, cache, raw=raw)

    # ─────────────────────────────────────────────────────────────────────────
    # Private: write helpers
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Private: entity extraction pipeline
    # ─────────────────────────────────────────────────────────────────────────

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
                entity.id = canonical_id
                local_id_map[local_key] = entity
                chunk_entities[entity.id] = entity

                if not await self._upsert_with_retry(
                    lambda e=entity: self._neo4j_adapter.merge_entity_node(e),
                    result.chunk_id,
                ):
                    chunk_failed = True
                    break

                if not await self._upsert_with_retry(
                    lambda cid=result.chunk_id, eid=entity.id: (
                        self._neo4j_adapter.merge_relationship(
                            cid, eid, "MENTIONS_ENTITY", {"confidence": 1.0}
                        )
                    ),
                    result.chunk_id,
                ):
                    chunk_failed = True
                    break

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

            if self._entity_chroma_adapter is not None:
                new_entities = [
                    e for e in chunk_entities.values() if e.id not in deduped_entities
                ]
                if new_entities:
                    try:
                        await asyncio.gather(
                            *[
                                self._entity_chroma_adapter.upsert_entity_embedding(
                                    entity_id=e.id,
                                    name=e.name,
                                    description=e.description,
                                    entity_type=e.entity_type,
                                )
                                for e in new_entities
                            ]
                        )
                    except Exception:
                        self._logger.exception(
                            "Failed to batch-upsert entity embeddings for chunk %s",
                            result.chunk_id,
                        )

            if not await self._upsert_with_retry(
                lambda cid=result.chunk_id: (
                    self._neo4j_adapter.update_chunk_extraction_status(cid, "EXTRACTED")
                ),
                result.chunk_id,
            ):
                continue

            for eid, entity in chunk_entities.items():
                deduped_entities[eid] = entity

        return list(deduped_entities.values())

    # ─────────────────────────────────────────────────────────────────────────
    # Private: entity resolution + persistence
    # ─────────────────────────────────────────────────────────────────────────

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

        # Support both dict (from node_props, e.g. taxonomy enrichment) and
        # object (from raw entity, e.g. user signal writeback).
        if isinstance(raw, dict):
            _description = raw.get("description")
            _ticker = raw.get("ticker") or None
            _nodeset_ids = list(raw.get("nodeset_ids") or [])
        elif raw is not None:
            _description = getattr(raw, "description", None)
            _ticker = getattr(raw, "ticker", None)
            _nodeset_ids = list(getattr(raw, "nodeset_ids", []) or [])
        else:
            _description = None
            _ticker = None
            _nodeset_ids = []

        node = EntityNode(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            description=normalize_entity_description(_description, name),
            ticker=_ticker,
            nodeset_ids=_nodeset_ids,
        )

        existing_id = await self._find_existing_entity(node)
        if existing_id:
            entity_cache[key] = existing_id
            return existing_id

        await self._persist_entity(node)
        # Only assign nodesets when there are actual IDs to assign —
        # taxonomy entities (Company, Industry, Sector, Market) carry no nodeset_ids.
        if _nodeset_ids:
            await self._nodeset_manager.assign_to_node(
                node.id, node.entity_type, _nodeset_ids[0]
            )

        entity_cache[key] = node.id
        return node.id

    async def _find_existing_entity(self, node: EntityNode) -> Optional[str]:
        if await self._neo4j_adapter.entity_exists(node.id):
            return node.id
        return await self.find_similar_entities(node)

    async def _persist_entity(self, node: EntityNode) -> None:
        await self._neo4j_adapter.merge_entity_node(node)
        if self._entity_chroma_adapter is not None:
            await self._entity_chroma_adapter.upsert_entity_embedding(
                entity_id=node.id,
                name=node.name,
                description=node.description,
                entity_type=node.entity_type,
            )

    async def find_similar_entities(self, entity: EntityNode) -> Optional[str]:
        # BUG FIX: was self._chroma_adapter (news chunks collection).
        # Entity embeddings are written to _entity_chroma_adapter, so the
        # similarity lookup must read from the same collection.
        if self._entity_chroma_adapter is None:
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
            self._logger.exception(
                "find_similar_entities: fuzzy candidate query failed."
            )

        try:
            results = await self._entity_chroma_adapter.query_entity_similar(
                text=text,
                entity_type=entity.entity_type,
                n_results=VECTOR_TOP_K,
            )
        except Exception:
            self._logger.exception(
                "find_similar_entities: entity embedding search failed."
            )
            return None

        for doc, distance in results:
            candidate_id = doc.id or (doc.metadata or {}).get("entity_id")
            if not candidate_id or distance is None:
                continue
            if candidate_ids is not None and candidate_id not in candidate_ids:
                continue
            if (1.0 - float(distance)) >= SEMANTIC_MERGE_THRESHOLD:
                return str(candidate_id)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Private: relationship property builder
    # ─────────────────────────────────────────────────────────────────────────

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
        props: Dict[str, object] = {
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

    async def _resolve_user_node(
        self,
        name: str,
        entity_type: str,
        entity_cache: Dict[Tuple[str, str], str],
        node_props: dict,
    ) -> Optional[str]:
        """
        Persist a user-scoped node (UserInterestDomain, UserInterestEdge, TurnNode).

        Unlike _resolve_entity, this path:
        - Does not run fuzzy/semantic dedup (IDs are deterministic UUIDs).
        - Does not write to Chroma (user nodes are not semantically retrieved).
        - Delegates each type to a specialised Neo4j merge method.
        - Handles BELONGS_TO_NODESET assignment for UserInterestDomain nodes.
        """
        key = entity_key(name, entity_type)
        if key in entity_cache:
            return entity_cache[key]

        node_id: Optional[str] = None

        if entity_type == "UserInterestDomain":
            node_id = node_props.get("id") or name
            await self._neo4j_adapter.merge_user_interest_domain(
                domain_id=node_id,
                props=node_props,
            )
            # Attach to user NodeSet (Q3: side-effect inside resolve, Option A)
            nodeset_id = node_props.get("nodeset_id")
            if nodeset_id:
                await self._nodeset_manager.assign_to_node(
                    node_id, "UserInterestDomain", nodeset_id
                )

        elif entity_type == "UserInterestEdge":
            node_id = node_props.get("id") or name
            operation = node_props.get("operation", "reinforce")
            weight_delta = float(node_props.get("weight_delta", 1.0))
            await self._neo4j_adapter.merge_user_interest_edge(
                edge_id=node_id,
                props=node_props,
                operation=operation,
                weight_delta=weight_delta,
            )

        elif entity_type == "TurnNode":
            node_id = node_props.get("id") or name
            await self._neo4j_adapter.merge_turn_node(
                turn_id=node_id,
                props=node_props,
            )

        if node_id:
            entity_cache[key] = node_id

        return node_id
