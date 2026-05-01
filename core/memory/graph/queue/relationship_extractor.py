"""LLM-backed extraction helpers for graph queue writes."""

from __future__ import annotations

import asyncio
import json
from typing import Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from core.config import settings
from core.logger import get_logger
from core.memory.graph.models import (
    _EXTRACTABLE_ENTITY_TYPES,
    FINANCIAL_CONCEPT_CATEGORIES,
    BatchEntityExtractionResult,
    ChunkEntityExtractionResult,
    EntityNode,
)
from core.memory.graph.utils import canonical_entity_id
from core.memory.graph.utils import parse_relationships_block

logger = get_logger(__name__)

_CHUNK_EXTRACTION_USER_TEMPLATE = (
    "Extract entities from the following news chunks. "
    "Each chunk is labeled with [CHUNK_ID: ...].\n\n"
    "{chunk_blocks}\n\n"
)


def _build_batch_extraction_schema_for_prompt() -> str:
    schema = BatchEntityExtractionResult.model_json_schema()
    rendered_schema = json.dumps(schema, indent=2, ensure_ascii=True, sort_keys=True)
    return rendered_schema.replace("{", "{{").replace("}", "}}")


_EXTERNAL_SOURCE_CHUNK_EXTRACTION_PROMPT = f"""\
        You are an information extraction system. Extract entities from each
        news chunk provided. Only use information explicitly stated in each chunk.
        Allowed entity types: {", ".join(_EXTRACTABLE_ENTITY_TYPES)}.
        Sector, Industry, Market and FinancialConceptCategory entities are managed by the taxonomy pipeline and must NOT
        be extracted from text. Each entity must include a short, single-sentence description
        drawn only from the chunk text.
        FinancialConcept entities must be insightful and provide useful learning experience for the user or context for future analysis.
        They should NOT be generic definitions easily found in a textbook.
        Each FinancialConcept must include 1 to 3 (at max) concept_categories chosen from: {", ".join(FINANCIAL_CONCEPT_CATEGORIES.keys())}.
        Return a JSON object matching the BatchEntityExtractionResult schema.
        Each result must echo the chunk_id exactly as provided.
        JSON Schema:\n
        {_build_batch_extraction_schema_for_prompt()}
""".strip()


def _build_chunk_extraction_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", _CHUNK_EXTRACTION_USER_TEMPLATE),
        ]
    )


class RelationshipExtractor:
    """Calls LLMs and adapters to extract graph relationships/entities."""

    def __init__(
        self,
        neo4j_adapter: Optional[object] = None,
        chroma_adapter: Optional[object] = None,
        nodeset_manager: Optional[object] = None,
        entity_resolver: Optional[object] = None,
        llm: Optional[object] = None,
        llm_provider: Optional[Callable[..., object]] = None,
        retry_attempts: int = settings.EXTRACTION_LLM_RETRY_ATTEMPTS,
    ) -> None:
        self._neo4j_adapter = neo4j_adapter
        self._chroma_adapter = chroma_adapter
        self._nodeset_manager = nodeset_manager
        self._entity_resolver = entity_resolver
        self._llm = llm
        self._llm_provider = llm_provider
        self._retry_attempts = max(1, int(retry_attempts))
        self._extraction_semaphore = asyncio.Semaphore(
            settings.EXTRACTION_MAX_CONCURRENCY
        )

    async def extract(
        self,
        *,
        mode: str = "relationships",
        text: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        llm: Optional[object] = None,
        system_prompt: Optional[str] = None,
        force: bool = False,
    ) -> List[dict] | List[EntityNode]:
        if mode == "relationships":
            return await self._extract_relationships(
                text=text or "",
                llm=llm,
                system_prompt=system_prompt or "",
            )

        if mode == "chunk_entities":
            return await self._extract_chunk_entities(
                chunk_ids=list(chunk_ids or []),
                llm=llm,
                system_prompt=system_prompt,
                force=force,
            )

        logger.warning("RelationshipExtractor.extract: unknown mode '%s'", mode)
        return []

    async def _extract_relationships(
        self,
        *,
        text: str,
        llm: Optional[object],
        system_prompt: str,
    ) -> List[dict]:
        """Extract relationships from text; returns [] on any failure."""
        if not text or not text.strip():
            return []
        if llm is None:
            logger.warning("RelationshipExtractor.extract: missing llm for relationships")
            return []
        if not system_prompt.strip():
            logger.warning(
                "RelationshipExtractor.extract: missing system_prompt for relationships"
            )
            return []

        try:
            if self._retry_attempts <= 1:
                return await self._extract_relationships_once(
                    text=text,
                    llm=llm,
                    system_prompt=system_prompt,
                )

            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._retry_attempts),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=False,
            ):
                with attempt:
                    return await self._extract_relationships_once(
                        text=text,
                        llm=llm,
                        system_prompt=system_prompt,
                    )
        except Exception:
            logger.exception("RelationshipExtractor.extract: all attempts failed")

        return []

    async def _extract_chunk_entities(
        self,
        *,
        chunk_ids: List[str],
        llm: Optional[object],
        system_prompt: Optional[str],
        force: bool = False,
    ) -> List[EntityNode]:
        """Extract entities for chunk IDs and upsert mentions/status metadata."""
        if not chunk_ids:
            return []

        if llm is None:
            llm = self._resolve_chunk_extraction_llm()
        if llm is None:
            logger.warning("Entity extraction requested but no LLM configured.")
            return []
        if (
            self._neo4j_adapter is None
            or self._chroma_adapter is None
            or self._nodeset_manager is None
            or self._entity_resolver is None
        ):
            logger.warning(
                "Entity extraction requested but required adapters are not configured."
            )
            return []

        try:
            return await self._run_chunk_entity_extraction(
                chunk_ids=chunk_ids,
                llm=llm,
                system_prompt=system_prompt or self.default_chunk_system_prompt(),
                force=force,
            )
        except Exception:
            logger.exception("Entity extraction failed for chunks.")
            return []

    async def _extract_relationships_once(
        self,
        *,
        text: str,
        llm: object,
        system_prompt: str,
    ) -> List[dict]:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text),
            ]
        )

        content_value = getattr(response, "content", "")
        content_raw = "" if content_value is None else str(content_value)
        parsed = self._parse_relationships(content_raw, source="content")
        if parsed is not None:
            return parsed

        text_value = getattr(response, "text", "")
        text_raw = "" if text_value is None else str(text_value)
        if text_raw and text_raw != content_raw:
            parsed = self._parse_relationships(text_raw, source="text")
            if parsed is not None:
                return parsed

        return []

    async def _run_chunk_entity_extraction(
        self,
        *,
        chunk_ids: List[str],
        llm: object,
        system_prompt: str,
        force: bool = False,
    ) -> List[EntityNode]:
        if not chunk_ids:
            return []

        financial_events_nodeset_id = (
            await self._nodeset_manager.get_global_financial_events_id()
        )

        if force:
            pending_chunk_ids = list(chunk_ids)
        else:
            status_map = await self._neo4j_adapter.get_chunk_extraction_status(chunk_ids)
            pending_chunk_ids = [
                cid for cid in chunk_ids if status_map.get(cid, "PENDING") == "PENDING"
            ]

        if not pending_chunk_ids:
            return []

        chunk_docs = await self._chroma_adapter.get_documents_by_ids(pending_chunk_ids)
        chunk_lookup: Dict[str, Dict[str, str]] = {}
        for doc in chunk_docs:
            cid = doc.id or (doc.metadata or {}).get("chunk_id")
            if cid:
                chunk_lookup[cid] = {"text": doc.page_content or ""}

        chunks_to_process = [
            {"chunk_id": cid, **chunk_lookup[cid]}
            for cid in pending_chunk_ids
            if cid in chunk_lookup
        ]
        if not chunks_to_process:
            return []

        prompt = _build_chunk_extraction_prompt(system_prompt)
        extraction_chain = prompt | llm.with_structured_output(BatchEntityExtractionResult)

        async def _extract_batch(chunks: List[dict]) -> BatchEntityExtractionResult:
            chunk_blocks = "\n\n".join(
                f"[CHUNK_ID:{chunk['chunk_id']}|{chunk['text']}]" for chunk in chunks
            )
            async with self._extraction_semaphore:
                return await extraction_chain.ainvoke({"chunk_blocks": chunk_blocks})

        batch_size = max(settings.EXTRACTION_BATCH_SIZE, 1)
        batches = [
            chunks_to_process[i : i + batch_size]
            for i in range(0, len(chunks_to_process), batch_size)
        ]
        batch_results = await asyncio.gather(*[_extract_batch(batch) for batch in batches])

        results: List[ChunkEntityExtractionResult] = [
            result
            for batch in batch_results
            for result in batch.results
            if result.chunk_id and result.chunk_id in chunk_lookup
        ]

        deduped_entities: Dict[str, EntityNode] = {}
        for result in results:
            chunk_failed = False
            chunk_entities: Dict[str, EntityNode] = {}

            for entity in result.entities:
                if not entity.description:
                    entity.description = entity.name

                if entity.entity_type == "FinancialEvent":
                    if financial_events_nodeset_id not in entity.nodeset_ids:
                        entity.nodeset_ids = list(entity.nodeset_ids) + [
                            financial_events_nodeset_id
                        ]

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
                chunk_entities[resolved_id] = entity

                if not await self._upsert_with_retry(
                    lambda cid=result.chunk_id, eid=resolved_id: self._neo4j_adapter.merge_relationship(
                        cid, eid, "MENTIONS_ENTITY", {"confidence": 1.0}
                    ),
                    result.chunk_id,
                ):
                    chunk_failed = True
                    break

                await self._link_financial_concept_categories(entity)
                await self._link_financial_event_to_nodeset(
                    entity,
                    financial_events_nodeset_id,
                )

            if chunk_failed:
                continue

            expected_entity_ids = list(chunk_entities.keys())
            if not await self._upsert_with_retry(
                lambda cid=result.chunk_id, expected_ids=expected_entity_ids: self._verify_chunk_entity_links(
                    chunk_id=cid,
                    expected_entity_ids=expected_ids,
                ),
                result.chunk_id,
            ):
                continue

            if not await self._upsert_with_retry(
                lambda cid=result.chunk_id: self._neo4j_adapter.update_chunk_extraction_status(
                    cid, "EXTRACTED"
                ),
                result.chunk_id,
            ):
                continue
            await self._mark_chunk_extracted_in_chroma(result.chunk_id)

            for entity_id, entity in chunk_entities.items():
                deduped_entities[entity_id] = entity

        return list(deduped_entities.values())

    async def _verify_chunk_entity_links(
        self,
        *,
        chunk_id: str,
        expected_entity_ids: List[str],
    ) -> None:
        if not expected_entity_ids:
            return

        rows = await self._neo4j_adapter.get_entities_for_chunks([chunk_id])
        linked_ids = {
            str(row.get("entity_id") or "").strip()
            for row in list(rows or [])
            if str(row.get("entity_id") or "").strip()
        }
        missing = [entity_id for entity_id in expected_entity_ids if entity_id not in linked_ids]
        if missing:
            missing_rendered = ", ".join(sorted(missing))
            raise RuntimeError(
                f"MENTIONS_ENTITY links missing for chunk {chunk_id}: {missing_rendered}"
            )

    async def _upsert_with_retry(
        self,
        coro_factory: Callable[[], object],
        chunk_id: Optional[str] = None,
        max_attempts: int = settings.EXTRACTION_NEO4J_RETRY_ATTEMPTS,
    ) -> bool:
        for attempt in range(max_attempts):
            try:
                await coro_factory()
                return True
            except Exception:
                if attempt == max_attempts - 1:
                    logger.error(
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
            await self._neo4j_adapter.update_chunk_extraction_status(chunk_id, "PENDING")
        except Exception:
            logger.exception("Failed to mark chunk %s pending in Neo4j.", chunk_id)

        try:
            docs = await self._chroma_adapter.get_documents_by_ids([chunk_id])
            if not docs:
                return
            metadata = dict(docs[0].metadata or {})
            metadata["extraction_status"] = "PENDING"
            await self._chroma_adapter.update_metadata([chunk_id], [metadata])
        except Exception:
            logger.exception("Failed to mark chunk %s pending in ChromaDB.", chunk_id)

    async def _mark_chunk_extracted_in_chroma(self, chunk_id: Optional[str]) -> None:
        if not chunk_id:
            return
        try:
            docs = await self._chroma_adapter.get_documents_by_ids([chunk_id])
            if not docs:
                return
            metadata = dict(docs[0].metadata or {})
            metadata["extraction_status"] = "EXTRACTED"
            await self._chroma_adapter.update_metadata([chunk_id], [metadata])
        except Exception:
            logger.exception("Failed to mark chunk %s extracted in ChromaDB.", chunk_id)

    async def _link_financial_concept_categories(self, entity: EntityNode) -> None:
        if entity.entity_type != "FinancialConcept":
            return

        raw_categories = list(entity.concept_categories or [])
        if not raw_categories:
            logger.warning(
                "FinancialConcept '%s' missing concept_categories from LLM output",
                entity.name,
            )
            return

        allowed = set(FINANCIAL_CONCEPT_CATEGORIES)
        categories = [category for category in raw_categories if category in allowed]
        if not categories:
            logger.warning(
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
                logger.exception(
                    "Failed to link FinancialConcept '%s' to category '%s'",
                    entity.name,
                    category,
                )

    async def _link_financial_event_to_nodeset(
        self,
        entity: EntityNode,
        nodeset_id: str,
    ) -> None:
        if entity.entity_type != "FinancialEvent":
            return
        try:
            await self._nodeset_manager.assign_to_node(entity.id, "Entity", nodeset_id)
        except Exception:
            logger.exception(
                "Failed to link FinancialEvent '%s' to GlobalFinancialEvents nodeset",
                entity.name,
            )

    def _resolve_chunk_extraction_llm(self) -> Optional[object]:
        if self._llm_provider is not None:
            try:
                return self._llm_provider(None)
            except TypeError:
                return self._llm_provider()
            except Exception:
                logger.exception("Failed to resolve LLM via llm_provider.")
                return None
        return self._llm

    @staticmethod
    def default_chunk_system_prompt() -> str:
        return _EXTERNAL_SOURCE_CHUNK_EXTRACTION_PROMPT

    @staticmethod
    def _parse_relationships(raw: str, *, source: str) -> Optional[List[dict]]:
        parsed = parse_relationships_block(raw)
        if parsed is None:
            if "<relationships" not in raw.lower():
                logger.debug(
                    "RelationshipExtractor: no <relationships> block found in %s",
                    source,
                )
            else:
                logger.warning(
                    "RelationshipExtractor: failed to parse relationships block as JSON array from %s",
                    source,
                )
            return None
        return parsed
