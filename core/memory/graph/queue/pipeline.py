from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from uuid import uuid4

from core.logger import get_logger
from core.memory.graph.entity_resolver import EntityResolver
from core.memory.graph.models import (
    _USER_SCOPED_TYPES,
    ALLOWED_ENTITY_TYPES,
    ALLOWED_RELATIONSHIP_TYPES,
)
from core.memory.graph.queue.prompt_registry import PromptRegistry
from core.memory.graph.queue.relationship_extractor import RelationshipExtractor
from core.memory.graph.queue.types import GraphTask
from core.memory.graph.queue.utils import (
    has_extractable_payload,
    is_scoped_extraction_task,
)
from core.memory.graph.utils import (
    entity_key,
    normalize_entity_name,
    normalize_entity_type,
    normalize_relationship_type,
)
from core.memory.stores.neo4j_adapter import Neo4jAdapter

logger = get_logger(__name__)

_NODE_ENDPOINT_FIELDS: Sequence[Tuple[str, str, str]] = (
    ("from_name", "from_type", "from_node_props"),
    ("to_name", "to_type", "to_node_props"),
)
_CHUNK_ENTITY_CONTEXT_HEADER = (
    "Known entities extracted for the referenced chunks (strict grounding scope):"
)


class GraphWritePipeline:
    def __init__(
        self,
        *,
        entity_resolver: EntityResolver,
        graph_writer: Neo4jAdapter,
        relationship_extractor: RelationshipExtractor,
        llm_provider: Callable[[Optional[dict]], Any],
        prompt_registry: PromptRegistry,
    ) -> None:
        self._resolver = entity_resolver
        self._writer = graph_writer
        self._extractor = relationship_extractor
        self._llm_provider = llm_provider
        self._prompt_registry = prompt_registry

    async def extract_relationships_for_task(
        self, task: GraphTask
    ) -> Tuple[List[dict], Optional[GraphTask]]:
        if not has_extractable_payload(task):
            return [], None
        prompt = self._prompt_registry.get(task.system_prompt_id)
        if not prompt:
            logger.warning(
                "GraphWritePipeline: missing system prompt for id '%s'",
                task.system_prompt_id,
            )
            return [], None

        extraction_text = task.extraction_text or ""
        allowed_entity_keys: Optional[set[Tuple[str, str]]] = None
        if is_scoped_extraction_task(task) and task.chunk_ids:
            entity_rows = await self._writer.get_entities_for_chunks(task.chunk_ids)
            entity_context, allowed_entity_keys = self._build_chunk_entity_context(
                entity_rows
            )
            if not allowed_entity_keys:
                retry_task = self._build_retry_task(task)
                if retry_task is not None:
                    logger.info(
                        "GraphWritePipeline: no chunk entities for task '%s'; scheduling retry %d/%d",
                        task.task_id,
                        retry_task.retry_count,
                        retry_task.max_retries,
                    )
                else:
                    logger.info(
                        "GraphWritePipeline: no chunk entities for task '%s'; retries exhausted (%d/%d)",
                        task.task_id,
                        task.retry_count,
                        task.max_retries,
                    )
                return [], retry_task
            extraction_text = f"{entity_context}\n\n{extraction_text}"

        try:
            llm = self._llm_provider(task.llm_config)
            relationships = await self._extractor.extract(
                mode="relationships",
                text=extraction_text,
                llm=llm,
                system_prompt=prompt,
                allowed_entity_types=list(task.allowed_entity_types or []),
                allowed_relationship_types=list(task.allowed_relationship_types or []),
            )
            normalized_relationships = list(relationships or [])
            if allowed_entity_keys is not None:
                normalized_relationships = self._filter_relationships_to_known_entities(
                    task=task,
                    relationships=normalized_relationships,
                    allowed_entity_keys=allowed_entity_keys,
                )
            return normalized_relationships, None
        except Exception:
            logger.exception(
                "GraphWritePipeline: extraction failed for task '%s'", task.task_id
            )
            return [], None

    async def process_tasks(self, tasks: List[GraphTask]) -> Dict[str, Any]:
        if not tasks:
            return {
                "domain_edges": 0,
                "user_edges": 0,
                "retry_tasks": [],
                "processed_task_ids": [],
            }

        await self._process_chunk_entity_tasks(tasks)

        prepared_groups: Dict[str, List[Tuple[GraphTask, List[dict]]]] = {}
        retry_tasks: List[GraphTask] = []
        processed_task_ids: List[str] = []
        for task in tasks:
            processed_task_ids.append(task.task_id)
            relationships = list(task.relationships or [])
            if not relationships and task.extraction_text:
                relationships, retry_task = await self.extract_relationships_for_task(
                    task
                )
                if retry_task is not None:
                    retry_tasks.append(retry_task)
                if relationships:
                    task.relationships = relationships
            if relationships and task.extraction_text:
                relationships = self._filter_relationships_for_task(task, relationships)
                task.relationships = relationships
            if relationships:
                group_key = task.conversation_id
                prepared_groups.setdefault(group_key, []).append((task, relationships))

        total_domain = 0
        total_user = 0
        for conversation_id, group in prepared_groups.items():
            if not group:
                continue
            merged_relationships: List[dict] = []
            for task, relationships in group:
                merged_relationships.extend(relationships)
            source_agent_label = "+".join(
                dict.fromkeys(task.source_agent for task, _relationships in group)
            )
            domain_count, user_count = await self.process_relationships(
                relationships=merged_relationships,
                conversation_id=conversation_id,
                source_agent=source_agent_label,
                allow_create=False,
            )
            total_domain += domain_count
            total_user += user_count

        return {
            "domain_edges": total_domain,
            "user_edges": total_user,
            "retry_tasks": retry_tasks,
            "processed_task_ids": processed_task_ids,
        }

    @staticmethod
    def _build_retry_task(task: GraphTask) -> Optional[GraphTask]:
        max_retries = max(0, int(task.max_retries))
        current_retry_count = max(0, int(task.retry_count))
        if current_retry_count >= max_retries:
            return None

        retry_delay_seconds = max(1, int(task.retry_delay_seconds))
        return GraphTask(
            task_id=str(uuid4()),
            turn_id=task.turn_id,
            conversation_id=task.conversation_id,
            source_agent=task.source_agent,
            immediate=task.immediate,
            task_kind=task.task_kind,
            chunk_ids=list(task.chunk_ids or []),
            relationships=[],
            extraction_text=task.extraction_text,
            system_prompt=task.system_prompt,
            system_prompt_id=task.system_prompt_id,
            chunk_system_prompt=task.chunk_system_prompt,
            chunk_system_prompt_id=task.chunk_system_prompt_id,
            allowed_entity_types=list(task.allowed_entity_types or []),
            allowed_relationship_types=list(task.allowed_relationship_types or []),
            llm_config=dict(task.llm_config or {}),
            retry_count=current_retry_count + 1,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            not_before=time.time() + retry_delay_seconds,
        )

    @staticmethod
    def _build_chunk_entity_context(
        entity_rows: List[dict],
    ) -> Tuple[str, set[Tuple[str, str]]]:
        normalized_rows: List[Tuple[str, str, str]] = []
        allowed_entity_keys: set[Tuple[str, str]] = set()

        for row in entity_rows or []:
            entity_name = normalize_entity_name(
                str(row.get("entity_name") or row.get("name") or "")
            )
            entity_type = normalize_entity_type(
                str(row.get("entity_type") or row.get("type") or "")
            )
            source_chunk_id = str(row.get("source_chunk_id") or "").strip() or "unknown"
            if not entity_name or not entity_type:
                continue
            allowed_entity_keys.add(entity_key(entity_name, entity_type))
            normalized_rows.append((entity_type, entity_name, source_chunk_id))

        if not normalized_rows:
            return "", set()

        rendered_rows = sorted(set(normalized_rows))
        lines = [f"- {name} ({entity_type})" for entity_type, name, _ in rendered_rows]
        return (
            f"{_CHUNK_ENTITY_CONTEXT_HEADER}\n"
            "Only extract relationships between entities in this list.\n"
            + "\n".join(lines),
            allowed_entity_keys,
        )

    def _filter_relationships_to_known_entities(
        self,
        *,
        task: GraphTask,
        relationships: List[dict],
        allowed_entity_keys: set[Tuple[str, str]],
    ) -> List[dict]:
        if not relationships or not allowed_entity_keys:
            return []

        filtered: List[dict] = []
        dropped = 0
        for rel in relationships:
            from_name = normalize_entity_name(str(rel.get("from_name") or ""))
            to_name = normalize_entity_name(str(rel.get("to_name") or ""))
            from_type = normalize_entity_type(str(rel.get("from_type") or ""))
            to_type = normalize_entity_type(str(rel.get("to_type") or ""))
            if not from_name or not to_name or not from_type or not to_type:
                dropped += 1
                continue
            if (
                entity_key(from_name, from_type) not in allowed_entity_keys
                or entity_key(to_name, to_type) not in allowed_entity_keys
            ):
                dropped += 1
                continue
            filtered.append(rel)

        if dropped:
            logger.info(
                "GraphWritePipeline: dropped %d relationship(s) outside chunk-entity scope for task '%s'",
                dropped,
                task.task_id,
            )
        return filtered

    async def process_relationships(
        self,
        *,
        relationships: List[dict],
        conversation_id: str,
        source_agent: str,
        allow_create: bool,
    ) -> Tuple[int, int]:
        if not relationships:
            return 0, 0

        domain_rels: List[dict] = []
        user_rels: List[dict] = []
        for rel in relationships:
            from_type = str(rel.get("from_type") or "").strip()
            to_type = str(rel.get("to_type") or "").strip()
            if from_type in _USER_SCOPED_TYPES or to_type in _USER_SCOPED_TYPES:
                user_rels.append(rel)
            else:
                domain_rels.append(rel)

        domain_entity_cache: Dict[Tuple[str, str], str] = {}
        resolved_domain_relationships: List[dict] = []
        if domain_rels:
            resolved_batch = await self._resolver.resolve_relationship_edges(
                domain_rels,
                allow_create=allow_create,
            )
            domain_entity_cache = dict(resolved_batch.entity_cache)
            resolved_domain_relationships = list(resolved_batch.relationships)
            if resolved_batch.skipped_relationships:
                logger.debug(
                    "GraphWritePipeline: skipped %d unresolved domain relationship(s)",
                    resolved_batch.skipped_relationships,
                )

        domain_written = 0
        if resolved_domain_relationships:
            domain_written = await self._writer.write_relationships(
                resolved_domain_relationships,
                conversation_id,
                source_agent,
                domain_entity_cache,
            )

        user_written = 0
        if user_rels:
            user_node_cache = await self._upsert_user_scoped_nodes(user_rels)
            user_domain_cache = await self._resolve_domain_entity_cache(
                relationships=user_rels,
                allow_create=allow_create,
                base_cache=domain_entity_cache,
            )
            combined_cache = {
                **domain_entity_cache,
                **user_domain_cache,
                **user_node_cache,
            }
            user_written = await self._writer.write_relationships(
                user_rels,
                conversation_id,
                source_agent,
                combined_cache,
            )

        return domain_written, user_written

    async def _process_chunk_entity_tasks(self, tasks: List[GraphTask]) -> None:
        chunk_groups: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        for task in tasks:
            if not is_scoped_extraction_task(task) or not task.chunk_ids:
                continue
            prompt_text = self._resolve_chunk_system_prompt(task)
            prompt_key = (
                task.chunk_system_prompt_id
                or (prompt_text.strip() if prompt_text and prompt_text.strip() else "")
                or "__default__"
            )
            llm_config_key = json.dumps(
                task.llm_config or {}, sort_keys=True, default=str
            )
            allowed_entity_types = list(task.allowed_entity_types or [])
            allowed_relationship_types = list(task.allowed_relationship_types or [])
            scope_key = json.dumps(
                {
                    "allowed_entity_types": allowed_entity_types,
                    "allowed_relationship_types": allowed_relationship_types,
                },
                sort_keys=True,
            )
            group_key = (prompt_key, llm_config_key, scope_key)
            group_entry = chunk_groups.setdefault(
                group_key,
                {
                    "chunk_ids": [],
                    "prompt_text": prompt_text,
                    "llm_config": task.llm_config,
                    "allowed_entity_types": allowed_entity_types,
                    "allowed_relationship_types": allowed_relationship_types,
                },
            )
            group_entry["chunk_ids"].extend(task.chunk_ids)

        if not chunk_groups:
            return

        for (_prompt_key, _llm_config_key, _scope_key), group in chunk_groups.items():
            chunk_ids = list(dict.fromkeys(group["chunk_ids"]))
            if not chunk_ids:
                continue
            try:
                llm = self._llm_provider(group["llm_config"])
                await self._extractor.extract(
                    mode="chunk_entities",
                    chunk_ids=chunk_ids,
                    llm=llm,
                    system_prompt=group["prompt_text"],
                    allowed_entity_types=list(group["allowed_entity_types"] or []),
                    allowed_relationship_types=list(
                        group["allowed_relationship_types"] or []
                    ),
                )
            except Exception:
                logger.exception(
                    "GraphWritePipeline: chunk entity extraction failed for %d chunk(s)",
                    len(chunk_ids),
                )

    def _resolve_chunk_system_prompt(self, task: GraphTask) -> Optional[str]:
        if task.chunk_system_prompt and task.chunk_system_prompt.strip():
            return task.chunk_system_prompt
        if task.chunk_system_prompt_id:
            prompt = self._prompt_registry.get(task.chunk_system_prompt_id)
            if prompt:
                return prompt
        return None

    async def _resolve_domain_entity_cache(
        self,
        *,
        relationships: List[dict],
        allow_create: bool,
        base_cache: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> Dict[Tuple[str, str], str]:
        cache: Dict[Tuple[str, str], str] = dict(base_cache or {})
        unique_entities: List[Tuple[str, str, Optional[dict]]] = []
        seen = set(cache.keys())

        for rel in relationships:
            for (
                entity_name,
                raw_type,
                raw_props,
            ) in self._iter_relationship_endpoints(rel):
                if not raw_type or raw_type in _USER_SCOPED_TYPES:
                    continue

                entity_type = normalize_entity_type(raw_type)
                if not entity_type:
                    continue

                if not entity_name:
                    continue

                key = entity_key(entity_name, entity_type)
                if key in seen:
                    continue

                seen.add(key)
                unique_entities.append((entity_name, entity_type, raw_props))

        if not unique_entities:
            return cache

        resolved = await self._resolver.resolve_entities(
            unique_entities,
            allow_create=allow_create,
        )
        for (entity_name, entity_type), resolution in resolved.items():
            if resolution.entity_id:
                cache[entity_key(entity_name, entity_type)] = resolution.entity_id
        return cache

    async def _upsert_user_scoped_nodes(
        self, relationships: List[dict]
    ) -> Dict[Tuple[str, str], str]:
        cache: Dict[Tuple[str, str], str] = {}
        seen: set[Tuple[str, str, str]] = set()

        for rel in relationships:
            for (
                node_name,
                node_type,
                raw_props,
            ) in self._iter_relationship_endpoints(rel):
                if node_type not in _USER_SCOPED_TYPES:
                    continue

                if not node_name:
                    continue

                node_props = dict(raw_props) if isinstance(raw_props, dict) else {}
                dedup_key = self._user_scoped_dedup_key(
                    node_name=node_name,
                    node_type=node_type,
                    node_props=node_props,
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                node_id = str(node_props.get("id") or node_name).strip()
                if not node_id:
                    continue

                await self._merge_user_scoped_node(
                    node_type=node_type,
                    node_id=node_id,
                    node_props=node_props,
                )

                cache[entity_key(node_name, node_type)] = node_id

        return cache

    @staticmethod
    def _user_scoped_dedup_key(
        *, node_name: str, node_type: str, node_props: dict
    ) -> Tuple[str, str, str]:
        user_email = str(node_props.get("user_email") or "").strip().lower()
        return node_name.lower(), node_type, user_email

    def _iter_relationship_endpoints(
        self, relationship: dict
    ) -> Iterator[Tuple[str, str, Optional[dict]]]:
        for name_key, type_key, props_key in _NODE_ENDPOINT_FIELDS:
            node_name = normalize_entity_name(str(relationship.get(name_key) or ""))
            node_type = str(relationship.get(type_key) or "").strip()
            yield node_name, node_type, relationship.get(props_key)

    async def _merge_user_scoped_node(
        self,
        *,
        node_type: str,
        node_id: str,
        node_props: dict,
    ) -> None:
        merged_props = {**node_props, "id": node_id}
        if node_type == "UserInterestDomain":
            await self._writer.merge_user_interest_domain(node_id, merged_props)
            return
        if node_type == "UserInterestEdge":
            operation, weight_delta = self._normalize_user_interest_edge_update(
                node_props
            )
            await self._writer.merge_user_interest_edge(
                edge_id=node_id,
                props=merged_props,
                operation=operation,
                weight_delta=weight_delta,
            )
            return
        if node_type == "UserInterestEvent":
            await self._writer.merge_user_interest_event(
                node_id=node_id, props=merged_props
            )
            return
        if node_type == "SessionNode":
            await self._writer.merge_session_node(node_id=node_id, props=merged_props)
            return

    @staticmethod
    def _normalize_user_interest_edge_update(node_props: dict) -> Tuple[str, float]:
        operation = str(node_props.get("operation") or "reinforce").strip()
        if operation not in {"reinforce", "invalidate"}:
            operation = "reinforce"
        try:
            weight_delta = float(node_props.get("weight_delta", 0.0))
        except (TypeError, ValueError):
            weight_delta = 0.0
        return operation, weight_delta

    def _filter_relationships_for_task(
        self,
        task: GraphTask,
        relationships: List[dict],
    ) -> List[dict]:
        if not relationships:
            return []

        allowed_entity_types = set(task.allowed_entity_types or ALLOWED_ENTITY_TYPES)
        allowed_relationship_types = set(
            task.allowed_relationship_types or ALLOWED_RELATIONSHIP_TYPES
        )
        scoped_relationships: List[dict] = []
        dropped = 0

        for rel in relationships:
            from_type = normalize_entity_type(str(rel.get("from_type") or "").strip())
            to_type = normalize_entity_type(str(rel.get("to_type") or "").strip())
            relation_type = normalize_relationship_type(
                str(rel.get("relation") or rel.get("relation_type") or "RELATED_TO")
            )
            if (
                not from_type
                or not to_type
                or from_type not in allowed_entity_types
                or to_type not in allowed_entity_types
                or relation_type not in allowed_relationship_types
            ):
                dropped += 1
                continue

            scoped_rel = dict(rel)
            scoped_rel["relation"] = relation_type
            scoped_relationships.append(scoped_rel)

        if dropped:
            logger.info(
                "GraphWritePipeline: dropped %d out-of-scope relationship(s) for task '%s'",
                dropped,
                task.task_id,
            )
        return scoped_relationships
