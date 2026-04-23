from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.logger import get_logger
from core.memory.graph.entity_resolver import EntityResolver
from core.memory.graph.models import _USER_SCOPED_TYPES
from core.memory.graph.queue.prompt_registry import PromptRegistry
from core.memory.graph.queue.types import TASK_KIND_CHUNK_ENTITIES, GraphTask
from core.memory.graph.relationship_extractor import RelationshipExtractor
from core.memory.graph.utils import (
    entity_key,
    normalize_entity_name,
    normalize_entity_type,
)
from core.memory.stores.neo4j_adapter import Neo4jAdapter

logger = get_logger(__name__)

_NODE_ENDPOINT_FIELDS: Sequence[Tuple[str, str, str]] = (
    ("from_name", "from_type", "from_node_props"),
    ("to_name", "to_type", "to_node_props"),
)


class GraphWritePipeline:
    def __init__(
        self,
        *,
        entity_resolver: EntityResolver,
        graph_writer: Neo4jAdapter,
        relationship_extractor: RelationshipExtractor,
        entity_extractor: Callable[[List[str]], Any],
        llm_provider: Callable[[Optional[dict]], Any],
        prompt_registry: PromptRegistry,
    ) -> None:
        self._resolver = entity_resolver
        self._writer = graph_writer
        self._extractor = relationship_extractor
        self._entity_extractor = entity_extractor
        self._llm_provider = llm_provider
        self._prompt_registry = prompt_registry

    async def extract_relationships_for_task(self, task: GraphTask) -> List[dict]:
        if task.task_kind == TASK_KIND_CHUNK_ENTITIES:
            return []
        if not task.extraction_text or not task.system_prompt_id:
            return []
        prompt = self._prompt_registry.get(task.system_prompt_id)
        if not prompt:
            logger.warning(
                "GraphWritePipeline: missing system prompt for id '%s'",
                task.system_prompt_id,
            )
            return []
        try:
            llm = self._llm_provider(task.llm_config)
            relationships = await self._extractor.extract(
                text=task.extraction_text,
                llm=llm,
                system_prompt=prompt,
                max_attempts=1,
            )
            return list(relationships or [])
        except Exception:
            logger.exception(
                "GraphWritePipeline: extraction failed for task '%s'", task.task_id
            )
            return []

    async def process_tasks(self, tasks: List[GraphTask]) -> Dict[str, int]:
        if not tasks:
            return {"domain_edges": 0, "user_edges": 0}

        await self._process_chunk_entity_tasks(tasks)

        prepared_groups: Dict[bool, List[Tuple[GraphTask, List[dict]]]] = {
            True: [],
            False: [],
        }
        for task in tasks:
            if task.task_kind == TASK_KIND_CHUNK_ENTITIES:
                continue
            relationships = list(task.relationships or [])
            if not relationships and task.extraction_text:
                relationships = await self.extract_relationships_for_task(task)
                if relationships:
                    task.relationships = relationships
            if relationships:
                prepared_groups[bool(task.allow_create)].append((task, relationships))

        total_domain = 0
        total_user = 0
        for allow_create, group in prepared_groups.items():
            if not group:
                continue
            merged_relationships: List[dict] = []
            source_agents: List[str] = []
            for task, relationships in group:
                merged_relationships.extend(relationships)
                if task.source_agent not in source_agents:
                    source_agents.append(task.source_agent)
            source_agent_label = "+".join(source_agents)
            conversation_id = group[0][0].conversation_id
            domain_count, user_count = await self.process_relationships(
                relationships=merged_relationships,
                conversation_id=conversation_id,
                source_agent=source_agent_label,
                allow_create=allow_create,
            )
            total_domain += domain_count
            total_user += user_count

        return {"domain_edges": total_domain, "user_edges": total_user}

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
        chunk_ids: List[str] = []
        for task in tasks:
            if task.task_kind == TASK_KIND_CHUNK_ENTITIES and task.chunk_ids:
                chunk_ids.extend(task.chunk_ids)
        chunk_ids = list(dict.fromkeys(chunk_ids))
        if not chunk_ids:
            return
        try:
            await self._entity_extractor(chunk_ids)
        except Exception:
            logger.exception(
                "GraphWritePipeline: chunk entity extraction failed for %d chunk(s)",
                len(chunk_ids),
            )

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
            for name_key, type_key, props_key in _NODE_ENDPOINT_FIELDS:
                raw_type = str(rel.get(type_key) or "").strip()
                if not raw_type or raw_type in _USER_SCOPED_TYPES:
                    continue

                entity_type = normalize_entity_type(raw_type)
                if not entity_type:
                    continue

                entity_name = normalize_entity_name(str(rel.get(name_key) or ""))
                if not entity_name:
                    continue

                key = entity_key(entity_name, entity_type)
                if key in seen:
                    continue

                seen.add(key)
                unique_entities.append((entity_name, entity_type, rel.get(props_key)))

        if not unique_entities:
            return cache

        resolved = await self._resolver.resolve_batch(
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
        seen: set[Tuple[str, str]] = set()

        for rel in relationships:
            for name_key, type_key, props_key in _NODE_ENDPOINT_FIELDS:
                node_type = str(rel.get(type_key) or "").strip()
                if node_type not in _USER_SCOPED_TYPES:
                    continue

                node_name = normalize_entity_name(str(rel.get(name_key) or ""))
                if not node_name:
                    continue

                cache_key = entity_key(node_name, node_type)
                if cache_key in seen:
                    continue
                seen.add(cache_key)

                raw_props = rel.get(props_key)
                node_props = dict(raw_props) if isinstance(raw_props, dict) else {}
                node_id = str(node_props.get("id") or node_name).strip()
                if not node_id:
                    continue

                if node_type == "UserInterestDomain":
                    domain_props = {**node_props, "id": node_id}
                    await self._writer.merge_user_interest_domain(node_id, domain_props)
                elif node_type == "UserInterestEdge":
                    operation = str(node_props.get("operation") or "reinforce").strip()
                    if operation not in {"reinforce", "invalidate"}:
                        operation = "reinforce"
                    try:
                        weight_delta = float(node_props.get("weight_delta", 0.0))
                    except (TypeError, ValueError):
                        weight_delta = 0.0
                    edge_props = {**node_props, "id": node_id}
                    await self._writer.merge_user_interest_edge(
                        edge_id=node_id,
                        props=edge_props,
                        operation=operation,
                        weight_delta=weight_delta,
                    )
                elif node_type == "TurnNode":
                    turn_props = {**node_props, "id": node_id}
                    await self._writer.merge_turn_node(node_id, turn_props)

                cache[cache_key] = node_id

        return cache
