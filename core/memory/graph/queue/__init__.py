from typing import List, Optional
from uuid import uuid4

from core.memory.graph.queue.manager import GraphQueueManager
from core.memory.graph.queue.types import (
    TASK_KIND_CHUNK_ENTITIES,
    TASK_KIND_RELATIONSHIPS,
    GraphTask,
)

from .utils import (
    build_scoped_system_prompt,
    default_allowed_entity_types,
    default_allowed_relationship_types,
    normalize_allowed_entity_types,
    normalize_allowed_relationship_types,
    prompt_id_from_text,
)

__all__ = [
    "GraphQueueManager",
    "GraphTask",
    "TASK_KIND_RELATIONSHIPS",
    "TASK_KIND_CHUNK_ENTITIES",
    "make_graph_task",
    "make_extraction_task",
    "prompt_id_from_text",
]


def make_graph_task(
    turn_id: str,
    conversation_id: str,
    source_agent: str,
    relationships: List[dict],
    immediate: bool = False,
    allow_create: Optional[bool] = None,
) -> GraphTask:
    return GraphTask(
        task_id=str(uuid4()),
        turn_id=turn_id,
        conversation_id=conversation_id,
        source_agent=source_agent,
        immediate=immediate,
        relationships=relationships,
        allow_create=allow_create,
    )


def make_extraction_task(
    turn_id: str,
    conversation_id: str,
    source_agent: str,
    extraction_text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    llm_config: Optional[dict] = None,
    immediate: bool = False,
    task_kind: str = TASK_KIND_RELATIONSHIPS,
    chunk_ids: Optional[List[str]] = None,
    allowed_entity_types: Optional[List[str]] = None,
    allowed_relationship_types: Optional[List[str]] = None,
    allow_create: Optional[bool] = None,
) -> GraphTask:
    normalized_entity_types = normalize_allowed_entity_types(allowed_entity_types)
    normalized_relationship_types = normalize_allowed_relationship_types(
        allowed_relationship_types
    )

    effective_system_prompt = system_prompt
    if task_kind == TASK_KIND_RELATIONSHIPS:
        if not normalized_entity_types:
            normalized_entity_types = default_allowed_entity_types()
        if not normalized_relationship_types:
            normalized_relationship_types = default_allowed_relationship_types()
        if effective_system_prompt:
            effective_system_prompt = build_scoped_system_prompt(
                base_prompt=effective_system_prompt,
                allowed_entity_types=normalized_entity_types,
                allowed_relationship_types=normalized_relationship_types,
            )

    prompt_id = (
        prompt_id_from_text(effective_system_prompt)
        if effective_system_prompt
        else None
    )
    return GraphTask(
        task_id=str(uuid4()),
        turn_id=turn_id,
        conversation_id=conversation_id,
        source_agent=source_agent,
        immediate=immediate,
        task_kind=task_kind,
        chunk_ids=chunk_ids,
        relationships=[],
        extraction_text=extraction_text,
        system_prompt=effective_system_prompt,
        system_prompt_id=prompt_id,
        allowed_entity_types=normalized_entity_types or None,
        allowed_relationship_types=normalized_relationship_types or None,
        llm_config=llm_config,
        allow_create=allow_create,
    )
