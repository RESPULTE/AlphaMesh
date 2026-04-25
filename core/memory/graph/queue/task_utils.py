from __future__ import annotations

from core.memory.graph.queue.types import TASK_KIND_CHUNK_ENTITIES, GraphTask


def is_chunk_entities_task(task: GraphTask) -> bool:
    return task.task_kind == TASK_KIND_CHUNK_ENTITIES


def has_chunk_ids(task: GraphTask) -> bool:
    return bool(task.chunk_ids)


def has_inline_relationships(task: GraphTask) -> bool:
    return bool(task.relationships)


def has_extraction_text(task: GraphTask) -> bool:
    return bool(task.extraction_text)


def has_extractable_payload(task: GraphTask) -> bool:
    return has_extraction_text(task) and bool(task.system_prompt_id)
