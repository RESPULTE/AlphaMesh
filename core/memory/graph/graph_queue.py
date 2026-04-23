"""
Legacy compatibility facade for graph queue APIs.

All implementation now lives under `core.memory.graph.queue`.
This file only re-exports the public symbols and can be removed
once downstream imports migrate to the new package.
"""

from core.memory.graph.queue import (
    GraphQueueManager,
    GraphTask,
    TASK_KIND_CHUNK_ENTITIES,
    TASK_KIND_RELATIONSHIPS,
    make_extraction_task,
    make_graph_task,
    prompt_id_from_text,
)

_TASK_KIND_RELATIONSHIPS = TASK_KIND_RELATIONSHIPS
_TASK_KIND_CHUNK_ENTITIES = TASK_KIND_CHUNK_ENTITIES
_prompt_id = prompt_id_from_text

__all__ = [
    "GraphQueueManager",
    "GraphTask",
    "make_graph_task",
    "make_extraction_task",
    "TASK_KIND_RELATIONSHIPS",
    "TASK_KIND_CHUNK_ENTITIES",
    "_TASK_KIND_RELATIONSHIPS",
    "_TASK_KIND_CHUNK_ENTITIES",
    "prompt_id_from_text",
    "_prompt_id",
]
