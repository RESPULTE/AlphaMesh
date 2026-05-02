"""
Legacy compatibility facade for graph queue APIs.

All implementation now lives under `core.memory.graph.queue`.
This file only re-exports the public symbols and can be removed
once downstream imports migrate to the new package.
"""

from core.memory.graph.queue import (
    GraphQueueManager,
    GraphTask,
    TASK_KIND_EXTRACTION,
    TASK_KIND_SCOPED_EXTRACTION,
    make_extraction_task,
    make_graph_task,
    prompt_id_from_text,
)

_TASK_KIND_EXTRACTION = TASK_KIND_EXTRACTION
_TASK_KIND_SCOPED_EXTRACTION = TASK_KIND_SCOPED_EXTRACTION
_prompt_id = prompt_id_from_text

__all__ = [
    "GraphQueueManager",
    "GraphTask",
    "make_graph_task",
    "make_extraction_task",
    "TASK_KIND_EXTRACTION",
    "TASK_KIND_SCOPED_EXTRACTION",
    "_TASK_KIND_EXTRACTION",
    "_TASK_KIND_SCOPED_EXTRACTION",
    "prompt_id_from_text",
    "_prompt_id",
]
