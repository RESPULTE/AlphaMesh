from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Sequence

from core.memory.graph.models import ALLOWED_ENTITY_TYPES, ALLOWED_RELATIONSHIP_TYPES
from core.memory.graph.queue.types import (
    TASK_KIND_EXTRACTION,
    TASK_KIND_SCOPED_EXTRACTION,
    GraphTask,
)

_SCOPE_PROMPT_HEADER = "Task-scoped extraction constraints:"
_DEFAULT_ALLOWED_ENTITY_TYPES: Sequence[str] = tuple(sorted(ALLOWED_ENTITY_TYPES))
_DEFAULT_ALLOWED_RELATIONSHIP_TYPES: Sequence[str] = tuple(
    sorted(ALLOWED_RELATIONSHIP_TYPES)
)


def is_extraction_task(task: GraphTask) -> bool:
    return task.task_kind == TASK_KIND_EXTRACTION


def is_scoped_extraction_task(task: GraphTask) -> bool:
    return task.task_kind == TASK_KIND_SCOPED_EXTRACTION


def is_supported_task_kind(task: GraphTask) -> bool:
    return task.task_kind in {TASK_KIND_EXTRACTION, TASK_KIND_SCOPED_EXTRACTION}


def has_chunk_ids(task: GraphTask) -> bool:
    return bool(task.chunk_ids)


def has_inline_relationships(task: GraphTask) -> bool:
    return bool(task.relationships)


def has_extraction_text(task: GraphTask) -> bool:
    return bool(task.extraction_text)


def has_extractable_payload(task: GraphTask) -> bool:
    return has_extraction_text(task) and bool(task.system_prompt_id)


def prompt_id_from_text(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def normalize_allowed_entity_types(values: Optional[Iterable[str]]) -> List[str]:
    """Normalize and validate configured entity types for extraction scope."""
    return _normalize_scope_values(
        values=values,
        allowed_values=ALLOWED_ENTITY_TYPES,
        value_label="entity_type",
    )


def normalize_allowed_relationship_types(values: Optional[Iterable[str]]) -> List[str]:
    """Normalize and validate configured relationship types for extraction scope."""
    return _normalize_scope_values(
        values=values,
        allowed_values=set(ALLOWED_RELATIONSHIP_TYPES),
        value_label="relationship_type",
    )


def default_allowed_entity_types() -> List[str]:
    return list(_DEFAULT_ALLOWED_ENTITY_TYPES)


def default_allowed_relationship_types() -> List[str]:
    return list(_DEFAULT_ALLOWED_RELATIONSHIP_TYPES)


def build_scoped_system_prompt(
    *,
    base_prompt: str,
    allowed_entity_types: Sequence[str],
    allowed_relationship_types: Sequence[str],
) -> str:
    """Append deterministic task-scoped constraints to a relationship prompt."""
    if _SCOPE_PROMPT_HEADER in base_prompt:
        return base_prompt

    entities_block = ", ".join(allowed_entity_types)
    relationships_block = ", ".join(allowed_relationship_types)
    constraints = (
        f"{_SCOPE_PROMPT_HEADER}\n"
        f"- Allowed entity types (strict): {entities_block}\n"
        f"- Allowed relationship types (strict): {relationships_block}\n"
        "- Reject any relationship where either endpoint type or relationship type is outside these allowed lists.\n"
        "- If no valid relationships remain, return an empty array in <relationships>."
    )
    return f"{base_prompt.rstrip()}\n\n{constraints}"


def _normalize_scope_values(
    *,
    values: Optional[Iterable[str]],
    allowed_values: set[str],
    value_label: str,
) -> List[str]:
    if values is None:
        return []

    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    unknown = [value for value in normalized if value not in allowed_values]
    if unknown:
        allowed_rendered = ", ".join(sorted(allowed_values))
        unknown_rendered = ", ".join(unknown)
        raise ValueError(
            f"Unknown {value_label}(s): {unknown_rendered}. Allowed values: {allowed_rendered}"
        )
    return normalized
