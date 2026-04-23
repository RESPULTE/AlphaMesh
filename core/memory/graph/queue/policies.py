from __future__ import annotations

from typing import Optional, Set


def parse_source_allowlist(raw_value: str) -> Set[str]:
    return {item.strip() for item in str(raw_value or "").split(",") if item.strip()}


def resolve_allow_create(
    *,
    source_agent: str,
    task_allow_create: Optional[bool],
    explicit_allow_create: Optional[bool],
    default_allow_create_sources: Set[str],
) -> bool:
    if explicit_allow_create is not None:
        return bool(explicit_allow_create)
    if task_allow_create is not None:
        return bool(task_allow_create)
    return source_agent in default_allow_create_sources
