"""Utility functions for graph entity normalization and ID generation."""

from __future__ import annotations

import uuid
from typing import Any, Optional, Tuple

from core.memory.graph.models import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_RELATIONSHIP_TYPES,
    ENTITY_NAMESPACE,
)


def normalize_relationship_type(value: str) -> str:
    """Normalize to canonical type or fall back to RELATED_TO."""
    normalized = str(value or "").strip().upper().replace(" ", "_")
    if normalized in ALLOWED_RELATIONSHIP_TYPES:
        return normalized
    return "RELATED_TO"


def generate_uuid5(key: str) -> str:
    """Generate a stable UUID v5 from a string key using ENTITY_NAMESPACE."""
    return str(uuid.uuid5(ENTITY_NAMESPACE, key))


def canonical_entity_id(name: str, entity_type: str) -> str:
    """Generate a stable UUID v5 for an entity based on its name and type."""
    key = f"{name.lower()}::{entity_type.lower()}"
    return generate_uuid5(key)


def canonical_nodeset_id(name: str) -> str:
    """Generate a stable UUID v5 for a nodeset based on its name."""
    return generate_uuid5(name)


def normalize_entity_type(value: Any) -> Optional[str]:
    """Validate and normalize entity type against ALLOWED_ENTITY_TYPES."""
    if not value:
        return None
    entity_type = str(value).strip()
    if entity_type in ALLOWED_ENTITY_TYPES:
        return entity_type
    return None


def normalize_entity_name(value: Any) -> str:
    """Clean and normalize entity name."""
    return str(value or "").strip()


def normalize_entity_description(value: Any, fallback: str) -> str:
    """Normalize entity description or use fallback."""
    description = str(value or "").strip()
    return description or fallback


def entity_key(name: str, entity_type: str) -> Tuple[str, str]:
    """Generate a standard key for entity caching (lowercase name, original type)."""
    return (name.lower(), entity_type)
