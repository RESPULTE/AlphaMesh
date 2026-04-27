"""Property helpers for entity resolution: conversion, merging, and node construction.

All functions here are pure/stateless.
"""

from __future__ import annotations

from typing import Any, List, Optional

from core.memory.graph.models import EntityNode
from core.memory.graph.utils import (
    normalize_entity_description,
)

from .types import _CONFIDENCE_RANK


def try_float_value(value: Any) -> Optional[float]:
    """Return *value* as float if it is numeric, else ``None``."""
    if isinstance(value, (int, float)):
        return float(value)
    return None


def merge_confidence(existing: Any, incoming: Any) -> Any:
    """Return the higher of two confidence values (numeric or textual).

    Numeric values take precedence over textual labels.  When both are textual
    the ranking is: ``low < medium < high``.
    """
    existing_numeric = try_float_value(existing)
    incoming_numeric = try_float_value(incoming)
    if existing_numeric is not None and incoming_numeric is not None:
        return max(existing_numeric, incoming_numeric)
    if existing_numeric is not None:
        return existing
    if incoming_numeric is not None:
        return incoming

    existing_text = str(existing or "low").strip().lower()
    incoming_text = str(incoming or "low").strip().lower()
    if _CONFIDENCE_RANK.get(incoming_text, 0) > _CONFIDENCE_RANK.get(existing_text, 0):
        return incoming
    return existing


def props_to_dict(props: Optional[Any]) -> dict:
    """Normalise arbitrary props into a plain ``dict``.

    Accepts a plain ``dict``, an object with known attributes, or ``None``.
    """
    if isinstance(props, dict):
        return dict(props)
    if props is None:
        return {}
    output: dict = {}
    for field in ("description", "ticker", "nodeset_ids"):
        value = getattr(props, field, None)
        if value is not None:
            output[field] = value
    return output


def merge_props(existing: dict, incoming: dict) -> dict:
    """Merge *incoming* props into *existing*, preferring richer values.

    Rules:
    - ``description``: keep the longer non-empty value.
    - ``ticker``: keep the first non-empty value.
    - ``nodeset_ids``: union (order-preserving, no duplicates).
    """
    if not incoming:
        return dict(existing)
    merged = dict(existing)

    incoming_desc = str(incoming.get("description") or "").strip()
    existing_desc = str(merged.get("description") or "").strip()
    if incoming_desc and len(incoming_desc) > len(existing_desc):
        merged["description"] = incoming_desc

    if not merged.get("ticker") and incoming.get("ticker"):
        merged["ticker"] = incoming["ticker"]

    existing_nodesets: List[str] = list(merged.get("nodeset_ids") or [])
    for nodeset_id in list(incoming.get("nodeset_ids") or []):
        if nodeset_id not in existing_nodesets:
            existing_nodesets.append(nodeset_id)
    if existing_nodesets:
        merged["nodeset_ids"] = existing_nodesets

    return merged


def build_entity_node(
    *,
    name: str,
    entity_type: str,
    entity_id: str,
    props: Optional[Any],
) -> EntityNode:
    """Construct an :class:`EntityNode` from normalised inputs and raw props."""
    prop_dict = props_to_dict(props)
    description = prop_dict.get("description")
    ticker = prop_dict.get("ticker") or None
    nodeset_ids = list(prop_dict.get("nodeset_ids") or [])

    return EntityNode(
        id=entity_id,
        name=name,
        entity_type=entity_type,  # type: ignore[arg-type]
        description=normalize_entity_description(description, name),
        ticker=ticker,
        nodeset_ids=nodeset_ids,
    )
