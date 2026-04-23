"""Shared helpers for relationship extraction prompt blocks."""

from __future__ import annotations

import json
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, Field, RootModel

from core.memory.graph.models import ChunkExtractedRelationship, EntityNode


def _get_model_field_choices(model_cls: type[BaseModel], field_name: str) -> list[str]:
    """Read allowed values for a field from the Pydantic model definition."""
    field_info = model_cls.model_fields.get(field_name)
    if field_info is None:
        return []

    if get_origin(field_info.annotation) is Literal:
        return [str(choice) for choice in get_args(field_info.annotation)]

    field_schema = (
        model_cls.model_json_schema().get("properties", {}).get(field_name, {})
    )
    enum_values = field_schema.get("enum", []) if isinstance(field_schema, dict) else []
    return [str(value) for value in enum_values]


_ENTITY_TYPE_CHOICES = _get_model_field_choices(EntityNode, "entity_type")
_RELATIONSHIP_TYPE_CHOICES = _get_model_field_choices(
    ChunkExtractedRelationship, "relationship_type"
)
_EXTRACTABLE_ENTITY_TYPES = [
    entity_type
    for entity_type in ("Company", "FinancialEvent", "FinancialConcept")
    if entity_type in _ENTITY_TYPE_CHOICES
]


class _RelationshipPromptEntry(BaseModel):
    """Schema for one relationship item expected inside <relationships>."""

    from_name: str
    from_type: str
    relation: str
    to_name: str
    to_type: str
    confidence: Literal["high", "low"] = Field(
        description='"high" for explicit evidence, "low" for inferred.'
    )
    reason: str = Field(description="1-2 short sentences explaining the relationship.")


class _RelationshipPromptArray(RootModel[list[_RelationshipPromptEntry]]):
    """Schema for the JSON array returned inside <relationships>."""


def _build_relationship_schema_for_prompt() -> str:
    schema = _RelationshipPromptArray.model_json_schema()
    rendered_schema = json.dumps(schema, indent=2, ensure_ascii=True, sort_keys=True)
    return rendered_schema.replace("{", "{{").replace("}", "}}")


def build_relationships_block(*, include_context_only_rule: bool = False) -> str:
    context_only_rule = (
        "\nOnly reference entity names that appear in the context. Do NOT create new entities."
        if include_context_only_rule
        else ""
    )

    return f"""\
    <relationships>
        [JSON array of relationships between entities already mentioned in the analysis.{context_only_rule}
        Output MUST match this JSON Schema:
        {_build_relationship_schema_for_prompt()}]
    </relationships>
""".strip()
