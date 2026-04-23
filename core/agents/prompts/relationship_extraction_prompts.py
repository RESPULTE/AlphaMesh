"""Shared helpers for relationship extraction prompt blocks."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from core.memory.graph.models import GlobalRelationshipType


class Relationship(BaseModel):
    """Schema for one relationship item expected inside <relationships>."""

    from_name: str
    from_type: str
    relationship_type: GlobalRelationshipType
    to_name: str
    to_type: str
    confidence: Literal["high", "low"] = Field(
        description='"high" for explicit evidence, "low" for inferred.'
    )
    reason: str = Field(description="1-2 short sentences explaining the relationship.")


def _build_relationship_schema_for_prompt() -> str:
    schema = Relationship.model_json_schema()
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


print(_build_relationship_schema_for_prompt())
