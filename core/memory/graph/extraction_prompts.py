"""Prompt templates for chunk-level entity extraction."""

from __future__ import annotations

import json
from typing import Literal, get_args, get_origin

from pydantic import BaseModel

from core.memory.graph.models import (
    FINANCIAL_CONCEPT_CATEGORIES,
    BatchExtractionResult,
    EntityNode,
    ExtractedRelationship,
)

# ---------------------------
# Model Schema / Enum Helpers
# ---------------------------


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


def _pipe_join(values: list[str]) -> str:
    return "|".join(values)


# ---------------------------
# Derived Type Choices
# ---------------------------

_ENTITY_TYPE_CHOICES = _get_model_field_choices(EntityNode, "entity_type")
_RELATIONSHIP_TYPE_CHOICES = _get_model_field_choices(
    ExtractedRelationship, "relationship_type"
)
_EXTRACTABLE_ENTITY_TYPES = [
    entity_type
    for entity_type in ("Company", "FinancialEvent", "FinancialConcept")
    if entity_type in _ENTITY_TYPE_CHOICES
]


# ---------------------------
# Relationship Prompt Helper
# ---------------------------


def _build_batch_extraction_schema_for_prompt() -> str:
    schema = BatchExtractionResult.model_json_schema()
    rendered_schema = json.dumps(schema, indent=2, ensure_ascii=True, sort_keys=True)
    return rendered_schema.replace("{", "{{").replace("}", "}}")


def _build_relationships_block(*, include_context_only_rule: bool = False) -> str:
    relationship_choices = _pipe_join(_RELATIONSHIP_TYPE_CHOICES)
    from_type_choices = _pipe_join(_EXTRACTABLE_ENTITY_TYPES)
    context_only_rule = (
        "\nOnly reference entity names that appear in the context. Do NOT create new entities."
        if include_context_only_rule
        else ""
    )

    return f"""\
    <relationships>
        [JSON array of relationships between entities already mentioned in the analysis.{context_only_rule}
        Each entry: {{"from_name": str, "from_type": "{from_type_choices}",
        "relation": "{relationship_choices}",
        "to_name": str, "to_type": str, "confidence": "high|low", "reason": "1-3 sentences"}}]
    </relationships>
""".strip()


# ---------------------------
# Module-Level Constants
# ---------------------------

CHUNK_EXTRACTION_PROMPT = f"""\
        You are an information extraction system. Extract entities and relationships from each 
        news chunk provided. Only use information explicitly stated in each chunk. 
        Do not infer relationships across multiple articles or chunks. 
        Allowed entity types: {", ".join(_EXTRACTABLE_ENTITY_TYPES)}. 
        Sector, Industry, Market and FinancialConceptCategory entities are managed by the taxonomy pipeline and must NOT 
        be extracted from text. Each entity must include a short, single-sentence description 
        drawn only from the chunk text. FinancialConcept entities MUST include concept_categories with 1-3 entries chosen only from: {_pipe_join(list(FINANCIAL_CONCEPT_CATEGORIES.keys()))}. 
        Extracted relationships MUST use relationship_type values from: {"|".join(_RELATIONSHIP_TYPE_CHOICES)}. 
        Return a JSON object matching the BatchExtractionResult schema. 
        Each entity must include a temporary local_id used by relationships; 
        relationships must reference entities by local_id. 
        Each result must echo the chunk_id exactly as provided. 
        JSON Schema:\n
        { _build_batch_extraction_schema_for_prompt()}
""".strip()


COMBINED_ANALYSIS_RELATIONSHIP_PROMPT = f"""\
You are a financial analyst. Given the context below, produce TWO sections in this exact format:

<analysis>
[Your detailed financial analysis here. Cite sources with [N] notation where applicable.]
</analysis>

{_build_relationships_block(include_context_only_rule=True)}

Rules:
- <analysis> must always be populated. Never leave it empty.
- <relationships> may be an empty array [] if no clear relationships exist.
- Confidence "high" = explicitly stated in context; "low" = inferred.
- reason field: 1-3 short sentences explaining why this relationship holds.
""".strip()


ANALYSIS_ONLY_RELATIONSHIP_PROMPT = f"""\
Given the analysis below, extract only relationships between entities already mentioned.
Return ONLY:
{_build_relationships_block()}

Analysis:
{{analysis_text}}
""".strip()


DEFERRED_RELATIONSHIP_SYSTEM_PROMPT = f"""\
Given the analysis text below, extract only relationships between entities already mentioned.
Return ONLY:
{_build_relationships_block()}
""".strip()
