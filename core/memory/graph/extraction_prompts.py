"""Prompt templates for chunk-level entity extraction."""

from __future__ import annotations

import json
from typing import Any, Literal, get_args, get_origin

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from core.memory.graph.models import (
    FINANCIAL_CONCEPT_CATEGORIES,
    BatchExtractionResult,
    EntityNode,
    ExtractedRelationship,
)


def _get_literal_values(annotation: Any) -> list[str]:
    """Extract Literal choices from a type annotation."""
    if get_origin(annotation) is Literal:
        return [str(choice) for choice in get_args(annotation)]
    return []


def _get_model_field_choices(model_cls: type[BaseModel], field_name: str) -> list[str]:
    """Read allowed values for a field from the Pydantic model definition."""
    field_info = model_cls.model_fields.get(field_name)
    if field_info is None:
        return []

    literal_choices = _get_literal_values(field_info.annotation)
    if literal_choices:
        return literal_choices

    field_schema = (
        model_cls.model_json_schema().get("properties", {}).get(field_name, {})
    )
    enum_values = field_schema.get("enum", []) if isinstance(field_schema, dict) else []
    return [str(value) for value in enum_values]


def _pipe_join(values: list[str]) -> str:
    return "|".join(values)


def _quoted_join(values: list[str]) -> str:
    return ", ".join(f'"{value}"' for value in values)


def _escape_braces_for_prompt_template(text: str) -> str:
    """
    Escape braces so LangChain ChatPromptTemplate treats JSON as literal text.
    """
    return text.replace("{", "{{").replace("}", "}}")


_ENTITY_TYPE_CHOICES = _get_model_field_choices(EntityNode, "entity_type")
_RELATIONSHIP_TYPE_CHOICES = _get_model_field_choices(
    ExtractedRelationship, "relationship_type"
)
_EXTRACTABLE_ENTITY_TYPES = [
    entity_type
    for entity_type in ("Company", "FinancialEvent", "FinancialConcept")
    if entity_type in _ENTITY_TYPE_CHOICES
]
_ANALYSIS_RELATIONSHIP_FROM_TYPES = [
    entity_type
    for entity_type in ("Company", "FinancialConcept", "FinancialEvent", "Sector")
    if entity_type in _ENTITY_TYPE_CHOICES
]
_DEFERRED_RELATIONSHIP_FROM_TYPES = [
    entity_type
    for entity_type in ("Company", "FinancialConcept", "FinancialEvent")
    if entity_type in _ENTITY_TYPE_CHOICES
]


def _build_batch_extraction_schema_for_prompt() -> str:
    schema = BatchExtractionResult.model_json_schema()
    rendered_schema = json.dumps(schema, indent=2, ensure_ascii=True, sort_keys=True)
    return _escape_braces_for_prompt_template(rendered_schema)


def build_chunk_extraction_system_prompt() -> str:
    """Build system prompt with runtime schema and enum values from Pydantic models."""
    concept_category_choices = _quoted_join(list(FINANCIAL_CONCEPT_CATEGORIES.keys()))
    relationship_choices = _pipe_join(_RELATIONSHIP_TYPE_CHOICES)
    extraction_schema = _build_batch_extraction_schema_for_prompt()
    allowed_entity_types = ", ".join(_EXTRACTABLE_ENTITY_TYPES)

    return (
        "You are an information extraction system. Extract entities and relationships from each "
        "news chunk provided. Only use information explicitly stated in each chunk. "
        "Do not infer relationships across multiple articles or chunks. "
        f"Allowed entity types: {allowed_entity_types}. "
        "Sector, Industry, Market and FinancialConceptCategory entities are managed by the taxonomy pipeline and must NOT "
        "be extracted from text. Each entity must include a short, single-sentence description "
        f"drawn only from the chunk text. FinancialConcept entities MUST include concept_categories with 1-3 entries chosen only from: {concept_category_choices}. "
        f"Extracted relationships MUST use relationship_type values from: {relationship_choices}. "
        "Return a JSON object matching the BatchExtractionResult schema. "
        "Each entity must include a temporary local_id used by relationships; "
        "relationships must reference entities by local_id. "
        "Each result must echo the chunk_id exactly as provided. "
        "JSON Schema:\n"
        f"{extraction_schema}"
    )


CHUNK_EXTRACTION_USER_TEMPLATE = (
    "Extract entities and relationships from the following news chunks. "
    "Each chunk is labeled with [CHUNK_ID: ...].\n\n"
    "{chunk_blocks}\n\n"
)


def build_extraction_prompt() -> ChatPromptTemplate:
    """Build the chat prompt template for chunk extraction."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", build_chunk_extraction_system_prompt()),
            ("user", CHUNK_EXTRACTION_USER_TEMPLATE),
        ]
    )


def build_combined_analysis_relationship_prompt() -> str:
    relationship_choices = _pipe_join(_RELATIONSHIP_TYPE_CHOICES)
    from_type_choices = _pipe_join(_ANALYSIS_RELATIONSHIP_FROM_TYPES)

    return f"""\
You are a financial analyst. Given the context below, produce TWO sections in this exact format:

<analysis>
[Your detailed financial analysis here. Cite sources with [N] notation where applicable.]
</analysis>

<relationships>
[JSON array of relationships between entities already mentioned in the analysis.
Only reference entity names that appear in the context. Do NOT create new entities.
Each entry: {{"from_name": str, "from_type": "{from_type_choices}",
 "relation": "{relationship_choices}",
 "to_name": str, "to_type": str, "confidence": "high|low", "reason": "1-3 sentences"}}]
</relationships>

Rules:
- <analysis> must always be populated. Never leave it empty.
- <relationships> may be an empty array [] if no clear relationships exist.
- Confidence "high" = explicitly stated in context; "low" = inferred.
- reason field: 1-3 short sentences explaining why this relationship holds.
""".strip()


def build_analysis_only_relationship_prompt() -> str:
    relationship_choices = _pipe_join(_RELATIONSHIP_TYPE_CHOICES)
    from_type_choices = _pipe_join(_ANALYSIS_RELATIONSHIP_FROM_TYPES)

    return f"""\
Given the analysis below, extract only relationships between entities already mentioned.
Return ONLY:
<relationships>
[JSON array of relationships between entities already mentioned in the analysis.
Each entry: {{"from_name": str, "from_type": "{from_type_choices}",
 "relation": "{relationship_choices}",
 "to_name": str, "to_type": str, "confidence": "high|low", "reason": "1-3 sentences"}}]
</relationships>

Analysis:
{{analysis_text}}
""".strip()


def build_deferred_relationship_system_prompt() -> str:
    relationship_choices = _pipe_join(_RELATIONSHIP_TYPE_CHOICES)
    from_type_choices = _pipe_join(_DEFERRED_RELATIONSHIP_FROM_TYPES)

    return f"""\
Given the analysis text below, extract only relationships between entities already mentioned.
Return ONLY:
<relationships>
[JSON array of relationships between entities already mentioned in the analysis.
Each entry: {{"from_name": str, "from_type": "{from_type_choices}",
 "relation": "{relationship_choices}",
 "to_name": str, "to_type": str, "confidence": "high|low", "reason": "1-3 sentences"}}]
</relationships>
""".strip()


COMBINED_ANALYSIS_RELATIONSHIP_PROMPT = build_combined_analysis_relationship_prompt()
ANALYSIS_ONLY_RELATIONSHIP_PROMPT = build_analysis_only_relationship_prompt()
DEFERRED_RELATIONSHIP_SYSTEM_PROMPT = build_deferred_relationship_system_prompt()
