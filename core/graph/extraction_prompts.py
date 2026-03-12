"""Prompt templates for chunk-level entity extraction."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

CHUNK_EXTRACTION_SYSTEM_PROMPT = (
    "You are an information extraction system. Extract entities and relationships from a "
    "single news chunk. Only use information explicitly stated in the chunk. "
    "Do not infer relationships across multiple articles. "
    "Allowed entity types: Company, Person, MacroIndicator, Event, GeoPoliticalRegion, Instrument. "
    "Return a JSON object matching the ChunkExtractionResult schema. "
    "Each entity must include a temporary local_id used by relationships; "
    "relationships must reference entities by local_id. "
    "Schema:\n"
    "{{\n"
    '  "chunk_id": "<str>",\n'
    '  "entities": [\n'
    "    {{\n"
    '      "local_id": "<str>",\n'
    '      "id": "<uuid or placeholder>",\n'
    '      "name": "<str>",\n'
    '      "entity_type": "<Company|Person|MacroIndicator|Event|GeoPoliticalRegion|Instrument>",\n'
    '      "aliases": ["<str>", "..."],\n'
    '      "nodeset_ids": ["<str>", "..."]\n'
    "    }}\n"
    "  ],\n"
    '  "relationships": [\n'
    "    {{\n"
    '      "source_entity_local_id": "<str>",\n'
    '      "target_entity_local_id": "<str>",\n'
    '      "relationship_type": "<str>",\n'
    '      "confidence": <float>\n'
    "    }}\n"
    "  ]\n"
    "}}"
)

CHUNK_EXTRACTION_USER_TEMPLATE = (
    "Extract entities and relationships from the following news chunk:\n\n"
    "{chunk_text}\n\n"
)


def build_extraction_prompt() -> ChatPromptTemplate:
    """Build the chat prompt template for chunk extraction."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", CHUNK_EXTRACTION_SYSTEM_PROMPT),
            ("user", CHUNK_EXTRACTION_USER_TEMPLATE),
        ]
    )
