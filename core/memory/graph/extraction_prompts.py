"""Prompt templates for chunk-level entity extraction."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

CHUNK_EXTRACTION_SYSTEM_PROMPT = (
    "You are an information extraction system. Extract entities and relationships from each "
    "news chunk provided. Only use information explicitly stated in each chunk. "
    "Do not infer relationships across multiple articles or chunks. "
    "Allowed entity types: Company, FinancialEvent, FinancialConcept, Sector. Each entity must include a short, single-sentence description drawn only from the chunk text. "
    "Return a JSON object matching the BatchExtractionResult schema. "
    "Each entity must include a temporary local_id used by relationships; "
    "relationships must reference entities by local_id. "
    "Each result must echo the chunk_id exactly as provided. "
    "Schema:\n"
    "{{\n"
    '  "results": [\n'
    "    {{\n"
    '      "chunk_id": "<str>",\n'
    '      "entities": [\n'
    "        {{\n"
    '          "local_id": "<str>",\n'
    '          "id": "<uuid or placeholder>",\n'
    '          "name": "<str>",\n'
    '          "entity_type": "<Company|FinancialEvent|FinancialConcept|Sector>",\n          "description": "<short summary from chunk text>",\n'
    '          "aliases": ["<str>", "..."],\n'
    '          "nodeset_ids": ["<str>", "..."]\n'
    "        }}\n"
    "      ],\n"
    '      "relationships": [\n'
    "        {{\n"
    '          "source_entity_local_id": "<str>",\n'
    '          "target_entity_local_id": "<str>",\n'
    '          "relationship_type": "<str>",\n'
    '          "confidence": <float>\n'
    "        }}\n"
    "      ]\n"
    "    }}\n"
    "  ]\n"
    "}}"
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
            ("system", CHUNK_EXTRACTION_SYSTEM_PROMPT),
            ("user", CHUNK_EXTRACTION_USER_TEMPLATE),
        ]
    )

