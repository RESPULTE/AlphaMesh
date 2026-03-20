"""Utility functions for graph entity normalization and ID generation."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from langchain_core.messages import BaseMessage
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from core.config import settings
from core.logger import get_logger
from core.memory.graph.models import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_RELATIONSHIP_TYPES,
    ENTITY_NAMESPACE,
)

logger = get_logger(__name__)

_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)
_REL_RE = re.compile(r"<relationships>(.*?)</relationships>", re.DOTALL | re.IGNORECASE)


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


@dataclass
class ExtractionResult:
    analysis: str
    relationships: List[dict]
    parse_success: bool


def parse_xml_blocks(raw: str) -> Tuple[str, Optional[List[dict]]]:
    analysis_match = _ANALYSIS_RE.search(raw or "")
    if not analysis_match:
        raise ValueError("Missing <analysis> block in LLM response.")

    analysis_text = analysis_match.group(1).strip()
    rel_match = _REL_RE.search(raw or "")
    if not rel_match:
        return analysis_text, None

    rel_text = rel_match.group(1).strip()
    if not rel_text:
        return analysis_text, []

    try:
        relationships = json.loads(rel_text)
    except json.JSONDecodeError:
        return analysis_text, None

    if not isinstance(relationships, list):
        return analysis_text, None
    return analysis_text, relationships


def parse_relationships_block(raw: str) -> Optional[List[dict]]:
    rel_match = _REL_RE.search(raw or "")
    if not rel_match:
        return None
    rel_text = rel_match.group(1).strip()
    if not rel_text:
        return []
    try:
        relationships = json.loads(rel_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(relationships, list):
        return None
    return relationships


async def extract_with_retry(
    llm,
    prompt_messages: List[BaseMessage],
    max_attempts: int = settings.EXTRACTION_LLM_RETRY_ATTEMPTS,
) -> ExtractionResult:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    ):
        with attempt:
            response = await llm.ainvoke(prompt_messages)
            analysis_text, relationships = parse_xml_blocks(response.content)
            if relationships is None:
                return ExtractionResult(
                    analysis=analysis_text,
                    relationships=[],
                    parse_success=False,
                )
            return ExtractionResult(
                analysis=analysis_text,
                relationships=relationships,
                parse_success=True,
            )

    return ExtractionResult(analysis="", relationships=[], parse_success=False)
