"""LLM wrapper for combined analysis + relationship extraction."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.messages import BaseMessage
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from core.config import settings
from core.logger import get_logger
from core.memory.graph.subgraph_builder import InMemorySubgraphBuilder
from core.memory.stores.subgraph_store import SubgraphStore

logger = get_logger(__name__)

_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)
_REL_RE = re.compile(r"<relationships>(.*?)</relationships>", re.DOTALL | re.IGNORECASE)


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


async def retry_relationships_only(
    llm,
    analysis_text: str,
    agent_name: str,
    conversation_id: str,
    builder: InMemorySubgraphBuilder,
    store: SubgraphStore,
    key: str,
    prompt: str,
) -> None:
    if not analysis_text.strip():
        return

    try:
        response = await llm.ainvoke(prompt.format(analysis_text=analysis_text))
        relationships = parse_relationships_block(response.content)
        if relationships is None:
            logger.warning(
                "Retry relationships parse failed for %s:%s", agent_name, conversation_id
            )
            return
        graph = await builder.build(relationships, source_agent=agent_name)
        await store.save(key, graph)
    except Exception:
        logger.exception(
            "Retry relationships extraction failed for %s:%s", agent_name, conversation_id
        )
        return


