"""
core/memory/graph/relationship_extractor.py

Single responsibility: call the LLM, parse the <relationships> XML block,
return a list of relationship dicts.

This is extracted from SubgraphExtractionService which previously mixed
LLM extraction, graph construction, and Neo4j persistence.
"""

from __future__ import annotations

import json
import re
from typing import List

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

_REL_RE = re.compile(r"<relationships>(.*?)</relationships>", re.DOTALL | re.IGNORECASE)


class RelationshipExtractor:
    """
    Calls the LLM with a caller-supplied system prompt and parses the
    <relationships> JSON block from the response.

    Returns [] on any failure — callers can always proceed without guards.
    """

    async def extract(
        self,
        text: str,
        llm,
        system_prompt: str,
        max_attempts: int = settings.EXTRACTION_LLM_RETRY_ATTEMPTS,
    ) -> List[dict]:
        """
        Extract relationship dicts from text via LLM.

        Parameters
        ----------
        text          : The source text to extract from.
        llm           : Any LangChain-compatible async LLM.
        system_prompt : Domain-specific extraction prompt (caller-owned).
        max_attempts  : Retry budget for transient LLM failures.

        Returns
        -------
        List of relationship dicts.  Empty list on any failure.
        """
        if not text or not text.strip():
            return []

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=False,
            ):
                with attempt:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    response = await llm.ainvoke(
                        [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=text),
                        ]
                    )
                    return self._parse_relationships(response.content or "")
        except Exception:
            logger.exception("RelationshipExtractor.extract: all attempts failed")

        return []

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_relationships(raw: str) -> List[dict]:
        match = _REL_RE.search(raw)
        if not match:
            logger.debug("RelationshipExtractor: no <relationships> block found")
            return []

        rel_text = match.group(1).strip()
        if not rel_text:
            return []

        try:
            result = json.loads(rel_text)
            if isinstance(result, list):
                return result
            logger.warning(
                "RelationshipExtractor: expected JSON array, got %s", type(result)
            )
        except json.JSONDecodeError:
            logger.warning(
                "RelationshipExtractor: JSON parse failed for relationships block"
            )

        return []
