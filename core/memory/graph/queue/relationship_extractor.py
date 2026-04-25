"""LLM-backed relationship extraction for graph queue writes."""

from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from core.config import settings
from core.logger import get_logger
from core.memory.graph.utils import parse_relationships_block

logger = get_logger(__name__)


class RelationshipExtractor:
    """Calls the LLM and returns parsed relationship dicts."""

    def __init__(
        self,
        retry_attempts: int = settings.EXTRACTION_LLM_RETRY_ATTEMPTS,
    ) -> None:
        self._retry_attempts = max(1, int(retry_attempts))

    async def extract(
        self,
        *,
        text: str,
        llm: object,
        system_prompt: str,
    ) -> List[dict]:
        """Extract relationships from text; returns [] on any failure."""
        if not text or not text.strip():
            return []

        try:
            if self._retry_attempts <= 1:
                return await self._extract_once(
                    text=text,
                    llm=llm,
                    system_prompt=system_prompt,
                )

            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._retry_attempts),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=False,
            ):
                with attempt:
                    return await self._extract_once(
                        text=text,
                        llm=llm,
                        system_prompt=system_prompt,
                    )
        except Exception:
            logger.exception("RelationshipExtractor.extract: all attempts failed")

        return []

    async def _extract_once(
        self,
        *,
        text: str,
        llm: object,
        system_prompt: str,
    ) -> List[dict]:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text),
            ]
        )
        return self._parse_relationships(str(response.content or ""))

    @staticmethod
    def _parse_relationships(raw: str) -> List[dict]:
        parsed = parse_relationships_block(raw)
        if parsed is None:
            if "<relationships" not in raw.lower():
                logger.debug("RelationshipExtractor: no <relationships> block found")
            else:
                logger.warning(
                    "RelationshipExtractor: failed to parse relationships block as JSON array"
                )
            return []
        return parsed
