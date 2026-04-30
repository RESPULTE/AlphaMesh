"""LLM-backed relationship extraction for graph queue writes."""

from __future__ import annotations

from typing import List, Optional

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

        content_raw = self._coerce_response_text(getattr(response, "content", ""))
        parsed = self._parse_relationships(content_raw, source="content")
        if parsed is not None:
            return parsed

        text_raw = self._coerce_response_text(getattr(response, "text", ""))
        if text_raw and text_raw != content_raw:
            parsed = self._parse_relationships(text_raw, source="text")
            if parsed is not None:
                return parsed

        return []

    @staticmethod
    def _coerce_response_text(value: object) -> str:
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _parse_relationships(raw: str, *, source: str) -> Optional[List[dict]]:
        parsed = parse_relationships_block(raw)
        if parsed is None:
            if "<relationships" not in raw.lower():
                logger.debug(
                    "RelationshipExtractor: no <relationships> block found in %s",
                    source,
                )
            else:
                logger.warning(
                    "RelationshipExtractor: failed to parse relationships block as JSON array from %s",
                    source,
                )
            return None
        return parsed
