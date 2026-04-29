"""Article chunking utilities for news ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Tuple
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import logger
from core.logger import get_logger
from core.memory.graph.models import DocumentMetadata
from core.memory.retrieval.models import RetrievedChunk

logger = get_logger(__name__)


class ArticleChunker:
    """Converts a raw article dict into chunk nodes."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the chunker with size and overlap settings."""
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._logger = get_logger(__name__)

    @staticmethod
    def _parse_published_at(value: object) -> datetime:
        """Parse publishedAt into a timezone-aware UTC datetime."""
        now_utc = datetime.now(timezone.utc)
        if not isinstance(value, str):
            return now_utc

        raw = value.strip()
        if not raw:
            return now_utc

        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                return now_utc

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def chunk_article(
        self, article: dict
    ) -> Tuple[DocumentMetadata, List[RetrievedChunk]]:
        """Split a NewsAPI article into document metadata and chunk nodes."""
        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()
        content = (article.get("content") or "").strip()
        source_url = (article.get("url") or "").strip()
        published_at = self._parse_published_at(article.get("publishedAt"))

        document_id = str(uuid4())
        full_text = "\n\n".join(
            [part for part in [title, description, content] if part]
        )
        chunks = self._splitter.split_text(full_text)

        document_meta = DocumentMetadata(
            document_id=document_id,
            title=title,
            source_url=source_url,
            published_at=published_at,
        )

        logger.info("Chunking article '%s' into %d chunks.", title, len(chunks))

        chunk_records: List[RetrievedChunk] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_records.append(
                RetrievedChunk(
                    source="vector",
                    document_id=document_id,
                    chunk_index=idx,
                    text=chunk_text,
                    chunk_id=str(uuid4()),
                    article_title=title,
                    source_url=source_url,
                    published_at=published_at,
                )
            )

        return document_meta, chunk_records
