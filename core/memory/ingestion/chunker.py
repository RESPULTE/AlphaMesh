"""Article chunking utilities for news ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Tuple
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from core import logger
from core.logger import get_logger

logger = get_logger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata contract for a document node."""

    model_config = ConfigDict(extra="ignore")

    document_id: str
    title: str
    source_url: str
    published_at: datetime
    companies_involved: List[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """Metadata and text for a single chunk."""

    model_config = ConfigDict(extra="ignore")

    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    article_title: str
    source_url: str
    published_at: datetime
    companies_involved: List[str] = Field(default_factory=list)


class ArticleChunker:
    """Converts a raw article dict into chunk records."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the chunker with size and overlap settings."""
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._logger = get_logger(__name__)

    def _parse_published_at(self, value: str) -> datetime:
        """Parse an ISO8601 timestamp from NewsAPI."""
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            self._logger.exception("Failed to parse publishedAt: %s", value)
            return datetime.now(timezone.utc)

    def chunk_article(
        self, article: dict, companies_involved: List[str]
    ) -> Tuple[DocumentMetadata, List[ChunkRecord]]:
        """Split a NewsAPI article into document metadata and chunk records."""
        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()
        content = (article.get("content") or "").strip()
        source_url = (article.get("url") or "").strip()
        published_at_raw = (article.get("publishedAt") or "").strip()

        published_at = (
            self._parse_published_at(published_at_raw)
            if published_at_raw
            else datetime.now(timezone.utc)
        )

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
            companies_involved=companies_involved,
        )

        logger.info("Chunking article '%s' into %d chunks.", title, len(chunks))

        chunk_records: List[ChunkRecord] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_records.append(
                ChunkRecord(
                    document_id=document_id,
                    chunk_id=str(uuid4()),
                    chunk_index=idx,
                    text=chunk_text,
                    article_title=title,
                    source_url=source_url,
                    published_at=published_at,
                    companies_involved=companies_involved,
                )
            )

        return document_meta, chunk_records
