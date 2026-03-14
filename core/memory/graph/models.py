"""Pydantic models and constants for the Neo4j graph schema."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ENTITY_NAMESPACE = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class DocumentNode(BaseModel):
    """Graph node contract for a document-level anchor."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    source_url: str
    published_at: datetime
    ingested_at: datetime
    companies_involved: List[str] = Field(default_factory=list)
    nodeset_ids: List[str] = Field(default_factory=list)


class ChunkNode(BaseModel):
    """Graph node contract for a chunk of a document."""

    model_config = ConfigDict(extra="ignore")

    id: str
    text: str
    chunk_index: int
    document_id: str
    companies_involved: List[str] = Field(default_factory=list)
    nodeset_ids: List[str] = Field(default_factory=list)
    extraction_status: Literal["PENDING", "EXTRACTED"]


class EntityNode(BaseModel):
    """Graph node contract for an extracted entity."""

    model_config = ConfigDict(extra="ignore")

    local_id: Optional[str] = Field(default=None, exclude=True)
    id: str
    name: str
    entity_type: Literal[
        "Company",
        "FinancialEvent",
        "FinancialConcept",
        "Sector",
    ]
    description: str
    aliases: List[str] = Field(default_factory=list)
    nodeset_ids: List[str] = Field(default_factory=list)


class ExtractedRelationship(BaseModel):
    """Intermediate relationship model returned by the extraction call."""

    model_config = ConfigDict(extra="ignore")

    source_entity_local_id: str
    target_entity_local_id: str
    relationship_type: str
    confidence: float


class ChunkExtractionResult(BaseModel):
    """Structured extraction output for a single chunk."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(
        default="", description="ignored by input, used for traceability in output"
    )
    entities: List[EntityNode] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


class BatchExtractionResult(BaseModel):
    """Structured extraction output for a batch of chunks."""

    model_config = ConfigDict(extra="ignore")

    results: List[ChunkExtractionResult] = Field(default_factory=list)
