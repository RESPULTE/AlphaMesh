"""Pydantic models and constants for the Neo4j graph schema."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ENTITY_NAMESPACE = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

GLOBAL_NODESET_NAME = "GLOBAL"

GLOBAL_ENTITY_NODESETS = {
    "Global Financial Wisdom": "A global anchor nodeset containing overarching FinancialConcept entities.",
    "Global Financial Events": "A global anchor nodeset containing broad FinancialEvent entities.",
}

ALL_MAIN_SECTORS = {
    "Technology": "Companies involved in research, development, and manufacturing of technologically based goods and services.",
    "Healthcare": "Companies providing medical services, manufacturing medical equipment or drugs, providing medical insurance.",
    "Financials": "Firms engaged in banking, investment services, insurance, and real estate.",
    "Consumer Discretionary": "Businesses that sell non-essential goods and services.",
    "Consumer Staples": "Companies that produce essential products used by consumers.",
    "Energy": "Companies involved in the exploration, production, refining, and marketing of oil, gas, and renewable energy.",
    "Materials": "Companies that discover, extract, and process raw materials.",
    "Industrials": "Firms that produce capital goods used in manufacturing, resource extraction, and construction.",
    "Utilities": "Companies providing essential public services such as water, gas, and electricity.",
    "Real Estate": "Companies involved in the development, operation, and management of real estate.",
    "Communication Services": "Companies that facilitate communication and offer entertainment content.",
    "Transportation": "Companies involved in the movement of goods and people.",
}


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
    article_title: str
    source_url: str
    published_at: datetime
    companies_involved: List[str] = Field(default_factory=list)
    nodeset_ids: List[str] = Field(default_factory=list)
    extraction_status: Literal["PENDING", "EXTRACTED"] = "PENDING"


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

