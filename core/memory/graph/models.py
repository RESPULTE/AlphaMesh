"""Pydantic models and constants for the Neo4j graph schema."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ENTITY_NAMESPACE = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

GLOBAL_ENTITY_NODESETS = {
    "Global Financial Wisdom": "A global anchor nodeset containing overarching FinancialConcept entities.",
    "Global Financial Events": "A global anchor nodeset containing broad FinancialEvent entities.",
}

# ── ALL_MAIN_SECTORS: update to yfinance canonical names ─────────────────────
ALL_MAIN_SECTORS = {
    "Technology": "Companies involved in research, development, and manufacturing of technologically based goods and services.",
    "Healthcare": "Companies providing medical services, manufacturing medical equipment or drugs, providing medical insurance.",
    "Financial Services": "Firms engaged in banking, investment services, insurance, and real estate.",
    "Consumer Cyclical": "Businesses that sell non-essential goods and services dependent on economic cycles.",
    "Consumer Defensive": "Companies that produce essential products used by consumers regardless of economic conditions.",
    "Energy": "Companies involved in the exploration, production, refining, and marketing of oil, gas, and renewable energy.",
    "Basic Materials": "Companies that discover, extract, and process raw materials.",
    "Industrials": "Firms that produce capital goods used in manufacturing, resource extraction, and construction.",
    "Utilities": "Companies providing essential public services such as water, gas, and electricity.",
    "Real Estate": "Companies involved in the development, operation, and management of real estate.",
    "Communication Services": "Companies that facilitate communication and offer entertainment content.",
}

# ── ALLOWED_ENTITY_TYPES: add Industry and Market ────────────────────────────
ALLOWED_ENTITY_TYPES = {
    "Company",
    "FinancialEvent",
    "FinancialConcept",
    "Sector",
    "Industry",
    "Market",
}

# ── ALLOWED_RELATIONSHIP_TYPES: add BELONGS_TO ───────────────────────────────
ALLOWED_RELATIONSHIP_TYPES = [
    "AFFECTS",
    "CAUSED_BY",
    "INCREASES",
    "DECREASES",
    "CORRELATED_WITH",
    "EXPOSES_TO",
    "MITIGATES",
    "COMPETES_WITH",
    "ACQUIRED_BY",
    "RELATED_TO",
    "BELONGS_TO",
    "HAS_INTEREST_IN",
    "TARGETS",
    "SOURCED_FROM",
    "INVALIDATED_BY",
]

# ── RelationshipType literal: add BELONGS_TO ─────────────────────────────────
RelationshipType = Literal[
    "AFFECTS",
    "CAUSED_BY",
    "INCREASES",
    "DECREASES",
    "CORRELATED_WITH",
    "EXPOSES_TO",
    "MITIGATES",
    "COMPETES_WITH",
    "ACQUIRED_BY",
    "HAS_INTEREST_IN",
    "TARGETS",
    "SOURCED_FROM",
    "INVALIDATED_BY",
    "REPORTED_BY",
    "RELATED_TO",
    "BELONGS_TO",
]

_RELATIONSHIP_WEIGHTS: dict[str, float] = {
    "AFFECTS": 1.0,
    "CAUSED_BY": 0.95,
    "BOOSTS": 0.85,
    "DRAGS": 0.85,
    "CORRELATED_WITH": 0.70,
    "EXPOSES_TO": 0.65,
    "MITIGATES": 0.55,
    "COMPETES_WITH": 0.45,
    "ACQUIRED_BY": 0.40,
    "RELATED_TO": 0.10,  # generic fallback — lowest priority
    "BELONGS_TO": 0.20,  # intermediate priority between RELATED_TO and specific causal/affective relationships
}

# ── _ENTITY_TYPE_WEIGHTS: add Industry and Market ────────────────────────────
_ENTITY_TYPE_WEIGHTS: dict[str, float] = {
    "Company": 1.0,
    "FinancialEvent": 0.90,
    "FinancialConcept": 0.65,
    "Industry": 0.55,
    "Sector": 0.45,
    "Market": 0.30,
}

# ── User-scoped node types: bypass fuzzy/semantic dedup in build_graph ─────────
_USER_SCOPED_TYPES = frozenset(
    {
        "UserInterestDomain",
        "UserInterestEdge",
        "TurnNode",
    }
)

# ── New models ─────────────────────────────────────────────────────────────────


class UserInterestDomain(BaseModel):
    """Semantic grouping node anchoring a user's interests by category."""

    model_config = ConfigDict(extra="ignore")
    id: str
    user_email: str
    domain_type: Literal["investment", "learning"]
    category: str  # sector name for investment; concept category for learning
    created_at: datetime


class UserInterestEdge(BaseModel):
    """
    Reified edge tracking a user's interest in a specific entity.
    Acts as a first-class node to carry provenance, weight, and status.
    """

    model_config = ConfigDict(extra="ignore")
    id: str
    user_email: str
    domain_type: Literal["investment", "learning"]
    category: str
    entity_id: str  # resolved UUID of the target entity
    weight: float = 0.0  # cumulative confidence signal; incremented on reinforce
    status: Literal["Active", "Invalidated", "Paused"] = "Active"
    invalidated: bool = False
    created_at: datetime
    last_updated_at: datetime


class TurnNode(BaseModel):
    """
    Represents a single conversation turn for full provenance tracking.
    UserInterestEdge nodes link to TurnNodes via SOURCED_FROM / INVALIDATED_BY.
    """

    model_config = ConfigDict(extra="ignore")
    id: str  # turn_id (uuid4 per OrchestratorAgent.run() call)
    conversation_id: str
    user_message_excerpt: str  # first 200 chars of user message
    created_at: datetime


class DocumentMetadata(BaseModel):
    """Metadata contract for a document node."""

    model_config = ConfigDict(extra="ignore")

    document_id: str
    title: str
    source_url: str
    published_at: datetime


class DocumentNode(BaseModel):
    """Graph node contract for a document-level anchor."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    source_url: str
    published_at: datetime
    ingested_at: datetime
    nodeset_ids: List[str] = Field(default_factory=list)


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
        "Industry",
        "Market",
    ]
    # For Company/Sector/Industry/Market: description is the canonical yfinance
    # value and is NEVER overwritten by LLM extraction (enforced in neo4j_adapter).
    description: str
    # Ticker symbol — only populated for Company entities via yfinance enrichment.
    ticker: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    nodeset_ids: List[str] = Field(default_factory=list)


class UserInvestmentInterestNode(BaseModel):
    """User-scoped investment interest node (not a domain entity)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    user_email: str
    status: Literal["Bought", "Interested", "Sold", "Avoids"]
    reason: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    updated_at: datetime
    target_entity_ids: List[str] = Field(default_factory=list)


class UserLearningInterestNode(BaseModel):
    """User-scoped learning interest node (not a domain entity)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    user_email: str
    status: Literal["Interested", "Understood", "Confused", "Not Interested"]
    reason: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    updated_at: datetime
    target_entity_ids: List[str] = Field(default_factory=list)


class ExtractedRelationship(BaseModel):
    """Intermediate relationship model returned by the extraction call."""

    model_config = ConfigDict(extra="ignore")

    source_entity_local_id: str
    target_entity_local_id: str
    relationship_type: RelationshipType
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
