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

# FinancialConcept category taxonomy (12-category analog to sectors).
FINANCIAL_CONCEPT_CATEGORIES = {
    "Monetary Policy & Rates": "Central bank policy, interest rates, yield curves, and rate expectations.",
    "Inflation & Prices": "Price levels, inflation measures, purchasing power, and cost dynamics.",
    "Growth & Business Cycle": "Macro growth, recessions, employment, and cycle indicators.",
    "Corporate Fundamentals": "Earnings, revenue, margins, cash flow, balance sheets, and guidance.",
    "Valuation & Pricing": "Valuation frameworks, multiples, intrinsic value, and pricing models.",
    "Market Structure & Trading": "Liquidity, volume, order flow, market microstructure, and execution.",
    "Risk & Volatility": "Risk metrics, volatility, drawdowns, correlation, and portfolio risk.",
    "Credit & Fixed Income": "Bonds, yields, spreads, duration, and credit quality.",
    "Derivatives & Hedging": "Options, futures, swaps, Greeks, hedging, and structured products.",
    "FX & Global Markets": "Currencies, exchange rates, cross-border flows, and global linkages.",
    "Commodities & Real Assets": "Energy, metals, agriculture, and real-asset exposures.",
    "Regulation & Governance": "Regulation, compliance, governance, and market oversight.",
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
    "FinancialConceptCategory",
    "Sector",
    "Industry",
    "Market",
}

_EXTRACTABLE_ENTITY_TYPES = {"Company", "FinancialEvent", "FinancialConcept"}

ENTITY_TYPE_EXTRACTION_GUIDANCE: dict[str, str] = {
    "Company": (
        "Use a legally or commercially recognized organization name explicitly stated in the text. "
        "Prefer canonical company names over abbreviations when both are present. "
        "Do not infer companies that are not directly mentioned."
    ),
    "FinancialEvent": (
        "Capture concrete finance-relevant events explicitly described in the text, such as earnings releases, guidance revisions, "
        "M&A announcements, rating actions, funding activity, or regulatory developments. "
        "Keep names concise and specific enough to anchor downstream relationships."
    ),
    "FinancialConcept": (
        "Extract substantive finance concepts that explain mechanism, risk, valuation, liquidity, profitability, or market behavior in context. "
        "Descriptions should be insight-oriented and tied to source context rather than generic textbook definitions. "
        "Avoid trivial or overly broad concepts with low analytical value."
    ),
    "FinancialConceptCategory": (
        "Use this type only for canonical taxonomy categories that already exist in the system. "
        "Do not derive this type from free text extraction unless explicitly required by task constraints. "
        "Prefer linking FinancialConcept nodes to known categories instead of creating new category entities."
    ),
    "Sector": (
        "Represent broad economic sectors using canonical market taxonomy labels. "
        "Only extract sectors explicitly stated in the source text. "
        "Do not substitute implied sectors from outside knowledge."
    ),
    "Industry": (
        "Represent industry-level groupings narrower than sectors when explicitly mentioned in the text. "
        "Use canonical industry phrasing when multiple variants appear. "
        "Do not infer industry labels solely from a company's known business profile."
    ),
    "Market": (
        "Represent explicit market-level entities such as regions, exchanges, or broad market aggregates stated in context. "
        "Use concise canonical labels and avoid synthetic market terms. "
        "Only extract market entities directly supported by source text."
    ),
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
    "HAS_EVENT",
    "OBSERVED_IN",
]

# ── RelationshipType literal: add BELONGS_TO ─────────────────────────────────
GlobalRelationshipType = Literal[
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
    "HAS_EVENT",
    "OBSERVED_IN",
    "REPORTED_BY",
    "RELATED_TO",
    "BELONGS_TO",
]

RELATIONSHIP_TYPE_METADATA: dict[str, dict[str, str | float]] = {
    "AFFECTS": {
        "description": "A directional influence where one entity impacts the state, outcomes, or behavior of another.",
        "weight": 1.0,
    },
    "CAUSED_BY": {
        "description": "A causal link where an effect entity is explicitly caused by the source entity or event.",
        "weight": 0.95,
    },
    "INCREASES": {
        "description": "A directional relationship where the source contributes to or is associated with an increase in the target.",
        "weight": 0.85,
    },
    "DECREASES": {
        "description": "A directional relationship where the source contributes to or is associated with a decrease in the target.",
        "weight": 0.85,
    },
    "CORRELATED_WITH": {
        "description": "A non-causal association where two entities are described as moving or occurring together.",
        "weight": 0.70,
    },
    "EXPOSES_TO": {
        "description": "A relationship where one entity increases another entity's exposure to a risk, factor, or condition.",
        "weight": 0.65,
    },
    "MITIGATES": {
        "description": "A relationship where one entity reduces risk, severity, or impact associated with the target.",
        "weight": 0.55,
    },
    "COMPETES_WITH": {
        "description": "A competitive relationship between peer entities operating in overlapping markets or products.",
        "weight": 0.45,
    },
    "ACQUIRED_BY": {
        "description": "A corporate action relationship where the source entity is acquired by the target entity.",
        "weight": 0.40,
    },
    "RELATED_TO": {
        "description": "A generic fallback relationship for relevance when no more specific edge type is justified.",
        "weight": 0.10,
    },
    "BELONGS_TO": {
        "description": "A membership or classification relationship where the source is categorized under the target.",
        "weight": 0.20,
    },
    "HAS_INTEREST_IN": {
        "description": "A user-interest relationship indicating an interest domain or edge targets a specific entity.",
        "weight": 0.35,
    },
    "TARGETS": {
        "description": "A directional user-interest relationship linking an interest edge to its underlying target entity.",
        "weight": 0.30,
    },
    "HAS_EVENT": {
        "description": "A provenance relationship linking an interest edge to a concrete interest observation event.",
        "weight": 0.25,
    },
    "OBSERVED_IN": {
        "description": "A provenance relationship linking an observed event to the session context in which it occurred.",
        "weight": 0.20,
    },
}

_RELATIONSHIP_WEIGHTS: dict[str, float] = {
    relationship_type: float(metadata["weight"])
    for relationship_type, metadata in RELATIONSHIP_TYPE_METADATA.items()
}

# ── _ENTITY_TYPE_WEIGHTS: add Industry and Market ────────────────────────────
_ENTITY_TYPE_WEIGHTS: dict[str, float] = {
    "Company": 1.0,
    "FinancialEvent": 0.90,
    "FinancialConcept": 0.65,
    "FinancialConceptCategory": 0.60,
    "Industry": 0.55,
    "Sector": 0.45,
    "Market": 0.30,
}

# ── User-scoped node types: bypass fuzzy/semantic dedup in build_graph ─────────
_USER_SCOPED_TYPES = frozenset(
    {
        "UserInterestDomain",
        "UserInterestEdge",
        "UserInterestEvent",
        "SessionNode",
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
    Acts as a first-class node to carry aggregate stance and influence metadata.
    """

    model_config = ConfigDict(extra="ignore")
    id: str
    user_email: str
    domain_type: Literal["investment", "learning"]
    category: str
    entity_id: str  # resolved UUID of the target entity
    cumulative_weight: float = 0.0
    reinforcement_count: int = 0
    invalidation_count: int = 0
    current_stance: Literal["positive", "negative"] = "positive"
    last_changed_at: Optional[datetime] = None
    created_at: datetime
    last_updated_at: datetime


class SessionNode(BaseModel):
    """Session-scoped provenance anchor for user-interest events."""

    model_config = ConfigDict(extra="ignore")
    id: str
    started_at: datetime
    user_email: str


class UserInterestEvent(BaseModel):
    """Immutable event emitted whenever user interest stance is observed."""

    model_config = ConfigDict(extra="ignore")
    id: str
    user_email: str
    domain_type: Literal["investment", "learning"]
    category: str
    entity_id: str
    stance: Literal["positive", "negative"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    observed_at: datetime
    source_excerpt: str = ""


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
        "FinancialConceptCategory",
        "Sector",
        "Industry",
        "Market",
    ]

    # For Company/Sector/Industry/Market: description is the canonical yfinance
    # value and is NEVER overwritten by LLM extraction (enforced in neo4j_adapter).
    description: str
    # Ticker symbol — only populated for Company entities via yfinance enrichment.
    ticker: Optional[str] = None
    concept_categories: List[str] = Field(default_factory=list)
    nodeset_ids: List[str] = Field(default_factory=list)


class RelationshipExtractionItem(BaseModel):
    """Schema for one relationship item expected inside <relationships> prompts."""

    from_name: str
    from_type: str
    relationship_type: str
    to_name: str
    to_type: str
    confidence: str = Field(
        description='"high" for explicit evidence, "low" for inferred.'
    )
    reason: str = Field(description="1-2 short sentences explaining the relationship.")


class ChunkEntityExtractionResult(BaseModel):
    """Structured entity extraction output for a single chunk."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(
        default="", description="ignored by input, used for traceability in output"
    )
    entities: List[EntityNode] = Field(default_factory=list)


class BatchEntityExtractionResult(BaseModel):
    """Structured entity extraction output for a batch of chunks."""

    model_config = ConfigDict(extra="ignore")

    results: List[ChunkEntityExtractionResult] = Field(default_factory=list)

