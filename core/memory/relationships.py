# core/memory/relationships.py
"""
Custom edge (relationship) types for the Financial Knowledge Memory Module.

Global edge types define relationships between entities in the shared knowledge base.
User-specific edge types capture personalized interactions and preferences.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ==============================================================================
# GLOBAL EDGE TYPES
# These relationships exist in the GLOBAL namespace between shared entities.
# ==============================================================================


class RelatedETF(BaseModel):
    """Relationship between a Company and an ETF that holds it."""

    weight_percentage: Optional[float] = Field(
        None, description="Weight of the company in the ETF as a percentage"
    )
    as_of_date: Optional[datetime] = Field(
        None, description="Date when this relationship was established/updated"
    )


class AffectsSecurity(BaseModel):
    """Relationship between a News/Event and a Company/ETF it impacts."""

    impact_type: Optional[str] = Field(
        None, description="Type of impact (direct, indirect, sector-wide)"
    )
    impact_magnitude: Optional[str] = Field(
        None, description="Magnitude of impact (high, medium, low)"
    )
    impact_direction: Optional[str] = Field(
        None, description="Direction of impact (positive, negative, neutral)"
    )


class RelatedConcept(BaseModel):
    """Relationship between two financial concepts."""

    relationship_type: Optional[str] = Field(
        None,
        description="Type of relationship (prerequisite, related, opposite, component)",
    )
    strength: Optional[str] = Field(
        None, description="Strength of relationship (strong, moderate, weak)"
    )


# ==============================================================================
# USER-SPECIFIC EDGE TYPES (POSITIVE)
# These relationships capture positive user interactions and preferences.
# ==============================================================================


class Holds(BaseModel):
    """User holds a position in a security."""

    quantity: Optional[float] = Field(None, description="Number of shares held")
    purchase_date: Optional[datetime] = Field(None, description="Date of purchase")
    purchase_price: Optional[float] = Field(None, description="Price per share at purchase")
    portfolio_name: Optional[str] = Field(
        None, description="Name of the portfolio containing this position"
    )


class Watches(BaseModel):
    """User is actively tracking/watching an entity."""

    start_date: Optional[datetime] = Field(None, description="When tracking started")
    watchlist_name: Optional[str] = Field(
        None, description="Name of the watchlist containing this item"
    )
    alert_threshold: Optional[float] = Field(
        None, description="Price threshold for alerts"
    )


class LearnedAbout(BaseModel):
    """User has learned about a financial concept."""

    comprehension_level: Optional[str] = Field(
        None, description="User's comprehension level (basic, intermediate, advanced)"
    )
    learning_date: Optional[datetime] = Field(None, description="When the user learned this")
    source: Optional[str] = Field(
        None, description="Source of learning (article, video, course)"
    )


class Researched(BaseModel):
    """User has researched an entity in depth."""

    research_depth: Optional[str] = Field(
        None, description="Depth of research (surface, moderate, deep)"
    )
    research_date: Optional[datetime] = Field(None, description="When the research occurred")
    notes: Optional[str] = Field(None, description="Brief notes from the research")


class Interested(BaseModel):
    """User has expressed interest in an entity."""

    interest_level: Optional[str] = Field(
        None, description="Level of interest (low, moderate, high)"
    )
    reason: Optional[str] = Field(None, description="Reason for interest")
    expressed_date: Optional[datetime] = Field(
        None, description="When interest was expressed"
    )


# ==============================================================================
# USER-SPECIFIC EDGE TYPES (NEGATIVE)
# These relationships capture negative user interactions and avoidances.
# ==============================================================================


class Sold(BaseModel):
    """User sold a position in a security."""

    quantity: Optional[float] = Field(None, description="Number of shares sold")
    sale_date: Optional[datetime] = Field(None, description="Date of sale")
    sale_price: Optional[float] = Field(None, description="Price per share at sale")
    reason: Optional[str] = Field(
        None, description="Reason for selling (profit-taking, loss-cutting, rebalancing)"
    )


class Avoids(BaseModel):
    """User actively avoids an entity (company, sector, etc.)."""

    reason: Optional[str] = Field(
        None, description="Reason for avoidance (ethical, risk, past loss)"
    )
    since_date: Optional[datetime] = Field(None, description="When avoidance started")
    severity: Optional[str] = Field(
        None, description="Severity of avoidance (soft, hard)"
    )


class Unfollowed(BaseModel):
    """User stopped tracking/following an entity."""

    unfollow_date: Optional[datetime] = Field(None, description="When tracking stopped")
    reason: Optional[str] = Field(
        None, description="Reason for unfollowing (lost interest, too volatile)"
    )
    previous_watchlist: Optional[str] = Field(
        None, description="Watchlist it was removed from"
    )


class DistrustedSector(BaseModel):
    """User distrusts a sector or category."""

    reason: Optional[str] = Field(
        None, description="Reason for distrust (volatility, past experience, news)"
    )
    since_date: Optional[datetime] = Field(None, description="When distrust started")
    distrust_level: Optional[str] = Field(
        None, description="Level of distrust (cautious, avoidant, blacklisted)"
    )


# ==============================================================================
# EDGE TYPE REGISTRIES
# Used when calling graphiti.add_episode() with custom edge types.
# ==============================================================================

GLOBAL_EDGE_TYPES = {
    "RelatedETF": RelatedETF,
    "AffectsSecurity": AffectsSecurity,
    "RelatedConcept": RelatedConcept,
}

USER_POSITIVE_EDGE_TYPES = {
    "Holds": Holds,
    "Watches": Watches,
    "LearnedAbout": LearnedAbout,
    "Researched": Researched,
    "Interested": Interested,
}

USER_NEGATIVE_EDGE_TYPES = {
    "Sold": Sold,
    "Avoids": Avoids,
    "Unfollowed": Unfollowed,
    "DistrustedSector": DistrustedSector,
}

USER_EDGE_TYPES = {**USER_POSITIVE_EDGE_TYPES, **USER_NEGATIVE_EDGE_TYPES}

ALL_EDGE_TYPES = {**GLOBAL_EDGE_TYPES, **USER_EDGE_TYPES}


# ==============================================================================
# EDGE TYPE MAPPINGS
# Define which edge types can exist between specific entity type pairs.
# ==============================================================================

GLOBAL_EDGE_TYPE_MAP = {
    ("Company", "ETF"): ["RelatedETF"],
    ("ETF", "Company"): ["RelatedETF"],
    ("FinancialNews", "Company"): ["AffectsSecurity"],
    ("FinancialNews", "ETF"): ["AffectsSecurity"],
    ("FinancialEvent", "Company"): ["AffectsSecurity"],
    ("FinancialEvent", "ETF"): ["AffectsSecurity"],
    ("FinancialConcept", "FinancialConcept"): ["RelatedConcept"],
    ("Entity", "Entity"): ["AffectsSecurity", "RelatedConcept"],  # Fallback
}

USER_EDGE_TYPE_MAP = {
    # Portfolio relationships
    ("Portfolio", "Position"): ["Holds"],
    ("Portfolio", "EntityReference"): ["Holds", "Sold"],
    # Watchlist relationships
    ("Watchlist", "EntityReference"): ["Watches", "Unfollowed"],
    # User learning and research
    ("UserPreference", "EntityReference"): ["Interested", "Avoids", "DistrustedSector"],
    ("UserPreference", "FinancialConcept"): ["LearnedAbout"],
    ("UserGoal", "Portfolio"): ["Interested"],
    # Generic fallback for user namespace
    ("Entity", "Entity"): [
        "Holds",
        "Watches",
        "LearnedAbout",
        "Researched",
        "Interested",
        "Sold",
        "Avoids",
        "Unfollowed",
    ],
}
