# core/memory/entities.py
"""
Custom entity types for the Financial Knowledge Memory Module.

Global entity types store detailed financial knowledge shared across all users.
User-specific entity types store lightweight references and personalized data.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ==============================================================================
# GLOBAL ENTITY TYPES
# These entities are stored in the GLOBAL namespace and shared across all users.
# They contain detailed, slowly-changing financial information.
# ==============================================================================


class Company(BaseModel):
    """A publicly traded company with comprehensive financial details."""

    ticker: Optional[str] = Field(None, description="Stock ticker symbol (e.g., AAPL)")
    sector: Optional[str] = Field(
        None, description="Business sector (e.g., Technology, Healthcare)"
    )
    exchange: Optional[str] = Field(
        None, description="Stock exchange (e.g., NYSE, NASDAQ)"
    )
    founded_year: Optional[int] = Field(None, description="Year the company was founded")
    employee_count: Optional[int] = Field(None, description="Approximate number of employees")


class ETF(BaseModel):
    """An Exchange-Traded Fund with key characteristics."""

    ticker: Optional[str] = Field(None, description="ETF ticker symbol (e.g., SPY)")
    sector: Optional[str] = Field(
        None, description="Primary sector focus (e.g., Technology, Broad Market)"
    )
    exchange: Optional[str] = Field(
        None, description="Stock exchange (e.g., NYSE, NASDAQ)"
    )
    expense_ratio: Optional[float] = Field(
        None, description="Annual expense ratio as a percentage"
    )
    holdings_count: Optional[int] = Field(
        None, description="Number of holdings in the ETF"
    )


class FinancialNews(BaseModel):
    """A financial news item or article."""

    category: Optional[str] = Field(
        None,
        description="News category (e.g., Earnings, Macro, Regulatory, M&A)",
    )
    sentiment: Optional[str] = Field(
        None, description="Sentiment analysis result (positive, negative, neutral)"
    )
    source: Optional[str] = Field(None, description="News source (e.g., Reuters, Bloomberg)")
    publish_date: Optional[datetime] = Field(None, description="Publication date and time")


class FinancialEvent(BaseModel):
    """A significant financial event (earnings, Fed decisions, IPOs, etc.)."""

    event_type: Optional[str] = Field(
        None,
        description="Type of event (e.g., Earnings, Fed Meeting, IPO, M&A)",
    )
    impact_level: Optional[str] = Field(
        None, description="Expected impact level (high, medium, low)"
    )
    affected_sectors: Optional[str] = Field(
        None, description="Comma-separated list of affected sectors"
    )
    event_date: Optional[datetime] = Field(None, description="Date of the event")


class FinancialConcept(BaseModel):
    """A financial concept, term, or strategy."""

    category: Optional[str] = Field(
        None,
        description="Concept category (e.g., Ratio, Strategy, Indicator, Term)",
    )
    complexity_level: Optional[str] = Field(
        None, description="Complexity level (beginner, intermediate, advanced)"
    )
    related_concepts: Optional[str] = Field(
        None, description="Comma-separated list of related concept names"
    )


# ==============================================================================
# USER-SPECIFIC ENTITY TYPES
# These entities are stored in user-specific namespaces (user_{id}).
# They contain personalized data and lightweight references to global entities.
# ==============================================================================


class User(BaseModel):
    """
    The user entity representing an individual investor.
    
    Profile information about the user. Note that interests and preferences
    toward specific entities should be captured via edges (e.g., INTERESTED_IN,
    AVOIDS) linking to global entities, not as fields here.
    """

    age: Optional[int] = Field(None, description="User's age in years")
    occupation: Optional[str] = Field(None, description="User's occupation or profession")
    income_range: Optional[str] = Field(
        None, description="Approximate income range (e.g., '50k-75k', '100k-150k')"
    )
    experience_level: Optional[str] = Field(
        None, description="Investment experience level (beginner, intermediate, advanced)"
    )
    monthly_expenditure: Optional[float] = Field(
        None, description="Approximate monthly expenditure in primary currency"
    )
    savings_rate: Optional[float] = Field(
        None, description="Percentage of income saved/invested monthly"
    )
    location: Optional[str] = Field(
        None, description="General location or region (e.g., 'US-West', 'Singapore')"
    )


class EntityReference(BaseModel):
    """
    A lightweight reference to a global entity.
    
    This allows user-specific graphs to reference shared knowledge without
    duplicating the detailed entity data. Uses global_entity_uuid as a foreign key.
    """

    global_entity_uuid: Optional[str] = Field(
        None, description="UUID of the referenced entity in the GLOBAL namespace"
    )
    reference_type: Optional[str] = Field(
        None,
        description="Type of reference (e.g., Company, ETF, FinancialConcept)",
    )
    semantic_context: Optional[str] = Field(
        None, description="Brief context for why this reference exists in user's graph"
    )


class Portfolio(BaseModel):
    """A user's investment portfolio."""

    portfolio_name: Optional[str] = Field(None, description="Name of the portfolio")
    creation_date: Optional[datetime] = Field(
        None, description="When the portfolio was created"
    )
    total_value: Optional[float] = Field(
        None, description="Current total value of the portfolio"
    )
    currency: Optional[str] = Field(None, description="Currency (e.g., USD, EUR)")


class Watchlist(BaseModel):
    """A user's watchlist for tracking securities of interest."""

    watchlist_name: Optional[str] = Field(None, description="Name of the watchlist")
    creation_date: Optional[datetime] = Field(
        None, description="When the watchlist was created"
    )
    description: Optional[str] = Field(None, description="Description of the watchlist purpose")


class Position(BaseModel):
    """
    A user's position in a specific security.
    
    References a global entity (Company or ETF) via global_entity_uuid.
    """

    global_entity_uuid: Optional[str] = Field(
        None, description="UUID of the Company/ETF in the GLOBAL namespace"
    )
    quantity: Optional[float] = Field(None, description="Number of shares held")
    average_cost: Optional[float] = Field(
        None, description="Average cost per share"
    )
    purchase_date: Optional[datetime] = Field(
        None, description="Date of initial purchase"
    )


class UserPreference(BaseModel):
    """User's financial preferences and risk profile."""

    risk_profile: Optional[str] = Field(
        None, description="Risk tolerance (conservative, moderate, aggressive)"
    )
    investment_horizon: Optional[str] = Field(
        None, description="Investment horizon (short-term, medium-term, long-term)"
    )
    preferred_sectors: Optional[str] = Field(
        None, description="Comma-separated list of preferred sectors"
    )
    avoided_sectors: Optional[str] = Field(
        None, description="Comma-separated list of sectors to avoid"
    )


class UserGoal(BaseModel):
    """A user's financial goal."""

    goal_name: Optional[str] = Field(None, description="Name of the goal")
    target_amount: Optional[float] = Field(None, description="Target amount in currency")
    current_amount: Optional[float] = Field(None, description="Current amount saved")
    target_date: Optional[datetime] = Field(None, description="Target date to achieve goal")
    priority: Optional[str] = Field(None, description="Priority level (high, medium, low)")


# ==============================================================================
# ENTITY TYPE REGISTRIES
# Used when calling graphiti.add_episode() with custom entity types.
# ==============================================================================

GLOBAL_ENTITY_TYPES = {
    "Company": Company,
    "ETF": ETF,
    "FinancialNews": FinancialNews,
    "FinancialEvent": FinancialEvent,
    "FinancialConcept": FinancialConcept,
}

USER_ENTITY_TYPES = {
    "User": User,
    "EntityReference": EntityReference,
    "Portfolio": Portfolio,
    "Watchlist": Watchlist,
    "Position": Position,
    "UserPreference": UserPreference,
    "UserGoal": UserGoal,
}

# Combined registry for unified episode processing
ALL_ENTITY_TYPES = {**GLOBAL_ENTITY_TYPES, **USER_ENTITY_TYPES}
