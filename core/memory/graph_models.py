"""
core/memory/graph_models.py

Custom Cognee DataPoint models for the financial AI assistant domain.

Architecture:
  - All entities inherit from Entity
  - `target_nodeset`: string enum set by the LLM during cognify ("GLOBAL" or "USER")
  - `belongs_to_set`: inherited from DataPoint; assigned in post-processing to an
                     actual NodeSet DataPoint — NEVER by the LLM
  - Uses Cognee's built-in NodeSet (cognee.modules.engine.models.node_set)
  - FinancialKnowledgeGraph: top-level graph model passed to cognify()
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional, Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------

# Base for all financial domain entities
# ---------------------------------------------------------------------------

USER_SPECIFIC_ENTITIES = ["UserInvestmentInterest", "UserLearningInterest"]
DATASET_NAME = "alphamese_financial"

# Dedicated global nodesets for non-sector shared entities
GLOBAL_NODESET_NAME = "Market"
GLOBAL_FINANCIAL_WISDOM_NODESET = "Global Financial Wisdom"
GLOBAL_FINANCIAL_EVENT_NODESET = "Global Financial Event"

# Registry consumed by nodeset_manager to bootstrap these nodes at startup
GLOBAL_ENTITY_NODESETS: dict[str, str] = {
    GLOBAL_FINANCIAL_WISDOM_NODESET: (
        "A shared knowledge base of financial concepts, metrics, and educational definitions "
        "(e.g. Interest Rates, P/E Ratio, Inflation). Entities here are always globally accessible."
    ),
    GLOBAL_FINANCIAL_EVENT_NODESET: (
        "A shared repository of significant financial market and economic events "
        "(e.g. Fed rate cuts, GDP reports, earnings surprises). Entities here are always globally accessible."
    ),
    GLOBAL_NODESET_NAME: (
        "The overall market, representing the aggregate of all companies and sectors."
    ),
}

# ---------------------------------------------------------------------------
# Domain DataPoint models
# ---------------------------------------------------------------------------


class Sector(NodeSet):
    """A specific economic sector."""

    name: str = Field(description="Name of the main economic sector.")
    description: str = Field(description="Explanation of the sector's main activities.")
    metadata: dict = {"index_fields": ["name", "description"]}


class Industry(NodeSet):
    """
    A specific industrial niche or specialized business category within a primary economic Sector.
    Used for granular classification (e.g., 'Cloud Infrastructure' as a Industry of 'Information Technology').
    """

    name: str = Field(
        description="The precise name of the industrial niche or sub-category."
    )
    description: str = Field(
        description="A focused explanation of the niche segment's core business activities and scope."
    )
    metadata: dict = {"index_fields": ["name", "description"]}


class Company(DataPoint):
    """A publicly traded company or investment entity."""

    ticker: str = Field(description="Stock ticker symbol (e.g., AAPL).")
    name: str = Field(description="Full corporate name of the company.")
    description: str = Field(
        description="A brief description of this specific company."
    )
    sector: str = Field(
        description="The specific economic sector this company belongs to. Must match one of the predefined standard sectors: Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate."
    )
    industry: Optional[str] = Field(
        description="The specific industrial niche or specialized business category within a primary economic Sector. Used for granular classification (e.g., 'Cloud Infrastructure' as a Industry of 'Information Technology')."
    )

    metadata: dict = {"index_fields": ["name", "ticker", "description"]}


class FinancialConcept(DataPoint):
    """A financial concept, term, or educational definition. Always GLOBAL."""

    name: str = Field(description="Name of the financial concept or metric.")
    description: str = Field(description="Clear description of the concept.")
    category: Literal[
        "valuation",
        "technical_analysis",
        "fundamental_analysis",
        "macroeconomics",
        "risk",
        "derivatives",
        "portfolio_management",
        "other",
    ] = Field(description="Category of the concept.")

    related_concepts: Optional[List["FinancialConcept"]] = Field(
        default=None, description="Other conceptually related financial terms."
    )

    metadata: dict = {"index_fields": ["name", "description"]}


class FinancialEvent(DataPoint):
    """A significant financial event."""

    name: str = Field(description="Name of the financial event.")
    description: str = Field(description="Description of the financial event.")
    date: datetime = Field(
        description="Date of the event. Use the current date if none is found",
        default_factory=datetime.now,
    )

    positively_impacted: Optional[List[Union[Company, Sector]]] = Field(
        default=None,
        description="Companies or Sectors that are positively impacted by this event.",
    )
    negatively_impacted: Optional[List[Union[Company, Sector]]] = Field(
        default=None,
        description="Companies or Sectors that are negatively impacted by this event.",
    )
    metadata: dict = {"index_fields": ["name", "description"]}


class UserInvestmentInterestStatus(DataPoint):
    """
    Represents the current actionable status of a user's investment thesis.
    """

    status: Literal["Bought", "Interested", "Sold", "Avoids"] = Field(
        description="The definitive state of the user's investment perspective regarding the associated targets. E.g., 'Bought' if they own it, 'Avoids' if they explicitly decide against it."
    )
    metadata: dict = {"index_fields": ["status"]}


class UserInvestmentInterest(DataPoint):
    """
    Represents a structured investment thesis or perspective articulated by the user.
    It captures what the user thinks about specific financial entities and why.
    """

    status: UserInvestmentInterestStatus = Field(
        description="The current status of this investment thesis."
    )

    reason: str = Field(
        description="A detailed explanation or rationale for the user's investment thesis or interest."
    )

    # Relationship fields (using SkipValidation[Any] to avoid forward reference issues)
    # targets: Connects to Company or Sector nodes.
    targets: list[Union[Company, Sector]] = Field(
        description="The specific Company or Sector nodes that this investment thesis targets. These are the focal points of the user's interest."
    )
    supporting_events: Optional[list[FinancialEvent]] = Field(
        default=None,
        description="Financial events that bolster or justify this thesis.",
    )
    threatening_events: Optional[list[FinancialEvent]] = Field(
        default=None,
        description="Financial events that challenge, threaten, or pose direct risks to this thesis.",
    )
    metadata: dict = {"index_fields": ["reason", "status"]}


class UserLearningInterestStatus(DataPoint):
    """
    Represents the current state of a user's educational or learning journey regarding a specific topic.
    """

    status: Literal["Interested", "Understood", "Confused", "Not Interested"] = Field(
        description="The current state of the user's comprehension or curiosity about the topic. E.g., 'Confused' if they ask for clarification, 'Understood' if they indicate comprehension."
    )
    metadata: dict = {"index_fields": ["status"]}


class UserLearningInterest(DataPoint):
    """
    Represents a topic, concept, or event that the user wants to learn more about or understand better.
    This tracks the user's educational progress and curiosity.
    """

    reason: str = Field(
        description="The specific questions, confusion, or curiosity the user expressed, explaining why they are interested in learning about these targets."
    )

    status: UserLearningInterestStatus = Field(
        description="The current comprehension status of this learning interest."
    )

    targets: list[Union[FinancialConcept, FinancialEvent]] = Field(
        description="The specific predefined FinancialConcept or FinancialEvent nodes that the user is trying to understand."
    )

    metadata: dict = {"index_fields": ["reason", "status"]}


# ---------------------------------------------------------------------------
# Rebuild all models (required for forward references)
# ---------------------------------------------------------------------------

Company.model_rebuild()
FinancialConcept.model_rebuild()
Sector.model_rebuild()
UserInvestmentInterest.model_rebuild()
UserLearningInterest.model_rebuild()
FinancialEvent.model_rebuild()
Industry.model_rebuild()

ALL_ENTITIES = {
    "Company",
    "Sector",
    "FinancialConcept",
    "FinancialEvent",
    "UserInvestmentInterest",
    "UserLearningInterest",
    "Industry",
}

ALL_MAIN_SECTORS = {
    "Energy": "Companies involved in the exploration, production, and distribution of oil, gas, and renewable energy.",
    "Materials": "Includes chemical, construction material, glass, paper, forest product, and mining companies.",
    "Industrials": "Manufacturers and distributors of capital goods, including aerospace, defense, and machinery.",
    "Consumer Discretionary": "Businesses that sell non-essential goods and services, such as automotive, apparel, and leisure.",
    "Consumer Staples": "Essential product providers, including food, beverage, personal products, and household goods.",
    "Health Care": "Pharmaceuticals, biotechnology, medical devices, and healthcare service providers.",
    "Financials": "Banks, investment firms, insurance companies, and real estate finance entities.",
    "Information Technology": "Software, hardware, semiconductors, and IT service providers.",
    "Communication Services": "Telecommunications providers, media, entertainment, and interactive service companies.",
    "Utilities": "Providers of basic services including electricity, gas, and water.",
    "Real Estate": "Companies engaged in real estate development, management, and REITs.",
}


# ---------------------------------------------------------------------------
# Top-level graph model — passed as `graph_model` to cognify()
# ---------------------------------------------------------------------------


class FinancialKnowledgeGraph(BaseModel):
    """
    The schema the LLM populates during knowledge graph extraction.
    Passed to cognify() as: cognify(graph_model=FinancialKnowledgeGraph)
    """

    entities: List[
        Union[
            Company,
            FinancialConcept,
            UserInvestmentInterest,
            UserLearningInterest,
            FinancialEvent,
            Industry,
        ]
    ] = Field(
        default_factory=list,
        description=("All extracted financial entities."),
    )


FinancialKnowledgeGraph.model_rebuild()
