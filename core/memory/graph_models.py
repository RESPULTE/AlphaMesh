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

USER_SPECIFIC_ENTITIES = ["InvestmentInterest"]

# ---------------------------------------------------------------------------
# Domain DataPoint models
# ---------------------------------------------------------------------------


class Sector(NodeSet):
    """A specific economic sector."""

    name: str = Field(description="Name of the economic sector.")
    description: str = Field(description="Explanation of the sector's main activities.")
    metadata: dict = {"index_fields": ["name", "description"]}


class Company(DataPoint):
    """A publicly traded company or investment entity."""

    ticker: str = Field(description="Stock ticker symbol (e.g., AAPL).")
    name: str = Field(description="Full corporate name of the company.")
    sector: str = Field(
        description="The specific economic sector this company belongs to. Must match one of the predefined standard sectors: Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate."
    )
    description: str = Field(
        description="A brief description of this specific company."
    )

    metadata: dict = {"index_fields": ["company_name", "ticker", "description"]}


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
    from_date: Optional[datetime] = Field(
        default=None, description="Start date of the event, if applicable."
    )
    to_date: Optional[datetime] = Field(
        default=None, description="End date of the event, if applicable."
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


class InvestmentInterest(DataPoint):
    """
    An individual's or agent's structured investment thesis.
    Links to targeted Entities (Companies/Sectors), and supporting/threatening events.
    """

    status: Literal["Bought", "Interested", "Sold", "Avoids"] = Field(
        description="The current status of this investment thesis."
    )

    reason: str = Field(
        description="The reason for this investment thesis as stated by the user."
    )

    # Relationship fields (using SkipValidation[Any] to avoid forward reference issues)
    # targets: Connects to Company or Sector nodes.
    targets: list[Union[Company, Sector]] = Field(
        description="The specific Company or Sector nodes this thesis targets."
    )
    supporting_events: Optional[list[FinancialEvent]] = Field(
        default=None, description="Financial events that support this thesis."
    )
    threatening_events: Optional[list[FinancialEvent]] = Field(
        default=None,
        description="Financial events that threaten or pose risks to this thesis.",
    )
    metadata: dict = {"index_fields": ["reason"]}


# ---------------------------------------------------------------------------
# Rebuild all models (required for forward references)
# ---------------------------------------------------------------------------

Company.model_rebuild()
FinancialConcept.model_rebuild()
Sector.model_rebuild()
InvestmentInterest.model_rebuild()
FinancialEvent.model_rebuild()

ALL_ENTITIES = {
    "Company",
    "Sector",
    "FinancialConcept",
    "FinancialEvent",
    "InvestmentInterest",
}

ALL_SECTORS = {
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
    "Market": "The overall market, representing the aggregate of all companies and sectors.",
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
            InvestmentInterest,
            FinancialEvent,
        ]
    ] = Field(
        default_factory=list,
        description=("All extracted financial entities."),
    )


FinancialKnowledgeGraph.model_rebuild()
