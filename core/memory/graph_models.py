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
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Base for all financial domain entities
# ---------------------------------------------------------------------------

USER_SPECIFIC_ENTITIES = ["InvestmentThesis"]

# ---------------------------------------------------------------------------
# Domain DataPoint models
# ---------------------------------------------------------------------------


class FinancialEntity(DataPoint):
    """A global entity."""

    name: str
    description: str
    metadata: dict = {"index_fields": ["name", "description"]}


class Sector(FinancialEntity):
    """A specific economic sector."""

    name: str
    description: str


class Company(FinancialEntity):
    """A publicly traded company or investment entity."""

    ticker: str
    name: str

    metadata: dict = {"index_fields": ["name", "ticker", "description"]}


class FinancialConcept(FinancialEntity):
    """A financial concept, term, or educational definition. Always GLOBAL."""

    name: str
    definition: str
    category: Literal[
        "valuation",
        "technical_analysis",
        "fundamental_analysis",
        "macroeconomics",
        "risk",
        "derivatives",
        "portfolio_management",
        "other",
    ]

    related_concepts: Optional[List[str]] = None

    metadata: dict = {"index_fields": ["name", "definition"]}


class FinancialEvent(FinancialEntity):
    """A significant financial event."""

    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

    positively_impacted: Optional[List[Union[Company, Sector]]] = None
    negatively_impacted: Optional[List[Union[Company, Sector]]] = None


class InvestmentThesis(DataPoint):
    """
    An individual's or agent's structured investment thesis.
    Links to targeted Entities (Companies/Sectors), and supporting/threatening events.
    """

    status: Literal["Bought", "Interested", "Sold", "Avoids"]

    # Relationship fields (using SkipValidation[Any] to avoid forward reference issues)
    # targets: Connects to Company or Sector nodes.
    targets: list[Union[Company, Sector]]
    supporting_events: Optional[list[FinancialEvent]] = None
    threatening_events: Optional[list[FinancialEvent]] = None


# ---------------------------------------------------------------------------
# Rebuild all models (required for forward references)
# ---------------------------------------------------------------------------

Company.model_rebuild()
FinancialConcept.model_rebuild()
Sector.model_rebuild()
InvestmentThesis.model_rebuild()
FinancialEvent.model_rebuild()
FinancialEntity.model_rebuild()


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
            InvestmentThesis,
            FinancialEvent,
            FinancialEntity,
        ]
    ] = Field(
        default_factory=list,
        description=("All extracted financial entities."),
    )


FinancialKnowledgeGraph.model_rebuild()
