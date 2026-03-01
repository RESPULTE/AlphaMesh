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
from cognee.modules.engine.models import Entity
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Base for all financial domain entities
# ---------------------------------------------------------------------------

USER_SPECIFIC_ENTITIES = ["InvestmentThesis"]

# ---------------------------------------------------------------------------
# Domain DataPoint models
# ---------------------------------------------------------------------------


class Company(Entity):
    """A publicly traded company or investment entity."""

    ticker: str
    name: str

    metadata: dict = {"index_fields": ["name", "ticker", "description"]}


class FinancialConcept(Entity):
    """A financial concept, term, or educational definition. Always GLOBAL."""

    name: str
    definition: str
    category: Optional[
        Literal[
            "valuation",
            "technical_analysis",
            "fundamental_analysis",
            "macroeconomics",
            "risk",
            "derivatives",
            "portfolio_management",
            "other",
        ]
    ] = None
    related_concepts: Optional[List[str]] = None
    examples: Optional[List[str]] = None

    metadata: dict = {"index_fields": ["name", "definition"]}


class FinancialEntity(Entity):
    """A global entity."""

    name: str
    related_to: list["FinancialEntity"]


class Sector(FinancialEntity):
    """A specific economic sector."""


class FinancialEvent(FinancialEntity):
    """A significant financial event."""

    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class GlobalInfluence(DataPoint):
    """
    Extracts named influence relationships between global entities.
    Examples: POSITIVE_AFFECT, NEGATIVE_AFFECT, INCREASES_RISK.
    """

    source_id: str
    target_id: str
    relationship_name: str
    weight: Optional[float] = None
    evidence: Optional[str] = None

    metadata: dict = {"index_fields": ["source_id", "target_id", "relationship_name"]}


class InvestmentThesis(DataPoint):
    """
    An individual's or agent's structured investment thesis.
    Links to targeted Entities (Companies/Sectors), and supporting/threatening events.
    """

    summary: str
    status: Literal["Active", "Dormant", "Archived"]
    metadata: dict = {"index_fields": ["summary"]}

    # Relationship fields (using SkipValidation[Any] to avoid forward reference issues)
    # targets: Connects to Company or Sector nodes.
    targets: list["Company"]


# ---------------------------------------------------------------------------
# Rebuild all models (required for forward references)
# ---------------------------------------------------------------------------

Entity.model_rebuild()
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
            Sector,
            InvestmentThesis,
            FinancialEvent,
            FinancialEntity,
        ]
    ] = Field(
        default_factory=list,
        description=("All extracted financial entities."),
    )


FinancialKnowledgeGraph.model_rebuild()
