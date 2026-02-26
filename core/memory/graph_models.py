"""
core/memory/graph_models.py

Custom Cognee DataPoint models for the financial AI assistant domain.

Architecture:
  - All entities inherit from FinancialBaseDataPoint
  - `target_nodeset`: string enum set by the LLM during cognify ("GLOBAL" or "USER")
  - `belongs_to_set`: inherited from DataPoint; assigned in post-processing to an
                     actual NodeSet DataPoint — NEVER by the LLM
  - Uses Cognee's built-in NodeSet (cognee.modules.engine.models.node_set)
  - FinancialKnowledgeGraph: top-level graph model passed to cognify()
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, SkipValidation

from cognee.infrastructure.engine import DataPoint, Edge
from cognee.modules.engine.models import Entity


# ---------------------------------------------------------------------------
# NodeSet target enum — only two legal values
# ---------------------------------------------------------------------------


class NodeSetTarget(str, Enum):
    GLOBAL = "GLOBAL"
    USER = "USER"


# ---------------------------------------------------------------------------
# Base for all financial domain entities
# ---------------------------------------------------------------------------


class FinancialBaseDataPoint(Entity):
    """
    Abstract base for every financial entity type.

    Adds:
        target_nodeset  — populated by the LLM during cognify.
                          Allowed: NodeSetTarget.GLOBAL | NodeSetTarget.USER
                          Post-processing validates this and assigns belongs_to_set.
    """

    target_nodeset: Optional[NodeSetTarget] = Field(
        default=None,
        description=(
            "REQUIRED: Set to 'GLOBAL' for public/shared data, "
            "or 'USER' for private per-user data. "
            "Populated by the LLM during entity extraction."
        ),
    )
    # `belongs_to_set` is inherited from DataPoint as Optional[List[DataPoint]]
    # Assigned by the post-processing task — never by the LLM.




# ---------------------------------------------------------------------------
# Domain DataPoint models
# ---------------------------------------------------------------------------


class Company(FinancialBaseDataPoint):
    """A publicly traded company or investment entity."""

    ticker: str
    name: str
    description: Optional[str] = None

    metadata: dict = {"index_fields": ["name", "ticker", "description"]}




class FinancialConcept(FinancialBaseDataPoint):
    """A financial concept, term, or educational definition. Always GLOBAL."""

    name: str
    definition: str
    category: Optional[
        Literal[
            "valuation", "technical_analysis", "fundamental_analysis",
            "macroeconomics", "risk", "derivatives", "portfolio_management", "other",
        ]
    ] = None
    related_concepts: Optional[List[str]] = None
    examples: Optional[List[str]] = None

    metadata: dict = {"index_fields": ["name", "definition"]}


class GlobalEntity(FinancialBaseDataPoint):
    """A global entity."""
    name: str
    description: Optional[str] = None
    related_to: list["GlobalEntity"]


class Sector(GlobalEntity):
    """A specific economic sector."""


class GlobalEvent(GlobalEntity):
    """A significant global event."""

class MacroTrend(GlobalEntity):
    """A macroeconomic trend."""

VALID_GLOBAL_INFLUENCE_TYPES = ["Company", "Sector", "GlobalEvent", "MacroTrend"]

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
    thesis_id: str
    summary: str
    status: Literal["Active", "Dormant", "Archived"]
    created_at: datetime
    metadata: dict = {"index_fields": ["summary"]}

    # Relationship fields (using SkipValidation[Any] to avoid forward reference issues)
    # targets: Connects to Company or Sector nodes.
    targets: list["Company"]






# ---------------------------------------------------------------------------
# Rebuild all models (required for forward references)
# ---------------------------------------------------------------------------

FinancialBaseDataPoint.model_rebuild()
Company.model_rebuild()
FinancialConcept.model_rebuild()
Sector.model_rebuild()
GlobalEvent.model_rebuild()
MacroTrend.model_rebuild()
InvestmentThesis.model_rebuild()
GlobalInfluence.model_rebuild()


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
            GlobalEvent, 
            MacroTrend, 
            InvestmentThesis,
            GlobalInfluence
        ]
    ] = Field(
        default_factory=list,
        description=(
            "All extracted financial entities. "
            "Each MUST have target_nodeset set to 'GLOBAL' or 'USER'."
        ),
    )




FinancialKnowledgeGraph.model_rebuild()
