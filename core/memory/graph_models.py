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

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from cognee.infrastructure.engine import DataPoint
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

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Domain DataPoint models
# ---------------------------------------------------------------------------


class Company(FinancialBaseDataPoint):
    """A publicly traded company or investment entity."""

    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    description: Optional[str] = None
    market_cap_usd: Optional[float] = None
    country: Optional[str] = None

    metadata: dict = {"index_fields": ["name", "ticker", "description"]}
    model_config = {"arbitrary_types_allowed": True}




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
    model_config = {"arbitrary_types_allowed": True}




# ---------------------------------------------------------------------------
# Rebuild all models (required for forward references)
# ---------------------------------------------------------------------------

FinancialBaseDataPoint.model_rebuild()
Company.model_rebuild()
FinancialConcept.model_rebuild()


# ---------------------------------------------------------------------------
# Top-level graph model — passed as `graph_model` to cognify()
# ---------------------------------------------------------------------------


class FinancialKnowledgeGraph(BaseModel):
    """
    The schema the LLM populates during knowledge graph extraction.
    Passed to cognify() as: cognify(graph_model=FinancialKnowledgeGraph)
    """

    entities: List[
        Union[Company, FinancialConcept]
    ] = Field(
        default_factory=list,
        description=(
            "All extracted financial entities. "
            "Each MUST have target_nodeset set to 'GLOBAL' or 'USER'."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}


FinancialKnowledgeGraph.model_rebuild()
