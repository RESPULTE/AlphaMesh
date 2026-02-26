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
    description: Optional[str] = None

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




class Sector(FinancialBaseDataPoint):
    """A specific economic sector."""
    name: str

class Industry(FinancialBaseDataPoint):
    """A specific industry within a sector."""
    name: str

class GlobalEvent(FinancialBaseDataPoint):
    """A significant global event."""
    name: str
    description: Optional[str] = None

class MacroTrend(FinancialBaseDataPoint):
    """A macroeconomic trend."""
    name: str
    description: Optional[str] = None

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
    targets: SkipValidation[Any] = None
    # supported_by: Connects to any other global nodes.
    supported_by: SkipValidation[Any] = None
    # threatened_by: Connects to any other global nodes.
    threatened_by: SkipValidation[Any] = None

    # Cognee Edge structures dictating how edge properties are attached:
    # - User NodeSet -> Thesis (Edge: `HoldsThesis`, property: `conviction_level`)
    #   Note: Do NOT create a distinct "User" Node; the user is represented as a Cognee NodeSet natively.
    # - Thesis -> GlobalEvent/MacroTrend (Edges: `SupportedBy` / `ThreatenedBy`, properties: `weight`, `added_on`)

    model_config = {"arbitrary_types_allowed": True}


'''
example of custom edge (relationship) between entities

from cognee.infrastructure.engine import Edge
from core.memory.graph_models import InvestmentThesis, Company
# 1. Instantiate the Target Node
nvidia_node = Company(
    ticker="NVDA",
    name="NVIDIA Corp",
    sector="Technology",
    description="Leading designer of AI accelerators."
)
# 2. Instantiate the Investment Thesis Node
thesis = InvestmentThesis(
    thesis_id="TH-NVDA-AI-2026",
    summary="NVIDIA will continue to dominate the AI accelerator market due to its CUDA moat.",
    status="Active"
)
# 3. Attach the target using Edge syntax
# You assign a tuple consisting of the Edge object and the target node(s).
thesis.targets = [
    (Edge(relationship_type="TARGETS"), nvidia_node)
]
# You can also use this same pattern for supported_by and threatened_by edges:
# thesis.supported_by = [(Edge(relationship_type="SUPPORTED_BY", weights=1.0), macro_trend_node)]
'''


# ---------------------------------------------------------------------------
# Rebuild all models (required for forward references)
# ---------------------------------------------------------------------------

FinancialBaseDataPoint.model_rebuild()
Company.model_rebuild()
FinancialConcept.model_rebuild()
Sector.model_rebuild()
Industry.model_rebuild()
GlobalEvent.model_rebuild()
MacroTrend.model_rebuild()
InvestmentThesis.model_rebuild()


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
            Industry, 
            GlobalEvent, 
            MacroTrend, 
            InvestmentThesis
        ]
    ] = Field(
        default_factory=list,
        description=(
            "All extracted financial entities. "
            "Each MUST have target_nodeset set to 'GLOBAL' or 'USER'."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}


FinancialKnowledgeGraph.model_rebuild()
