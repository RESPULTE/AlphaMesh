from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GlobalEntityType(str, Enum):
    """Entities allowed in the global shared knowledge graph."""
    FINANCIAL_INSTRUMENT = "FinancialInstrument"
    ORGANIZATION = "Organization"
    PERSON = "Person"
    MARKET_EVENT = "MarketEvent"
    FINANCIAL_CONCEPT = "FinancialConcept"
    ECONOMIC_INDICATOR = "EconomicIndicator"
    REGULATION = "Regulation"


class UserEntityType(str, Enum):
    """Entities allowed in the user-specific private knowledge graph."""
    USER_PROFILE = "UserProfile"
    PORTFOLIO = "Portfolio"
    FINANCIAL_GOAL = "FinancialGoal"
    RISK_PROFILE = "RiskProfile"
    TRANSACTION = "Transaction"
    GLOBAL_REF = "GlobalRef"  # Stub referring to a global entity


class EntityModel(BaseModel):
    name: str
    entity_type: str
    description: str

    def to_extraction_hint(self) -> str:
        """Returns a string description for LLM prompt injection."""
        return f"{self.entity_type}: {self.description}"


GLOBAL_ENTITY_TYPES = [e.value for e in GlobalEntityType]
USER_ENTITY_TYPES = [e.value for e in UserEntityType]


class IngestionResult(BaseModel):
    user_id: str
    content_type: str  # 'conversation' or 'document'
    status: str = "pending"
    global_track_id: Optional[str] = None
    user_track_id: Optional[str] = None
    message: str = ""


class IngestionStatus(BaseModel):
    track_id: str
    namespace: str
    status: str
    message: Optional[str] = None
    error: Optional[str] = None


class QueryResult(BaseModel):
    user_id: str
    query: str
    mode: str
    global_context: str
    user_context: str
    merged_context: str
    cross_references: List[Dict[str, Any]] = []
