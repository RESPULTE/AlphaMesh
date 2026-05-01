from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserInterestEntityRef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity_name: str
    entity_type: Literal[
        "Company",
        "FinancialEvent",
        "FinancialConcept",
        "Sector",
    ]


class UserInterestQuerySpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    domain_type: Optional[Literal["investment", "learning"]] = None
    category: Optional[str] = None
    target_entities: List[UserInterestEntityRef] = Field(default_factory=list)
    hops: int = Field(default=0)
    broad_fallback: bool = Field(default=False)
    risk_or_avoidance_intent: bool = Field(default=False)

    @field_validator("hops", mode="before")
    @classmethod
    def _clamp_hops(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 0
        return max(0, min(2, parsed))


class UserInterestQueryResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_spec: Optional[UserInterestQuerySpec] = None
    context_block: str = "(none)"
    debug_payload: Dict[str, Any] = Field(default_factory=dict)
