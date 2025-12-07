from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BaseAgentInput(BaseModel):
    """
    Common input schema for all financial agents.
    Standardizes ticker, query, and date handling.
    """

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL, NVDA).")
    query: str = Field(
        description="The user's original query or a specific sub-question."
    )
    metrics: Optional[List[str]] = Field(
        description="List of financial metrics to analyze."
    )

    start_date: Optional[datetime] = Field(
        default=None,
        description="Start date for the analysis window (format: YYYY-MM-DD).",
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End date for the analysis window (format: YYYY-MM-DD).",
    )

    @field_validator("start_date", "end_date", mode="before")
    def parse_dates(cls, v):
        """
        Universal date parser that handles strings, datetimes, and None.
        """
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                # Fallback for LLMs that might output ISO format or others
                try:
                    return datetime.fromisoformat(v)
                except ValueError:
                    raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        return v
