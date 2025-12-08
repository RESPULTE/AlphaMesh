from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BaseAgentInput(BaseModel):
    """
    The unified input schema shared by the Orchestrator and all Sub-Agents.
    """

    query: str = Field(description="The original user query for context.")
    vector_query: str = Field(
        description="The query optimized for vector store retrieval (e.g., 'news about AAPL stock price drop')."
    )
    ticker: Optional[str] = Field(description="The stock ticker symbol (e.g., AAPL).")
    metrics: Optional[List[str]] = Field(
        default_factory=list,
        description="List of financial metrics to analyze (if applicable).",
    )
    start_date: Optional[datetime] = Field(
        default=None, description="Start date (format: YYYY-MM-DD)."
    )
    end_date: Optional[datetime] = Field(
        default=None, description="End date (format: YYYY-MM-DD)."
    )

    @field_validator("start_date", "end_date", mode="before")
    def parse_dates(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                return datetime.fromisoformat(v)
        return v


class BaseAgentOutput(BaseModel, ABC):
    """
    An abstract base class for agent outputs. It enforces that each output
    type must know how to format itself into a string for the final LLM analyst.
    """

    agent_name: str = Field(
        description="The name of the agent that produced this output."
    )
    analysis: str = Field(
        description="The detailed analysis or primary output of the agent."
    )

    @abstractmethod
    def get_llm_context_str(self) -> str:
        """
        Formats the output's data into a string suitable for an LLM context.
        """
        raise NotImplementedError
