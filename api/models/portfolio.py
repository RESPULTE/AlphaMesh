"""
api/models/portfolio.py

Request/response schemas for portfolio and ticker search endpoints.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


AssetType = Literal["equity", "etf"]


class PortfolioHolding(BaseModel):
    ticker: str = Field(
        description="Ticker symbol (canonical uppercase form).",
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$",
    )
    company_name: str = Field(
        description="Human-readable company or fund name.",
        min_length=1,
        max_length=200,
    )
    exchange: Optional[str] = Field(
        default=None,
        description="Primary listing exchange when available.",
        max_length=40,
    )
    asset_type: AssetType = Field(description="Asset class for this holding.")
    shares: float = Field(
        description="Total shares currently held.",
        gt=0,
        allow_inf_nan=False,
    )


class UpsertPortfolioHoldingRequest(BaseModel):
    user_email: str = Field(
        description="User identity key for per-user portfolio files.",
        min_length=3,
        max_length=320,
    )
    ticker: str = Field(
        description="Ticker symbol to upsert (case-insensitive).",
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$",
    )
    company_name: str = Field(
        description="Company/fund display name for this ticker.",
        min_length=1,
        max_length=200,
    )
    exchange: Optional[str] = Field(
        default=None,
        description="Primary listing exchange when available.",
        max_length=40,
    )
    asset_type: AssetType = Field(description="Asset type for ticker search results.")
    shares: float = Field(
        description="Absolute total shares to set for this ticker.",
        gt=0,
        allow_inf_nan=False,
    )


class PortfolioResponse(BaseModel):
    user_email: str
    holdings: List[PortfolioHolding] = Field(default_factory=list)


class TickerSearchResult(BaseModel):
    ticker: str
    company_name: str
    exchange: Optional[str] = None
    asset_type: AssetType


class TickerSearchResponse(BaseModel):
    results: List[TickerSearchResult] = Field(default_factory=list)

