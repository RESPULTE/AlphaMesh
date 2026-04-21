"""
api/models/requests.py

Inbound request schemas for the AlphaMesh FastAPI layer.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body for POST /api/v1/chat — starts one analysis turn."""

    message: str = Field(
        description="The user's natural-language query.",
        min_length=1,
        max_length=4_000,
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing conversation to continue. "
            "Omit (or pass null) to start a fresh conversation."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing login session id. "
            "When omitted, server creates or reuses an active session."
        ),
    )
    user_email: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated compatibility field. "
            "Use authenticated token identity instead."
        ),
    )
