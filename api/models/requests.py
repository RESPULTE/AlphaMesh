"""
api/models/requests.py

Inbound request schemas for the AlphaMesh FastAPI layer.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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


class AuthEmailRequest(BaseModel):
    """Body for login/sign-up endpoints."""

    email: str = Field(
        description="User email used as identity key.",
        min_length=3,
        max_length=320,
    )

    @field_validator("email")
    @classmethod
    def validate_and_normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL_RE.fullmatch(normalized):
            raise ValueError("Invalid email format")
        return normalized


class AuthRefreshRequest(BaseModel):
    """Body for token refresh endpoint."""

    refresh_token: str = Field(
        description="Refresh token issued by login/sign-up/refresh.",
        min_length=1,
        max_length=4096,
    )
