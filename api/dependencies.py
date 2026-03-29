"""
api/dependencies.py

FastAPI Depends() providers for shared service singletons.

All services are created once during application startup (via the lifespan
context manager in main.py) and stored on `app.state`.  Routers retrieve
them through these dependency functions rather than importing globals directly,
which keeps each router independently testable.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth.adapter import get_auth_adapter
from api.services.analysis_runner import AnalysisRunner
from api.services.conversation_store import ConversationStore
from api.services.event_broadcaster import EventBroadcaster
from api.services.market_data_service import MarketDataService
from api.services.session_service import SessionService

bearer = HTTPBearer(auto_error=False)


def get_broadcaster(request: Request) -> EventBroadcaster:
    return request.app.state.broadcaster


def get_store(request: Request) -> ConversationStore:
    return request.app.state.store


def get_runner(request: Request) -> AnalysisRunner:
    return request.app.state.runner


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str:
    """
    Extract and validate user identity from Authorization: Bearer <token>.

    Returns the user's email string on success.
    Raises HTTP 401 on missing or invalid token.
    """
    if not creds:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    email = get_auth_adapter().verify_access_token(creds.credentials)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    return email


def get_current_user_from_query(
    token: str = Query(..., description="JWT access token (for SSE endpoints)"),
) -> str:
    """
    Extract and validate user identity from the `token` query parameter.

    SSE endpoints (GET /api/analyze) must use this variant because the
    browser EventSource API does not allow setting custom request headers.

    Returns the user's email string on success.
    Raises HTTP 401 on missing or invalid token.
    """
    email = get_auth_adapter().verify_access_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    return email


# ── Service dependencies (instantiated per-request via Depends) ───────────────
# Keeping them as Depends factories decouples routers from concrete classes and
# makes unit testing straightforward — swap the dependency override.


def get_market_data_service() -> MarketDataService:
    """Provide a MarketDataService instance (backed by the shared module-level cache)."""
    return MarketDataService()


def get_session_service(request: Request) -> SessionService:
    """Provide the SessionService singleton (initialised during lifespan startup)."""
    return request.app.state.session_service
