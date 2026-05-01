"""
api/dependencies.py

FastAPI Depends() providers for shared service singletons.

All services are created once during application startup (via the lifespan
context manager in main.py) and stored on `app.state`. Routers retrieve
them through these dependency functions rather than importing globals directly,
which keeps each router independently testable.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth.adapter import get_auth_adapter
from api.services.analysis_runner import AnalysisRunner
from api.services.conversation_service import ConversationStore
from api.services.event_broadcaster import EventBroadcaster
from api.services.portfolio_json_store import PortfolioJsonStore
from api.services.session_service import SessionService
from core.market_data_service import MarketDataService
from core.services import service_manager

bearer = HTTPBearer(auto_error=False)


def _extract_token(
    request: Request,
    creds: HTTPAuthorizationCredentials | None,
) -> str | None:
    if creds and creds.credentials:
        return creds.credentials.strip()
    query_token = (request.query_params.get("token") or "").strip()
    return query_token or None


def get_broadcaster(request: Request) -> EventBroadcaster:
    return request.app.state.broadcaster


def get_store(request: Request) -> ConversationStore:
    return request.app.state.store


def get_runner(request: Request) -> AnalysisRunner:
    return request.app.state.runner


def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str:
    """
    Extract and validate user identity from Authorization: Bearer <token>.

    Returns the user's email string on success.
    Raises HTTP 401 on missing or invalid token.
    """
    token = _extract_token(request, creds)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    email = get_auth_adapter().verify_access_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    return email


def get_current_user_optional(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str | None:
    """
    Best-effort user extraction from Authorization header.
    Returns None when header is missing or invalid.
    """
    token = _extract_token(request, creds)
    if not token:
        return None
    return get_auth_adapter().verify_access_token(token)


# -- Service dependencies -----------------------------------------------------


def get_market_data_service() -> MarketDataService:
    """Provide a MarketDataService instance backed by the shared cache."""
    return service_manager.get_market_data_service()


def get_session_service(request: Request) -> SessionService:
    """Provide the SessionService singleton (initialised during lifespan startup)."""
    return request.app.state.session_service


def get_portfolio_store(request: Request) -> PortfolioJsonStore:
    """Provide the per-user JSON portfolio store singleton."""
    return request.app.state.portfolio_store
