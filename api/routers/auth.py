from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.auth.adapter import get_auth_adapter
from api.dependencies import get_session_service
from api.models.requests import AuthEmailRequest, AuthRefreshRequest
from api.models.responses import AuthResponse, LogoutResponse
from api.services.session_service import SessionService
from core.config import settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _issue_tokens(email: str) -> tuple[str, str]:
    adapter = get_auth_adapter()
    access_token = adapter.create_access_token(email)
    refresh_token = adapter.create_refresh_token(email)
    return access_token, refresh_token


async def _build_auth_response(email: str, svc: SessionService) -> AuthResponse:
    access_token, refresh_token = _issue_tokens(email)
    session_id = await svc.ensure_session(user_id=email)
    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_email=email,
        session_id=session_id,
    )


@router.post("/signup", response_model=AuthResponse, summary="Sign up with email")
async def signup(
    body: AuthEmailRequest,
    svc: SessionService = Depends(get_session_service),
) -> AuthResponse:
    return await _build_auth_response(body.email, svc)


@router.post("/login", response_model=AuthResponse, summary="Login with email")
async def login(
    body: AuthEmailRequest,
    svc: SessionService = Depends(get_session_service),
) -> AuthResponse:
    return await _build_auth_response(body.email, svc)


@router.post("/refresh", response_model=AuthResponse, summary="Refresh access token")
async def refresh(
    body: AuthRefreshRequest,
    svc: SessionService = Depends(get_session_service),
) -> AuthResponse:
    email = get_auth_adapter().verify_refresh_token(body.refresh_token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    return await _build_auth_response(email, svc)


@router.post("/logout", response_model=LogoutResponse, summary="Logout current client session")
async def logout() -> LogoutResponse:
    return LogoutResponse(status="ok")