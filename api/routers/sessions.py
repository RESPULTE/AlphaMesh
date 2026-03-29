from fastapi import APIRouter, Depends

from api.dependencies import get_current_user, get_session_service
from api.services.session_service import SessionService

router = APIRouter(tags=["sessions"])


@router.get("/")
async def list_sessions(
    user_email: str = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
):
    return await svc.get_sessions(user_email)


@router.get("/{ticker}")
async def sessions_by_ticker(
    ticker: str,
    user_email: str = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
):
    return await svc.get_sessions_by_ticker(user_email, ticker)
