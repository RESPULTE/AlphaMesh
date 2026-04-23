from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user, get_session_service
from api.services.session_service import SessionService

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("", summary="Create (or reuse) active login session")
async def create_session(
    user_id: str = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
):
    session_id = await svc.ensure_session(user_id=user_id)
    return {"session_id": session_id}


@router.get("", summary="List login sessions")
async def list_sessions(
    user_id: str = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
):
    return await svc.get_sessions(user_id=user_id)


@router.delete("/{session_id}", summary="End a login session")
async def end_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
):
    ended = await svc.end_session(user_id=user_id, session_id=session_id)
    if not ended:
        raise HTTPException(status_code=404, detail="session_id not found")
    return {"session_id": session_id, "status": "ended"}
