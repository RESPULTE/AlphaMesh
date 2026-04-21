"""
api/routers/chat.py

POST /api/v1/chat — starts one analysis turn.

Returns a ChatAck immediately (request_id + conversation_id).
The client should then open GET /api/v1/stream/{request_id} to receive
incremental progress events and the final structured result.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_current_user_optional,
    get_runner,
    get_session_service,
)
from api.models.requests import ChatRequest
from api.models.responses import ChatAck
from api.services.analysis_runner import AnalysisRunner
from core.memory.sessions.session_service import SessionService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatAck,
    status_code=202,
    summary="Start an analysis turn",
    description=(
        "Enqueues an analysis request and returns request_id + conversation_id "
        "immediately (HTTP 202 Accepted).  Connect to GET /api/v1/stream/{request_id} "
        "to receive Server-Sent Events with progress updates and the final result."
    ),
)
async def post_chat(
    body: ChatRequest,
    runner: AnalysisRunner = Depends(get_runner),
    user_id_from_token: str | None = Depends(get_current_user_optional),
    session_svc: SessionService = Depends(get_session_service),
) -> ChatAck:
    from uuid import uuid4

    user_id = user_id_from_token or body.user_email
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Missing user identity: provide Bearer token or deprecated user_email",
        )

    session_id = await session_svc.ensure_session(
        user_id=user_id,
        session_id=body.session_id,
    )
    request_id = str(uuid4())
    conversation_id = runner.launch(
        request_id,
        body,
        user_id=user_id,
        session_id=session_id,
    )
    return ChatAck(
        request_id=request_id,
        conversation_id=conversation_id,
        session_id=session_id,
    )
