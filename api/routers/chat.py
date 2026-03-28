"""
api/routers/chat.py

POST /api/v1/chat — starts one analysis turn.

Returns a ChatAck immediately (request_id + conversation_id).
The client should then open GET /api/v1/stream/{request_id} to receive
incremental progress events and the final structured result.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_runner
from api.models.requests import ChatRequest
from api.models.responses import ChatAck
from api.services.analysis_runner import AnalysisRunner

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
) -> ChatAck:
    from uuid import uuid4

    request_id = str(uuid4())
    conversation_id = runner.launch(request_id, body)
    return ChatAck(request_id=request_id, conversation_id=conversation_id)
