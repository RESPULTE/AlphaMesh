from uuid import uuid4

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from api.dependencies import get_current_user_from_query, get_session_service
from api.services.analysis_stream_service import AnalysisStreamService
from core.memory.sessions.session_service import SessionService

router = APIRouter(tags=["analyze"])


@router.get("/analyze")
async def analyze(
    query: str = Query(..., min_length=1, max_length=1000),
    ticker: str | None = Query(default=None, max_length=10),
    conversation_id: str | None = Query(default=None),
    user_email: str = Depends(get_current_user_from_query),
    session_svc: SessionService = Depends(get_session_service),
):
    cid = conversation_id or str(uuid4())
    svc = AnalysisStreamService()  # per-request instantiation
    return StreamingResponse(
        svc.stream(
            query=query,
            user_email=user_email,
            conversation_id=cid,
            ticker=ticker,
            session_svc=session_svc,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

