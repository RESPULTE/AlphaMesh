"""
api/routers/stream.py

GET /api/v1/stream/{request_id} — Server-Sent Events stream.

Protocol
────────
Every event is a JSON object on the `data:` line of an SSE frame, followed
by two newlines.  The frontend should parse each `data:` line as JSON.

Event types (see api/models/responses.py):
  progress — incremental status update
  complete  — final structured result; stream terminates
  error     — unrecoverable failure; stream terminates

Keepalives
──────────
A `: keepalive` comment is sent every 25 seconds to prevent proxies and
browsers from closing idle connections.

Reconnect handling
──────────────────
Missed events are not replayed on reconnect (accepted loss; see design doc).
The final `complete` event contains the full FinalResult, so even a dropped
connection delivers the answer once the stream is re-opened.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import get_broadcaster, get_current_user
from api.services.event_broadcaster import EventBroadcaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["stream"])

_KEEPALIVE_INTERVAL_S = 25.0
_TERMINAL_EVENT_TYPES = frozenset({"complete", "error"})


@router.get(
    "/stream/{request_id}",
    summary="SSE stream for an analysis request",
    description=(
        "Opens a Server-Sent Events stream for the given request_id.  "
        "Returns 404 if the request_id is unknown or has already expired."
    ),
    responses={
        200: {"content": {"text/event-stream": {}}},
        404: {"description": "request_id not found"},
    },
)
async def stream_events(
    request_id: str,
    _user_id: str = Depends(get_current_user),
    broadcaster: EventBroadcaster = Depends(get_broadcaster),
) -> StreamingResponse:
    queue = broadcaster.get(request_id)
    if queue is None:
        raise HTTPException(
            status_code=404, detail=f"request_id '{request_id}' not found or expired."
        )

    return StreamingResponse(
        _generate(request_id, queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


async def _generate(request_id: str, queue: asyncio.Queue):
    """Async generator that yields SSE-formatted strings."""
    try:
        while True:
            try:
                event_dict = await asyncio.wait_for(
                    queue.get(), timeout=_KEEPALIVE_INTERVAL_S
                )
                queue.task_done()
            except asyncio.TimeoutError:
                # Send a comment-only keepalive to prevent proxy disconnection
                yield ": keepalive\n\n"
                continue

            yield f"data: {json.dumps(event_dict, default=str)}\n\n"

            if event_dict.get("event_type") in _TERMINAL_EVENT_TYPES:
                break

    except asyncio.CancelledError:
        # Client disconnected — the analysis task continues running in background
        logger.debug(
            "SSE stream cancelled for request %s (client disconnected)", request_id
        )
    except Exception:
        logger.exception("SSE stream error for request %s", request_id)
        yield f"data: {json.dumps({'event_type': 'error', 'request_id': request_id, 'error': 'Internal stream error'})}\n\n"
