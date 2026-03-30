"""
api/sinks/sse_sink.py

EventQueue sink that bridges the synchronous EventQueue fire path to the
asyncio.Queue consumed by each request's SSE stream.

Thread-safety
─────────────
`on_event()` is called synchronously from EventQueue._fire(), which itself is
called from within a coroutine running in the asyncio event loop.  However,
to be safe against any future path where publish() might be called from a
thread pool (e.g. inside asyncio.to_thread), we always use
`loop.call_soon_threadsafe` which is safe from both in-loop and out-of-loop
callers.

Filtering
─────────
Each SSESink is bound to a specific `response_id`.  Because EventQueue is a
global singleton shared across all concurrent requests, a sink must ignore
events from other concurrent runs.  The `response_id` is obtained from the
ResponseGroup created by `EventQueue.start_response()` and is a stable UUID
that never collides across runs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SSESink:
    """
    Bridges one request's EventQueue events to its asyncio.Queue.

    Created by AnalysisRunner immediately before OrchestratorAgent.run() is
    called and removed (via EventQueue.remove_sink) in the finally block.
    """

    def __init__(
        self,
        request_id: str,
        response_id: str,
        event_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._request_id = request_id
        self._response_id = response_id
        self._event_queue = event_queue
        self._loop = loop

    # ── EventQueue sink protocol ──────────────────────────────────────────────

    def on_event(self, event: Any) -> None:  # event: core.event_queue.Event
        """Filter to this request's response group, then enqueue for SSE."""
        if event.response_id != self._response_id:
            return
        data = event.data or {}
        event_type = data.get("event_type")
        if event_type:
            payload = {
                "event_type": event_type,
                "request_id": self._request_id,
                **{k: v for k, v in data.items() if k != "event_type"},
            }
        else:
            payload = {
                "event_type": "progress",
                "request_id": self._request_id,
                "source": event.source,
                "level": event.level.label,
                "message": event.message,
                "timestamp": event.timestamp.isoformat(),
            }
        # call_soon_threadsafe is safe from both in-loop and thread contexts
        try:
            self._loop.call_soon_threadsafe(self._safe_put, payload)
        except RuntimeError:
            # Loop is closed (e.g. shutdown during a request) — silently discard
            pass

    def on_group_opened(self, group: Any) -> None:
        pass  # No-op; we don't expose group lifecycle to the frontend

    def on_group_closed(self, group: Any) -> None:
        pass  # No-op

    # ── Internal ─────────────────────────────────────────────────────────────

    def _safe_put(self, payload: dict) -> None:
        """Called from the event loop via call_soon_threadsafe."""
        try:
            self._event_queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.debug(
                "SSESink: event queue full for request %s — dropping event",
                self._request_id,
            )
