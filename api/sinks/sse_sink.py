"""
api/sinks/sse_sink.py

EventQueue sink that bridges the synchronous EventQueue fire path to the
asyncio.Queue consumed by each request's SSE stream.
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
        self._pending_puts: set[asyncio.Task] = set()

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
        try:
            # Fast path: if already on the owning loop, enqueue immediately to
            # preserve strict ordering.
            if asyncio.get_running_loop() is self._loop:
                self._safe_put(payload)
                return
        except RuntimeError:
            # No running loop in this thread: fall back to thread-safe callback.
            pass
        try:
            self._loop.call_soon_threadsafe(self._safe_put, payload)
        except RuntimeError:
            # Loop is closed (e.g. shutdown during a request). Discard silently.
            pass

    def on_group_opened(self, group: Any) -> None:
        pass

    def on_group_closed(self, group: Any) -> None:
        pass

    def _safe_put(self, payload: dict) -> None:
        """Called from the event loop to enqueue one SSE payload."""
        try:
            self._event_queue.put_nowait(payload)
        except asyncio.QueueFull:
            # For token chunks, preserve delivery by waiting for queue space.
            if payload.get("event_type") == "analysis_chunk":
                task = self._loop.create_task(self._event_queue.put(payload))
                self._pending_puts.add(task)
                task.add_done_callback(self._pending_puts.discard)
                return
            logger.debug(
                "SSESink: event queue full for request %s - dropping event",
                self._request_id,
            )

    async def drain(self) -> None:
        """
        Wait for all pending token enqueue tasks to complete.

        This is used by AnalysisRunner before sending terminal events so queued
        token chunks are not silently reordered behind `complete`.
        """
        if not self._pending_puts:
            return
        await asyncio.gather(*list(self._pending_puts), return_exceptions=True)
