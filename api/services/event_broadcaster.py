"""
api/services/event_broadcaster.py

Registry of per-request asyncio.Queues that feed SSE endpoints.

Lifecycle
---------
1. AnalysisRunner calls `broadcaster.create(request_id)` and gets a queue.
2. SSE endpoint calls `broadcaster.get(request_id)` and reads that queue.
3. AnalysisRunner enqueues the terminal `complete` or `error` event and schedules delayed cleanup.
4. After `_CLEANUP_DELAY_S`, the queue is evicted even if no client attached.

Queue capacity
--------------
`maxsize=512` is intentionally generous. Queue overflow handling is owned by
the sink: `analysis_chunk` events use backpressure and are not dropped for
connected clients, while non-token events may be dropped under pressure.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_QUEUE_MAX_SIZE = 512
_CLEANUP_DELAY_S = 120.0  # seconds to keep queue after the final event


class EventBroadcaster:
    """
    Maintains a per-request asyncio.Queue for SSE event delivery.

    Thread-safe: all access is from within the asyncio event loop.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.Queue] = {}

    def create(self, request_id: str) -> asyncio.Queue:
        """
        Create and register a new Queue for `request_id`.

        Must be called before the background analysis task is launched so
        that events are never dropped between task start and SSE connection.
        Idempotent: returns the existing Queue if already created.
        """
        if request_id not in self._queues:
            self._queues[request_id] = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
            logger.debug("EventBroadcaster: created queue for request %s", request_id)
        return self._queues[request_id]

    def get(self, request_id: str) -> Optional[asyncio.Queue]:
        """Return the Queue for `request_id`, or None if it does not exist."""
        return self._queues.get(request_id)

    def remove(self, request_id: str) -> None:
        """Evict the Queue for `request_id`."""
        if self._queues.pop(request_id, None) is not None:
            logger.debug("EventBroadcaster: removed queue for request %s", request_id)

    def schedule_cleanup(
        self, request_id: str, delay: float = _CLEANUP_DELAY_S
    ) -> None:
        """
        Schedule removal of `request_id`'s queue after `delay` seconds.

        Called after the final event is enqueued so abandoned connections do
        not keep queues alive indefinitely.
        """

        async def _cleanup() -> None:
            await asyncio.sleep(delay)
            self.remove(request_id)

        asyncio.create_task(_cleanup(), name=f"broadcaster_cleanup_{request_id[:8]}")

