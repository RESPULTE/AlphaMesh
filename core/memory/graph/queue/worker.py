from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Dict, List, Optional

from core.logger import get_logger
from core.memory.graph.queue.types import (
    GraphTask,
    QueueItem,
    SHUTDOWN_TURN_ID,
    SentinelTask,
)

logger = get_logger(__name__)


class ConversationQueueWorker:
    def __init__(
        self,
        *,
        conversation_id: str,
        batch_processor: Callable[[List[GraphTask], str], Awaitable[None]],
        idle_timeout_seconds: int,
    ) -> None:
        self.conversation_id = conversation_id
        self._batch_processor = batch_processor
        self._idle_timeout_seconds = idle_timeout_seconds

        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self._pending: Dict[str, List[GraphTask]] = {}
        self._last_activity = time.monotonic()
        self._consumer_task: Optional[asyncio.Task] = None
        self._state = "open"

    def start(self) -> None:
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(
                self._consume(), name=f"graph_consumer_{self.conversation_id[:8]}"
            )

    async def put(self, item: QueueItem) -> None:
        if self._state != "open":
            raise RuntimeError(
                f"ConversationQueueWorker for '{self.conversation_id}' is not open"
            )
        self._last_activity = time.monotonic()
        await self._queue.put(item)

    async def flush_turn(self, turn_id: str) -> None:
        await self.put(
            SentinelTask(turn_id=turn_id, conversation_id=self.conversation_id)
        )

    async def drain_and_stop(self, timeout_seconds: float = 30.0) -> None:
        if self._state == "closed":
            return
        self._state = "closing"
        await self._queue.put(
            SentinelTask(turn_id=SHUTDOWN_TURN_ID, conversation_id=self.conversation_id)
        )
        if self._consumer_task:
            try:
                await asyncio.wait_for(self._consumer_task, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(
                    "ConversationQueueWorker: consumer for '%s' did not stop in %.1fs",
                    self.conversation_id,
                    timeout_seconds,
                )
                self._consumer_task.cancel()
        self._state = "closed"

    @property
    def is_idle(self) -> bool:
        return (
            self._state == "open"
            and (time.monotonic() - self._last_activity) > self._idle_timeout_seconds
            and self._queue.empty()
            and not self._pending
        )

    async def _consume(self) -> None:
        logger.info(
            "ConversationQueueWorker: consumer started for '%s'", self.conversation_id
        )
        try:
            while True:
                item = await self._queue.get()
                try:
                    if isinstance(item, SentinelTask):
                        if item.turn_id == SHUTDOWN_TURN_ID:
                            await self._flush_all_pending()
                            break
                        await self._flush_turn_pending(item.turn_id)
                    else:
                        self._pending.setdefault(item.turn_id, []).append(item)
                        self._last_activity = time.monotonic()
                except Exception:
                    logger.exception(
                        "ConversationQueueWorker: failed to process queue item for '%s'",
                        self.conversation_id,
                    )
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            logger.info(
                "ConversationQueueWorker: consumer cancelled for '%s'",
                self.conversation_id,
            )
        finally:
            self._state = "closed"
            logger.info(
                "ConversationQueueWorker: consumer stopped for '%s'",
                self.conversation_id,
            )

    async def _flush_turn_pending(self, turn_id: str) -> None:
        tasks = self._pending.pop(turn_id, [])
        if tasks:
            await self._batch_processor(tasks, turn_id)

    async def _flush_all_pending(self) -> None:
        for turn_id, tasks in list(self._pending.items()):
            self._pending.pop(turn_id, None)
            if tasks:
                await self._batch_processor(tasks, turn_id)
