import asyncio
from asyncio.log import logger
from typing import Optional


def _safe_create_task(coro) -> Optional[asyncio.Task]:
    """Create an asyncio task only when a running loop exists."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        logger.warning("_safe_create_task: no running event loop — task skipped.")
        return None
