"""
core/memory/sessions/session_service.py

Stores per-user analysis sessions via a persistence adapter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from core.logger import get_logger
from core.memory.stores.base import SessionPersistenceAdapter

logger = get_logger(__name__)


class SessionService:
    """
    Stores and retrieves user analysis session records.

    Each record captures: who ran it (user_email), what they asked (query),
    which ticker was targeted, a short summary, and the conversation_id that
    links back to the full message history in ConversationStore.
    """

    def __init__(self, adapter: SessionPersistenceAdapter) -> None:
        self._adapter = adapter

    async def initialize(self) -> None:
        """
        Initialize the underlying persistence adapter.

        Called once by service_manager.startup � safe to call multiple times.
        """

        await self._adapter.initialize()
        logger.info("SessionService: initialised")

    async def save_analysis(
        self,
        *,
        user_email: str,
        conversation_id: str,
        query: str,
        ticker: Optional[str],
        summary_text: str,
    ) -> str:
        """
        Persist one analysis session record.

        Returns the generated session_id for tracing.
        Summary is truncated to 500 chars to keep the row lightweight
        (full content lives in ConversationStore).
        """
        session_id = str(uuid4())
        await self._adapter.save_session(
            session_id=session_id,
            conversation_id=conversation_id,
            user_email=user_email,
            ticker=ticker.upper() if ticker else None,
            query=query,
            summary=(summary_text or "")[:500],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.debug(
            "SessionService: saved session '%s' for user '%s'", session_id, user_email
        )
        return session_id

    async def get_sessions(
        self,
        user_email: str,
        limit: int = 20,
    ) -> List[dict]:
        """Return the most recent `limit` sessions for a user, newest first."""
        return await self._adapter.list_sessions(user_email=user_email, limit=limit)

    async def get_sessions_by_ticker(
        self,
        user_email: str,
        ticker: str,
        limit: int = 10,
    ) -> List[dict]:
        """Return sessions for a specific ticker, newest first."""
        return await self._adapter.list_sessions_by_ticker(
            user_email=user_email,
            ticker=ticker.upper(),
            limit=limit,
        )
