"""
api/services/session_service.py

Login-session lifecycle and session-conversation linking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from api.services.session_sql_store import SQLiteSessionStore
from core.logger import get_logger

logger = get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionService:
    """
    Manages login-scoped sessions and their conversation links.

    A single login session may link to many conversations.
    A conversation may be linked to many sessions (same user only).
    """

    def __init__(self, db: SQLiteSessionStore) -> None:
        self._sql_db = db

    async def initialize(self) -> None:
        await self._sql_db.initialize()
        logger.info("SessionService: initialised")

    async def create_session(self, user_id: str) -> str:
        session_id = str(uuid4())
        await self._sql_db.create_login_session(
            session_id=session_id,
            user_id=user_id,
            created_at=_utc_now(),
        )
        logger.debug(
            "SessionService: created session '%s' for '%s'", session_id, user_id
        )
        return session_id

    async def ensure_session(
        self,
        *,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Resolve a valid active session id for the user.

        If session_id is provided but not valid for the user, create a new one.
        If session_id is missing, reuse the latest active session when available.
        """
        now = _utc_now()
        if session_id:
            touched = await self._sql_db.touch_login_session(
                session_id=session_id,
                user_id=user_id,
                last_seen_at=now,
            )
            if touched:
                return session_id

        latest = await self._sql_db.get_latest_active_session(user_id=user_id)
        if latest:
            await self._sql_db.touch_login_session(
                session_id=latest,
                user_id=user_id,
                last_seen_at=now,
            )
            return latest
        return await self.create_session(user_id=user_id)

    async def end_session(self, *, user_id: str, session_id: str) -> bool:
        return await self._sql_db.end_login_session(
            session_id=session_id,
            user_id=user_id,
            ended_at=_utc_now(),
        )

    async def link_conversation(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_id: str,
    ) -> None:
        await self._sql_db.link_session_conversation(
            session_id=session_id,
            user_id=user_id,
            conversation_id=conversation_id,
            linked_at=_utc_now(),
        )

    async def user_has_conversation(
        self, *, user_id: str, conversation_id: str
    ) -> bool:
        return await self._sql_db.user_has_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def get_sessions(self, user_id: str, limit: int = 20) -> List[dict]:
        return await self._sql_db.list_sessions(user_id=user_id, limit=limit)
