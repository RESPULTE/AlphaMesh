"""Session persistence contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class SessionPersistenceAdapter(ABC):
    """Abstract adapter for login-session persistence."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema and indexes. Must be idempotent."""

    @abstractmethod
    async def create_login_session(
        self,
        session_id: str,
        user_id: str,
        created_at: str,
    ) -> None:
        """Create one login session."""

    @abstractmethod
    async def touch_login_session(
        self,
        session_id: str,
        user_id: str,
        last_seen_at: str,
    ) -> bool:
        """Update session last-seen timestamp. False when session does not exist."""

    @abstractmethod
    async def end_login_session(
        self,
        session_id: str,
        user_id: str,
        ended_at: str,
    ) -> bool:
        """Mark an existing session ended. False when session does not exist."""

    @abstractmethod
    async def link_session_conversation(
        self,
        session_id: str,
        user_id: str,
        conversation_id: str,
        linked_at: str,
    ) -> None:
        """Link a conversation to a session for the same user."""

    @abstractmethod
    async def user_has_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> bool:
        """Return True when a conversation is linked to the user."""

    @abstractmethod
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        """List recent sessions for user."""

    @abstractmethod
    async def get_latest_active_session(
        self,
        user_id: str,
    ) -> Optional[str]:
        """Return most recent active session id, or None."""

