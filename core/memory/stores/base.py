"""
core/memory/stores/base.py

Abstract persistence adapters for conversation and session storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class ConversationPersistenceAdapter(ABC):
    """
    Abstract adapter for durable conversation storage.

    All methods are async and must be safe to call concurrently.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema / indices. Idempotent safe to call multiple times."""

    @abstractmethod
    async def ensure_conversation(
        self,
        conversation_id: str,
        user_email: Optional[str],
    ) -> None:
        """
        Create the conversation record if it does not already exist.
        Update `last_message_at` on every call.
        """

    @abstractmethod
    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: str,
    ) -> None:
        """Persist a single message. `role` is 'user' or 'assistant'."""

    @abstractmethod
    async def load_messages(
        self,
        conversation_id: str,
    ) -> List[dict]:
        """
        Return all messages for a conversation in chronological order.

        Each dict has keys: role, content, timestamp (ISO 8601 string).
        """

    @abstractmethod
    async def list_conversations(
        self,
        user_email: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """
        Return conversation metadata, most-recent first.

        Each dict has keys: conversation_id, created_at, last_message_at,
        message_count. Filtered by user_email when supplied.
        """


class SessionPersistenceAdapter(ABC):
    """
    Abstract adapter for session metadata storage.

    Keeps SessionService decoupled from the underlying store so we can
    swap SQLite for Redis/external APIs later without touching routers.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema / indices. Idempotent � safe to call multiple times."""

    @abstractmethod
    async def create_login_session(
        self,
        session_id: str,
        user_id: str,
        created_at: str,
    ) -> None:
        """Create a login-scoped session."""

    @abstractmethod
    async def touch_login_session(
        self,
        session_id: str,
        user_id: str,
        last_seen_at: str,
    ) -> bool:
        """Update last_seen_at for an active session. Returns False if not found."""

    @abstractmethod
    async def end_login_session(
        self,
        session_id: str,
        user_id: str,
        ended_at: str,
    ) -> bool:
        """Mark a login session ended. Returns False if not found."""

    @abstractmethod
    async def link_session_conversation(
        self,
        session_id: str,
        user_id: str,
        conversation_id: str,
        linked_at: str,
    ) -> None:
        """Create a many-to-many link between a login session and a conversation."""

    @abstractmethod
    async def user_has_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> bool:
        """Return True when the conversation is linked to the given user."""

    @abstractmethod
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        """Return most recent login sessions for a user, newest first."""

    @abstractmethod
    async def get_latest_active_session(
        self,
        user_id: str,
    ) -> Optional[str]:
        """Return latest active session id for user, or None."""
