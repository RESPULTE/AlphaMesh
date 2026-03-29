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
    async def save_session(
        self,
        session_id: str,
        conversation_id: str,
        user_email: str,
        ticker: Optional[str],
        query: str,
        summary: str,
        created_at: str,
    ) -> None:
        """Persist a single session record."""

    @abstractmethod
    async def list_sessions(
        self,
        user_email: str,
        limit: int = 20,
    ) -> List[dict]:
        """Return the most recent `limit` sessions for a user, newest first."""

    @abstractmethod
    async def list_sessions_by_ticker(
        self,
        user_email: str,
        ticker: str,
        limit: int = 10,
    ) -> List[dict]:
        """Return sessions for a specific ticker, newest first."""
