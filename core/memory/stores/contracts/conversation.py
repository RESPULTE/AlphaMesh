"""Conversation persistence contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class ConversationPersistenceAdapter(ABC):
    """Abstract adapter for durable conversation storage."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema and indexes. Must be idempotent."""

    @abstractmethod
    async def ensure_conversation(
        self,
        conversation_id: str,
        user_email: Optional[str],
    ) -> None:
        """Create conversation when missing and refresh last-message timestamp."""

    @abstractmethod
    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: str,
    ) -> None:
        """Persist one conversation message."""

    @abstractmethod
    async def load_messages(self, conversation_id: str) -> List[dict]:
        """Load all messages for a conversation ordered by created time."""

    @abstractmethod
    async def list_conversations(
        self,
        user_email: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """List conversation metadata, newest first."""

