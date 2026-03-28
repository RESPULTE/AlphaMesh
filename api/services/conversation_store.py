"""
api/services/conversation_store.py

Two-tier conversation storage:
  Tier 1 — in-memory dict (O(1) reads, no I/O)
  Tier 2 — pluggable persistence adapter (SQLite by default)

Write-through: every mutation is written to both tiers atomically.
Read-through:  if a conversation is not in memory, it is loaded from the
               adapter on first access and cached for the session lifetime.

LangChain message ↔ plain dict serialisation happens here so the rest of
the API layer never needs to import langchain types.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from api.models.responses import (
    ConversationHistoryResponse,
    ConversationMessage,
    ConversationSummary,
)
from api.persistence.base import ConversationPersistenceAdapter

logger = logging.getLogger(__name__)

# ── Serialisation helpers ──────────────────────────────────────────────────────


def _to_dict(msg: BaseMessage) -> dict:
    """Convert a LangChain BaseMessage to a plain dict for storage."""
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _from_dict(d: dict) -> BaseMessage:
    """Reconstruct a LangChain BaseMessage from a stored plain dict."""
    return (
        HumanMessage(content=d["content"])
        if d["role"] == "user"
        else AIMessage(content=d["content"])
    )


# ── Store ──────────────────────────────────────────────────────────────────────


class ConversationStore:
    """
    Write-through, read-through conversation store.

    Concurrency
    ───────────
    One asyncio.Lock per conversation prevents concurrent writes from
    interleaving.  The lock is only held during dict mutations and adapter
    I/O, never across agent invocations.
    """

    def __init__(self, adapter: ConversationPersistenceAdapter) -> None:
        self._adapter = adapter
        # conversation_id → List[dict]  (serialised messages)
        self._cache: Dict[str, List[dict]] = {}
        # conversation_id → asyncio.Lock
        self._locks: Dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Delegate schema creation to the persistence adapter."""
        await self._adapter.initialize()

    def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        if conversation_id not in self._locks:
            self._locks[conversation_id] = asyncio.Lock()
        return self._locks[conversation_id]

    async def _load_from_adapter(self, conversation_id: str) -> List[dict]:
        """Load messages from the adapter and populate the in-memory cache."""
        messages = await self._adapter.load_messages(conversation_id)
        self._cache[conversation_id] = messages
        return messages

    # ── Public API ─────────────────────────────────────────────────────────────

    async def ensure_conversation(
        self,
        conversation_id: str,
        user_email: Optional[str] = None,
    ) -> None:
        """
        Guarantee that a conversation record exists (in memory + adapter).
        Safe to call multiple times (idempotent).
        """
        async with self._get_lock(conversation_id):
            if conversation_id not in self._cache:
                self._cache[conversation_id] = []
            await self._adapter.ensure_conversation(conversation_id, user_email)

    async def add_messages(
        self,
        conversation_id: str,
        messages: List[BaseMessage],
    ) -> None:
        """
        Append one or more LangChain messages (write-through).

        Called once per turn: [HumanMessage(user query), AIMessage(synthesis)].
        """
        async with self._get_lock(conversation_id):
            for msg in messages:
                d = _to_dict(msg)
                self._cache.setdefault(conversation_id, []).append(d)
                await self._adapter.save_message(
                    conversation_id,
                    role=d["role"],
                    content=d["content"],
                    timestamp=d["timestamp"],
                )

    async def get_langchain_messages(self, conversation_id: str) -> List[BaseMessage]:
        """
        Return the full message history as LangChain BaseMessage objects.
        Populates the in-memory cache from the adapter on first access.
        """
        async with self._get_lock(conversation_id):
            if conversation_id not in self._cache:
                await self._load_from_adapter(conversation_id)
        return [_from_dict(d) for d in self._cache.get(conversation_id, [])]

    async def get_history_response(
        self, conversation_id: str
    ) -> ConversationHistoryResponse:
        """Return the full message history formatted for the REST endpoint."""
        async with self._get_lock(conversation_id):
            if conversation_id not in self._cache:
                await self._load_from_adapter(conversation_id)
        messages = [
            ConversationMessage(
                role=d["role"],
                content=d["content"],
                timestamp=d.get("timestamp", ""),
            )
            for d in self._cache.get(conversation_id, [])
        ]
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
        )

    async def list_conversations(
        self,
        user_email: Optional[str] = None,
        limit: int = 50,
    ) -> List[ConversationSummary]:
        """List recent conversations, delegating to the persistence adapter."""
        rows = await self._adapter.list_conversations(
            user_email=user_email, limit=limit
        )
        return [
            ConversationSummary(
                conversation_id=r["conversation_id"],
                created_at=r["created_at"],
                last_message_at=r["last_message_at"],
                message_count=r["message_count"],
            )
            for r in rows
        ]
