"""
api/services/conversation_service.py

Two-tier conversation storage:
  Tier 1 - in-memory dict (O(1) reads, no I/O)
  Tier 2 - pluggable persistence adapter (JSONL by default)

Write-through: every mutation is written to both tiers atomically.
Read-through:  if a conversation is not in memory, it is loaded from the
               adapter on first access and cached for the session lifetime.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from api.services.conversation_jsonl_store import JsonlConversationStore

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _from_dict(d: dict) -> BaseMessage:
    """Reconstruct a LangChain BaseMessage from a stored plain dict."""
    return (
        HumanMessage(content=d["content"])
        if d["role"] == "user"
        else AIMessage(content=d["content"])
    )


class ConversationStore:
    """
    Write-through, read-through conversation store.

    Concurrency
    -----------
    One asyncio.Lock per conversation prevents concurrent writes from
    interleaving. The lock is only held during dict mutations and adapter
    I/O, never across agent invocations.
    """

    def __init__(self, db: JsonlConversationStore) -> None:
        self._db = db
        # conversation_id -> List[dict] (serialized messages)
        self._message_cache: Dict[str, List[dict]] = {}
        # conversation_id -> List[dict] (rich turn records)
        self._turn_cache: Dict[str, List[dict]] = {}
        # conversation_id -> user_email owner
        self._owners: Dict[str, str] = {}
        # conversation_id -> asyncio.Lock
        self._locks: Dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Delegate schema creation to the persistence adapter."""
        await self._db.initialize()

    async def ensure_user_workspace(self, user_email: str) -> int:
        """
        Ensure per-user chatlog workspace exists and return conversation count.
        """
        return await self._db.ensure_user_workspace(user_email)

    def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        if conversation_id not in self._locks:
            self._locks[conversation_id] = asyncio.Lock()
        return self._locks[conversation_id]

    def _resolve_owner(self, conversation_id: str, user_email: Optional[str]) -> str:
        if user_email:
            self._owners[conversation_id] = user_email
            return user_email
        cached_owner = self._owners.get(conversation_id)
        if cached_owner:
            return cached_owner
        raise ValueError(
            f"user_email is required when conversation '{conversation_id}' is not cached"
        )

    @staticmethod
    def _messages_from_turns(turns: List[dict]) -> List[dict]:
        messages: List[dict] = []
        for turn in turns:
            timestamp = str(turn.get("created_at") or _utc_now_iso())
            user_message = str(turn.get("user_message") or "")
            assistant_text = str(turn.get("assistant_synthesis") or "")

            if user_message:
                messages.append(
                    {
                        "role": "user",
                        "content": user_message,
                        "timestamp": timestamp,
                    }
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "timestamp": timestamp,
                }
            )
        return messages

    async def _load_from_adapter(
        self,
        conversation_id: str,
        user_email: Optional[str] = None,
    ) -> List[dict]:
        """Load turns from adapter and populate in-memory turn/message caches."""
        owner = self._resolve_owner(conversation_id, user_email)
        turns = await self._db.load_turns(conversation_id=conversation_id, user_email=owner)
        self._turn_cache[conversation_id] = turns
        self._message_cache[conversation_id] = self._messages_from_turns(turns)
        return turns

    async def ensure_conversation(
        self,
        conversation_id: str,
        user_email: Optional[str] = None,
    ) -> None:
        """
        Guarantee that a conversation record exists (in memory + adapter).
        Safe to call multiple times (idempotent).
        """
        if not user_email:
            raise ValueError("user_email is required to ensure a conversation")
        async with self._get_lock(conversation_id):
            self._owners[conversation_id] = user_email
            await self._db.ensure_conversation(conversation_id, user_email)

    async def append_turn(
        self,
        conversation_id: str,
        user_email: str,
        turn: dict,
    ) -> None:
        """
        Append one rich turn record and update projected message history.

        Called once per turn from AnalysisRunner after final result is built.
        """
        async with self._get_lock(conversation_id):
            self._owners[conversation_id] = user_email
            if conversation_id not in self._turn_cache:
                await self._load_from_adapter(conversation_id, user_email=user_email)
            self._turn_cache.setdefault(conversation_id, []).append(dict(turn))
            self._message_cache[conversation_id] = self._messages_from_turns(
                self._turn_cache[conversation_id]
            )
            await self._db.append_turn(
                conversation_id=conversation_id,
                user_email=user_email,
                turn=turn,
            )

    async def get_langchain_messages(
        self,
        conversation_id: str,
        user_email: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        Return the full message history as LangChain BaseMessage objects.
        Populates the in-memory cache from the adapter on first access.
        """
        async with self._get_lock(conversation_id):
            if conversation_id not in self._message_cache:
                await self._load_from_adapter(conversation_id, user_email=user_email)
        return [_from_dict(d) for d in self._message_cache.get(conversation_id, [])]

    async def get_history(
        self,
        conversation_id: str,
        user_email: Optional[str] = None,
    ) -> List[dict]:
        """Return the full message history as plain dicts."""
        async with self._get_lock(conversation_id):
            if conversation_id not in self._message_cache:
                await self._load_from_adapter(conversation_id, user_email=user_email)
        return list(self._message_cache.get(conversation_id, []))

    async def get_turns(
        self,
        conversation_id: str,
        user_email: Optional[str] = None,
    ) -> List[dict]:
        """Return full structured turn history for a conversation."""
        async with self._get_lock(conversation_id):
            if conversation_id not in self._turn_cache:
                await self._load_from_adapter(conversation_id, user_email=user_email)
        return list(self._turn_cache.get(conversation_id, []))

    async def get_turns_paginated(
        self,
        conversation_id: str,
        user_email: Optional[str] = None,
        *,
        limit: Optional[int] = None,
        before_turn_id: Optional[str] = None,
    ) -> Tuple[List[dict], bool, Optional[str]]:
        """
        Return paginated structured turn history in chronological order.

        Pagination semantics:
        - `limit=None`: return the full conversation history.
        - First page (`before_turn_id=None`): return the newest `limit` turns.
        - Older pages (`before_turn_id=<id>`): return turns strictly older than <id>.
        """
        turns = await self.get_turns(conversation_id, user_email=user_email)
        if limit is None:
            return turns, False, None

        if not turns or limit <= 0:
            return [], False, None

        if before_turn_id:
            slice_end = next(
                (
                    idx
                    for idx, turn in enumerate(turns)
                    if str(turn.get("turn_id") or "") == before_turn_id
                ),
                None,
            )
            if slice_end is None or slice_end <= 0:
                return [], False, None
        else:
            slice_end = len(turns)

        start = max(0, slice_end - limit)
        page = turns[start:slice_end]
        has_more = start > 0
        next_before_turn_id = (
            str(page[0].get("turn_id") or "")
            if has_more and page and page[0].get("turn_id")
            else None
        )
        return list(page), has_more, next_before_turn_id

    async def list_conversations(
        self,
        user_email: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """List recent conversations, delegating to the persistence adapter."""
        if not user_email:
            return []
        return await self._db.list_conversations(user_email=user_email, limit=limit)
