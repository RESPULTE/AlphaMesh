from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Generic, List, Tuple, TypeVar

from core.memory.retrieval.models import RetrievedChunk

TurnMemoryT = TypeVar("TurnMemoryT", bound="TurnRelevantMemoryBase")
ConversationMemoryT = TypeVar(
    "ConversationMemoryT", bound="ConversationWorkingMemoryBase"
)


@dataclass
class TurnRelevantMemoryBase:
    turn_id: str
    query: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConversationWorkingMemoryBase(Generic[TurnMemoryT]):
    agent_memory_context: str = ""
    working_chunks: List[RetrievedChunk] = field(default_factory=list)
    turn_records: List[TurnMemoryT] = field(default_factory=list)


class ConversationWorkingMemoryManagerBase(Generic[TurnMemoryT, ConversationMemoryT]):
    def __init__(
        self,
        *,
        max_chunks: int,
        max_turns: int,
        conversation_factory: Callable[[], ConversationMemoryT],
    ) -> None:
        self._max_chunks = max_chunks
        self._max_turns = max_turns
        self._conversation_factory = conversation_factory
        self._conversation_memory: Dict[str, ConversationMemoryT] = {}

    def get_conversation_memory(self, conversation_id: str) -> ConversationMemoryT:
        return self._conversation_memory.setdefault(
            conversation_id, self._conversation_factory()
        )

    def get_existing_conversation_memory(
        self, conversation_id: str
    ) -> ConversationMemoryT | None:
        return self._conversation_memory.get(conversation_id)

    def resolve_agent_memory_context(
        self,
        *,
        conversation_id: str,
        incoming_memory_context: str | None,
    ) -> str:
        memory = self.get_conversation_memory(conversation_id)
        incoming = (incoming_memory_context or "").strip()
        if incoming and incoming != memory.agent_memory_context:
            memory.agent_memory_context = incoming
        return incoming or memory.agent_memory_context

    def persist_agent_memory_summary(
        self, *, conversation_id: str, rendered_summary: str
    ) -> None:
        if not conversation_id:
            return
        summary = (rendered_summary or "").strip()
        if not summary:
            return
        memory = self.get_conversation_memory(conversation_id)
        memory.agent_memory_context = summary

    def get_working_memory_chunks(self, conversation_id: str) -> List[RetrievedChunk]:
        if not conversation_id:
            return []
        memory = self.get_conversation_memory(conversation_id)
        return list(memory.working_chunks)

    def merge_working_chunks(
        self, *, conversation_id: str, chunks: List[RetrievedChunk]
    ) -> None:
        if not conversation_id:
            return
        memory = self.get_conversation_memory(conversation_id)
        chunk_map: Dict[str, RetrievedChunk] = {
            chunk.chunk_id: chunk for chunk in memory.working_chunks if chunk.chunk_id
        }
        for chunk in chunks:
            if chunk.chunk_id:
                chunk_map[chunk.chunk_id] = chunk
        merged = list(chunk_map.values())
        if len(merged) > self._max_chunks:
            merged = merged[-self._max_chunks :]
        memory.working_chunks = merged

    def append_turn_record(self, *, conversation_id: str, record: TurnMemoryT) -> None:
        if not conversation_id:
            return
        memory = self.get_conversation_memory(conversation_id)
        memory.turn_records.append(record)
        if len(memory.turn_records) > self._max_turns:
            memory.turn_records = memory.turn_records[-self._max_turns :]

    @staticmethod
    def normalize_turn_timestamp(turn: dict) -> str:
        return str(turn.get("created_at") or turn.get("timestamp") or "").strip()

    @classmethod
    def collect_agent_summaries_from_turns(
        cls,
        *,
        turns: List[dict],
        agent_name: str,
    ) -> List[Tuple[str, dict]]:
        rows: List[Tuple[str, dict]] = []
        for turn in turns:
            summaries = turn.get("agent_memory_summaries") or {}
            if not isinstance(summaries, dict):
                continue
            payload = summaries.get(agent_name)
            if not isinstance(payload, dict):
                continue
            ts = cls.normalize_turn_timestamp(turn) or "unknown_time"
            rows.append((ts, payload))
        return rows

    @staticmethod
    def render_memory_summary_fallback(
        memory_summary: dict, *, max_chars: int = 350
    ) -> str:
        text = json.dumps(memory_summary or {}, ensure_ascii=True)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."
