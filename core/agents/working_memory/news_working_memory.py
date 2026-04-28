from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from core.agents.working_memory.base import (
    ConversationWorkingMemoryBase,
    ConversationWorkingMemoryManagerBase,
    TurnRelevantMemoryBase,
)
from core.memory.retrieval.models import RetrievedChunk


@dataclass
class NewsTurnRelevantMemory(TurnRelevantMemoryBase):
    chunk_ids: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    score_unavailable: bool = False


@dataclass
class NewsConversationWorkingMemory(
    ConversationWorkingMemoryBase[NewsTurnRelevantMemory]
):
    turn_records: List[NewsTurnRelevantMemory] = field(default_factory=list)


class NewsWorkingMemoryManager(
    ConversationWorkingMemoryManagerBase[
        NewsTurnRelevantMemory, NewsConversationWorkingMemory
    ]
):
    def __init__(self, *, max_chunks: int = 100, max_turns: int = 20) -> None:
        super().__init__(
            max_chunks=max_chunks,
            max_turns=max_turns,
            conversation_factory=NewsConversationWorkingMemory,
        )

    def persist_finalized_turn(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        query: str,
        chunks: List[RetrievedChunk],
        score_unavailable: bool,
        source_key_fn: Callable[[RetrievedChunk], str],
    ) -> None:
        if not conversation_id:
            return
        self.merge_working_chunks(conversation_id=conversation_id, chunks=chunks)
        source_urls = list(
            dict.fromkeys(
                source_key_fn(chunk) for chunk in chunks if source_key_fn(chunk)
            )
        )
        self.append_turn_record(
            conversation_id=conversation_id,
            record=NewsTurnRelevantMemory(
                turn_id=turn_id,
                query=query,
                chunk_ids=[chunk.chunk_id for chunk in chunks if chunk.chunk_id],
                source_urls=source_urls,
                score_unavailable=score_unavailable,
            ),
        )

    def render_working_memory_block(
        self, conversation_id: str, *, turn_limit: int = 4
    ) -> str:
        if not conversation_id:
            return "(none)"
        memory = self.get_existing_conversation_memory(conversation_id)
        if memory is None or not memory.turn_records:
            return "(none)"
        lines: List[str] = []
        for row in memory.turn_records[-turn_limit:]:
            ts = row.created_at.isoformat()
            lines.append(
                f"- turn={row.turn_id} at={ts}\n"
                f"  query={row.query}\n"
                f"  relevant_chunks={len(row.chunk_ids)}\n"
                f"  relevant_sources={len(row.source_urls)}\n"
                f"  score_unavailable={row.score_unavailable}"
            )
        return "\n".join(lines)
