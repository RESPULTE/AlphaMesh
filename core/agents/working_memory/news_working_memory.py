from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from core.agents.working_memory.base import (
    ConversationWorkingMemoryBase,
    ConversationWorkingMemoryManagerBase,
    TurnRelevantMemoryBase,
)
from core.agents.utils import trim_text
from core.memory.retrieval.models import RetrievedChunk


@dataclass
class NewsTurnRelevantMemory(TurnRelevantMemoryBase):
    turn_index: int = 0
    chunk_ids: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)


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
    AGENT_NAME = "news_agent"

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
        turn_index: int,
        chunks: List[RetrievedChunk],
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
                turn_id=str(turn_index),
                query="",
                turn_index=turn_index,
                chunk_ids=[chunk.chunk_id for chunk in chunks if chunk.chunk_id],
                source_urls=source_urls,
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
            lines.append(
                f"- turn_index={row.turn_index}\n"
                f"  relevant_chunks={len(row.chunk_ids)}\n"
                f"  relevant_sources={len(row.source_urls)}"
            )
        return "\n".join(lines)

    @staticmethod
    def render_memory_summary(memory_summary: dict) -> str:
        if not memory_summary:
            return ""
        actions = (
            memory_summary.get("research_actions")
            or memory_summary.get("tools_used")
            or []
        )
        if not isinstance(actions, list):
            actions = []
        sentiment = memory_summary.get("sentiment") or {}
        sentiment_label = ""
        if isinstance(sentiment, dict):
            sentiment_label = str(sentiment.get("label") or "").strip()
        source_count = int(memory_summary.get("source_count") or 0)
        summary_text = trim_text(
            memory_summary.get("missing_information_goal")
            or memory_summary.get("findings_summary")
            or "",
            max_chars=200,
        )
        return (
            f"actions={','.join(str(a) for a in actions[:4]) or 'none'}; "
            f"sources={source_count}; sentiment={sentiment_label or 'N/A'}; "
            f"summary={summary_text or 'N/A'}"
        )

    @classmethod
    def build_context_from_history_summaries(
        cls,
        turns: List[dict],
        window: int = 8,
    ) -> str:
        rows = cls.collect_agent_summaries_from_turns(
            turns=turns,
            agent_name=cls.AGENT_NAME,
        )
        if not rows:
            return ""
        lines: List[str] = []
        for ts, payload in rows[-window:]:
            rendered = cls.render_memory_summary(payload)
            if not rendered:
                rendered = cls.render_memory_summary_fallback(payload)
            lines.append(f"- [{ts}] {rendered}")
        return "\n".join(lines)
