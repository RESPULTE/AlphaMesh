from __future__ import annotations

from dataclasses import dataclass, field

from core.agents.working_memory.base import (
    ConversationWorkingMemoryBase,
    ConversationWorkingMemoryManagerBase,
    TurnRelevantMemoryBase,
)
from core.agents.working_memory.news_working_memory import NewsWorkingMemoryManager
from core.memory.retrieval.models import RetrievedChunk


def _make_chunk(chunk_id: str, url: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        source="vector",
        metadata={"source_url": url, "article_title": chunk_id},
        source_url=url,
        article_title=chunk_id,
    )


@dataclass
class _TestTurnRecord(TurnRelevantMemoryBase):
    note: str = ""


@dataclass
class _TestConversationMemory(ConversationWorkingMemoryBase[_TestTurnRecord]):
    turn_records: list[_TestTurnRecord] = field(default_factory=list)


class _TestWorkingMemoryManager(
    ConversationWorkingMemoryManagerBase[_TestTurnRecord, _TestConversationMemory]
):
    def __init__(self) -> None:
        super().__init__(
            max_chunks=2,
            max_turns=2,
            conversation_factory=_TestConversationMemory,
        )


def test_base_manager_resolves_and_persists_agent_memory_context() -> None:
    manager = _TestWorkingMemoryManager()

    assert (
        manager.resolve_agent_memory_context(
            conversation_id="conv-1",
            incoming_memory_context=None,
        )
        == ""
    )

    manager.persist_agent_memory_summary(
        conversation_id="conv-1",
        rendered_summary="summary-1",
    )
    assert (
        manager.resolve_agent_memory_context(
            conversation_id="conv-1",
            incoming_memory_context=None,
        )
        == "summary-1"
    )
    assert (
        manager.resolve_agent_memory_context(
            conversation_id="conv-1",
            incoming_memory_context="incoming-override",
        )
        == "incoming-override"
    )
    assert (
        manager.resolve_agent_memory_context(
            conversation_id="conv-1",
            incoming_memory_context=None,
        )
        == "incoming-override"
    )


def test_base_manager_merges_dedupes_and_truncates_working_chunks() -> None:
    manager = _TestWorkingMemoryManager()
    manager.merge_working_chunks(
        conversation_id="conv-2",
        chunks=[_make_chunk("chunk-a", "https://a"), _make_chunk("chunk-b", "https://b")],
    )
    manager.merge_working_chunks(
        conversation_id="conv-2",
        chunks=[
            _make_chunk("chunk-b", "https://b-v2"),
            _make_chunk("chunk-c", "https://c"),
        ],
    )

    chunk_ids = [chunk.chunk_id for chunk in manager.get_working_memory_chunks("conv-2")]
    assert chunk_ids == ["chunk-b", "chunk-c"]
    assert manager.get_working_memory_chunks("conv-2")[0].source_url == "https://b-v2"


def test_base_manager_appends_and_truncates_turn_records() -> None:
    manager = _TestWorkingMemoryManager()
    manager.append_turn_record(
        conversation_id="conv-3",
        record=_TestTurnRecord(turn_id="t1", query="q1"),
    )
    manager.append_turn_record(
        conversation_id="conv-3",
        record=_TestTurnRecord(turn_id="t2", query="q2"),
    )
    manager.append_turn_record(
        conversation_id="conv-3",
        record=_TestTurnRecord(turn_id="t3", query="q3"),
    )

    memory = manager.get_conversation_memory("conv-3")
    assert [row.turn_id for row in memory.turn_records] == ["t2", "t3"]


def test_news_manager_persists_turn_with_deduped_sources_and_score_flag() -> None:
    manager = NewsWorkingMemoryManager(max_chunks=10, max_turns=10)

    manager.persist_finalized_turn(
        conversation_id="conv-news",
        turn_id="turn-1",
        query="AAPL catalysts",
        chunks=[
            _make_chunk("chunk-1", "https://news/1"),
            _make_chunk("chunk-2", "https://news/1"),
            _make_chunk("chunk-3", "https://news/2"),
        ],
        score_unavailable=True,
        source_key_fn=lambda chunk: str(chunk.source_url or "").strip(),
    )

    memory = manager.get_conversation_memory("conv-news")
    assert len(memory.turn_records) == 1
    row = memory.turn_records[0]
    assert row.chunk_ids == ["chunk-1", "chunk-2", "chunk-3"]
    assert row.source_urls == ["https://news/1", "https://news/2"]
    assert row.score_unavailable is True


def test_news_manager_render_working_memory_block_is_stable() -> None:
    manager = NewsWorkingMemoryManager(max_chunks=10, max_turns=10)
    manager.persist_finalized_turn(
        conversation_id="conv-render",
        turn_id="turn-1",
        query="MSFT guidance",
        chunks=[_make_chunk("chunk-1", "https://news/msft")],
        score_unavailable=False,
        source_key_fn=lambda chunk: str(chunk.source_url or "").strip(),
    )

    block = manager.render_working_memory_block("conv-render", turn_limit=4)
    assert "turn=turn-1" in block
    assert "query=MSFT guidance" in block
    assert "relevant_chunks=1" in block
    assert "relevant_sources=1" in block
    assert "score_unavailable=False" in block
