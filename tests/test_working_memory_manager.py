from __future__ import annotations

from dataclasses import dataclass, field

from core.agents.working_memory.base import (
    ConversationWorkingMemoryBase,
    ConversationWorkingMemoryManagerBase,
    TurnRelevantMemoryBase,
)
from core.agents.working_memory.news_working_memory import NewsWorkingMemoryManager
from core.agents.working_memory.fundamental_working_memory import (
    FundamentalWorkingMemoryManager,
)
from core.agents.models.fundamental_agent_models import (
    ExecutorBatchLog,
    ExecutorToolLog,
)
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
        turn_index=1,
        chunks=[
            _make_chunk("chunk-1", "https://news/1"),
            _make_chunk("chunk-2", "https://news/1"),
            _make_chunk("chunk-3", "https://news/2"),
        ],
        source_key_fn=lambda chunk: str(chunk.source_url or "").strip(),
    )

    memory = manager.get_conversation_memory("conv-news")
    assert len(memory.turn_records) == 1
    row = memory.turn_records[0]
    assert row.turn_id == "1"
    assert row.chunk_ids == ["chunk-1", "chunk-2", "chunk-3"]
    assert row.source_urls == ["https://news/1", "https://news/2"]
    assert memory.seen_url_keys == ["https://news/1", "https://news/2"]
    assert memory.seen_chunk_ids == ["chunk-1", "chunk-2", "chunk-3"]


def test_news_manager_render_working_memory_block_is_stable() -> None:
    manager = NewsWorkingMemoryManager(max_chunks=10, max_turns=10)
    manager.persist_finalized_turn(
        conversation_id="conv-render",
        turn_index=1,
        chunks=[_make_chunk("chunk-1", "https://news/msft")],
        source_key_fn=lambda chunk: str(chunk.source_url or "").strip(),
    )

    block = manager.render_working_memory_block("conv-render", turn_limit=4)
    assert "turn_index=1" in block
    assert "relevant_chunks=1" in block
    assert "relevant_sources=1" in block


def test_news_manager_build_context_from_history_summaries_is_stable() -> None:
    turns = [
        {
            "created_at": "2026-04-26T00:00:00+00:00",
            "agent_memory_summaries": {
                "news_agent": {
                    "research_actions": ["newsapi"],
                    "source_count": 2,
                    "sentiment": {"label": "BUY"},
                    "main_catalyst": "Raised guidance.",
                }
            },
        },
        {
            "created_at": "2026-04-26T00:01:00+00:00",
            "agent_memory_summaries": {
                "news_agent": {
                    "research_actions": ["web_search"],
                    "source_count": 1,
                    "sentiment": {"label": "NEUTRAL"},
                    "main_catalyst": "Mixed follow-through.",
                }
            },
        },
    ]

    block = NewsWorkingMemoryManager.build_context_from_history_summaries(
        turns, window=2
    )
    assert "[2026-04-26T00:00:00+00:00]" in block
    assert "actions=newsapi" in block
    assert "actions=web_search" in block
    assert "sentiment=NEUTRAL" in block


def test_news_manager_seen_history_canonicalizes_urls_and_enforces_caps() -> None:
    manager = NewsWorkingMemoryManager(
        max_chunks=10,
        max_turns=10,
        seen_url_cap=2,
        seen_chunk_cap=3,
    )
    manager.merge_seen_history(
        conversation_id="conv-cap",
        url_keys=[
            manager.canonicalize_url_key("https://EXAMPLE.com/a?utm_source=x#fragment"),
            manager.canonicalize_url_key("https://example.com/b?x=1"),
        ],
        chunk_ids=["c1", "c2"],
    )
    manager.merge_seen_history(
        conversation_id="conv-cap",
        url_keys=[
            manager.canonicalize_url_key("https://example.com/a?again=1"),
            manager.canonicalize_url_key("https://example.com/c"),
        ],
        chunk_ids=["c2", "c3", "c4"],
    )

    assert manager.get_seen_url_keys("conv-cap") == [
        "https://example.com/b",
        "https://example.com/c",
    ]
    assert manager.get_seen_chunk_ids("conv-cap") == ["c2", "c3", "c4"]


def test_news_manager_default_working_chunk_cap_is_config_driven() -> None:
    manager = NewsWorkingMemoryManager()
    chunks = [_make_chunk(f"chunk-{idx}", f"https://news/{idx}") for idx in range(40)]
    manager.merge_working_chunks(conversation_id="conv-limit", chunks=chunks)
    stored = manager.get_working_memory_chunks("conv-limit")
    assert len(stored) == 30
    assert stored[0].chunk_id == "chunk-10"


def test_fundamental_manager_persists_counts_and_batch_records() -> None:
    manager = FundamentalWorkingMemoryManager(max_turns=10)
    manager.persist_finalized_turn(
        conversation_id="conv-fund",
        turn_id="turn-1",
        query="Assess margins",
        task_completed=False,
        task_completion_reason="Need one more ratio.",
        computed_row_labels=["gross_margin", "operating_margin"],
        executor_logs=[
            ExecutorBatchLog(
                batch_index=0,
                batch_reasoning="First pass",
                calls=[
                    ExecutorToolLog(
                        tool_name="profitability_ratios",
                        parameters={"revenue_metric": "Revenues"},
                        success=True,
                        summary="Computed margin rows.",
                        output_row_labels=["gross_margin"],
                        added_row_count=1,
                    ),
                    ExecutorToolLog(
                        tool_name="dcf_intrinsic_value",
                        parameters={"fcf_metric": "FreeCashFlow"},
                        success=False,
                        error="Missing FreeCashFlow",
                        summary="Failed",
                        added_row_count=0,
                    ),
                ],
            )
        ],
    )

    row = manager.get_conversation_memory("conv-fund").turn_records[0]
    assert row.tool_call_count == 2
    assert row.successful_tool_call_count == 1
    assert row.failed_tool_call_count == 1
    assert row.computed_row_labels == ["gross_margin", "operating_margin"]
    assert row.batch_records[0].calls[0].tool_name == "profitability_ratios"


def test_fundamental_manager_truncates_turns_and_renders_stably() -> None:
    manager = FundamentalWorkingMemoryManager(max_turns=2)
    for idx in range(1, 4):
        manager.persist_finalized_turn(
            conversation_id="conv-trunc",
            turn_id=f"turn-{idx}",
            query=f"query-{idx}",
            task_completed=True,
            task_completion_reason="",
            computed_row_labels=[f"row-{idx}"],
            executor_logs=[
                ExecutorBatchLog(
                    batch_index=idx,
                    batch_reasoning=f"batch-{idx}",
                    calls=[
                        ExecutorToolLog(
                            tool_name="cagr",
                            parameters={"metric": "Revenues"},
                            success=True,
                            summary=f"summary-{idx}",
                            added_row_count=1,
                        )
                    ],
                )
            ],
        )

    memory = manager.get_conversation_memory("conv-trunc")
    assert [row.turn_id for row in memory.turn_records] == ["turn-2", "turn-3"]

    block = manager.render_planner_working_memory_block(
        "conv-trunc", turn_limit=4, max_calls_per_turn=12
    )
    assert "turn=turn-2" in block
    assert "turn=turn-3" in block
    assert "turn=turn-1" not in block
    assert "call=cagr" in block


def test_fundamental_manager_build_context_from_history_summaries_is_stable() -> None:
    turns = [
        {
            "created_at": "2026-04-26T01:00:00+00:00",
            "agent_memory_summaries": {
                "fundamentals_agent": {
                    "tools_used": ["profitability_ratios"],
                    "key_rows": ["Revenues"],
                    "task_completed": False,
                    "main_conclusion": "Need one more ratio.",
                }
            },
        },
        {
            "created_at": "2026-04-26T01:01:00+00:00",
            "agent_memory_summaries": {
                "fundamentals_agent": {
                    "tools_used": ["cagr"],
                    "key_rows": ["NetIncomeLoss"],
                    "task_completed": True,
                    "main_conclusion": "Trend is positive.",
                }
            },
        },
    ]

    block = FundamentalWorkingMemoryManager.build_context_from_history_summaries(
        turns, window=2
    )
    assert "[2026-04-26T01:00:00+00:00]" in block
    assert "tools=profitability_ratios" in block
    assert "tools=cagr" in block
    assert "conclusion=Trend is positive." in block
