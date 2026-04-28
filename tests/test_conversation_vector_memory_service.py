from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from core.memory.conversation import ConversationVectorMemoryService


class _FakeLLM:
    def __init__(self, token_count: int) -> None:
        self._token_count = token_count

    def get_num_tokens(self, _text: str) -> int:
        return self._token_count


def _turn(turn_id: str, user: str, assistant: str) -> dict:
    return {
        "turn_id": turn_id,
        "created_at": "2026-04-28T10:00:00+00:00",
        "user_message": user,
        "assistant_synthesis": assistant,
        "agent_memory_summaries": {
            "news_agent": {"source_count": 3, "main_catalyst": "Guidance"}
        },
    }


def test_ensure_index_skips_when_below_token_threshold(
    chroma_adapter_stub,
    monkeypatch,
) -> None:
    adapter, vectorstore = chroma_adapter_stub
    monkeypatch.setattr("core.config.settings.CONVERSATION_MEMORY_TOKEN_LIMIT", 5000)

    service = ConversationVectorMemoryService(adapter, _FakeLLM(token_count=20))
    turns = [_turn("t-1", "Quick question", "Quick answer")]

    indexed = asyncio.run(
        service.ensure_index(
            conversation_id="conv-1",
            user_email="alpha@example.com",
            turns=turns,
        )
    )

    assert indexed is False
    assert vectorstore.last_payload == {}


def test_ensure_index_and_retrieve_reuses_existing_vector_settings(
    chroma_adapter_stub,
    monkeypatch,
) -> None:
    adapter, vectorstore = chroma_adapter_stub
    monkeypatch.setattr("core.config.settings.CONVERSATION_MEMORY_TOKEN_LIMIT", 1)
    monkeypatch.setattr("core.config.settings.CHUNK_SIZE", 40)
    monkeypatch.setattr("core.config.settings.CHUNK_OVERLAP", 5)
    monkeypatch.setattr("core.config.settings.MEMORY_VECTOR_TOP_K", 7)
    monkeypatch.setattr("core.config.settings.MEMORY_SIMILARITY_THRESHOLD", 0.8)

    captured_query_kwargs: dict = {}

    async def _fake_query(**kwargs):
        captured_query_kwargs.update(kwargs)
        return [
            (
                Document(
                    page_content="User prefers valuation-first framing.",
                    metadata={
                        "chunk_id": "cid-1",
                        "turn_id": "t-1",
                        "created_at": "2026-04-28T10:00:00+00:00",
                    },
                    id="cid-1",
                ),
                0.92,
            ),
            (
                Document(
                    page_content="Low-score chunk should be filtered out.",
                    metadata={
                        "chunk_id": "cid-2",
                        "turn_id": "t-2",
                        "created_at": "2026-04-28T10:02:00+00:00",
                    },
                    id="cid-2",
                ),
                0.6,
            ),
        ]

    monkeypatch.setattr(adapter, "query", _fake_query)
    service = ConversationVectorMemoryService(adapter, _FakeLLM(token_count=2000))

    block, hits = asyncio.run(
        service.ensure_index_and_retrieve(
            conversation_id="conv-9",
            user_email="alpha@example.com",
            turns=[
                _turn(
                    "t-1",
                    "I want valuation focused answers with less macro noise.",
                    "Understood. I will focus on valuation framing.",
                )
            ],
            query="How should we frame AAPL valuation now?",
        )
    )

    assert vectorstore.last_payload["ids"], "Expected chunk upsert to occur"
    assert captured_query_kwargs["n_results"] == 7
    assert captured_query_kwargs["where"] == {"conversation_id": "conv-9"}
    assert len(hits) == 1
    assert "valuation-first framing" in block


def test_per_user_private_collection_isolation(monkeypatch) -> None:
    from core.memory.stores.chroma_adapter import ChromaDBAdapter

    monkeypatch.setattr("core.config.settings.CONVERSATION_MEMORY_TOKEN_LIMIT", 1)
    adapter = ChromaDBAdapter(
        collection_name="news_chunks",
        persist_directory=".test",
        embedding_function=object(),
    )

    captured_collections: list[str] = []

    async def _fake_upsert_chunks(*, collection_name=None, **_kwargs):
        captured_collections.append(str(collection_name))

    monkeypatch.setattr(adapter, "upsert_chunks", _fake_upsert_chunks)
    service = ConversationVectorMemoryService(adapter, _FakeLLM(token_count=2000))
    turns = [_turn("t-1", "Question", "Answer")]

    asyncio.run(
        service.ensure_index(
            conversation_id="conv-1",
            user_email="alpha@example.com",
            turns=turns,
        )
    )
    asyncio.run(
        service.ensure_index(
            conversation_id="conv-1",
            user_email="beta@example.com",
            turns=turns,
        )
    )

    assert len(captured_collections) == 2
    assert captured_collections[0] != captured_collections[1]
    assert all(name.startswith("conv_private_") for name in captured_collections)


def test_incremental_indexing_and_threshold_cross_backfill(monkeypatch) -> None:
    from core.memory.stores.chroma_adapter import ChromaDBAdapter

    adapter = ChromaDBAdapter(
        collection_name="news_chunks",
        persist_directory=".test",
        embedding_function=object(),
    )

    captured_ids: list[list[str]] = []

    async def _fake_upsert_chunks(*, chunk_ids, **_kwargs):
        captured_ids.append(list(chunk_ids))

    monkeypatch.setattr(adapter, "upsert_chunks", _fake_upsert_chunks)
    service = ConversationVectorMemoryService(adapter, _FakeLLM(token_count=3000))

    turns = [
        _turn("t-1", "First", "First answer"),
        _turn("t-2", "Second", "Second answer"),
    ]

    monkeypatch.setattr("core.config.settings.CONVERSATION_MEMORY_TOKEN_LIMIT", 5000)
    first = asyncio.run(
        service.ensure_index(
            conversation_id="conv-77",
            user_email="alpha@example.com",
            turns=turns,
        )
    )
    assert first is False
    assert captured_ids == []

    monkeypatch.setattr("core.config.settings.CONVERSATION_MEMORY_TOKEN_LIMIT", 1)
    second = asyncio.run(
        service.ensure_index(
            conversation_id="conv-77",
            user_email="alpha@example.com",
            turns=turns,
        )
    )
    assert second is True
    assert len(captured_ids) == 1
    assert any("::t-1::" in cid for cid in captured_ids[0])
    assert any("::t-2::" in cid for cid in captured_ids[0])

    turns.append(_turn("t-3", "Third", "Third answer"))
    third = asyncio.run(
        service.ensure_index(
            conversation_id="conv-77",
            user_email="alpha@example.com",
            turns=turns,
        )
    )
    assert third is True
    assert len(captured_ids) == 2
    assert all("::t-3::" in cid for cid in captured_ids[1])
