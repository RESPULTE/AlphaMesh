from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from api.services.conversation_jsonl_store import JsonlConversationStore
from api.services.conversation_service import ConversationStore


def conversation_store(tmp_path) -> ConversationStore:
    return ConversationStore(JsonlConversationStore(base_path=str(tmp_path / "chatlogs")))


async def _seed_turns(
    store: ConversationStore,
    *,
    conversation_id: str,
    user_email: str,
    count: int,
) -> None:
    await store.initialize()
    await store.ensure_conversation(conversation_id, user_email)
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    for idx in range(count):
        turn = {
            "turn_id": f"turn-{idx}",
            "request_id": f"req-{idx}",
            "conversation_id": conversation_id,
            "user_email": user_email,
            "session_id": "session-1",
            "created_at": (base + timedelta(minutes=idx)).isoformat(),
            "duration_ms": float(1000 + idx),
            "user_message": f"user prompt {idx}",
            "assistant_synthesis": f"assistant response {idx}",
            "agent_analyses": {},
            "agent_memory_summaries": {},
            "ticker_results": [],
            "tickers": [],
        }
        await store.append_turn(conversation_id, user_email, turn)


def test_get_turns_paginated_without_limit_returns_full_history(tmp_path) -> None:
    async def _run() -> None:
        store = conversation_store(tmp_path)
        user_email = "demo@alphamesh.local"
        conversation_id = "conv-1"
        await _seed_turns(
            store,
            conversation_id=conversation_id,
            user_email=user_email,
            count=20,
        )

        turns, has_more, next_before = await store.get_turns_paginated(
            conversation_id,
            user_email=user_email,
            limit=None,
        )

        assert len(turns) == 20
        assert turns[0]["turn_id"] == "turn-0"
        assert turns[-1]["turn_id"] == "turn-19"
        assert has_more is False
        assert next_before is None

    asyncio.run(_run())


def test_get_turns_paginated_latest_page_limit_8(tmp_path) -> None:
    async def _run() -> None:
        store = conversation_store(tmp_path)
        user_email = "demo@alphamesh.local"
        conversation_id = "conv-1"
        await _seed_turns(
            store,
            conversation_id=conversation_id,
            user_email=user_email,
            count=20,
        )

        turns, has_more, next_before = await store.get_turns_paginated(
            conversation_id,
            user_email=user_email,
            limit=8,
        )

        assert [turn["turn_id"] for turn in turns] == [f"turn-{idx}" for idx in range(12, 20)]
        assert has_more is True
        assert next_before == "turn-12"

    asyncio.run(_run())


def test_get_turns_paginated_before_turn_returns_previous_page_without_overlap(
    tmp_path,
) -> None:
    async def _run() -> None:
        store = conversation_store(tmp_path)
        user_email = "demo@alphamesh.local"
        conversation_id = "conv-1"
        await _seed_turns(
            store,
            conversation_id=conversation_id,
            user_email=user_email,
            count=20,
        )

        latest_turns, _, first_next_before = await store.get_turns_paginated(
            conversation_id,
            user_email=user_email,
            limit=8,
        )
        prev_turns, has_more, next_before = await store.get_turns_paginated(
            conversation_id,
            user_email=user_email,
            limit=8,
            before_turn_id=first_next_before,
        )

        latest_ids = {turn["turn_id"] for turn in latest_turns}
        prev_ids = [turn["turn_id"] for turn in prev_turns]

        assert prev_ids == [f"turn-{idx}" for idx in range(4, 12)]
        assert latest_ids.isdisjoint(set(prev_ids))
        assert has_more is True
        assert next_before == "turn-4"

    asyncio.run(_run())


def test_get_turns_paginated_invalid_before_turn_id_returns_empty_page(
    tmp_path,
) -> None:
    async def _run() -> None:
        store = conversation_store(tmp_path)
        user_email = "demo@alphamesh.local"
        conversation_id = "conv-1"
        await _seed_turns(
            store,
            conversation_id=conversation_id,
            user_email=user_email,
            count=20,
        )

        turns, has_more, next_before = await store.get_turns_paginated(
            conversation_id,
            user_email=user_email,
            limit=8,
            before_turn_id="missing-turn-id",
        )

        assert turns == []
        assert has_more is False
        assert next_before is None

    asyncio.run(_run())
