from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage

from api.services.conversation_jsonl_store import JsonlConversationStore
from api.services.conversation_service import ConversationStore


def _sample_turn(*, request_id: str, message: str, synthesis: str) -> dict:
    return {
        "turn_id": f"conv-1:{request_id}",
        "request_id": request_id,
        "conversation_id": "conv-1",
        "user_email": "alpha@example.com",
        "session_id": "session-1",
        "created_at": "2026-04-26T00:00:00+00:00",
        "duration_ms": 10.5,
        "user_message": message,
        "assistant_synthesis": synthesis,
        "agent_analyses": {"news_agent": "news", "fundamentals_agent": "funds"},
        "ticker_results": [],
        "tickers": ["AAPL"],
    }


def test_jsonl_store_creates_index_and_conversation_file(tmp_path) -> None:
    store = JsonlConversationStore(str(tmp_path / "chatlogs"))
    asyncio.run(store.initialize())
    asyncio.run(store.ensure_conversation("conv-1", "alpha@example.com"))

    user_dir = tmp_path / "chatlogs" / "alpha_example.com"
    assert (user_dir / "index.json").exists()
    assert (user_dir / "conv-1.jsonl").exists()

    index_payload = json.loads((user_dir / "index.json").read_text(encoding="utf-8"))
    assert isinstance(index_payload, list)
    assert index_payload[0]["conversation_id"] == "conv-1"
    assert index_payload[0]["message_count"] == 0


def test_jsonl_store_append_turn_updates_counts_and_loads_turns(tmp_path) -> None:
    store = JsonlConversationStore(str(tmp_path / "chatlogs"))
    asyncio.run(store.initialize())
    asyncio.run(store.ensure_conversation("conv-1", "alpha@example.com"))
    asyncio.run(
        store.append_turn(
            conversation_id="conv-1",
            user_email="alpha@example.com",
            turn=_sample_turn(request_id="req-1", message="hi", synthesis="hello"),
        )
    )

    turns = asyncio.run(store.load_turns("conv-1", "alpha@example.com"))
    assert len(turns) == 1
    assert turns[0]["request_id"] == "req-1"
    assert turns[0]["assistant_synthesis"] == "hello"

    rows = asyncio.run(store.list_conversations("alpha@example.com", limit=10))
    assert len(rows) == 1
    assert rows[0]["message_count"] == 2
    assert rows[0]["turn_count"] == 1


def test_conversation_service_projects_turns_to_message_history(tmp_path) -> None:
    adapter = JsonlConversationStore(str(tmp_path / "chatlogs"))
    service = ConversationStore(adapter)
    asyncio.run(service.initialize())
    asyncio.run(service.ensure_conversation("conv-1", "alpha@example.com"))
    asyncio.run(
        service.append_turn(
            conversation_id="conv-1",
            user_email="alpha@example.com",
            turn=_sample_turn(request_id="req-1", message="Should I buy AAPL?", synthesis="Balanced view."),
        )
    )

    history = asyncio.run(service.get_history("conv-1", user_email="alpha@example.com"))
    assert [row["role"] for row in history] == ["user", "assistant"]
    assert history[0]["content"] == "Should I buy AAPL?"
    assert history[1]["content"] == "Balanced view."

    langchain_messages = asyncio.run(
        service.get_langchain_messages("conv-1", user_email="alpha@example.com")
    )
    assert isinstance(langchain_messages[0], HumanMessage)
    assert isinstance(langchain_messages[1], AIMessage)


def test_jsonl_store_load_turns_self_heals_missing_chatlog_file(tmp_path) -> None:
    store = JsonlConversationStore(str(tmp_path / "chatlogs"))
    asyncio.run(store.initialize())

    turns = asyncio.run(store.load_turns("conv-missing", "alpha@example.com"))
    assert turns == []

    chatlog_path = tmp_path / "chatlogs" / "alpha_example.com" / "conv-missing.jsonl"
    assert chatlog_path.exists()


def test_jsonl_store_ensure_user_workspace_creates_index_only(tmp_path) -> None:
    store = JsonlConversationStore(str(tmp_path / "chatlogs"))
    asyncio.run(store.initialize())

    count = asyncio.run(store.ensure_user_workspace("alpha@example.com"))
    user_dir = tmp_path / "chatlogs" / "alpha_example.com"

    assert count == 0
    assert (user_dir / "index.json").exists()
    assert list(user_dir.glob("*.jsonl")) == []


def test_jsonl_store_ensure_user_workspace_is_idempotent(tmp_path) -> None:
    store = JsonlConversationStore(str(tmp_path / "chatlogs"))
    asyncio.run(store.initialize())
    asyncio.run(store.ensure_conversation("conv-1", "alpha@example.com"))

    first = asyncio.run(store.ensure_user_workspace("alpha@example.com"))
    second = asyncio.run(store.ensure_user_workspace("alpha@example.com"))

    assert first == 1
    assert second == 1

    index_payload = json.loads(
        (tmp_path / "chatlogs" / "alpha_example.com" / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(index_payload) == 1
    assert index_payload[0]["conversation_id"] == "conv-1"
