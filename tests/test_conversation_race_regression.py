from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.models.requests import ChatRequest
from api.routers import auth as auth_router
from api.routers import chat as chat_router
from api.routers import conversations as conversations_router
from api.services.analysis_runner import AnalysisRunner
from api.services.conversation_jsonl_store import JsonlConversationStore
from api.services.conversation_service import ConversationStore
from api.services.event_broadcaster import EventBroadcaster
from api.services.session_service import SessionService
from api.services.session_sql_store import SQLiteSessionStore
from core.config import settings


class _NoOpOrchestrator:
    async def run(self, **_kwargs):
        return None


class _NoOpAnalysisRunner(AnalysisRunner):
    async def _run(
        self,
        request_id: str,
        conversation_id: str,
        chat_request: ChatRequest,
        *,
        user_id: str,
        session_id: str,
    ) -> None:
        _ = (request_id, conversation_id, chat_request, user_id, session_id)
        return None


class _RecordingStore:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    async def ensure_conversation(self, conversation_id: str, user_email: str) -> None:
        _ = (conversation_id, user_email)
        self._trace.append("ensure_conversation")


class _RecordingSessionService:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    async def link_conversation(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_id: str,
    ) -> None:
        _ = (user_id, session_id, conversation_id)
        self._trace.append("link_conversation")


def test_launch_prepares_conversation_before_return() -> None:
    trace: list[str] = []
    runner = _NoOpAnalysisRunner(
        broadcaster=EventBroadcaster(),
        store=_RecordingStore(trace),  # type: ignore[arg-type]
        session_service=_RecordingSessionService(trace),  # type: ignore[arg-type]
        orchestrator=_NoOpOrchestrator(),  # type: ignore[arg-type]
    )

    async def _scenario() -> None:
        conversation_id = await runner.launch(
            request_id="req-1",
            chat_request=ChatRequest(
                message="Analyze AAPL",
                conversation_id="conv-1",
                user_email="user@example.com",
            ),
            user_id="user@example.com",
            session_id="session-1",
        )
        assert conversation_id == "conv-1"
        assert trace[:2] == ["ensure_conversation", "link_conversation"]

    asyncio.run(_scenario())


def test_chat_then_immediate_turns_fetch_does_not_404(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "DEV_ALLOW_USER_EMAIL_FALLBACK", True)

    store = ConversationStore(JsonlConversationStore(str(tmp_path / "chatlogs")))
    session_service = SessionService(
        SQLiteSessionStore(str(tmp_path / "conversations.db"))
    )
    asyncio.run(store.initialize())
    asyncio.run(session_service.initialize())

    runner = _NoOpAnalysisRunner(
        broadcaster=EventBroadcaster(),
        store=store,
        session_service=session_service,
        orchestrator=_NoOpOrchestrator(),  # type: ignore[arg-type]
    )

    app = FastAPI()
    app.state.store = store
    app.state.session_service = session_service
    app.state.runner = runner
    app.include_router(chat_router.router)
    app.include_router(conversations_router.router)
    client = TestClient(app)

    chat = client.post(
        "/api/v1/chat",
        json={
            "message": "Analyze AAPL",
            "conversation_id": None,
            "session_id": None,
            "user_email": "race@test.com",
        },
    )
    assert chat.status_code == 202
    conversation_id = chat.json()["conversation_id"]
    assert conversation_id

    turns = client.get(
        f"/api/v1/conversations/{conversation_id}/turns",
        params={"user_email": "race@test.com", "limit": 8},
    )
    assert turns.status_code == 200
    payload = turns.json()
    assert payload["conversation_id"] == conversation_id
    assert payload["turns"] == []


def test_bootstrap_endpoint_creates_user_workspace(tmp_path) -> None:
    store = ConversationStore(JsonlConversationStore(str(tmp_path / "chatlogs")))
    session_service = SessionService(
        SQLiteSessionStore(str(tmp_path / "conversations.db"))
    )
    asyncio.run(store.initialize())
    asyncio.run(session_service.initialize())

    app = FastAPI()
    app.state.store = store
    app.state.session_service = session_service
    app.include_router(auth_router.router)
    app.include_router(conversations_router.router)
    client = TestClient(app)

    login = client.post(
        "/api/v1/auth/login",
        json={"email": "bootstrap@test.com"},
    )
    assert login.status_code == 200
    access_token = login.json()["access_token"]

    bootstrap = client.post(
        "/api/v1/conversations/bootstrap",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    assert payload["status"] == "ok"
    assert payload["conversation_count"] == 0

    user_dir = tmp_path / "chatlogs" / "bootstrap_test.com"
    assert (user_dir / "index.json").exists()

    bootstrap_again = client.post(
        "/api/v1/conversations/bootstrap",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert bootstrap_again.status_code == 200
    assert bootstrap_again.json()["conversation_count"] == 0
