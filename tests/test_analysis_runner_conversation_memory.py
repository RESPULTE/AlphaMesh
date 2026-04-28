from __future__ import annotations

import asyncio
from types import SimpleNamespace

from api.models.requests import ChatRequest
from api.services.analysis_runner import AnalysisRunner
from api.services.event_broadcaster import EventBroadcaster


class _FakeStore:
    def __init__(self) -> None:
        self.appended_turns: list[dict] = []

    async def ensure_conversation(self, _conversation_id: str, _user_email: str) -> None:
        return None

    async def get_langchain_messages(self, _conversation_id: str, user_email: str):
        _ = user_email
        return []

    async def get_turns(self, _conversation_id: str, user_email: str):
        _ = user_email
        return [
            {
                "turn_id": "old-turn-1",
                "created_at": "2026-04-28T10:00:00+00:00",
                "user_message": "Old question",
                "assistant_synthesis": "Old answer",
                "agent_memory_summaries": {},
            }
        ]

    async def append_turn(self, conversation_id: str, user_email: str, turn: dict) -> None:
        _ = (conversation_id, user_email)
        self.appended_turns.append(dict(turn))


class _FakeSessionService:
    async def link_conversation(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_id: str,
    ) -> None:
        _ = (user_id, session_id, conversation_id)
        return None


class _FakeOrchestrator:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.last_kwargs = None

    async def run(self, **kwargs):
        self.trace.append("orchestrator_run")
        self.last_kwargs = dict(kwargs)
        return SimpleNamespace(
            summary="done",
            tickers=[],
            agent_analyses={},
            sources=[],
            fundamental_data=None,
            fundamentals_visualization=None,
            fundamentals_raw_display_data=None,
            fundamentals_task_completed=True,
            fundamentals_task_completion_reason="",
            agent_memory_summaries={},
            turn_id="turn-new-1",
        )


class _FakeConversationMemoryService:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.refresh_started = asyncio.Event()
        self.refresh_done = asyncio.Event()

    async def ensure_index_and_retrieve(self, **_kwargs):
        self.trace.append("memory_retrieve")
        return (
            "1. [2026-04-28T10:00:00+00:00] turn=old-turn-1 score=0.91\n   Old preference.",
            [{"chunk_id": "cid-1", "turn_id": "old-turn-1", "score": 0.91}],
        )

    async def ensure_index(self, **_kwargs):
        self.trace.append("memory_refresh_start")
        self.refresh_started.set()
        await asyncio.sleep(0.2)
        self.trace.append("memory_refresh_done")
        self.refresh_done.set()
        return True


def test_runner_passes_memory_to_orchestrator_and_refresh_is_non_blocking(monkeypatch) -> None:
    from core.services import service_manager

    trace: list[str] = []
    broadcaster = EventBroadcaster()
    request_id = "req-memory-1"
    broadcaster.create(request_id)

    store = _FakeStore()
    session_service = _FakeSessionService()
    orchestrator = _FakeOrchestrator(trace)
    memory_service = _FakeConversationMemoryService(trace)

    runner = AnalysisRunner(
        broadcaster=broadcaster,
        store=store,
        session_service=session_service,
        orchestrator=orchestrator,
    )

    monkeypatch.setattr(
        service_manager,
        "get_conversation_memory_service",
        lambda: memory_service,
    )

    async def _scenario() -> None:
        await runner._run(
            request_id=request_id,
            conversation_id="conv-memory-1",
            chat_request=ChatRequest(
                message="Use my prior context please.",
                conversation_id="conv-memory-1",
                user_email="demo@alphamesh.local",
            ),
            user_id="demo@alphamesh.local",
            session_id="session-1",
        )

        # Retrieval must run before orchestrator invocation.
        assert trace.index("memory_retrieve") < trace.index("orchestrator_run")

        assert orchestrator.last_kwargs is not None
        assert (
            orchestrator.last_kwargs["conversation_memory_block"].startswith("1. [2026-04-28")
        )
        assert orchestrator.last_kwargs["conversation_memory_hits"][0]["chunk_id"] == "cid-1"

        # Refresh is intentionally backgrounded and not awaited by _run.
        await asyncio.sleep(0)
        assert memory_service.refresh_started.is_set()
        assert not memory_service.refresh_done.is_set()
        await asyncio.sleep(0.25)
        assert memory_service.refresh_done.is_set()

    asyncio.run(_scenario())
