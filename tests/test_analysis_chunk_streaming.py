from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
import sys
import types

import pytest

from api.models.responses import StreamEvent
from api.sinks.sse_sink import SSESink
from core.agents.analysis_token_stream import AnalysisChunkStreamer
from core.services import ServiceManager


def _analysis_chunk_event(seq: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        response_id="resp-1",
        source="news_agent",
        level=SimpleNamespace(label="INFO"),
        message="chunk",
        timestamp=datetime.now(timezone.utc),
        data={
            "event_type": "analysis_chunk",
            "agent": "news_agent",
            "node": "_analyse_news_node",
            "stream_id": "stream-1",
            "phase": "delta",
            "seq": seq,
            "delta": "abc",
        },
    )


def test_stream_event_accepts_analysis_chunk() -> None:
    event = StreamEvent(
        event_type="analysis_chunk",
        request_id="req-1",
        agent="news_agent",
        node="_analyse_news_node",
        stream_id="stream-1",
        phase="delta",
        seq=3,
        delta="hello",
    )
    assert event.event_type == "analysis_chunk"
    assert event.seq == 3


def test_sse_sink_blocks_for_analysis_chunk_when_queue_is_full() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        await queue.put({"event_type": "existing"})
        sink = SSESink(
            request_id="req-1",
            response_id="resp-1",
            event_queue=queue,
            loop=loop,
        )

        sink.on_event(_analysis_chunk_event(seq=1))
        await asyncio.sleep(0)

        first = await queue.get()
        queue.task_done()
        assert first["event_type"] == "existing"

        await asyncio.sleep(0.01)
        second = await queue.get()
        queue.task_done()
        assert second["event_type"] == "analysis_chunk"
        assert second["seq"] == 1

    asyncio.run(_run())


def test_sse_sink_drain_preserves_chunk_order_before_terminal_event() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        await queue.put({"event_type": "existing"})
        sink = SSESink(
            request_id="req-1",
            response_id="resp-1",
            event_queue=queue,
            loop=loop,
        )

        sink.on_event(_analysis_chunk_event(seq=2))
        await asyncio.sleep(0)

        first = await queue.get()
        queue.task_done()
        assert first["event_type"] == "existing"

        await sink.drain()
        complete_put = asyncio.create_task(queue.put({"event_type": "complete"}))

        second = await queue.get()
        queue.task_done()
        await complete_put
        third = await queue.get()
        queue.task_done()

        assert second["event_type"] == "analysis_chunk"
        assert second["seq"] == 2
        assert third["event_type"] == "complete"

    asyncio.run(_run())


def test_analysis_chunk_streamer_emits_start_delta_end(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str, dict]] = []

    def _capture(source: str, event_type: str, payload: dict) -> None:
        captured.append((source, event_type, payload))

    monkeypatch.setattr(
        "core.agents.analysis_token_stream.publish_frontend_event",
        _capture,
    )
    streamer = AnalysisChunkStreamer(
        source="news_agent",
        agent="news_agent",
        node="_analyse_news_node",
        enabled=True,
        flush_chars=3,
        flush_interval_s=10.0,
    )
    streamer.start()
    streamer.add_delta("ab")
    streamer.add_delta("cd")
    streamer.end(final_text="abcd")

    phases = [payload["phase"] for _, _, payload in captured]
    assert phases == ["start", "delta", "end"]
    assert captured[1][2]["delta"] == "abcd"
    assert captured[2][2]["text"] == "abcd"
    assert captured[2][2]["is_final"] is True


def test_service_manager_get_agent_returns_separate_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        def __init__(self, **kwargs):
            self.temperature = kwargs.get("temperature")

    fake_module = types.SimpleNamespace(ChatGoogleGenerativeAI=_FakeLLM)
    monkeypatch.setitem(sys.modules, "langchain_google_genai.chat_models", fake_module)

    manager = ServiceManager()
    first = manager.get_agent(temperature=0.1)
    second = manager.get_agent(temperature=0.9)
    assert first is not second
    assert first.temperature == 0.1
    assert second.temperature == 0.9
