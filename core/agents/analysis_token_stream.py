from __future__ import annotations

import time
from typing import Any, List, Sequence
from uuid import uuid4

from langchain_core.messages import BaseMessage

from core.event_queue import publish_frontend_event


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) else ""
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return ""


def chunk_text(chunk: Any) -> str:
    """Extract plain text from a streaming model chunk."""
    return _coerce_text(getattr(chunk, "content", chunk))


class AnalysisChunkStreamer:
    """
    Emits coalesced token chunks to SSE via EventQueue frontend events.
    """

    def __init__(
        self,
        *,
        source: str,
        agent: str,
        node: str,
        enabled: bool,
        flush_chars: int = 96,
        flush_interval_s: float = 0.08,
    ) -> None:
        self._source = source
        self._agent = agent
        self._node = node
        self._enabled = enabled
        self._flush_chars = max(1, int(flush_chars))
        self._flush_interval_s = max(0.01, float(flush_interval_s))
        self._stream_id = str(uuid4())
        self._seq = 0
        self._buffer: List[str] = []
        self._buffer_chars = 0
        self._last_flush = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _emit(
        self,
        *,
        phase: str,
        delta: str | None = None,
        text: str | None = None,
        is_final: bool | None = None,
        error: str | None = None,
    ) -> None:
        if not self._enabled:
            return
        payload: dict[str, Any] = {
            "agent": self._agent,
            "node": self._node,
            "stream_id": self._stream_id,
            "phase": phase,
            "seq": self._seq,
        }
        if delta is not None:
            payload["delta"] = delta
        if text is not None:
            payload["text"] = text
        if is_final is not None:
            payload["is_final"] = is_final
        if error is not None:
            payload["error"] = error
        publish_frontend_event(self._source, "analysis_chunk", payload)
        self._seq += 1

    def start(self) -> None:
        self._emit(phase="start")

    def add_delta(self, text: str) -> None:
        if not self._enabled:
            return
        if not text:
            return
        self._buffer.append(text)
        self._buffer_chars += len(text)
        now = time.monotonic()
        if (
            self._buffer_chars >= self._flush_chars
            or (now - self._last_flush) >= self._flush_interval_s
        ):
            self.flush()

    def flush(self) -> None:
        if not self._enabled:
            return
        if not self._buffer:
            return
        delta = "".join(self._buffer)
        self._buffer.clear()
        self._buffer_chars = 0
        self._last_flush = time.monotonic()
        self._emit(phase="delta", delta=delta)

    def end(self, *, final_text: str) -> None:
        if not self._enabled:
            return
        self.flush()
        self._emit(phase="end", text=final_text, is_final=True)

    def error(self, message: str) -> None:
        if not self._enabled:
            return
        self.flush()
        self._emit(phase="error", error=message, is_final=True)


async def stream_model_text(
    *,
    llm: Any,
    messages: Sequence[BaseMessage],
    streamer: AnalysisChunkStreamer,
) -> str:
    parts: List[str] = []
    async for chunk in llm.astream(messages):
        text = chunk_text(chunk)
        if not text:
            continue
        parts.append(text)
        streamer.add_delta(text)
    return "".join(parts)
