"""
event_queue.py
==============

A lightweight event queue that lets agents publish progress messages to
the end user.  Every message is stored, printed to the console the moment
it arrives, and optionally forwarded to any other sink you configure
(Streamlit, WebSocket, file, …).

Messages are automatically grouped into response buckets — response_1,
response_2, … — so you can review exactly what happened during each
individual agent run.

─────────────────────────────────────────────────────────────────────
QUICK START
─────────────────────────────────────────────────────────────────────

    from event_queue import get_queue, EventLevel

    queue = get_queue()

    # 1. Open a new response group (once per user request)
    queue.start_response("orchestrator")

    # 2. Agents publish events anywhere in their code
    queue.publish("news_agent",     "Fetching articles from NewsAPI …")
    queue.publish("news_agent",     "Ingested 12 chunks into ChromaDB", level=EventLevel.SUCCESS)
    queue.publish("fundamentals",   "Retrieved P/E ratio: 28.4x")
    queue.publish("orchestrator",   "Synthesis complete", level=EventLevel.SUCCESS)

    # 3. Close the group when the request is done
    queue.end_response()

    # 4. Inspect what was recorded
    queue.print_response_log("response_1")
    all_data = queue.export()          # list of dicts, JSON-serialisable

─────────────────────────────────────────────────────────────────────
SINKS  (configure once at startup)
─────────────────────────────────────────────────────────────────────

    from event_queue import get_queue, StreamlitSink, FileSink

    queue = get_queue()
    queue.add_sink(StreamlitSink())         # streams into st.status() blocks
    queue.add_sink(FileSink("run.log"))     # appends every event to a file
    queue.add_sink(my_custom_sink)          # any callable(event_dict) works

─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

# ─────────────────────────────────────────────────────────────
# Event levels
# ─────────────────────────────────────────────────────────────


class EventLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40

    @property
    def label(self) -> str:
        return self.name

    @property
    def icon(self) -> str:
        return {
            EventLevel.DEBUG: "·",
            EventLevel.INFO: "→",
            EventLevel.SUCCESS: "✓",
            EventLevel.WARNING: "⚠",
            EventLevel.ERROR: "✗",
        }[self]

    @property
    def ansi(self) -> str:
        return {
            EventLevel.DEBUG: "\033[90m",  # grey
            EventLevel.INFO: "\033[36m",  # cyan
            EventLevel.SUCCESS: "\033[32m",  # green
            EventLevel.WARNING: "\033[33m",  # yellow
            EventLevel.ERROR: "\033[31m",  # red
        }[self]


_RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────
# Event — a single published message
# ─────────────────────────────────────────────────────────────


class Event:
    """One message published by an agent."""

    __slots__ = ("id", "response_id", "source", "message", "level", "data", "timestamp")

    def __init__(
        self,
        response_id: str,
        source: str,
        message: str,
        level: EventLevel = EventLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = str(uuid4())
        self.response_id = response_id
        self.source = source
        self.message = message
        self.level = level
        self.data = data
        self.timestamp = datetime.now(timezone.utc)

    # ── Formatting ──────────────────────────────────────────

    def to_console_line(self, *, color: bool = True) -> str:
        ts = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        parts = (
            f"[{ts}]"
            f"  [{self.response_id}]"
            f"  {self.level.icon}  {self.level.label:<7}"
            f"  [{self.source}]"
            f"  {self.message}"
        )
        if color:
            return f"{self.level.ansi}{parts}{_RESET}"
        return parts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "response_id": self.response_id,
            "source": self.source,
            "message": self.message,
            "level": self.level.label,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return f"<Event [{self.level.label}] {self.source}: {self.message!r}>"


# ─────────────────────────────────────────────────────────────
# Response group — a labelled bucket of events
# ─────────────────────────────────────────────────────────────


class ResponseGroup:
    """All events that belong to one response cycle."""

    def __init__(self, response_id: str, label: str, opened_by: str) -> None:
        self.response_id = response_id
        self.label = label  # e.g. "response_1"
        self.opened_by = opened_by
        self.opened_at = datetime.now(timezone.utc)
        self.closed_at: Optional[datetime] = None
        self.events: List[Event] = []

    def add(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed_at = datetime.now(timezone.utc)

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.closed_at is None:
            return None
        return (self.closed_at - self.opened_at).total_seconds() * 1000

    def summary_line(self) -> str:
        counts: Dict[str, int] = defaultdict(int)
        for e in self.events:
            counts[e.level.label] += 1
        count_str = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        dur = f"{self.duration_ms:.0f}ms" if self.duration_ms is not None else "open"
        return (
            f"{self.label}  |  {len(self.events)} events"
            f"  [{count_str}]  duration={dur}"
            f"  opened_by={self.opened_by}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_id": self.response_id,
            "label": self.label,
            "opened_by": self.opened_by,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "duration_ms": self.duration_ms,
            "events": [e.to_dict() for e in self.events],
        }


# ─────────────────────────────────────────────────────────────
# Sinks — where events get forwarded
# ─────────────────────────────────────────────────────────────


class ConsoleSink:
    """
    Prints every event to stdout the moment it is published.

    This is added automatically by EventQueue — you don't need to add it
    manually unless you want to customise min_level or disable colour.
    """

    def __init__(
        self,
        min_level: EventLevel = EventLevel.DEBUG,
        color: bool = True,
        show_data: bool = False,
        show_group_banners: bool = True,
    ) -> None:
        self.min_level = min_level
        self.color = color and _tty_supports_color()
        self.show_data = show_data
        self.show_group_banners = show_group_banners

    def on_event(self, event: Event) -> None:
        if event.level < self.min_level:
            return
        print(event.to_console_line(color=self.color), flush=True)
        if self.show_data and event.data:
            import json

            print(
                "  "
                + json.dumps(event.data, indent=2, default=str).replace("\n", "\n  "),
                flush=True,
            )

    def on_group_opened(self, group: ResponseGroup) -> None:
        if not self.show_group_banners:
            return
        w = 68
        line = f"  ▶  {group.label.upper()}  ·  {group.opened_by}  "
        bar = "─" * w
        text = f"\n{bar}\n{line}\n{bar}"
        print((_color(text, "\033[36m") if self.color else text), flush=True)

    def on_group_closed(self, group: ResponseGroup) -> None:
        if not self.show_group_banners:
            return
        w = 68
        line = f"  ◀  {group.summary_line()}  "
        bar = "─" * w
        text = f"{bar}\n{line}\n{bar}\n"
        print((_color(text, "\033[32m") if self.color else text), flush=True)


class FileSink:
    """
    Appends every event to a plain-text log file.

        queue.add_sink(FileSink("logs/agent_run.log"))
    """

    def __init__(
        self,
        path: str,
        min_level: EventLevel = EventLevel.DEBUG,
        json_lines: bool = False,
    ) -> None:
        self.min_level = min_level
        self.path = path
        self.json_lines = json_lines
        import os

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

    def on_event(self, event: Event) -> None:
        if event.level < self.min_level:
            return
        import json

        line = (
            json.dumps(event.to_dict(), default=str)
            if self.json_lines
            else event.to_console_line(color=False)
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_group_opened(self, group: ResponseGroup) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"\n# ── {group.label.upper()} OPENED  ({group.opened_by}) ──\n")

    def on_group_closed(self, group: ResponseGroup) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"# ── {group.summary_line()} ──\n\n")


class CallableSink:
    """
    Wrap any plain callable so it acts as a sink.

        def my_handler(event_dict: dict) -> None:
            requests.post("https://my-log-server/events", json=event_dict)

        queue.add_sink(CallableSink(my_handler))
    """

    def __init__(
        self,
        fn: Callable[[Dict[str, Any]], None],
        min_level: EventLevel = EventLevel.DEBUG,
    ) -> None:
        self.min_level = min_level
        self._fn = fn

    def on_event(self, event: Event) -> None:
        if event.level >= self.min_level:
            try:
                self._fn(event.to_dict())
            except Exception:
                pass

    def on_group_opened(self, group: ResponseGroup) -> None:
        pass

    def on_group_closed(self, group: ResponseGroup) -> None:
        pass


# ─────────────────────────────────────────────────────────────
# EventQueue — the main class
# ─────────────────────────────────────────────────────────────


class EventQueue:
    """
    Central store and dispatcher for agent events.

    One instance is shared across the whole application via get_queue().
    Thread-safe; does NOT require asyncio.

    Key concepts
    ────────────
    • call start_response(source)  before each new user request
    • agents call queue.publish(source, message) throughout their work
    • call end_response()          when the request is fully done
    • all events are stored forever (per process lifetime) for inspection
    """

    def __init__(self, auto_console: bool = True) -> None:
        self._lock = threading.Lock()
        self._sinks: list = []
        self._groups: List[ResponseGroup] = []
        self._by_label: Dict[str, ResponseGroup] = {}
        self._counter: int = 0
        self._active_group: Optional[ResponseGroup] = None

        if auto_console:
            self._console_sink = ConsoleSink()
            self._sinks.append(self._console_sink)
        else:
            self._console_sink = None

    # ── Sink management ─────────────────────────────────────

    def add_sink(self, sink: Any) -> "EventQueue":
        """Register an additional output sink.  Returns self for chaining."""
        with self._lock:
            self._sinks.append(sink)
        return self

    def remove_sink(self, sink: Any) -> None:
        with self._lock:
            self._sinks = [s for s in self._sinks if s is not sink]

    def configure_console(
        self,
        *,
        min_level: EventLevel = EventLevel.DEBUG,
        color: bool = True,
        show_data: bool = False,
        show_group_banners: bool = True,
    ) -> "EventQueue":
        """Reconfigure the built-in console sink."""
        if self._console_sink:
            self._console_sink.min_level = min_level
            self._console_sink.color = color and _tty_supports_color()
            self._console_sink.show_data = show_data
            self._console_sink.show_group_banners = show_group_banners
        return self

    # ── Group management ────────────────────────────────────

    def start_response(self, opened_by: str = "system") -> str:
        """
        Open a new response group.  Call this once at the start of each
        user request, before any agent starts publishing.

        Returns the response label (e.g. "response_3") for reference.
        """
        with self._lock:
            self._counter += 1
            label = f"response_{self._counter}"
            group = ResponseGroup(
                response_id=str(uuid4()),
                label=label,
                opened_by=opened_by,
            )
            self._groups.append(group)
            self._by_label[label] = group
            self._active_group = group

        self._fire("on_group_opened", group)
        return label

    def end_response(self) -> None:
        """
        Seal the current response group.  Call this once the full agent
        pipeline has finished for this user request.
        """
        with self._lock:
            group = self._active_group
            if group is None:
                return
            group.close()
            self._active_group = None

        self._fire("on_group_closed", group)

    # ── Publishing ──────────────────────────────────────────

    def publish(
        self,
        source: str,
        message: str,
        level: EventLevel = EventLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
        response_label: Optional[str] = None,
    ) -> Event:
        """
        Publish one event.

        Parameters
        ──────────
        source          Name of the component emitting this event,
                        e.g. "news_agent", "orchestrator.planner".
        message         Human-readable description of what just happened.
        level           Severity — DEBUG / INFO / SUCCESS / WARNING / ERROR.
        data            Optional structured payload (must be JSON-serialisable).
        response_label  Pin to a specific group by label (e.g. "response_2").
                        Defaults to the currently active group.
        """
        with self._lock:
            if response_label:
                group = self._by_label.get(response_label)
            else:
                group = self._active_group

            if group is None:
                # Auto-open an ungrouped bucket so nothing is ever lost
                self._counter += 1
                label = f"response_{self._counter}"
                group = ResponseGroup(
                    response_id=str(uuid4()),
                    label=label,
                    opened_by="auto",
                )
                self._groups.append(group)
                self._by_label[label] = group
                self._active_group = group

            event = Event(
                response_id=group.response_id,
                source=source,
                message=message,
                level=level,
                data=data,
            )
            group.add(event)

        # Fire sinks outside the lock so they can call publish() themselves
        self._fire("on_event", event)
        return event

    # ── Convenience publish methods ─────────────────────────

    def debug(self, source: str, message: str, **kwargs) -> Event:
        return self.publish(source, message, EventLevel.DEBUG, **kwargs)

    def info(self, source: str, message: str, **kwargs) -> Event:
        return self.publish(source, message, EventLevel.INFO, **kwargs)

    def success(self, source: str, message: str, **kwargs) -> Event:
        return self.publish(source, message, EventLevel.SUCCESS, **kwargs)

    def warning(self, source: str, message: str, **kwargs) -> Event:
        return self.publish(source, message, EventLevel.WARNING, **kwargs)

    def error(self, source: str, message: str, **kwargs) -> Event:
        return self.publish(source, message, EventLevel.ERROR, **kwargs)

    # ── Inspection ──────────────────────────────────────────

    def get_response(self, label: str) -> Optional[ResponseGroup]:
        """Return a response group by its label, e.g. 'response_2'."""
        return self._by_label.get(label)

    def all_responses(self) -> List[ResponseGroup]:
        """Return every response group recorded so far."""
        return list(self._groups)

    def latest_response(self) -> Optional[ResponseGroup]:
        """Return the most recently opened response group."""
        return self._groups[-1] if self._groups else None

    def print_response_log(self, label: str, *, color: bool = True) -> None:
        """Pretty-print every event in a given response group to stdout."""
        group = self._by_label.get(label)
        if group is None:
            print(f"[EventQueue] No response found with label '{label}'")
            return

        w = 68
        bar = "─" * w
        header = f"\n{bar}\n  {group.summary_line()}\n{bar}"
        print(_color(header, "\033[36m") if color else header)

        for event in group.events:
            print(event.to_console_line(color=color))

        footer = bar
        print(_color(footer, "\033[36m") if color else footer, "\n")

    def print_all_logs(self, *, color: bool = True) -> None:
        """Print every response group and all of its events."""
        for group in self._groups:
            self.print_response_log(group.label, color=color)

    def export(self) -> List[Dict[str, Any]]:
        """Return all groups and events as a JSON-serialisable list of dicts."""
        return [g.to_dict() for g in self._groups]

    def export_response(self, label: str) -> Optional[Dict[str, Any]]:
        """Return one response group as a JSON-serialisable dict."""
        group = self._by_label.get(label)
        return group.to_dict() if group else None

    def clear(self) -> None:
        """Wipe all stored groups and reset the counter (useful between tests)."""
        with self._lock:
            self._groups.clear()
            self._by_label.clear()
            self._counter = 0
            self._active_group = None

    # ── Internal ────────────────────────────────────────────

    def _fire(self, method: str, arg: Any) -> None:
        for sink in list(self._sinks):
            fn = getattr(sink, method, None)
            if fn:
                try:
                    fn(arg)
                except Exception as exc:
                    # A broken sink must never crash the agent
                    print(
                        f"[EventQueue] Sink {sink.__class__.__name__}.{method} raised: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )


# ─────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────

_QUEUE: Optional[EventQueue] = None
_QUEUE_LOCK = threading.Lock()


def get_queue() -> EventQueue:
    """Return the process-wide EventQueue singleton."""
    global _QUEUE
    if _QUEUE is None:
        with _QUEUE_LOCK:
            if _QUEUE is None:
                _QUEUE = EventQueue(auto_console=True)
    return _QUEUE


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _tty_supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _color(text: str, ansi_code: str) -> str:
    return f"{ansi_code}{text}{_RESET}"


# ─────────────────────────────────────────────────────────────
# Agent integration helper (optional mixin)
# ─────────────────────────────────────────────────────────────


class AgentPublisher:
    """
    Drop-in helper that gives any agent a clean publish() interface
    without importing EventQueue directly.

    Usage
    ─────
        class NewsAnalysisAgent:
            def __init__(self):
                self._pub = AgentPublisher("news_agent")

            async def _fetch_news_node(self, state):
                self._pub.info("Fetching articles …")
                ...
                self._pub.success(f"Got {len(articles)} articles")
    """

    def __init__(self, source: str) -> None:
        self.source = source

    def _q(self) -> EventQueue:
        return get_queue()

    def debug(self, message: str, data: Optional[Dict] = None) -> None:
        self._q().debug(self.source, message, data=data)

    def info(self, message: str, data: Optional[Dict] = None) -> None:
        self._q().info(self.source, message, data=data)

    def success(self, message: str, data: Optional[Dict] = None) -> None:
        self._q().success(self.source, message, data=data)

    def warning(self, message: str, data: Optional[Dict] = None) -> None:
        self._q().warning(self.source, message, data=data)

    def error(self, message: str, data: Optional[Dict] = None) -> None:
        self._q().error(self.source, message, data=data)


from contextvars import ContextVar as _ContextVar
from typing import Optional as _Opt

_current_response_label: _ContextVar[_Opt[str]] = _ContextVar(
    "alphaMesh_response_label", default=None
)
"""
Holds the EventQueue response-group label for the current async task tree.
 
Set by AnalysisRunner before calling OrchestratorAgent.run() and propagated
automatically to sub-tasks (asyncio copies ContextVar context on task creation).
Agents read this via the publish helpers below — they never set it themselves.
"""


def _agent_publish(
    source: str, message: str, level: "EventLevel"
) -> None:  # noqa: F821
    """Internal: publish one event using the active response label (if any)."""
    label = _current_response_label.get()
    get_queue().publish(source, message, level=level, response_label=label)


def publish_progress(source: str, message: str) -> None:
    """Publish an INFO-level progress event from an agent node."""
    _agent_publish(source, message, EventLevel.INFO)  # noqa: F821


def publish_success(source: str, message: str) -> None:
    """Publish a SUCCESS-level event from an agent node."""
    _agent_publish(source, message, EventLevel.SUCCESS)  # noqa: F821


def publish_warning(source: str, message: str) -> None:
    """Publish a WARNING-level event from an agent node."""
    _agent_publish(source, message, EventLevel.WARNING)  # noqa: F821


def publish_error_event(source: str, message: str) -> None:
    """Publish an ERROR-level event from an agent node."""
    _agent_publish(source, message, EventLevel.ERROR)  # noqa: F821


# ─────────────────────────────────────────────────────────────
# Self-contained demo  (python event_queue.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    q = get_queue()
    q.configure_console(color=True, show_group_banners=True)

    # ── Simulate two separate user requests ──────────────────

    print("\n=== Simulating request 1 ===")
    q.start_response("orchestrator")

    q.info("orchestrator", "Received query: 'What is AAPL doing?'")
    q.info("orchestrator", "Routing to: news_agent, fundamentals_agent")
    time.sleep(0.05)

    q.info("news_agent", "Fetching from NewsAPI — ticker=AAPL")
    q.debug("news_agent", "Date range: 2025-03-07 → 2025-03-14")
    q.success("news_agent", "Fetched 34 articles")
    time.sleep(0.05)

    q.info("news_agent", "Ingesting into ChromaDB …")
    q.success("news_agent", "Ingested 89 chunks")
    time.sleep(0.05)

    q.info("fundamentals", "Pulling financial data for AAPL")
    q.success(
        "fundamentals",
        "P/E=28.4  Revenue=$124B  YoY=+6.1%",
        data={"pe": 28.4, "revenue": "124B"},
    )
    time.sleep(0.05)

    q.info("retriever", "Running dual-store retrieval …")
    q.success("retriever", "Retrieved 12 chunks (vector=8, graph=4)")
    q.info("orchestrator", "Synthesising final answer …")
    q.success("orchestrator", "Response ready")

    q.end_response()

    # ── Simulate a second request ─────────────────────────────

    print("\n=== Simulating request 2 ===")
    q.start_response("orchestrator")
    q.info("orchestrator", "Received query: 'Compare MSFT vs GOOGL'")
    q.warning("news_agent", "NewsAPI rate limit close: 8 requests remaining")
    q.info("news_agent", "Fetched 21 articles for MSFT")
    q.info("news_agent", "Fetched 18 articles for GOOGL")
    q.error("fundamentals", "Timeout fetching GOOGL fundamentals — retrying")
    q.success("fundamentals", "GOOGL data retrieved on retry")
    q.success("orchestrator", "Comparison response ready")
    q.end_response()

    # ── Replay a specific group ───────────────────────────────
    print("\n=== Replaying response_1 ===")
    q.print_response_log("response_1")

    # ── Export to dict ────────────────────────────────────────
    import json

    data = q.export()
    print(
        f"\n=== Exported {len(data)} response group(s), "
        f"{sum(len(g['events']) for g in data)} total events ==="
    )
