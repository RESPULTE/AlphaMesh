"""
api/dependencies.py

FastAPI Depends() providers for shared service singletons.

All services are created once during application startup (via the lifespan
context manager in main.py) and stored on `app.state`.  Routers retrieve
them through these dependency functions rather than importing globals directly,
which keeps each router independently testable.
"""

from __future__ import annotations

from fastapi import Request

from api.services.analysis_runner import AnalysisRunner
from api.services.conversation_store import ConversationStore
from api.services.event_broadcaster import EventBroadcaster


def get_broadcaster(request: Request) -> EventBroadcaster:
    return request.app.state.broadcaster


def get_store(request: Request) -> ConversationStore:
    return request.app.state.store


def get_runner(request: Request) -> AnalysisRunner:
    return request.app.state.runner
