"""
api/main.py

FastAPI application factory for AlphaMesh.

Startup sequence (lifespan)
────────────────────────────
1. Initialize backend services (Neo4j, ChromaDB, graph queue) via service_manager.startup().
2. Initialize the SQLite persistence adapter for conversations.
3. Wire services into app.state so Depends() providers can inject them.
4. Register all routers under /api/v1.

Shutdown sequence
─────────────────
1. Drain the graph write queue gracefully.
2. Any other cleanup delegated to service_manager.shutdown().

Run
───
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Environment
───────────
All configuration is read from the project's .env file via core.config.Settings.
No additional environment variables are needed for the API layer.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.persistence.sqlite_adapter import SQLiteConversationAdapter
from api.routers import chat, conversations, health, stream
from api.services.analysis_runner import AnalysisRunner
from api.services.conversation_store import ConversationStore
from api.services.event_broadcaster import EventBroadcaster

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of all service singletons."""
    logger.info("AlphaMesh API: starting up…")

    # ── Backend services (Neo4j, ChromaDB, graph queue, etc.) ─────────────────
    from core.services import service_manager

    await service_manager.startup()

    # In lifespan, after service_manager.startup():
    from api.services.session_service import SessionService

    _session_svc = SessionService()
    await _session_svc.initialize()

    # Replace CORS allowed_origins:
    allow_origins = (settings.allowed_origins_list,)

    # Add new routers:
    from api.middleware.error_handling import register_exception_handlers
    from api.middleware.rate_limiting import RateLimitMiddleware
    from api.routers import analyze as analyze_router
    from api.routers import market as market_router
    from api.routers import sessions as sessions_router

    app.include_router(analyze_router.router, prefix="/api")
    app.include_router(market_router.router, prefix="/api/market")
    app.include_router(sessions_router.router, prefix="/api/sessions")
    app.add_middleware(RateLimitMiddleware)
    register_exception_handlers(app)

    # ── Conversation persistence ───────────────────────────────────────────────
    from core.config import settings

    adapter = SQLiteConversationAdapter(db_path="./data/conversations.db")
    await adapter.initialize()

    # ── API-layer singletons ───────────────────────────────────────────────────
    broadcaster = EventBroadcaster()
    store = ConversationStore(adapter=adapter)
    await store.initialize()
    runner = AnalysisRunner(broadcaster=broadcaster, store=store)

    app.state.broadcaster = broadcaster
    app.state.store = store
    app.state.runner = runner

    logger.info("AlphaMesh API: ready.")
    yield

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    logger.info("AlphaMesh API: shutting down…")
    await service_manager.shutdown()
    logger.info("AlphaMesh API: shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AlphaMesh API",
        description=(
            "Financial research assistant API. "
            "POST /api/v1/chat to start an analysis turn, "
            "then stream results from GET /api/v1/stream/{request_id}."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Restrict `allow_origins` to your frontend domain in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(stream.router)
    app.include_router(conversations.router)

    return app


app = create_app()
