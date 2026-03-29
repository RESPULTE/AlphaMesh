"""
api/main.py

FastAPI application factory for AlphaMesh.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, conversations, health, stream
from api.services.analysis_runner import AnalysisRunner
from api.services.event_broadcaster import EventBroadcaster

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of all service singletons."""
    logger.info("AlphaMesh API: starting up...")

    # Backend services (Neo4j, ChromaDB, graph queue, etc.)
    from core.services import service_manager

    await service_manager.startup()

    # API-layer singletons
    broadcaster = EventBroadcaster()
    store = service_manager.get_conversation_store()
    runner = AnalysisRunner(broadcaster=broadcaster, store=store)
    session_service = service_manager.get_session_service()

    app.state.broadcaster = broadcaster
    app.state.store = store
    app.state.runner = runner
    app.state.session_service = session_service

    logger.info("AlphaMesh API: ready.")
    yield

    # Graceful shutdown
    logger.info("AlphaMesh API: shutting down...")
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

    # Middleware must be registered before startup.
    from api.middleware.rate_limiting import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware)

    # Restrict `allow_origins` to your frontend domain in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers and exception handlers must be registered before startup.
    from api.middleware.error_handling import register_exception_handlers
    from api.routers import analyze as analyze_router
    from api.routers import market as market_router
    from api.routers import sessions as sessions_router

    app.include_router(analyze_router.router, prefix="/api")
    app.include_router(market_router.router, prefix="/api/market")
    app.include_router(sessions_router.router, prefix="/api/sessions")

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(stream.router)
    app.include_router(conversations.router)

    register_exception_handlers(app)

    return app


app = create_app()
