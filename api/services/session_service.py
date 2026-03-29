"""
api/services/session_service.py

Stores per-user analysis sessions in a dedicated SQLite database so the
History tab can display real past sessions.

Lifecycle
─────────
`SessionService` is a singleton whose `initialize()` method must be called
during FastAPI lifespan startup (api/main.py) before the first request
arrives.  This ensures the table and index exist without per-request overhead.

The `_initialized` flag is guarded by an asyncio.Lock to prevent duplicate
CREATE TABLE calls under concurrent startup probing.

Separation from other stores
─────────────────────────────
• `data/financial_data.db` — EDGAR financial statement rows (FinancialDatabase)
• `data/conversations.db`  — full message history (ConversationStore)
• `data/graph_tasks.db`    — graph write queue (GraphQueueManager)
• `data/sessions.db`       — THIS FILE: per-user analysis session metadata
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

import aiosqlite

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class SessionService:
    """
    Stores and retrieves user analysis session records.

    Each record captures: who ran it (user_email), what they asked (query),
    which ticker was targeted, a short summary, and the conversation_id that
    links back to the full message history in ConversationStore.
    """

    def __init__(self) -> None:
        self._db = settings.SESSIONS_DB_PATH
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Create the sessions table and index if they do not exist.

        Called once by api/main.py lifespan — safe to call multiple times
        (idempotent due to CREATE TABLE IF NOT EXISTS).
        """
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self._db) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id       TEXT PRIMARY KEY,
                        conversation_id  TEXT NOT NULL,
                        user_email       TEXT NOT NULL,
                        ticker           TEXT,
                        query            TEXT NOT NULL,
                        summary          TEXT,
                        created_at       TEXT NOT NULL
                    )
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sessions_user_created
                    ON sessions (user_email, created_at DESC)
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sessions_ticker
                    ON sessions (user_email, ticker, created_at DESC)
                    """
                )
                await db.commit()
            self._initialized = True
            logger.info("SessionService: initialised at '%s'", self._db)

    async def save_analysis(
        self,
        *,
        user_email: str,
        conversation_id: str,
        query: str,
        ticker: Optional[str],
        summary_text: str,
    ) -> str:
        """
        Persist one analysis session record.

        Returns the generated session_id for tracing.
        Summary is truncated to 500 chars to keep the row lightweight
        (full content lives in ConversationStore).
        """
        session_id = str(uuid4())
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                """
                INSERT INTO sessions
                    (session_id, conversation_id, user_email, ticker, query, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    conversation_id,
                    user_email,
                    ticker.upper() if ticker else None,
                    query,
                    (summary_text or "")[:500],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
        logger.debug(
            "SessionService: saved session '%s' for user '%s'", session_id, user_email
        )
        return session_id

    async def get_sessions(
        self,
        user_email: str,
        limit: int = 20,
    ) -> List[dict]:
        """Return the most recent `limit` sessions for a user, newest first."""
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT session_id, conversation_id, ticker, query, summary, created_at
                FROM sessions
                WHERE user_email = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_email, limit),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def get_sessions_by_ticker(
        self,
        user_email: str,
        ticker: str,
        limit: int = 10,
    ) -> List[dict]:
        """Return sessions for a specific ticker, newest first."""
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT session_id, conversation_id, query, summary, created_at
                FROM sessions
                WHERE user_email = ? AND ticker = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_email, ticker.upper(), limit),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(row) for row in rows]
