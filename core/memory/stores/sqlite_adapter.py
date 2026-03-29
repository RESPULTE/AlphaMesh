"""
core/memory/stores/sqlite_adapter.py

SQLite implementations for conversation and session persistence.
"""

from __future__ import annotations

import time
from typing import List, Optional

import aiosqlite

from core.memory.stores.base import (
    ConversationPersistenceAdapter,
    SessionPersistenceAdapter,
)

_DEFAULT_DB_PATH = "./data/conversations.db"
_DEFAULT_SESSIONS_DB_PATH = "./data/sessions.db"


class SQLiteConversationAdapter(ConversationPersistenceAdapter):
    """
    Stores conversations and messages in a local SQLite file.

    Schema
    ------
    conversations (conversation_id PK, user_email, created_at, last_message_at)
    messages      (id AUTOINCREMENT, conversation_id FK, role, content, created_at)
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_email      TEXT,
                    created_at      REAL NOT NULL,
                    last_message_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT    NOT NULL,
                    role            TEXT    NOT NULL,
                    content         TEXT    NOT NULL,
                    created_at      REAL    NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations (conversation_id)
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_msg_conv "
                "ON messages (conversation_id, created_at)"
            )
            await db.commit()
        self._initialized = True

    async def ensure_conversation(
        self,
        conversation_id: str,
        user_email: Optional[str],
    ) -> None:
        now = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO conversations (conversation_id, user_email, created_at, last_message_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (conversation_id) DO UPDATE
                    SET last_message_at = excluded.last_message_at
                """,
                (conversation_id, user_email, now, now),
            )
            await db.commit()

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: str,
    ) -> None:
        now = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, now),
            )
            await db.execute(
                "UPDATE conversations SET last_message_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            await db.commit()

    async def load_messages(self, conversation_id: str) -> List[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": str(row["created_at"]),
            }
            for row in rows
        ]

    async def list_conversations(
        self,
        user_email: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        if user_email:
            query = (
                "SELECT c.conversation_id, c.created_at, c.last_message_at, "
                "COUNT(m.id) AS message_count "
                "FROM conversations c "
                "LEFT JOIN messages m ON c.conversation_id = m.conversation_id "
                "WHERE c.user_email = ? "
                "GROUP BY c.conversation_id "
                "ORDER BY c.last_message_at DESC "
                "LIMIT ?"
            )
            params = (user_email, limit)
        else:
            query = (
                "SELECT c.conversation_id, c.created_at, c.last_message_at, "
                "COUNT(m.id) AS message_count "
                "FROM conversations c "
                "LEFT JOIN messages m ON c.conversation_id = m.conversation_id "
                "GROUP BY c.conversation_id "
                "ORDER BY c.last_message_at DESC "
                "LIMIT ?"
            )
            params = (limit,)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
        return [
            {
                "conversation_id": row["conversation_id"],
                "created_at": str(row["created_at"]),
                "last_message_at": str(row["last_message_at"]),
                "message_count": row["message_count"],
            }
            for row in rows
        ]


class SQLiteSessionAdapter(SessionPersistenceAdapter):
    """
    Stores analysis session metadata in a local SQLite file.

    Schema
    ------
    sessions (session_id PK, conversation_id, user_email, ticker, query, summary, created_at)
    """

    def __init__(self, db_path: str = _DEFAULT_SESSIONS_DB_PATH) -> None:
        self._db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as db:
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

    async def save_session(
        self,
        session_id: str,
        conversation_id: str,
        user_email: str,
        ticker: Optional[str],
        query: str,
        summary: str,
        created_at: str,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
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
                    ticker,
                    query,
                    summary,
                    created_at,
                ),
            )
            await db.commit()

    async def list_sessions(
        self,
        user_email: str,
        limit: int = 20,
    ) -> List[dict]:
        async with aiosqlite.connect(self._db_path) as db:
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

    async def list_sessions_by_ticker(
        self,
        user_email: str,
        ticker: str,
        limit: int = 10,
    ) -> List[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT session_id, conversation_id, query, summary, created_at
                FROM sessions
                WHERE user_email = ? AND ticker = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_email, ticker, limit),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(row) for row in rows]
