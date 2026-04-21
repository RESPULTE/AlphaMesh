"""
core/memory/stores/sqlite_adapter.py

SQLite implementations for conversation and session persistence.
"""

from __future__ import annotations

import time
from datetime import datetime
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
        created_at = time.time()
        if timestamp:
            try:
                created_at = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                pass
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, created_at),
            )
            await db.execute(
                "UPDATE conversations SET last_message_at = ? WHERE conversation_id = ?",
                (created_at, conversation_id),
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
    Stores login session metadata and session-conversation links in SQLite.

    Schema
    ------
    login_sessions        (session_id PK, user_id, created_at, last_seen_at, ended_at, status)
    session_conversations (session_id, conversation_id, user_id, linked_at)

    Legacy compatibility table:
    sessions (session_id, conversation_id, user_email, ticker, query, summary, created_at)
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
                CREATE TABLE IF NOT EXISTS login_sessions (
                    session_id    TEXT PRIMARY KEY,
                    user_id       TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    last_seen_at  TEXT NOT NULL,
                    ended_at      TEXT,
                    status        TEXT NOT NULL DEFAULT 'active'
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS session_conversations (
                    session_id      TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_id         TEXT NOT NULL,
                    linked_at       TEXT NOT NULL,
                    PRIMARY KEY (session_id, conversation_id)
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_login_sessions_user_created
                ON login_sessions (user_id, created_at DESC)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sc_user_conversation
                ON session_conversations (user_id, conversation_id)
                """
            )
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

    async def create_login_session(
        self,
        session_id: str,
        user_id: str,
        created_at: str,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO login_sessions (session_id, user_id, created_at, last_seen_at, status)
                VALUES (?, ?, ?, ?, 'active')
                """,
                (session_id, user_id, created_at, created_at),
            )
            await db.commit()

    async def touch_login_session(
        self,
        session_id: str,
        user_id: str,
        last_seen_at: str,
    ) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                UPDATE login_sessions
                SET last_seen_at = ?, status = 'active'
                WHERE session_id = ? AND user_id = ?
                """,
                (last_seen_at, session_id, user_id),
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def end_login_session(
        self,
        session_id: str,
        user_id: str,
        ended_at: str,
    ) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                UPDATE login_sessions
                SET ended_at = ?, status = 'ended'
                WHERE session_id = ? AND user_id = ?
                """,
                (ended_at, session_id, user_id),
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def link_session_conversation(
        self,
        session_id: str,
        user_id: str,
        conversation_id: str,
        linked_at: str,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO session_conversations (session_id, conversation_id, user_id, linked_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (session_id, conversation_id) DO UPDATE
                SET linked_at = excluded.linked_at
                """,
                (session_id, conversation_id, user_id, linked_at),
            )
            await db.commit()

    async def user_has_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT 1
                FROM session_conversations
                WHERE user_id = ? AND conversation_id = ?
                LIMIT 1
                """,
                (user_id, conversation_id),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                async with db.execute(
                    """
                    SELECT 1
                    FROM conversations
                    WHERE user_email = ? AND conversation_id = ?
                    LIMIT 1
                    """,
                    (user_id, conversation_id),
                ) as cur:
                    row = await cur.fetchone()
        return row is not None

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
        """
        Legacy compatibility write path.
        """
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
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT session_id, user_id, created_at, last_seen_at, ended_at, status
                FROM login_sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def get_latest_active_session(
        self,
        user_id: str,
    ) -> Optional[str]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT session_id
                FROM login_sessions
                WHERE user_id = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id,),
            ) as cur:
                row = await cur.fetchone()
        return row["session_id"] if row else None
