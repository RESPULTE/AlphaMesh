"""SQLite session persistence store."""

from __future__ import annotations

from typing import List, Optional

from core.memory.stores.contracts.session import SessionPersistenceAdapter
from core.memory.stores.sqlite_adapter import SQLiteAdapter


class SQLiteSessionStore(SessionPersistenceAdapter):
    """Stores login sessions and session-conversation links in SQLite."""

    def __init__(self, db_path: str) -> None:
        self._sql = SQLiteAdapter(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._sql.execute(
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
        await self._sql.execute(
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
        await self._sql.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_login_sessions_user_created
            ON login_sessions (user_id, created_at DESC)
            """
        )
        await self._sql.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sc_user_conversation
            ON session_conversations (user_id, conversation_id)
            """
        )
        self._initialized = True

    async def create_login_session(
        self,
        session_id: str,
        user_id: str,
        created_at: str,
    ) -> None:
        await self._sql.execute(
            """
            INSERT INTO login_sessions (session_id, user_id, created_at, last_seen_at, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            (session_id, user_id, created_at, created_at),
        )

    async def touch_login_session(
        self,
        session_id: str,
        user_id: str,
        last_seen_at: str,
    ) -> bool:
        rows = await self._sql.execute_with_rowcount(
            """
            UPDATE login_sessions
            SET last_seen_at = ?, status = 'active'
            WHERE session_id = ? AND user_id = ?
            """,
            (last_seen_at, session_id, user_id),
        )
        return rows > 0

    async def end_login_session(
        self,
        session_id: str,
        user_id: str,
        ended_at: str,
    ) -> bool:
        rows = await self._sql.execute_with_rowcount(
            """
            UPDATE login_sessions
            SET ended_at = ?, status = 'ended'
            WHERE session_id = ? AND user_id = ?
            """,
            (ended_at, session_id, user_id),
        )
        return rows > 0

    async def link_session_conversation(
        self,
        session_id: str,
        user_id: str,
        conversation_id: str,
        linked_at: str,
    ) -> None:
        await self._sql.execute(
            """
            INSERT INTO session_conversations (session_id, conversation_id, user_id, linked_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (session_id, conversation_id) DO UPDATE
            SET linked_at = excluded.linked_at
            """,
            (session_id, conversation_id, user_id, linked_at),
        )

    async def user_has_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> bool:
        row = await self._sql.fetchone(
            """
            SELECT 1
            FROM session_conversations
            WHERE user_id = ? AND conversation_id = ?
            LIMIT 1
            """,
            (user_id, conversation_id),
        )
        return row is not None

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        return await self._sql.fetchall(
            """
            SELECT session_id, user_id, created_at, last_seen_at, ended_at, status
            FROM login_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )

    async def get_latest_active_session(
        self,
        user_id: str,
    ) -> Optional[str]:
        row = await self._sql.fetchone(
            """
            SELECT session_id
            FROM login_sessions
            WHERE user_id = ? AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        return str(row["session_id"]) if row else None

