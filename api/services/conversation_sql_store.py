"""SQLite conversation persistence store."""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional

from core.memory.stores.sqlite_adapter import SQLiteAdapter


class SQLiteConversationStore(SQLiteAdapter):
    """Stores conversations and messages in SQLite."""

    def __init__(self, db_path: str) -> None:
        super().__init__(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_email      TEXT,
                created_at      REAL NOT NULL,
                last_message_at REAL NOT NULL
            )
            """
        )
        await self.execute(
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
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages (conversation_id, created_at)"
        )
        self._initialized = True

    async def ensure_conversation(
        self,
        conversation_id: str,
        user_email: Optional[str],
    ) -> None:
        now = time.time()
        await self.execute(
            """
            INSERT INTO conversations (conversation_id, user_email, created_at, last_message_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (conversation_id) DO UPDATE
                SET last_message_at = excluded.last_message_at
            """,
            (conversation_id, user_email, now, now),
        )

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
        await self.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, created_at),
        )
        await self.execute(
            "UPDATE conversations SET last_message_at = ? WHERE conversation_id = ?",
            (created_at, conversation_id),
        )

    async def load_messages(self, conversation_id: str) -> List[dict]:
        rows = await self.fetchall(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
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
            rows = await self.fetchall(
                "SELECT c.conversation_id, c.created_at, c.last_message_at, "
                "COUNT(m.id) AS message_count "
                "FROM conversations c "
                "LEFT JOIN messages m ON c.conversation_id = m.conversation_id "
                "WHERE c.user_email = ? "
                "GROUP BY c.conversation_id "
                "ORDER BY c.last_message_at DESC "
                "LIMIT ?",
                (user_email, limit),
            )
        else:
            rows = await self.fetchall(
                "SELECT c.conversation_id, c.created_at, c.last_message_at, "
                "COUNT(m.id) AS message_count "
                "FROM conversations c "
                "LEFT JOIN messages m ON c.conversation_id = m.conversation_id "
                "GROUP BY c.conversation_id "
                "ORDER BY c.last_message_at DESC "
                "LIMIT ?",
                (limit,),
            )
        return [
            {
                "conversation_id": row["conversation_id"],
                "created_at": str(row["created_at"]),
                "last_message_at": str(row["last_message_at"]),
                "message_count": row["message_count"],
            }
            for row in rows
        ]
