"""Generic async SQLite adapter."""

from __future__ import annotations

from typing import Iterable, Sequence

import aiosqlite


class SQLiteAdapter:
    """Thin helper around aiosqlite for generic read/write operations."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def execute(self, sql: str, params: Sequence | None = None) -> None:
        """Execute one statement and commit."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(sql, tuple(params or ()))
            await db.commit()

    async def executemany(
        self, sql: str, seq_of_params: Iterable[Sequence | None]
    ) -> None:
        """Execute many statements and commit."""
        normalized = [tuple(params or ()) for params in seq_of_params]
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(sql, normalized)
            await db.commit()

    async def fetchall(self, sql: str, params: Sequence | None = None) -> list[dict]:
        """Return all rows as dictionaries."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, tuple(params or ())) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetchone(self, sql: str, params: Sequence | None = None) -> dict | None:
        """Return one row as a dictionary or None."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, tuple(params or ())) as cursor:
                row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def execute_with_rowcount(
        self, sql: str, params: Sequence | None = None
    ) -> int:
        """Execute one statement, commit, and return affected row count."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(sql, tuple(params or ()))
            await db.commit()
            return int(cursor.rowcount or 0)
