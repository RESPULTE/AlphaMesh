"""SQLite-backed persistence for graph queue tasks and prompt registry."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from core.memory.stores.contracts.graph_tasks import GraphTaskPersistenceAdapter
from core.memory.stores.sqlite_adapter import SQLiteAdapter


class GraphTaskSqlStore(GraphTaskPersistenceAdapter):
    """Durable storage for queued graph tasks."""

    def __init__(self, db_path: str) -> None:
        self._sql = SQLiteAdapter(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._sql.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_tasks (
                task_id         TEXT PRIMARY KEY,
                turn_id         TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                source_agent    TEXT NOT NULL,
                relationships   TEXT NOT NULL,
                extraction_text TEXT,
                system_prompt_id TEXT,
                llm_config      TEXT,
                task_kind       TEXT,
                chunk_ids       TEXT,
                status          TEXT NOT NULL DEFAULT 'PENDING',
                created_at      REAL NOT NULL,
                processed_at    REAL,
                error_message   TEXT
            )
            """
        )
        await self._sql.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_prompt_registry (
                prompt_id   TEXT PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                created_at  REAL NOT NULL
            )
            """
        )
        await self._sql.execute(
            "CREATE INDEX IF NOT EXISTS idx_gt_status_created ON graph_tasks(status, created_at)"
        )
        await self._sql.execute(
            "CREATE INDEX IF NOT EXISTS idx_gt_conversation ON graph_tasks(conversation_id, turn_id)"
        )
        self._initialized = True

    async def persist_task(self, task_payload: Dict[str, Any]) -> None:
        await self._sql.execute(
            """INSERT OR IGNORE INTO graph_tasks
               (task_id, turn_id, conversation_id, source_agent, relationships, extraction_text, system_prompt_id, llm_config, task_kind, chunk_ids, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)""",
            (
                task_payload["task_id"],
                task_payload["turn_id"],
                task_payload["conversation_id"],
                task_payload["source_agent"],
                json.dumps(task_payload.get("relationships") or []),
                task_payload.get("extraction_text"),
                task_payload.get("system_prompt_id"),
                (
                    json.dumps(task_payload["llm_config"])
                    if task_payload.get("llm_config") is not None
                    else None
                ),
                task_payload.get("task_kind"),
                (
                    json.dumps(task_payload["chunk_ids"])
                    if task_payload.get("chunk_ids")
                    else None
                ),
                float(task_payload.get("created_at", time.time())),
            ),
        )

    async def mark_processed(self, task_ids: List[str]) -> None:
        if not task_ids:
            return
        now = time.time()
        await self._sql.executemany(
            "UPDATE graph_tasks SET status='PROCESSED', processed_at=? WHERE task_id=?",
            [(now, task_id) for task_id in task_ids],
        )

    async def load_pending_tasks(self) -> List[Dict[str, Any]]:
        rows = await self._sql.fetchall(
            "SELECT task_id, turn_id, conversation_id, source_agent, relationships, extraction_text, system_prompt_id, llm_config, task_kind, chunk_ids, created_at "
            "FROM graph_tasks WHERE status='PENDING' ORDER BY created_at ASC"
        )
        tasks: List[Dict[str, Any]] = []
        for row in rows:
            tasks.append(
                {
                    "task_id": row["task_id"],
                    "turn_id": row["turn_id"],
                    "conversation_id": row["conversation_id"],
                    "source_agent": row["source_agent"],
                    "relationships": json.loads(row["relationships"] or "[]"),
                    "extraction_text": row["extraction_text"],
                    "system_prompt_id": row["system_prompt_id"],
                    "llm_config": json.loads(row["llm_config"])
                    if row["llm_config"]
                    else None,
                    "task_kind": row["task_kind"],
                    "chunk_ids": json.loads(row["chunk_ids"])
                    if row["chunk_ids"]
                    else None,
                    "created_at": row["created_at"],
                }
            )
        return tasks

    async def save_prompt(self, prompt_id: str, prompt_text: str) -> None:
        await self._sql.execute(
            "INSERT OR IGNORE INTO graph_prompt_registry (prompt_id, prompt_text, created_at) VALUES (?, ?, ?)",
            (prompt_id, prompt_text, time.time()),
        )

    async def load_prompts(self) -> Dict[str, str]:
        rows = await self._sql.fetchall(
            "SELECT prompt_id, prompt_text FROM graph_prompt_registry"
        )
        return {str(row["prompt_id"]): str(row["prompt_text"]) for row in rows}

    async def purge_processed_older_than(self, cutoff_epoch: float) -> None:
        await self._sql.execute(
            "DELETE FROM graph_tasks WHERE status='PROCESSED' AND processed_at < ?",
            (cutoff_epoch,),
        )

