"""SQLite-backed persistence for graph queue tasks and prompt registry."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from core.memory.stores.sqlite_adapter import SQLiteAdapter


class GraphTaskSqlStore(SQLiteAdapter):
    """Durable storage for queued graph tasks."""

    def __init__(self, db_path: str) -> None:
        super().__init__(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_tasks (
                task_id         TEXT PRIMARY KEY,
                turn_id         TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                source_agent    TEXT NOT NULL,
                relationships   TEXT NOT NULL,
                extraction_text TEXT,
                system_prompt_id TEXT,
                chunk_system_prompt_id TEXT,
                allowed_entity_types TEXT,
                allowed_relationship_types TEXT,
                llm_config      TEXT,
                task_kind       TEXT,
                chunk_ids       TEXT,
                retry_count     INTEGER NOT NULL DEFAULT 0,
                max_retries     INTEGER NOT NULL DEFAULT 3,
                retry_delay_seconds INTEGER NOT NULL DEFAULT 300,
                not_before      REAL,
                status          TEXT NOT NULL DEFAULT 'PENDING',
                created_at      REAL NOT NULL,
                processed_at    REAL,
                error_message   TEXT
            )
            """
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_gt_status_created ON graph_tasks(status, created_at)"
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_gt_status_not_before_created ON graph_tasks(status, not_before, created_at)"
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_gt_conversation ON graph_tasks(conversation_id, turn_id)"
        )
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_prompt_registry (
                prompt_id   TEXT PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                created_at  REAL NOT NULL
            )
            """
        )
        await self._ensure_graph_task_column(
            "chunk_system_prompt_id",
            "ALTER TABLE graph_tasks ADD COLUMN chunk_system_prompt_id TEXT",
        )
        await self._ensure_graph_task_column(
            "retry_count",
            "ALTER TABLE graph_tasks ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0",
        )
        await self._ensure_graph_task_column(
            "max_retries",
            "ALTER TABLE graph_tasks ADD COLUMN max_retries INTEGER NOT NULL DEFAULT 3",
        )
        await self._ensure_graph_task_column(
            "retry_delay_seconds",
            "ALTER TABLE graph_tasks ADD COLUMN retry_delay_seconds INTEGER NOT NULL DEFAULT 300",
        )
        await self._ensure_graph_task_column(
            "not_before",
            "ALTER TABLE graph_tasks ADD COLUMN not_before REAL",
        )
        self._initialized = True

    async def _ensure_graph_task_column(self, column_name: str, ddl: str) -> None:
        rows = await self.fetchall("PRAGMA table_info(graph_tasks)")
        existing = {str(row.get("name") or "") for row in rows}
        if column_name in existing:
            return
        await self.execute(ddl)

    async def persist_task(self, task_payload: Dict[str, Any]) -> None:
        await self.execute(
            """INSERT OR IGNORE INTO graph_tasks
               (task_id, turn_id, conversation_id, source_agent, relationships, extraction_text, system_prompt_id, chunk_system_prompt_id, allowed_entity_types, allowed_relationship_types, llm_config, task_kind, chunk_ids, retry_count, max_retries, retry_delay_seconds, not_before, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)""",
            (
                task_payload["task_id"],
                task_payload["turn_id"],
                task_payload["conversation_id"],
                task_payload["source_agent"],
                json.dumps(task_payload.get("relationships") or []),
                task_payload.get("extraction_text"),
                task_payload.get("system_prompt_id"),
                task_payload.get("chunk_system_prompt_id"),
                (
                    json.dumps(task_payload["allowed_entity_types"])
                    if task_payload.get("allowed_entity_types") is not None
                    else None
                ),
                (
                    json.dumps(task_payload["allowed_relationship_types"])
                    if task_payload.get("allowed_relationship_types") is not None
                    else None
                ),
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
                int(task_payload.get("retry_count") or 0),
                int(task_payload.get("max_retries") or 3),
                int(task_payload.get("retry_delay_seconds") or 300),
                (
                    float(task_payload["not_before"])
                    if task_payload.get("not_before") is not None
                    else None
                ),
                float(task_payload.get("created_at", time.time())),
            ),
        )

    async def mark_processed(self, task_ids: List[str]) -> None:
        if not task_ids:
            return
        now = time.time()
        await self.executemany(
            "UPDATE graph_tasks SET status='PROCESSED', processed_at=? WHERE task_id=?",
            [(now, task_id) for task_id in task_ids],
        )

    async def load_pending_tasks(self) -> List[Dict[str, Any]]:
        rows = await self.fetchall(
            "SELECT task_id, turn_id, conversation_id, source_agent, relationships, extraction_text, system_prompt_id, chunk_system_prompt_id, allowed_entity_types, allowed_relationship_types, llm_config, task_kind, chunk_ids, retry_count, max_retries, retry_delay_seconds, not_before, created_at "
            "FROM graph_tasks WHERE status='PENDING' ORDER BY created_at ASC"
        )
        return self._decode_task_rows(rows)

    async def load_pending_tasks_due(self, due_epoch: float) -> List[Dict[str, Any]]:
        rows = await self.fetchall(
            "SELECT task_id, turn_id, conversation_id, source_agent, relationships, extraction_text, system_prompt_id, chunk_system_prompt_id, allowed_entity_types, allowed_relationship_types, llm_config, task_kind, chunk_ids, retry_count, max_retries, retry_delay_seconds, not_before, created_at "
            "FROM graph_tasks WHERE status='PENDING' AND (not_before IS NULL OR not_before <= ?) ORDER BY created_at ASC",
            (float(due_epoch),),
        )
        return self._decode_task_rows(rows)

    async def load_pending_delayed_tasks_due(
        self, due_epoch: float
    ) -> List[Dict[str, Any]]:
        rows = await self.fetchall(
            "SELECT task_id, turn_id, conversation_id, source_agent, relationships, extraction_text, system_prompt_id, chunk_system_prompt_id, allowed_entity_types, allowed_relationship_types, llm_config, task_kind, chunk_ids, retry_count, max_retries, retry_delay_seconds, not_before, created_at "
            "FROM graph_tasks WHERE status='PENDING' AND not_before IS NOT NULL AND not_before <= ? ORDER BY created_at ASC",
            (float(due_epoch),),
        )
        return self._decode_task_rows(rows)

    def _decode_task_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                    "chunk_system_prompt_id": row.get("chunk_system_prompt_id"),
                    "allowed_entity_types": (
                        json.loads(row["allowed_entity_types"])
                        if row["allowed_entity_types"]
                        else None
                    ),
                    "allowed_relationship_types": (
                        json.loads(row["allowed_relationship_types"])
                        if row["allowed_relationship_types"]
                        else None
                    ),
                    "llm_config": (
                        json.loads(row["llm_config"]) if row["llm_config"] else None
                    ),
                    "task_kind": row["task_kind"],
                    "chunk_ids": (
                        json.loads(row["chunk_ids"]) if row["chunk_ids"] else None
                    ),
                    "retry_count": int(row.get("retry_count") or 0),
                    "max_retries": int(row.get("max_retries") or 3),
                    "retry_delay_seconds": int(
                        row.get("retry_delay_seconds") or 300
                    ),
                    "not_before": (
                        float(row["not_before"])
                        if row.get("not_before") is not None
                        else None
                    ),
                    "created_at": row["created_at"],
                }
            )
        return tasks

    async def save_prompt(self, prompt_id: str, prompt_text: str) -> None:
        await self.execute(
            "INSERT OR IGNORE INTO graph_prompt_registry (prompt_id, prompt_text, created_at) VALUES (?, ?, ?)",
            (prompt_id, prompt_text, time.time()),
        )

    async def load_prompts(self) -> Dict[str, str]:
        rows = await self.fetchall(
            "SELECT prompt_id, prompt_text FROM graph_prompt_registry"
        )
        return {str(row["prompt_id"]): str(row["prompt_text"]) for row in rows}

    async def purge_processed_older_than(self, cutoff_epoch: float) -> None:
        await self.execute(
            "DELETE FROM graph_tasks WHERE status='PROCESSED' AND processed_at < ?",
            (cutoff_epoch,),
        )
