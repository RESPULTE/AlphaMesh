"""
api/services/conversation_jsonl_store.py

Per-user JSONL chatlog persistence for conversation turns.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlConversationStore:
    """
    Stores one JSONL file per conversation and one index JSON per user.

    Layout:
      <base>/<safe_user>/index.json
      <base>/<safe_user>/<conversation_id>.jsonl
    """

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        await asyncio.to_thread(
            self._base.mkdir,
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _sanitize_user_email(user_email: str) -> str:
        value = (user_email or "").strip().lower()
        safe = re.sub(r"[^a-z0-9._-]+", "_", value).strip("._-")
        if not safe:
            raise ValueError("Invalid user_email")
        return safe

    def _get_user_dir(self, user_email: str) -> Path:
        safe_user = self._sanitize_user_email(user_email)
        return self._base / safe_user

    def _get_index_path(self, user_email: str) -> Path:
        return self._get_user_dir(user_email) / "index.json"

    def _get_chatlog_path(self, user_email: str, conversation_id: str) -> Path:
        return self._get_user_dir(user_email) / f"{conversation_id}.jsonl"

    @staticmethod
    def _read_index_sync(path: Path) -> List[dict]:
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        payload = json.loads(raw)
        if not isinstance(payload, list):
            return []
        return [row for row in payload if isinstance(row, dict)]

    @staticmethod
    def _write_index_sync(path: Path, rows: List[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        temp_path.replace(path)

    @staticmethod
    def _ensure_index_file_sync(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        path.write_text("[]", encoding="utf-8")

    @staticmethod
    def _ensure_conversation_file_sync(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        path.write_text("", encoding="utf-8")

    async def ensure_user_workspace(self, user_email: str) -> int:
        """
        Ensure the user chatlog directory + index file exist.

        Returns the current conversation count from index.json.
        """
        index_path = self._get_index_path(user_email)
        async with self._lock:
            await asyncio.to_thread(self._ensure_index_file_sync, index_path)
            rows = await asyncio.to_thread(self._read_index_sync, index_path)
        return len(rows)

    async def ensure_conversation(
        self,
        conversation_id: str,
        user_email: str,
    ) -> None:
        now = _utc_now_iso()
        index_path = self._get_index_path(user_email)
        chatlog_path = self._get_chatlog_path(user_email, conversation_id)

        async with self._lock:
            rows = await asyncio.to_thread(self._read_index_sync, index_path)
            row = next(
                (r for r in rows if r.get("conversation_id") == conversation_id), None
            )
            if row is None:
                rows.append(
                    {
                        "conversation_id": conversation_id,
                        "created_at": now,
                        "last_message_at": now,
                        "message_count": 0,
                        "turn_count": 0,
                    }
                )
            await asyncio.to_thread(self._write_index_sync, index_path, rows)
            await asyncio.to_thread(self._ensure_conversation_file_sync, chatlog_path)

    async def append_turn(
        self,
        conversation_id: str,
        user_email: str,
        turn: dict,
    ) -> None:
        index_path = self._get_index_path(user_email)
        chatlog_path = self._get_chatlog_path(user_email, conversation_id)
        now = _utc_now_iso()
        created_at = str(turn.get("created_at") or now)

        async with self._lock:
            rows = await asyncio.to_thread(self._read_index_sync, index_path)
            row = next(
                (r for r in rows if r.get("conversation_id") == conversation_id), None
            )
            if row is None:
                row = {
                    "conversation_id": conversation_id,
                    "created_at": created_at,
                    "last_message_at": created_at,
                    "message_count": 0,
                    "turn_count": 0,
                }
                rows.append(row)

            await asyncio.to_thread(self._ensure_conversation_file_sync, chatlog_path)
            line = json.dumps(turn, ensure_ascii=True)
            await asyncio.to_thread(self._append_line_sync, chatlog_path, line)

            row["last_message_at"] = created_at
            row["message_count"] = int(row.get("message_count", 0)) + 2
            row["turn_count"] = int(row.get("turn_count", 0)) + 1
            await asyncio.to_thread(self._write_index_sync, index_path, rows)

    @staticmethod
    def _append_line_sync(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def load_turns(
        self,
        conversation_id: str,
        user_email: str,
    ) -> List[dict]:
        chatlog_path = self._get_chatlog_path(user_email, conversation_id)

        def _load_sync(path: Path) -> List[dict]:
            if not path.exists():
                return []
            turns: List[dict] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        turns.append(payload)
            return turns

        async with self._lock:
            await asyncio.to_thread(self._ensure_conversation_file_sync, chatlog_path)
            return await asyncio.to_thread(_load_sync, chatlog_path)

    async def list_conversations(
        self,
        user_email: str,
        limit: int = 50,
    ) -> List[dict]:
        index_path = self._get_index_path(user_email)
        rows = await asyncio.to_thread(self._read_index_sync, index_path)
        rows_sorted = sorted(
            rows,
            key=lambda r: str(r.get("last_message_at") or ""),
            reverse=True,
        )
        normalized: List[dict] = []
        for row in rows_sorted[: max(1, limit)]:
            normalized.append(
                {
                    "conversation_id": str(row.get("conversation_id") or ""),
                    "created_at": str(row.get("created_at") or ""),
                    "last_message_at": str(row.get("last_message_at") or ""),
                    "message_count": int(row.get("message_count") or 0),
                    "turn_count": int(row.get("turn_count") or 0),
                }
            )
        return normalized
