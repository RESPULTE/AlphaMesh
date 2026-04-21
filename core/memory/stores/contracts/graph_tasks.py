"""Graph-task persistence contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GraphTaskPersistenceAdapter(ABC):
    """Abstract adapter for durable graph task queue persistence."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema and indexes. Must be idempotent."""

    @abstractmethod
    async def persist_task(self, task_payload: Dict[str, Any]) -> None:
        """Persist a queued graph task."""

    @abstractmethod
    async def mark_processed(self, task_ids: List[str]) -> None:
        """Mark task ids processed."""

    @abstractmethod
    async def load_pending_tasks(self) -> List[Dict[str, Any]]:
        """Load all pending tasks ordered by creation timestamp."""

    @abstractmethod
    async def save_prompt(self, prompt_id: str, prompt_text: str) -> None:
        """Persist prompt text by id."""

    @abstractmethod
    async def load_prompts(self) -> Dict[str, str]:
        """Load prompt registry entries."""

    @abstractmethod
    async def purge_processed_older_than(self, cutoff_epoch: float) -> None:
        """Delete processed tasks older than cutoff timestamp."""

