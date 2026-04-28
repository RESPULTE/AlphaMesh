from __future__ import annotations

from typing import Dict, Optional

from core.logger import get_logger
from core.memory.graph.queue.utils import prompt_id_from_text
from core.memory.graph.sql_store import GraphTaskSqlStore

logger = get_logger(__name__)


class PromptRegistry:
    def __init__(self, store: GraphTaskSqlStore) -> None:
        self._store = store
        self._prompts: Dict[str, str] = {}

    async def load(self) -> None:
        try:
            prompts = await self._store.load_prompts()
            self._prompts.update(prompts)
        except Exception:
            logger.exception("PromptRegistry: failed to load prompt registry")

    async def register(self, prompt_text: str) -> str:
        prompt_id = prompt_id_from_text(prompt_text)
        if self._prompts.get(prompt_id) == prompt_text:
            return prompt_id
        self._prompts[prompt_id] = prompt_text
        try:
            await self._store.save_prompt(prompt_id=prompt_id, prompt_text=prompt_text)
        except Exception:
            logger.exception(
                "PromptRegistry: failed to persist prompt_id '%s'", prompt_id
            )
        return prompt_id

    def get(self, prompt_id: str) -> Optional[str]:
        return self._prompts.get(prompt_id)
