from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TASK_KIND_RELATIONSHIPS = "relationships"
TASK_KIND_CHUNK_ENTITIES = "chunk_entities"
SHUTDOWN_TURN_ID = "__SHUTDOWN__"


@dataclass
class GraphTask:
    task_id: str
    turn_id: str
    conversation_id: str
    source_agent: str
    immediate: bool = False
    task_kind: str = TASK_KIND_RELATIONSHIPS
    chunk_ids: Optional[List[str]] = None
    relationships: List[dict] = field(default_factory=list)
    extraction_text: Optional[str] = None
    system_prompt: Optional[str] = None
    system_prompt_id: Optional[str] = None
    allowed_entity_types: Optional[List[str]] = None
    allowed_relationship_types: Optional[List[str]] = None
    llm_config: Optional[dict] = None
    allow_create: Optional[bool] = None
    created_at: float = field(default_factory=time.time)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "source_agent": self.source_agent,
            "task_kind": self.task_kind,
            "chunk_ids": self.chunk_ids,
            "relationships": self.relationships,
            "extraction_text": self.extraction_text,
            "system_prompt_id": self.system_prompt_id,
            "allowed_entity_types": self.allowed_entity_types,
            "allowed_relationship_types": self.allowed_relationship_types,
            "llm_config": self.llm_config,
            "allow_create": self.allow_create,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> GraphTask:
        allow_create = payload.get("allow_create")
        if allow_create is not None:
            allow_create = bool(allow_create)
        allowed_entity_types = payload.get("allowed_entity_types")
        if allowed_entity_types is not None:
            allowed_entity_types = list(allowed_entity_types)
        allowed_relationship_types = payload.get("allowed_relationship_types")
        if allowed_relationship_types is not None:
            allowed_relationship_types = list(allowed_relationship_types)
        return cls(
            task_id=str(payload["task_id"]),
            turn_id=str(payload["turn_id"]),
            conversation_id=str(payload["conversation_id"]),
            source_agent=str(payload["source_agent"]),
            task_kind=str(payload.get("task_kind") or TASK_KIND_RELATIONSHIPS),
            chunk_ids=payload.get("chunk_ids"),
            relationships=list(payload.get("relationships") or []),
            extraction_text=payload.get("extraction_text"),
            system_prompt_id=payload.get("system_prompt_id"),
            allowed_entity_types=allowed_entity_types,
            allowed_relationship_types=allowed_relationship_types,
            llm_config=payload.get("llm_config"),
            allow_create=allow_create,
            created_at=float(payload.get("created_at", time.time())),
        )


@dataclass
class SentinelTask:
    turn_id: str
    conversation_id: str


QueueItem = GraphTask | SentinelTask
