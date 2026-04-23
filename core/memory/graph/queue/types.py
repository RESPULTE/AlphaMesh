from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

TASK_KIND_RELATIONSHIPS = "relationships"
TASK_KIND_CHUNK_ENTITIES = "chunk_entities"
SHUTDOWN_TURN_ID = "__SHUTDOWN__"


def prompt_id_from_text(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


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
            "llm_config": self.llm_config,
            "allow_create": self.allow_create,
            "created_at": self.created_at,
        }


@dataclass
class SentinelTask:
    turn_id: str
    conversation_id: str


QueueItem = GraphTask | SentinelTask


def graph_task_from_payload(payload: Dict[str, Any]) -> GraphTask:
    allow_create = payload.get("allow_create")
    if allow_create is not None:
        allow_create = bool(allow_create)
    return GraphTask(
        task_id=str(payload["task_id"]),
        turn_id=str(payload["turn_id"]),
        conversation_id=str(payload["conversation_id"]),
        source_agent=str(payload["source_agent"]),
        task_kind=str(payload.get("task_kind") or TASK_KIND_RELATIONSHIPS),
        chunk_ids=payload.get("chunk_ids"),
        relationships=list(payload.get("relationships") or []),
        extraction_text=payload.get("extraction_text"),
        system_prompt_id=payload.get("system_prompt_id"),
        llm_config=payload.get("llm_config"),
        allow_create=allow_create,
        created_at=float(payload.get("created_at", time.time())),
    )


def make_graph_task(
    turn_id: str,
    conversation_id: str,
    source_agent: str,
    relationships: List[dict],
    immediate: bool = False,
    allow_create: Optional[bool] = None,
) -> GraphTask:
    return GraphTask(
        task_id=str(uuid4()),
        turn_id=turn_id,
        conversation_id=conversation_id,
        source_agent=source_agent,
        immediate=immediate,
        relationships=relationships,
        allow_create=allow_create,
    )


def make_extraction_task(
    turn_id: str,
    conversation_id: str,
    source_agent: str,
    extraction_text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    llm_config: Optional[dict] = None,
    immediate: bool = False,
    task_kind: str = TASK_KIND_RELATIONSHIPS,
    chunk_ids: Optional[List[str]] = None,
    allow_create: Optional[bool] = None,
) -> GraphTask:
    prompt_id = prompt_id_from_text(system_prompt) if system_prompt else None
    return GraphTask(
        task_id=str(uuid4()),
        turn_id=turn_id,
        conversation_id=conversation_id,
        source_agent=source_agent,
        immediate=immediate,
        task_kind=task_kind,
        chunk_ids=chunk_ids,
        relationships=[],
        extraction_text=extraction_text,
        system_prompt=system_prompt,
        system_prompt_id=prompt_id,
        llm_config=llm_config,
        allow_create=allow_create,
    )
