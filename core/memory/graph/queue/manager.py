from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import settings
from core.logger import get_logger
from core.memory.graph.entity_resolver import EntityResolver
from core.memory.graph.queue.pipeline import GraphWritePipeline
from core.memory.graph.queue.policies import (
    parse_source_allowlist,
    resolve_allow_create,
)
from core.memory.graph.queue.prompt_registry import PromptRegistry
from core.memory.graph.queue.relationship_extractor import RelationshipExtractor
from core.memory.graph.queue.types import TASK_KIND_RELATIONSHIPS, GraphTask
from core.memory.graph.queue.utils import (
    default_allowed_entity_types,
    default_allowed_relationship_types,
    has_chunk_ids,
    has_extractable_payload,
    has_extraction_text,
    has_inline_relationships,
    is_chunk_entities_task,
    normalize_allowed_entity_types,
    normalize_allowed_relationship_types,
)
from core.memory.graph.queue.worker import ConversationQueueWorker
from core.memory.graph.sql_store import GraphTaskSqlStore
from core.memory.stores.neo4j_adapter import Neo4jAdapter

logger = get_logger(__name__)

_TTL_SECONDS = 1800
_CLEANUP_INTERVAL_SECONDS = 300
_PROCESSED_TASK_RETENTION_SECONDS = 86400


class GraphQueueManager:
    def __init__(
        self,
        entity_resolver: EntityResolver,
        graph_writer: Neo4jAdapter,
        relationship_extractor: RelationshipExtractor,
        entity_extractor: Callable[[List[str]], Any],
        llm_provider: Callable[[Optional[dict]], Any],
        db_path: str = settings.GRAPH_QUEUE_DB_PATH,
    ) -> None:
        self._store = GraphTaskSqlStore(db_path=db_path)
        self._prompt_registry = PromptRegistry(store=self._store)
        self._pipeline = GraphWritePipeline(
            entity_resolver=entity_resolver,
            graph_writer=graph_writer,
            relationship_extractor=relationship_extractor,
            entity_extractor=entity_extractor,
            llm_provider=llm_provider,
            prompt_registry=self._prompt_registry,
        )

        self._allow_create_sources = parse_source_allowlist(
            settings.GRAPH_ALLOW_CREATE_SOURCES
        )

        self._queues: Dict[str, ConversationQueueWorker] = {}
        self._queues_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False

    async def open_session(self, conversation_id: str) -> None:
        _worker, created = await self._ensure_worker(conversation_id, lazy=False)
        if created:
            logger.info("GraphQueueManager: opened session '%s'", conversation_id)

    async def close_session(self, conversation_id: str) -> None:
        async with self._queues_lock:
            worker = self._queues.pop(conversation_id, None)
        if worker is None:
            return
        await worker.drain_and_stop()
        logger.info("GraphQueueManager: closed session '%s'", conversation_id)

    async def enqueue(
        self,
        task: GraphTask,
    ) -> str:
        if not await self._prepare_task(task):
            return task.task_id

        await self._store.persist_task(task.to_payload())

        worker, _created = await self._ensure_worker(task.conversation_id, lazy=True)
        await worker.put(task)

        return task.task_id

    async def flush_turn(self, conversation_id: str, turn_id: str) -> None:
        async with self._queues_lock:
            worker = self._queues.get(conversation_id)
        if worker is None:
            logger.debug(
                "GraphQueueManager.flush_turn: no session for '%s', skipping",
                conversation_id,
            )
            return
        await worker.flush_turn(turn_id)

    async def start(self) -> None:
        if self._started:
            return

        await self._store.initialize()
        await self._prompt_registry.load()
        await self._recover_pending_tasks()

        self._cleanup_task = asyncio.create_task(
            self._ttl_cleanup_loop(), name="graph_queue_ttl_cleanup"
        )
        self._started = True
        logger.info("GraphQueueManager: started")

    async def shutdown(self) -> None:
        cleanup_task = self._cleanup_task
        self._cleanup_task = None
        if cleanup_task:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task

        async with self._queues_lock:
            conversation_ids = list(self._queues.keys())

        for conversation_id in conversation_ids:
            await self.close_session(conversation_id)

        self._started = False
        logger.info("GraphQueueManager: shutdown complete")

    def _create_worker(self, conversation_id: str) -> ConversationQueueWorker:
        return ConversationQueueWorker(
            conversation_id=conversation_id,
            batch_processor=self._process_worker_batch,
            idle_timeout_seconds=_TTL_SECONDS,
        )

    async def _ensure_worker(
        self, conversation_id: str, lazy: bool
    ) -> Tuple[ConversationQueueWorker, bool]:
        async with self._queues_lock:
            worker = self._queues.get(conversation_id)
            if worker is not None:
                return worker, False
            if lazy:
                logger.warning(
                    "GraphQueueManager.enqueue: no open session for '%s', creating lazily",
                    conversation_id,
                )
            worker = self._create_worker(conversation_id)
            worker.start()
            self._queues[conversation_id] = worker
            return worker, True

    async def _process_worker_batch(self, tasks: List[GraphTask], turn_id: str) -> None:
        task_ids = [task.task_id for task in tasks]
        try:
            results = await self._pipeline.process_tasks(tasks)
            logger.info(
                "GraphQueueManager: processed turn '%s' tasks=%d domain_edges=%d user_edges=%d",
                turn_id,
                len(tasks),
                results.get("domain_edges", 0),
                results.get("user_edges", 0),
            )
            await self._store.mark_processed(task_ids)
        except Exception:
            logger.exception(
                "GraphQueueManager: failed to process turn '%s' for conversation '%s'",
                turn_id,
                tasks[0].conversation_id if tasks else "",
            )

    async def _recover_pending_tasks(self) -> None:
        try:
            rows = await self._store.load_pending_tasks()
        except Exception:
            logger.exception(
                "GraphQueueManager: failed to query PENDING tasks at startup"
            )
            return

        if not rows:
            return

        logger.info("GraphQueueManager: recovering %d pending task(s)", len(rows))

        tasks: List[GraphTask] = [GraphTask.from_payload(row) for row in rows]
        for task in tasks:
            self._apply_allow_create_policy(task)

        grouped: Dict[Tuple[str, str], List[GraphTask]] = {}
        for task in tasks:
            grouped.setdefault((task.conversation_id, task.turn_id), []).append(task)

        for (_conversation_id, turn_id), group_tasks in grouped.items():
            await self._process_worker_batch(group_tasks, turn_id)

    async def _ttl_cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
                await self._evict_idle_queues()
                await self._purge_old_processed_tasks()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("GraphQueueManager: TTL cleanup error")

    async def _evict_idle_queues(self) -> None:
        async with self._queues_lock:
            to_evict = [cid for cid, worker in self._queues.items() if worker.is_idle]
        for conversation_id in to_evict:
            logger.info(
                "GraphQueueManager: TTL evicting idle session '%s'", conversation_id
            )
            await self.close_session(conversation_id)

    async def _purge_old_processed_tasks(self) -> None:
        cutoff = time.time() - _PROCESSED_TASK_RETENTION_SECONDS
        try:
            await self._store.purge_processed_older_than(cutoff)
        except Exception:
            logger.exception("GraphQueueManager: failed to purge old processed tasks")

    def _apply_allow_create_policy(self, task: GraphTask) -> None:
        task.allow_create = resolve_allow_create(
            source_agent=task.source_agent,
            task_allow_create=task.allow_create,
            explicit_allow_create=None,
            default_allow_create_sources=self._allow_create_sources,
        )

    async def _prepare_task(self, task: GraphTask) -> bool:
        is_chunk_entities = is_chunk_entities_task(task)

        if task.task_kind == TASK_KIND_RELATIONSHIPS:
            task.allowed_entity_types = self._resolve_task_entity_scope(task)
            task.allowed_relationship_types = self._resolve_task_relationship_scope(
                task
            )

        if task.system_prompt is not None and not is_chunk_entities:
            task.system_prompt_id = await self._prompt_registry.register(
                task.system_prompt
            )

        self._apply_allow_create_policy(task)

        if is_chunk_entities and not has_chunk_ids(task):
            logger.warning(
                "GraphQueueManager.enqueue: missing chunk_ids for '%s'",
                task.task_id,
            )
            return False

        if (
            (not is_chunk_entities)
            and has_extraction_text(task)
            and not task.system_prompt_id
        ):
            logger.warning(
                "GraphQueueManager.enqueue: missing system_prompt_id for task '%s'",
                task.task_id,
            )
            return False

        has_work = (
            has_chunk_ids(task)
            if is_chunk_entities
            else has_inline_relationships(task) or has_extractable_payload(task)
        )
        if not has_work:
            logger.debug(
                "GraphQueueManager.enqueue: skipping empty task from '%s'",
                task.source_agent,
            )
            return False

        if (
            (not is_chunk_entities)
            and task.system_prompt_id
            and not self._prompt_registry.get(task.system_prompt_id)
        ):
            logger.warning(
                "GraphQueueManager.enqueue: prompt_id '%s' not registered; extraction may fail",
                task.system_prompt_id,
            )
        return True

    @staticmethod
    def _resolve_task_entity_scope(task: GraphTask) -> List[str]:
        if task.allowed_entity_types is None:
            return default_allowed_entity_types()
        normalized = normalize_allowed_entity_types(task.allowed_entity_types)
        if not normalized:
            return default_allowed_entity_types()
        return normalized

    @staticmethod
    def _resolve_task_relationship_scope(task: GraphTask) -> List[str]:
        if task.allowed_relationship_types is None:
            return default_allowed_relationship_types()
        normalized = normalize_allowed_relationship_types(
            task.allowed_relationship_types
        )
        if not normalized:
            return default_allowed_relationship_types()
        return normalized
