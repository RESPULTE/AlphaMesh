"""
core/memory/graph/graph_queue.py

Centralised, non-blocking graph write pipeline.

Architecture

GraphTask       — payload dataclass: relationship dicts + routing metadata
_SentinelTask   — poison-pill: signals consumer to flush one turn's accumulated tasks
ConversationQueue — one per active conversation_id: owns asyncio.Queue + consumer coroutine
GraphQueueManager — singleton: manages all ConversationQueues, SQLite persistence, startup/shutdown

Flow (normal path)

  1. Agent calls GraphQueueManager.enqueue(GraphTask)
     → Task persisted to SQLite (status=PENDING)
     → Task put on ConversationQueue._queue
  2. Consumer accumulates GraphTask items in _pending[turn_id]
  3. Orchestrator calls GraphQueueManager.flush_turn(conversation_id, turn_id)
     → _SentinelTask put on queue
  4. Consumer pops sentinel → processes all tasks for that turn_id:
     a. Merge all relationship lists
     b. Separate user-scoped from domain relationships
     c. EntityResolver.resolve_batch() — type-scoped dedup + Neo4j/Chroma upsert
     d. Neo4jAdapter.write_relationships() — domain edges
     e. Neo4jAdapter.write_relationships() — user-scoped edges (no dedup)
     f. Mark all GraphTasks for this turn PROCESSED in SQLite
  5. Consumer waits for next sentinel

Flow (bypass path — system tasks)

  GraphQueueManager.enqueue(task, immediate=True)
  → EntityResolver.resolve_batch()
  → Neo4jAdapter.write_relationships()
  → (fire-and-forget asyncio.Task, no SQLite, no queue)

Recovery on restart

  GraphQueueManager.start()
  → Scan SQLite for PENDING tasks
  → Process each orphaned task directly (bypass queue) sorted by turn_id
  → Mark processed

Session lifecycle

  open_session(conversation_id)  — creates ConversationQueue + starts consumer
  close_session(conversation_id) — drains queue, stops consumer, evicts from _queues
  TTL cleanup (safety net)       — every 5 min, evicts idle queues (30 min inactivity)

Multi-user safety

  Each conversation_id has its own ConversationQueue with its own asyncio.Queue
  and consumer coroutine.  The only shared state across conversations is:
    - GraphQueueManager._queues dict (protected by asyncio.Lock)
    - EntityResolver's global LRU cache (protected by its own asyncio.Lock)
  Neither lock is held during I/O, preventing cross-conversation head-of-line blocking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite

from core.logger import get_logger
from core.memory.graph.entity_resolver import EntityResolver
from core.memory.graph.models import _USER_SCOPED_TYPES
from core.memory.graph.relationship_extractor import RelationshipExtractor
from core.memory.graph.utils import (
    entity_key,
    normalize_entity_name,
    normalize_entity_type,
)
from core.memory.stores.neo4j_adapter import Neo4jAdapter

logger = get_logger(__name__)

# SQLite path — co-located with the financial data DB
_GRAPH_TASKS_DB = "./data/graph_tasks.db"
_TTL_SECONDS = 1800  # 30 min inactivity before queue eviction
_CLEANUP_INTERVAL = 300  # 5 min between TTL sweeps

# Global in-memory mapping of prompt_id -> prompt text
_PROMPT_REGISTRY: Dict[str, str] = {}


def _prompt_id(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def _register_prompt_local(prompt_id: str, prompt: str) -> None:
    _PROMPT_REGISTRY[prompt_id] = prompt


def _get_prompt(prompt_id: str) -> Optional[str]:
    return _PROMPT_REGISTRY.get(prompt_id)


#
# Data structures
#


@dataclass
class GraphTask:
    """Payload for one agent's graph write request."""

    task_id: str  # uuid4; returned to caller for tracing
    turn_id: str  # groups tasks within one orchestrator turn
    conversation_id: str
    source_agent: str
    relationships: List[dict]  # already-extracted dicts, no nx objects
    extraction_text: Optional[str] = None
    system_prompt_id: Optional[str] = None
    llm_config: Optional[dict] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class _SentinelTask:
    """Poison-pill: tells the consumer to flush all accumulated tasks for turn_id."""

    turn_id: str
    conversation_id: str


# Union type for queue items
_QueueItem = GraphTask | _SentinelTask


#
# ConversationQueue — one per active conversation
#


class ConversationQueue:
    """
    Owns: asyncio.Queue, consumer Task, per-turn task accumulator.
    Created by GraphQueueManager.open_session().
    """

    def __init__(
        self,
        conversation_id: str,
        entity_resolver: EntityResolver,
        graph_writer: Neo4jAdapter,
        relationship_extractor: RelationshipExtractor,
        llm_provider: Callable[[Optional[dict]], Any],
        prompt_resolver: Callable[[str], Optional[str]],
        db_path: str,
    ) -> None:
        self.conversation_id = conversation_id
        self._resolver = entity_resolver
        self._writer = graph_writer
        self._extractor = relationship_extractor
        self._llm_provider = llm_provider
        self._prompt_resolver = prompt_resolver
        self._db_path = db_path

        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._pending: Dict[str, List[GraphTask]] = {}  # turn_id → [GraphTask, ...]
        self._last_activity: float = time.monotonic()
        self._consumer_task: Optional[asyncio.Task] = None
        self._closing = False

    def start(self) -> None:
        """Start the consumer coroutine.  Must be called from a running event loop."""
        self._consumer_task = asyncio.create_task(
            self._consume(), name=f"graph_consumer_{self.conversation_id[:8]}"
        )

    async def put(self, item: _QueueItem) -> None:
        """Put a task or sentinel onto the queue."""
        self._last_activity = time.monotonic()
        await self._queue.put(item)

    async def drain_and_stop(self) -> None:
        """
        Enqueue a shutdown sentinel, then wait for the consumer to finish.
        Called by GraphQueueManager.close_session().
        """
        self._closing = True
        # Enqueue a global shutdown sentinel (turn_id="" is the shutdown signal)
        await self._queue.put(
            _SentinelTask(turn_id="__SHUTDOWN__", conversation_id=self.conversation_id)
        )
        if self._consumer_task:
            try:
                await asyncio.wait_for(self._consumer_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "ConversationQueue: consumer for %s did not finish within 30s, cancelling",
                    self.conversation_id,
                )
                self._consumer_task.cancel()

    @property
    def is_idle(self) -> bool:
        return (
            time.monotonic() - self._last_activity > _TTL_SECONDS
            and self._queue.empty()
            and not self._pending
        )

    async def _consume(self) -> None:
        """
        Main consumer coroutine.  Runs until a shutdown sentinel is received.
        """
        logger.info(
            "ConversationQueue: consumer started for '%s'", self.conversation_id
        )
        try:
            while True:
                item = await self._queue.get()
                try:
                    if isinstance(item, _SentinelTask):
                        if item.turn_id == "__SHUTDOWN__":
                            # Drain any remaining pending tasks before stopping
                            for turn_id, tasks in list(self._pending.items()):
                                if tasks:
                                    await self._process_batch(tasks, turn_id)
                            break
                        # Normal flush sentinel
                        tasks = self._pending.pop(item.turn_id, [])
                        if tasks:
                            await self._process_batch(tasks, item.turn_id)
                    else:
                        self._pending.setdefault(item.turn_id, []).append(item)
                        self._last_activity = time.monotonic()
                except Exception:
                    logger.exception(
                        "ConversationQueue: error processing item for conversation '%s'",
                        self.conversation_id,
                    )
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            logger.info(
                "ConversationQueue: consumer cancelled for '%s'", self.conversation_id
            )
        finally:
            logger.info(
                "ConversationQueue: consumer stopped for '%s'", self.conversation_id
            )

    #
    # Batch processing
    #

    async def _extract_relationships_for_task(self, task: GraphTask) -> List[dict]:
        if not task.extraction_text or not task.system_prompt_id:
            return []
        prompt = self._prompt_resolver(task.system_prompt_id)
        if not prompt:
            logger.warning(
                "ConversationQueue: missing system prompt for id '%s'",
                task.system_prompt_id,
            )
            return []
        try:
            llm = self._llm_provider(task.llm_config)
            return await self._extractor.extract(
                text=task.extraction_text,
                llm=llm,
                system_prompt=prompt,
                max_attempts=1,
            )
        except Exception:
            logger.exception(
                "ConversationQueue: extraction failed for task '%s'", task.task_id
            )
            return []

    async def _process_batch(self, tasks: List[GraphTask], turn_id: str) -> None:
        """
        Process all GraphTasks for one turn as a single merged batch.

        Steps:
          1. Merge all relationship lists from all tasks
          2. Separate user-scoped from domain relationships
          3. Collect unique domain entity descriptors
          4. EntityResolver.resolve_batch() — type-scoped dedup + Neo4j/Chroma upsert
          5. Neo4jAdapter.write_relationships() — domain edges
          6. Build user_entity_cache for user-scoped nodes
          7. EntityResolver.resolve_user_node() — for each unique user-scoped entity
          8. Neo4jAdapter.write_relationships() — user-scoped edges
          9. Mark all tasks PROCESSED in SQLite
        """
        task_ids = [t.task_id for t in tasks]
        logger.info(
            "ConversationQueue: processing batch — turn_id='%s' tasks=%d",
            turn_id,
            len(tasks),
        )

        #  Step 1: Merge all relationships
        all_relationships: List[dict] = []
        source_agents: List[str] = []
        for task in tasks:
            rels = task.relationships or []
            if not rels and task.extraction_text:
                rels = await self._extract_relationships_for_task(task)
            if rels:
                all_relationships.extend(rels)
                if task.source_agent not in source_agents:
                    source_agents.append(task.source_agent)

        if not all_relationships:
            await self._mark_processed(task_ids)
            return

        conversation_id = tasks[0].conversation_id
        source_agent_label = "+".join(source_agents)

        #  Step 2: Separate user-scoped from domain relationships
        domain_rels: List[dict] = []
        user_rels: List[dict] = []
        for rel in all_relationships:
            from_type = str(rel.get("from_type") or "").strip()
            to_type = str(rel.get("to_type") or "").strip()
            if from_type in _USER_SCOPED_TYPES or to_type in _USER_SCOPED_TYPES:
                user_rels.append(rel)
            else:
                domain_rels.append(rel)

        #  Step 3 & 4: Resolve domain entities
        domain_entity_cache: Dict[Tuple[str, str], str] = {}
        if domain_rels:
            unique_domain_entities: List[Tuple[str, str, Optional[dict]]] = []
            seen_domain: set = set()
            for rel in domain_rels:
                for name_key, type_key, props_key in [
                    ("from_name", "from_type", "from_node_props"),
                    ("to_name", "to_type", "to_node_props"),
                ]:
                    name = normalize_entity_name(str(rel.get(name_key) or ""))
                    etype_raw = str(rel.get(type_key) or "").strip()
                    etype = normalize_entity_type(etype_raw)
                    if not name or not etype:
                        continue
                    key = (name.lower(), etype)
                    if key not in seen_domain:
                        seen_domain.add(key)
                        unique_domain_entities.append(
                            (name, etype, rel.get(props_key) or None)
                        )

            resolved = await self._resolver.resolve_batch(unique_domain_entities)
            for (name, etype), entity_id in resolved.items():
                domain_entity_cache[entity_key(name, etype)] = entity_id

        #  Step 5: Write domain edges
        domain_written = 0
        if domain_rels:
            domain_written = await self._writer.write_relationships(
                domain_rels, conversation_id, source_agent_label, domain_entity_cache
            )

        #  Step 6 & 7: Resolve user-scoped nodes
        user_entity_cache: Dict[Tuple[str, str], str] = {}
        if user_rels:
            # Collect unique user-scoped entity descriptors
            user_nodes_to_resolve: List[Tuple[str, str, dict]] = []
            seen_user: set = set()
            for rel in user_rels:
                for name_key, type_key, props_key in [
                    ("from_name", "from_type", "from_node_props"),
                    ("to_name", "to_type", "to_node_props"),
                ]:
                    name = str(rel.get(name_key) or "").strip()
                    etype = str(rel.get(type_key) or "").strip()
                    if not name or not etype:
                        continue
                    # Domain entities referenced by user rels go through domain resolver
                    if etype not in _USER_SCOPED_TYPES:
                        etype_norm = normalize_entity_type(etype)
                        if etype_norm:
                            name_norm = normalize_entity_name(name)
                            ck = entity_key(name_norm, etype_norm)
                            if ck not in user_entity_cache:
                                # Try domain cache first, then resolve
                                if ck in domain_entity_cache:
                                    user_entity_cache[ck] = domain_entity_cache[ck]
                                else:
                                    eid = await self._resolver.resolve(
                                        name_norm, etype_norm
                                    )
                                    if eid:
                                        user_entity_cache[ck] = eid
                        continue

                    key = (name.lower(), etype)
                    if key not in seen_user:
                        seen_user.add(key)
                        props = rel.get(props_key) or {}
                        user_nodes_to_resolve.append((name, etype, props))

            # Resolve user-scoped nodes (deterministic, no dedup)
            for name, etype, props in user_nodes_to_resolve:
                node_id = await self._resolver.resolve_user_node(name, etype, props)
                if node_id:
                    user_entity_cache[entity_key(name, etype)] = node_id

        #  Step 8: Write user-scoped edges
        user_written = 0
        if user_rels:
            # Merge domain + user entity caches for edge writing
            combined_cache = {**domain_entity_cache, **user_entity_cache}
            user_written = await self._writer.write_relationships(
                user_rels, conversation_id, source_agent_label, combined_cache
            )

        logger.info(
            "ConversationQueue: batch done — turn_id='%s' domain_edges=%d user_edges=%d",
            turn_id,
            domain_written,
            user_written,
        )

        #  Step 9: Mark tasks processed
        await self._mark_processed(task_ids)

    async def _mark_processed(self, task_ids: List[str]) -> None:
        if not task_ids:
            return
        try:
            async with aiosqlite.connect(self._db_path) as db:
                now = time.time()
                await db.executemany(
                    "UPDATE graph_tasks SET status='PROCESSED', processed_at=? WHERE task_id=?",
                    [(now, tid) for tid in task_ids],
                )
                await db.commit()
        except Exception:
            logger.exception(
                "ConversationQueue: failed to mark tasks PROCESSED in SQLite"
            )


#
# GraphQueueManager — singleton managed by service_manager
#


class GraphQueueManager:
    """
    Singleton that manages all ConversationQueue instances.

    Public API

    open_session(conversation_id)          → create queue + start consumer
    close_session(conversation_id)         → drain queue + stop consumer
    enqueue(task: GraphTask) -> str        → enqueue task, return task_id
    flush_turn(conversation_id, turn_id)   → enqueue sentinel
    enqueue(task, immediate=True)    → bypass queue, direct write
    start()                                → init SQLite + recover orphans + start cleanup
    shutdown()                             → drain all queues gracefully
    """

    def __init__(
        self,
        entity_resolver: EntityResolver,
        graph_writer: Neo4jAdapter,
        relationship_extractor: RelationshipExtractor,
        llm_provider: Callable[[Optional[dict]], Any],
        db_path: str = _GRAPH_TASKS_DB,
    ) -> None:
        self._resolver = entity_resolver
        self._writer = graph_writer
        self._extractor = relationship_extractor
        self._llm_provider = llm_provider
        self._prompt_registry: Dict[str, str] = {}
        self._db_path = db_path

        self._queues: Dict[str, ConversationQueue] = {}
        self._queues_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False

    #
    # Session lifecycle
    #

    async def open_session(self, conversation_id: str) -> None:
        """
        Create a ConversationQueue for conversation_id and start its consumer.
        Idempotent: no-op if session already open.
        """
        async with self._queues_lock:
            if conversation_id in self._queues:
                return
            cq = ConversationQueue(
                conversation_id=conversation_id,
                entity_resolver=self._resolver,
                graph_writer=self._writer,
                relationship_extractor=self._extractor,
                llm_provider=self._llm_provider,
                prompt_resolver=_get_prompt,
                db_path=self._db_path,
            )
            cq.start()
            self._queues[conversation_id] = cq
            logger.info("GraphQueueManager: opened session '%s'", conversation_id)

    async def close_session(self, conversation_id: str) -> None:
        """
        Drain the ConversationQueue for conversation_id and stop its consumer.
        No-op if session is not open.
        """
        async with self._queues_lock:
            cq = self._queues.pop(conversation_id, None)
        if cq is None:
            return
        await cq.drain_and_stop()
        logger.info("GraphQueueManager: closed session '%s'", conversation_id)

    #
    # Enqueue and flush
    #

    async def enqueue(
        self,
        task: GraphTask,
        immediate: bool = False,
        extraction_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        llm_config: Optional[dict] = None,
    ) -> str:
        """
        Persist task to SQLite, then either enqueue it for turn-scoped batching
        or process immediately if immediate=True.
        Creates the session lazily if not already open (safety net).
        Returns the task_id for tracing.
        """
        if extraction_text is not None:
            task.extraction_text = extraction_text
        if llm_config is not None:
            task.llm_config = llm_config
        if system_prompt is not None:
            task.system_prompt_id = await self._register_prompt(system_prompt)

        has_relationships = bool(task.relationships)
        has_extraction = bool(task.extraction_text and task.system_prompt_id)

        if not has_relationships and not has_extraction:
            logger.debug(
                "GraphQueueManager.enqueue: skipping empty task from '%s'",
                task.source_agent,
            )
            return task.task_id

        if task.extraction_text and not task.system_prompt_id:
            logger.warning(
                "GraphQueueManager.enqueue: missing system_prompt_id for task '%s'",
                task.task_id,
            )
            return task.task_id

        if task.system_prompt_id and task.system_prompt_id not in self._prompt_registry:
            logger.warning(
                "GraphQueueManager.enqueue: prompt_id '%s' not registered; extraction may fail",
                task.system_prompt_id,
            )

        # Persist to SQLite first (durability)
        await self._persist_task(task)

        if immediate:
            try:
                relationships = task.relationships
                if not relationships and has_extraction:
                    relationships = await self._extract_relationships_for_task(task)
                await self._process_relationships_immediate(
                    relationships,
                    task.conversation_id,
                    task.source_agent,
                )
            except Exception:
                logger.exception(
                    "GraphQueueManager.enqueue: immediate write failed for '%s'",
                    task.source_agent,
                )
            await self._mark_processed([task.task_id])
            return task.task_id

        # Lazy session creation (safety net — callers should call open_session explicitly)
        async with self._queues_lock:
            cq = self._queues.get(task.conversation_id)
            if cq is None:
                logger.warning(
                    "GraphQueueManager.enqueue: no open session for '%s', creating lazily",
                    task.conversation_id,
                )
                cq = ConversationQueue(
                    conversation_id=task.conversation_id,
                    entity_resolver=self._resolver,
                    graph_writer=self._writer,
                    relationship_extractor=self._extractor,
                    llm_provider=self._llm_provider,
                    prompt_resolver=_get_prompt,
                    db_path=self._db_path,
                )
                cq.start()
                self._queues[task.conversation_id] = cq

        await cq.put(task)
        return task.task_id

    async def flush_turn(self, conversation_id: str, turn_id: str) -> None:
        """
        Enqueue a sentinel for this turn.  No-op if no session is open.
        This triggers the consumer to process all accumulated tasks for turn_id.
        Call this from orchestrator.run() after graph.ainvoke() returns.
        """
        async with self._queues_lock:
            cq = self._queues.get(conversation_id)
        if cq is None:
            logger.debug(
                "GraphQueueManager.flush_turn: no session for '%s', skipping",
                conversation_id,
            )
            return
        await cq.put(_SentinelTask(turn_id=turn_id, conversation_id=conversation_id))

    async def _extract_relationships_for_task(self, task: GraphTask) -> List[dict]:
        if not task.extraction_text or not task.system_prompt_id:
            return []
        prompt = _get_prompt(task.system_prompt_id)
        if not prompt:
            logger.warning(
                "GraphQueueManager: missing system prompt for id '%s'",
                task.system_prompt_id,
            )
            return []
        try:
            llm = self._llm_provider(task.llm_config)
            return await self._extractor.extract(
                text=task.extraction_text,
                llm=llm,
                system_prompt=prompt,
                max_attempts=1,
            )
        except Exception:
            logger.exception(
                "GraphQueueManager: extraction failed for task '%s'", task.task_id
            )
            return []

    async def _process_relationships_immediate(
        self,
        relationships: List[dict],
        conversation_id: str,
        source_agent: str,
    ) -> None:
        """
        Process relationships immediately without using the queue.
        Used when enqueue(immediate=True) is requested.
        """
        if not relationships:
            return

        try:
            # Separate and resolve (same logic as ConversationQueue._process_batch)
            domain_rels = [
                r
                for r in relationships
                if str(r.get("from_type") or "") not in _USER_SCOPED_TYPES
                and str(r.get("to_type") or "") not in _USER_SCOPED_TYPES
            ]
            user_rels = [r for r in relationships if r not in domain_rels]

            entity_cache: Dict[Tuple[str, str], str] = {}

            if domain_rels:
                unique_entities: List[Tuple[str, str, Optional[dict]]] = []
                seen: set = set()
                for rel in domain_rels:
                    for name_key, type_key, props_key in [
                        ("from_name", "from_type", "from_node_props"),
                        ("to_name", "to_type", "to_node_props"),
                    ]:
                        name = normalize_entity_name(str(rel.get(name_key) or ""))
                        etype = normalize_entity_type(
                            str(rel.get(type_key) or "").strip()
                        )
                        if not name or not etype:
                            continue
                        k = (name.lower(), etype)
                        if k not in seen:
                            seen.add(k)
                            unique_entities.append((name, etype, rel.get(props_key)))
                resolved = await self._resolver.resolve_batch(unique_entities)
                for (name, etype), eid in resolved.items():
                    entity_cache[entity_key(name, etype)] = eid
                await self._writer.write_relationships(
                    domain_rels, conversation_id, source_agent, entity_cache
                )

            if user_rels:
                user_cache: Dict[Tuple[str, str], str] = {**entity_cache}
                for rel in user_rels:
                    for name_key, type_key, props_key in [
                        ("from_name", "from_type", "from_node_props"),
                        ("to_name", "to_type", "to_node_props"),
                    ]:
                        name = str(rel.get(name_key) or "").strip()
                        etype = str(rel.get(type_key) or "").strip()
                        if not name or not etype or etype not in _USER_SCOPED_TYPES:
                            continue
                        k = entity_key(name, etype)
                        if k not in user_cache:
                            props = rel.get(props_key) or {}
                            nid = await self._resolver.resolve_user_node(
                                name, etype, props
                            )
                            if nid:
                                user_cache[k] = nid
                await self._writer.write_relationships(
                    user_rels, conversation_id, source_agent, user_cache
                )

            logger.info(
                "GraphQueueManager.enqueue(immediate=True): wrote %d relationships for '%s'",
                len(relationships),
                source_agent,
            )
        except Exception:
            logger.exception(
                "GraphQueueManager.enqueue(immediate=True): failed for agent '%s'",
                source_agent,
            )

    async def _mark_processed(self, task_ids: List[str]) -> None:
        if not task_ids:
            return
        try:
            async with aiosqlite.connect(self._db_path) as db:
                now = time.time()
                await db.executemany(
                    "UPDATE graph_tasks SET status='PROCESSED', processed_at=? WHERE task_id=?",
                    [(now, tid) for tid in task_ids],
                )
                await db.commit()
        except Exception:
            logger.exception("GraphQueueManager: failed to mark tasks processed")

    async def start(self) -> None:
        """
        Initialize SQLite, recover orphaned PENDING tasks, start TTL cleanup.
        Must be called once at application startup (services.py startup()).
        """
        if self._started:
            return
        self._started = True

        await self._init_db()
        await self._recover_pending_tasks()

        self._cleanup_task = asyncio.create_task(
            self._ttl_cleanup_loop(), name="graph_queue_ttl_cleanup"
        )
        logger.info("GraphQueueManager: started")

    async def shutdown(self) -> None:
        """Drain all open queues gracefully.  Call at application teardown."""
        if self._cleanup_task:
            self._cleanup_task.cancel()

        async with self._queues_lock:
            conv_ids = list(self._queues.keys())

        for cid in conv_ids:
            await self.close_session(cid)

        logger.info("GraphQueueManager: shutdown complete")

    #
    async def _register_prompt(self, prompt: str) -> str:
        prompt_id = _prompt_id(prompt)
        if self._prompt_registry.get(prompt_id) == prompt:
            return prompt_id
        self._prompt_registry[prompt_id] = prompt
        _register_prompt_local(prompt_id, prompt)
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "INSERT OR IGNORE INTO graph_prompt_registry (prompt_id, prompt_text, created_at) VALUES (?, ?, ?)",
                    (prompt_id, prompt, time.time()),
                )
                await db.commit()
        except Exception:
            logger.exception("GraphQueueManager: failed to persist prompt registry")
        return prompt_id

    async def _load_prompt_registry(self) -> None:
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT prompt_id, prompt_text FROM graph_prompt_registry"
                ) as cursor:
                    rows = await cursor.fetchall()
            for pid, prompt in rows:
                self._prompt_registry[pid] = prompt
                _register_prompt_local(pid, prompt)
        except Exception:
            logger.exception("GraphQueueManager: failed to load prompt registry")

    # SQLite persistence
    #

    async def _init_db(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
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
                    status          TEXT NOT NULL DEFAULT 'PENDING',
                    created_at      REAL NOT NULL,
                    processed_at    REAL,
                    error_message   TEXT
                )
            """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_prompt_registry (
                    prompt_id   TEXT PRIMARY KEY,
                    prompt_text TEXT NOT NULL,
                    created_at  REAL NOT NULL
                )
                """
            )

            # Ensure new columns exist for deferred extraction
            async with db.execute("PRAGMA table_info(graph_tasks)") as cursor:
                cols = {row[1] for row in await cursor.fetchall()}
            for col, col_type in [
                ("extraction_text", "TEXT"),
                ("system_prompt_id", "TEXT"),
                ("llm_config", "TEXT"),
            ]:
                if col not in cols:
                    await db.execute(
                        f"ALTER TABLE graph_tasks ADD COLUMN {col} {col_type}"
                    )

            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_gt_status_created ON graph_tasks(status, created_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_gt_conversation ON graph_tasks(conversation_id, turn_id)"
            )
            await db.commit()

        await self._load_prompt_registry()
        logger.info("GraphQueueManager: SQLite initialized at '%s'", self._db_path)

    async def _persist_task(self, task: GraphTask) -> None:
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """INSERT OR IGNORE INTO graph_tasks
                       (task_id, turn_id, conversation_id, source_agent, relationships, extraction_text, system_prompt_id, llm_config, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)""",
                    (
                        task.task_id,
                        task.turn_id,
                        task.conversation_id,
                        task.source_agent,
                        json.dumps(task.relationships),
                        task.extraction_text,
                        task.system_prompt_id,
                        (
                            json.dumps(task.llm_config)
                            if task.llm_config is not None
                            else None
                        ),
                        task.created_at,
                    ),
                )
                await db.commit()
        except Exception:
            logger.exception(
                "GraphQueueManager: failed to persist task '%s'", task.task_id
            )

    async def _recover_pending_tasks(self) -> None:
        """
        On startup, find PENDING tasks from previous sessions and process them
        directly (bypass queue � no session needed).
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT task_id, turn_id, conversation_id, source_agent, relationships, extraction_text, system_prompt_id, llm_config, created_at "
                    "FROM graph_tasks WHERE status='PENDING' ORDER BY created_at ASC"
                ) as cursor:
                    rows = await cursor.fetchall()
        except Exception:
            logger.exception(
                "GraphQueueManager: failed to query PENDING tasks at startup"
            )
            return

        if not rows:
            return

        logger.info(
            "GraphQueueManager: recovering %d orphaned PENDING task(s)", len(rows)
        )

        tasks: List[GraphTask] = []
        for row in rows:
            llm_config = json.loads(row[7]) if row[7] else None
            task = GraphTask(
                task_id=row[0],
                turn_id=row[1],
                conversation_id=row[2],
                source_agent=row[3],
                relationships=json.loads(row[4]) if row[4] else [],
                extraction_text=row[5],
                system_prompt_id=row[6],
                llm_config=llm_config,
                created_at=row[8],
            )
            if (
                not task.relationships
                and task.extraction_text
                and task.system_prompt_id
            ):
                task.relationships = await self._extract_relationships_for_task(task)
            tasks.append(task)

        # Group by conversation_id + turn_id and process each group
        grouped: Dict[Tuple[str, str], List[GraphTask]] = {}
        for task in tasks:
            key = (task.conversation_id, task.turn_id)
            grouped.setdefault(key, []).append(task)

        for (conv_id, turn_id), group_tasks in grouped.items():
            all_rels: List[dict] = []
            for t in group_tasks:
                all_rels.extend(t.relationships or [])
            if not all_rels:
                await self._mark_processed([t.task_id for t in group_tasks])
                continue

            source = "+".join({t.source_agent for t in group_tasks})
            try:
                await self._process_relationships_immediate(all_rels, conv_id, source)
                await self._mark_processed([t.task_id for t in group_tasks])
            except Exception:
                logger.exception(
                    "GraphQueueManager: recovery failed for conversation '%s' turn '%s'",
                    conv_id,
                    turn_id,
                )

    # TTL cleanup
    #

    async def _ttl_cleanup_loop(self) -> None:
        """Background task: evict idle queues every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(_CLEANUP_INTERVAL)
                await self._evict_idle_queues()
                await self._purge_old_processed_tasks()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("GraphQueueManager: TTL cleanup error")

    async def _evict_idle_queues(self) -> None:
        async with self._queues_lock:
            to_evict = [
                cid
                for cid, cq in self._queues.items()
                if cq.is_idle
                and (cq._consumer_task is None or cq._consumer_task.done())
            ]

        for cid in to_evict:
            logger.info("GraphQueueManager: TTL evicting idle session '%s'", cid)
            await self.close_session(cid)

    async def _purge_old_processed_tasks(self) -> None:
        """Delete PROCESSED tasks older than 24 hours."""
        cutoff = time.time() - 86400  # 24 hours
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "DELETE FROM graph_tasks WHERE status='PROCESSED' AND processed_at < ?",
                    (cutoff,),
                )
                await db.commit()
        except Exception:
            logger.exception("GraphQueueManager: failed to purge old processed tasks")


def make_graph_task(
    turn_id: str,
    conversation_id: str,
    source_agent: str,
    relationships: List[dict],
) -> GraphTask:
    """Convenience constructor that auto-generates task_id."""
    return GraphTask(
        task_id=str(uuid4()),
        turn_id=turn_id,
        conversation_id=conversation_id,
        source_agent=source_agent,
        relationships=relationships,
    )


def make_extraction_task(
    turn_id: str,
    conversation_id: str,
    source_agent: str,
    extraction_text: str,
    system_prompt: str,
    llm_config: Optional[dict] = None,
) -> GraphTask:
    """Convenience constructor for deferred extraction tasks."""
    return GraphTask(
        task_id=str(uuid4()),
        turn_id=turn_id,
        conversation_id=conversation_id,
        source_agent=source_agent,
        relationships=[],
        extraction_text=extraction_text,
        system_prompt_id=_prompt_id(system_prompt),
        llm_config=llm_config,
    )
