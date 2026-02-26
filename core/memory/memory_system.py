"""
core/memory/memory_system.py

Main entry point for the AlphaMesh multi-tenant financial memory system.

Key design principles (aligned with actual Cognee API):
  - cognee.add()     handles dataset creation internally — no manual dataset management
  - cognee.cognify() with our custom pipeline handles graph building
  - cognee.search()  uses node_type=NodeSet, node_name=[...] for per-user filtering
  - setup()          initializes Cognee's DB tables (idempotent)
  - get_default_user() provides the Cognee system user for setup tasks
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import cognee
from cognee.modules.engine.operations.setup import setup as cognee_setup
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.search.types import SearchType
from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor

from core.memory.exceptions import (
    DatasetInitError,
    IngestionError,
    QueryError,
    MemorySystemError,
)
from core.memory.nodeset_manager import (
    DATASET_NAME,
    initialize_cognee,
    get_or_create_global_nodeset,
    get_or_create_user_nodeset,
    get_user_nodeset_names,
    get_user_nodeset_name,
    GLOBAL_NODESET_NAME
)
from core.memory.pipeline_tasks import build_financial_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight context object for per-user state
# ---------------------------------------------------------------------------


@dataclass
class UserMemoryContext:
    """Cached resolution of a user's NodeSet context."""
    user_email: str
    nodeset_name: str
    user_nodeset: NodeSet
    global_nodeset: NodeSet
    allowed_nodeset_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.allowed_nodeset_names = [self.global_nodeset.name, self.user_nodeset.name]


# ---------------------------------------------------------------------------
# Ingestion item for batch ingestion
# ---------------------------------------------------------------------------


@dataclass
class IngestionItem:
    """
    One piece of content to batch-ingest.

    Fields:
        content:    Raw text or file path to add.
        data_type:  Cognee data type hint ("text", "pdf", etc.). Default "text".
        user_email: If set, this item belongs to a specific user (creates their NodeSet).
                    If None, treated as shared GLOBAL data.
    """
    content: str
    data_type: str = "text"
    user_email: Optional[str] = None


# ---------------------------------------------------------------------------
# Main memory system class
# ---------------------------------------------------------------------------


class FinancialMemorySystem:
    """
    Multi-tenant financial memory system built on Cognee.

    Manages one shared dataset with per-user NodeSet isolation.
    All public methods validate inputs, raise typed exceptions, and log
    appropriately — no silent failures.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._global_nodeset: Optional[NodeSet] = None
        self._user_context_cache: dict[str, UserMemoryContext] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize the memory system.

        - Calls Cognee setup() to create DB tables (idempotent)
        - Creates (or loads) the GLOBAL NodeSet

        Must be called before any other method. Safe to call multiple times.

        Raises:
            DatasetInitError: If setup fails.
            NodeSetCreationError: If GLOBAL NodeSet creation fails.
        """
        if self._initialized:
            logger.debug("FinancialMemorySystem already initialized — skipping.")
            return

        logger.info("Initializing FinancialMemorySystem …")

        # Initialize Cognee DB tables
        await initialize_cognee()

        # Ensure GLOBAL NodeSet exists in the graph
        self._global_nodeset = await get_or_create_global_nodeset()

        self._initialized = True
        logger.info(
            "FinancialMemorySystem ready. Dataset='%s', GLOBAL NodeSet id=%s.",
            DATASET_NAME, self._global_nodeset.id,
        )

    def _require_initialized(self) -> None:
        """Raise if initialize() has not been called."""
        if not self._initialized:
            raise MemorySystemError(
                "FinancialMemorySystem.initialize() must be called before use."
            )

    # ------------------------------------------------------------------
    # User context
    # ------------------------------------------------------------------

    async def get_user_context(self, user_email: str) -> UserMemoryContext:
        """
        Resolve and cache a user's NodeSet context.

        Creates the USER_<hash> NodeSet if it does not yet exist.

        Args:
            user_email: The authenticated user's email.

        Returns:
            UserMemoryContext with .allowed_nodeset_names = ["GLOBAL", "USER_<hash>"].
        """
        self._require_initialized()
        normalized = user_email.strip().lower()

        if normalized in self._user_context_cache:
            return self._user_context_cache[normalized]

        nodeset_name, user_nodeset = await get_or_create_user_nodeset(normalized)
        global_nodeset = self._global_nodeset or await get_or_create_global_nodeset()

        ctx = UserMemoryContext(
            user_email=normalized,
            nodeset_name=nodeset_name,
            user_nodeset=user_nodeset,
            global_nodeset=global_nodeset,
        )
        self._user_context_cache[normalized] = ctx
        logger.info(
            "User context resolved for '%s' → NodeSet '%s'.", normalized, nodeset_name
        )
        return ctx

    # ------------------------------------------------------------------
    # Ingestion helpers — cognee.add() handles dataset creation internally
    # ------------------------------------------------------------------

    async def _add_to_cognee(self, text: str, node_set: list[str] | None = None) -> None:
        """Internal: call cognee.add() and propagate errors with typed exceptions."""
        try:
            await cognee.add(text, dataset_name=DATASET_NAME, node_set=node_set)
        except Exception as exc:
            raise IngestionError(str(exc)) from exc

    async def ingest_conversation(
        self,
        user_email: str,
        messages: List[dict],
    ) -> None:
        """
        Ingest a conversation session.

        The conversation is formatted as plain text and classified as USER-scoped
        by the LLM during cognify().

        Args:
            user_email: Owner of this conversation.
            messages:   List of {role, content} dicts (OpenAI-style turns).

        Raises:
            IngestionError: If cognee.add() fails.
            ValueError: If messages is empty or malformed.
        """
        self._require_initialized()

        if not messages:
            raise ValueError("messages must be a non-empty list.")

        # Ensure the user's NodeSet is pre-created before cognify runs
        await self.get_user_context(user_email)

        lines = [f"[{msg.get('role', 'unknown').upper()}]: {msg.get('content', '')}"
                 for msg in messages]
        text = "\n".join(lines)

        logger.info(
            "Ingesting conversation for '%s' (%d messages, %d chars).",
            user_email, len(messages), len(text),
        )
        nodeset_name = get_user_nodeset_name(user_email)
        await self._add_to_cognee(text, node_set=[nodeset_name])


    # ------------------------------------------------------------------
    # Batch ingestion
    # ------------------------------------------------------------------

    async def ingest_batch(
        self,
        items: List[IngestionItem],
        concurrency_limit: int = 5,
    ) -> int:
        """
        Ingest multiple items concurrently with a concurrency cap.

        Args:
            items:             List of IngestionItem to ingest.
            concurrency_limit: Max parallel cognee.add() calls (default 5).

        Returns:
            Number of items successfully ingested.

        Raises:
            IngestionError: If any individual item fails (others continue).
        """
        self._require_initialized()

        if not items:
            return 0

        semaphore = asyncio.Semaphore(concurrency_limit)
        errors: list[tuple[int, Exception]] = []
        success_count = 0

        async def _ingest_one(idx: int, item: IngestionItem) -> None:
            nonlocal success_count
            async with semaphore:
                try:
                    if item.user_email:
                        await self.get_user_context(item.user_email)
                        ns_name = get_user_nodeset_name(item.user_email)
                        await self._add_to_cognee(item.content, node_set=[ns_name])
                    else:
                        await self._add_to_cognee(item.content, node_set=[GLOBAL_NODESET_NAME])
                    
                    success_count += 1
                    logger.debug("Batch item %d ingested (%d chars).", idx, len(item.content))
                except Exception as exc:
                    errors.append((idx, exc))
                    logger.error("Batch item %d failed: %s", idx, exc)

        await asyncio.gather(*[_ingest_one(i, item) for i, item in enumerate(items)])

        if errors:
            failed = [str(i) for i, _ in errors]
            raise IngestionError(
                f"{len(errors)} item(s) failed (indices: {', '.join(failed)}). "
                f"{success_count} succeeded."
            )

        logger.info("Batch ingestion complete: %d items.", success_count)
        return success_count

    # ------------------------------------------------------------------
    # Cognify
    # ------------------------------------------------------------------

    async def cognify(
        self,
        run_in_background: bool = False,
        chunks_per_batch: int = 100,
        chunk_size: Optional[int] = None,
    ) -> Any:
        """
        Run cognify over the entire dataset with the custom financial pipeline.

        Uses:
          - FinancialKnowledgeGraph as the graph_model (via build_financial_pipeline)
          - Constant system prompt injected
          - assign_nodeset_from_target task inserted after entity extraction
            to read the correct user NodeSet from the document lineage.

        Args:
            run_in_background:  If True, returns immediately (fire-and-forget).
            chunks_per_batch:   Batching parameter.
            chunk_size:         Max tokens per chunk (auto if None).

        Returns:
            PipelineRunInfo dict (blocking) or task handle (background).
        """
        self._require_initialized()

        logger.info(
            "Starting global cognify for all ingested data (background=%s).", run_in_background
        )

        tasks = await build_financial_pipeline(
            chunks_per_batch=chunks_per_batch,
            chunk_size=chunk_size,
        )

        try:
            pipeline_executor_func = get_pipeline_executor(
                run_in_background=run_in_background
            )
            result = await pipeline_executor_func(
                pipeline=run_pipeline,
                tasks=tasks,
                datasets=DATASET_NAME,
                pipeline_name="financial_cognify_pipeline",
            )
            logger.info("Global cognify completed.")
            return result
        except Exception as exc:
            raise MemorySystemError(f"Cognify failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Memify
    # ------------------------------------------------------------------

    async def memify(self) -> Any:
        """
        Run Cognee's memify for the shared dataset.
        """
        self._require_initialized()

        logger.info("Starting memify for global dataset.")
        try:
            result = await cognee.memify(datasets=DATASET_NAME)
            logger.info("Memify completed for global dataset.")
            return result
        except Exception as exc:
            raise MemorySystemError(f"Memify failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Query — strictly filtered per user
    # ------------------------------------------------------------------

    async def query(
        self,
        user_email: str,
        query_text: str,
        search_type: SearchType = SearchType.GRAPH_COMPLETION,
        top_k: int = 10,
    ) -> List[Any]:
        """
        Query the knowledge graph with strict NodeSet isolation.

        Uses Cognee's node_type + node_name filtering to restrict results to
        ONLY the GLOBAL NodeSet and the user's own USER_<hash> NodeSet.

        Under no circumstances will this query include another user's NodeSet.

        Args:
            user_email:  Authenticated user's email.
            query_text:  Natural language query string.
            search_type: Cognee search type (default GRAPH_COMPLETION).
            top_k:       Max results to return.

        Returns:
            List of search results.

        Raises:
            QueryError: If query fails.
            ValueError: If query_text is empty.
        """
        self._require_initialized()

        if not query_text or not query_text.strip():
            raise ValueError("query_text must not be empty.")

        # Deterministically derive the two authorized nodeset names — no DB call needed
        nodeset_names = get_user_nodeset_names(user_email)

        logger.info(
            "Query for '%s': type=%s, nodesets=%s, text='%.100s'",
            user_email,
            search_type.value if hasattr(search_type, "value") else search_type,
            nodeset_names,
            query_text,
        )

        try:
            # Cognee search filters by node_type=NodeSet and node_name=[names]
            # This ensures the graph traversal starts ONLY from the allowed NodeSet nodes
            results = await cognee.search(
                query_text=query_text,
                query_type=search_type,
                datasets=DATASET_NAME,
                node_type=NodeSet,
                node_name=nodeset_names,  # EXACTLY ["GLOBAL", "USER_<hash>"]
                top_k=top_k,
            )
            result_list = results or []
            logger.info(
                "Query returned %d results for '%s'.", len(result_list), user_email
            )
            return result_list

        except Exception as exc:
            raise QueryError(str(exc)) from exc
