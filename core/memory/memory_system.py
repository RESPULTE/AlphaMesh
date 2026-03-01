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
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines import run_pipeline
from cognee.modules.run_custom_pipeline import run_custom_pipeline
from cognee.modules.search.types import SearchType
from core.memory.exceptions import IngestionError, MemorySystemError, QueryError
from core.memory.nodeset_manager import (
    DATASET_NAME,
    GLOBAL_NODESET_NAME,
    get_or_create_global_nodeset,
    get_or_create_user_nodeset,
    get_user_nodeset_name,
    get_user_nodeset_names,
    initialize_cognee,
)

# Initialize predefined Sector entities
from core.memory.graph_models import Sector
from cognee.tasks.storage.add_data_points import (
    add_data_points as cognee_add_dp,
)
from core.memory.pipeline_tasks import get_canonical_id
from core.memory.pipeline_tasks import build_financial_pipeline
from cognee.infrastructure.databases.graph import get_graph_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight context object for per-user state
# ---------------------------------------------------------------------------

SECTORS = {
    "Energy": "Companies involved in the exploration, production, and distribution of oil, gas, and renewable energy.",
    "Materials": "Includes chemical, construction material, glass, paper, forest product, and mining companies.",
    "Industrials": "Manufacturers and distributors of capital goods, including aerospace, defense, and machinery.",
    "Consumer Discretionary": "Businesses that sell non-essential goods and services, such as automotive, apparel, and leisure.",
    "Consumer Staples": "Essential product providers, including food, beverage, personal products, and household goods.",
    "Health Care": "Pharmaceuticals, biotechnology, medical devices, and healthcare service providers.",
    "Financials": "Banks, investment firms, insurance companies, and real estate finance entities.",
    "Information Technology": "Software, hardware, semiconductors, and IT service providers.",
    "Communication Services": "Telecommunications providers, media, entertainment, and interactive service companies.",
    "Utilities": "Providers of basic services including electricity, gas, and water.",
    "Real Estate": "Companies engaged in real estate development, management, and REITs.",
}


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
        self.graph_client: Optional[Any] = None

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

        # Initialize the graph client to attach on self
        self.graph_client = await get_graph_engine()
        self._initialized = True

        sector_nodes = []
        for name, description in SECTORS.items():
            s_node = Sector(name=name, description=description, related_to=[])
            s_node.id = get_canonical_id(name)
            s_node.belongs_to_set = [self._global_nodeset]
            sector_nodes.append(s_node)

        logger.info(
            "FinancialMemorySystem ready. Dataset='%s', GLOBAL NodeSet id=%s.",
            DATASET_NAME,
            self._global_nodeset.id,
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

    async def _add_to_cognee(
        self, text: str, node_set: list[str] | None = None
    ) -> None:
        """Internal: call cognee.add() and propagate errors with typed exceptions."""
        try:
            await cognee.add(text, dataset_name=DATASET_NAME, node_set=node_set)
        except Exception as exc:
            raise IngestionError(str(exc)) from exc

    async def ingest_news(
        self,
        articles: List[dict],
        is_global: bool = True,
    ) -> None:
        """
        Ingest financial news articles.

        Args:
            articles:  List of dicts with 'headline' and 'summary' keys.
            is_global: True (default) = shared GLOBAL data.

        Raises:
            IngestionError: On Cognee failure.
            ValueError: If articles list is empty or yields no valid content.
        """
        self._require_initialized()

        if not articles:
            raise ValueError("articles must be a non-empty list.")

        text_blocks: list[str] = []
        for art in articles:
            headline = art.get("headline", "")
            summary = art.get("summary", "")
            if not headline and not summary:
                logger.warning("Skipping article with no headline or summary.")
                continue
            parts = [f"HEADLINE: {headline}", f"SUMMARY: {summary}"]
            if art.get("source"):
                parts.append(f"SOURCE: {art['source']}")
            if art.get("published_at"):
                parts.append(f"DATE: {art['published_at']}")
            text_blocks.append("\n".join(parts))

        if not text_blocks:
            raise ValueError("No valid articles after filtering.")

        combined = "\n\n---\n\n".join(text_blocks)
        logger.info(
            "Ingesting %d news articles (%s, %d chars total).",
            len(text_blocks),
            "GLOBAL" if is_global else "USER",
            len(combined),
        )
        node_set = [GLOBAL_NODESET_NAME] if is_global else None
        await self._add_to_cognee(combined, node_set=node_set)

    async def ingest_financial_report(
        self,
        ticker: str,
        report_type: str,
        content: str,
        period: Optional[str] = None,
        is_global: bool = True,
    ) -> None:
        """
        Ingest an SEC filing or financial report.

        Args:
            ticker:      Stock ticker (e.g. "AAPL").
            report_type: "10-K", "10-Q", "8-K", "annual", "quarterly", "earnings".
            content:     Full text or summary of the report.
            period:      Reporting period (e.g. "Q3 2024").
            is_global:   True for public SEC filings (default).
        """
        self._require_initialized()

        if not ticker or not content:
            raise ValueError("ticker and content are required.")

        header = f"FINANCIAL REPORT\nTICKER: {ticker.upper()}\nTYPE: {report_type}\n"
        if period:
            header += f"PERIOD: {period}\n"
        text = header + "\n" + content

        logger.info(
            "Ingesting %s report for %s (%d chars).",
            report_type,
            ticker.upper(),
            len(text),
        )
        node_set = [GLOBAL_NODESET_NAME] if is_global else None
        await self._add_to_cognee(text, node_set=node_set)

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

        lines = [
            f"[{msg.get('role', 'unknown').upper()}]: {msg.get('content', '')}"
            for msg in messages
        ]
        text = "\n".join(lines)

        logger.info(
            "Ingesting conversation for '%s' (%d messages, %d chars).",
            user_email,
            len(messages),
            len(text),
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
                        await self._add_to_cognee(
                            item.content, node_set=[GLOBAL_NODESET_NAME]
                        )

                    success_count += 1
                    logger.debug(
                        "Batch item %d ingested (%d chars).", idx, len(item.content)
                    )
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
            "Starting global cognify for all ingested data (background=%s).",
            run_in_background,
        )

        tasks = await build_financial_pipeline(
            chunks_per_batch=chunks_per_batch,
            chunk_size=chunk_size,
        )

        try:
            result = await run_custom_pipeline(
                tasks=tasks,
                dataset=DATASET_NAME,
                pipeline_name="financial_cognify_pipeline",
                incremental_loading=True,
            )
            logger.info(
                "Global cognify completed. Entity merging ran as pipeline step 7."
            )
            return result
        except Exception as exc:
            raise MemorySystemError(f"Cognify failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Query — strictly filtered per user
    # ------------------------------------------------------------------

    async def query(
        self,
        user_email: str,
        query_text: str,
        search_type: SearchType = SearchType.GRAPH_COMPLETION,
        top_k: int = 10,
        only_context: bool = False,
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
                only_context=only_context,
            )
            result_list = results or []
            logger.info(
                "Query returned %d results for '%s'.", len(result_list), user_email
            )
            return result_list

        except Exception as exc:
            raise QueryError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Graph Edge Property and State Triggers
    # ------------------------------------------------------------------

    async def adjust_edge_property(
        self,
        source_id: str,
        target_id: str,
        edge_label: str,
        property_name: str,
        delta: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> float:
        """
        Gracefully increments or decrements a property of a graph edge.
        Clamps the updated value strictly between min_val and max_val.

        Designed for Multi-Agent systems: agents provide continuous feedback (e.g., +0.05 or -0.1)
        on relationship strength, and this handles the boundary controls deterministically.

        Args:
            graph_client: The Cognee graph engine adapter client.
            source_id: Node ID of the relationship source.
            target_id: Node ID of the relationship target.
            edge_label: The relationship type (e.g., 'HoldsThesis', 'SupportedBy').
            property_name: The edge property to adjust (e.g., 'conviction_level').
            delta: Float value to add to the existing property value (can be negative).
            min_val: Hard lower bound for the property.
            max_val: Hard upper bound for the property.

        Returns:
            The newly calculated and clamped property value as a float.
            Returns 0.0 if the edge was not found or execution failed but didn't crash.
        """
        self._require_initialized()
        graph_client = self.graph_client

        logger.info(
            "Adjusting '%s' for edge '%s' between '%s' and '%s' by delta %.2f.",
            property_name,
            edge_label,
            source_id,
            target_id,
            delta,
        )

        # Cypher implementation applying boundary controls
        # Uses type(e) to check edge label safely.
        query = f"""
        MATCH (s {{id: $source_id}})-[e]->(t {{id: $target_id}})
        WHERE type(e) = $edge_label
        WITH e, coalesce(e.{property_name}, 0.0) AS current_val
        WITH e, current_val, current_val + $delta AS raw_new_val
        WITH e,
             CASE
                WHEN raw_new_val < $min_val THEN $min_val
                WHEN raw_new_val > $max_val THEN $max_val
                ELSE raw_new_val
             END AS final_val
        SET e.{property_name} = final_val
        RETURN final_val AS new_val
        """
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_label": edge_label,
            "delta": float(delta),
            "min_val": float(min_val),
            "max_val": float(max_val),
        }

        try:
            # Resolving the execution method defensively based on typical Cognee adapter shapes
            execute_fn = None
            if hasattr(graph_client, "query"):
                execute_fn = graph_client.query
            elif hasattr(graph_client, "execute"):
                execute_fn = graph_client.execute
            elif hasattr(graph_client, "graph") and hasattr(
                graph_client.graph, "execute"
            ):
                execute_fn = graph_client.graph.execute

            if not execute_fn:
                logger.warning(
                    "Provided graph_client lacks a query() or execute() method for Cypher queries."
                )
                return 0.0

            results = await execute_fn(query, params)

            # Simple parse to return the updated value safely
            if results and isinstance(results, list) and len(results) > 0:
                first_record = results[0]
                if isinstance(first_record, dict) and "new_val" in first_record:
                    return float(first_record["new_val"])
                # Fallback for adapters returning simple tuples or scalars
                return float(first_record) if first_record is not None else 0.0

            logger.warning(
                "Edge '%s' not found between '%s' and '%s' for property update.",
                edge_label,
                source_id,
                target_id,
            )
            return 0.0

        except Exception as exc:
            logger.error("Failed to adjust edge property %s: %s", property_name, exc)
            raise MemorySystemError(f"Edge property adjustment failed: {exc}") from exc
