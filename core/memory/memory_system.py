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
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.operations.setup import setup
from cognee.modules.run_custom_pipeline import run_custom_pipeline

from core.memory.exceptions import (
    DatasetInitError,
    IngestionError,
    MemorySystemError,
    QueryError,
)
from core.memory.financial_retriever import FinancialGraphRetriever, QueryScope
from core.memory.graph_models import DATASET_NAME
from core.memory.nodeset_manager import (
    GLOBAL_NODESET_NAME,
    get_or_create_all_sector_nodesets,
    get_or_create_global_nodeset,
    get_or_create_user_nodeset,
    get_user_nodeset_name,
    get_user_nodeset_names,
)

# Initialize predefined Sector entities
from core.memory.pipeline_tasks import (
    build_financial_pipeline,
    build_lean_document_pipeline,
)
from core.memory.prompts import get_search_system_prompt

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


class UserContextRecord(NamedTuple):
    """
    One user-specific interest node returned by
    :meth:`FinancialMemorySystem.get_user_knowledge_context`.

    Fields
    ------
    node_id:
        The graph node's UUID string.
    node_type:
        ``"UserInvestmentInterest"`` or ``"UserLearningInterest"``.
    status:
        Status string from the linked grouping node, e.g. ``"Bought"``
        or ``"Understood"``.
    reason:
        Textual rationale / question stored on the interest node.
    updated_at:
        Timestamp parsed from the node's ``updated_at`` property, or
        ``None`` when the property is absent.
    raw_properties:
        Full property dict returned from the graph (diagnostic use).
    """

    node_id: str
    node_type: str
    status: str
    reason: str
    updated_at: Optional[datetime]
    targets: List[Dict[str, Any]]
    supporting_events: List[Dict[str, Any]]
    threatening_events: List[Dict[str, Any]]
    raw_properties: Dict[str, Any]


# ---------------------------------------------------------------------------
# User context retrieval — module-level constants (shared, not instance state)
# ---------------------------------------------------------------------------

# Cache keys for the nested _user_context_cache
USER_NODESET_CONTEXT = "nodeset_context"
USER_KNOWLEDGE_CONTEXT = "knowledge_context"

# Priority order for ranking interest nodes by status (lower = higher ranked).
_STATUS_RANK: Dict[str, int] = {
    "Bought": 0,
    "Interested": 1,
    "Understood": 2,
    "Confused": 3,
    "Sold": 4,
    "Avoids": 5,
    "Not Interested": 6,
}

# Specialized queries for user interest types
_USER_INVESTMENT_CONTEXT_CYPHER = """
MATCH (ns:NodeSet {name: $nodeset_name})
MATCH (ns)-[:belongs_to_set|BELONGS_TO_SET]-(n)
WHERE n.type = 'UserInvestmentInterest'
OPTIONAL MATCH (n)-[:status|STATUS]->(s)
RETURN
    n.id          AS node_id,
    n.type        AS node_type,
    n.reason      AS reason,
    n.updated_at  AS updated_at,
    s.status      AS status,
    properties(n) AS props,
    [(n)-[:targets|TARGETS]->(t) | {id: t.id, name: t.name, type: t.type}] AS targets,
    [(n)-[:supporting_events|SUPPORTING_EVENTS]->(se) | {id: se.id, name: se.name, type: se.type}] AS supporting_events,
    [(n)-[:threatening_events|THREATENING_EVENTS]->(te) | {id: te.id, name: te.name, type: te.type}] AS threatening_events
LIMIT 500
"""

_USER_LEARNING_CONTEXT_CYPHER = """
MATCH (ns:NodeSet {name: $nodeset_name})
MATCH (ns)-[:belongs_to_set|BELONGS_TO_SET]-(n)
WHERE n.type = 'UserLearningInterest'
OPTIONAL MATCH (n)-[:status|STATUS]->(s)
RETURN
    n.id          AS node_id,
    n.type        AS node_type,
    n.reason      AS reason,
    n.updated_at  AS updated_at,
    s.status      AS status,
    properties(n) AS props,
    [(n)-[:targets|TARGETS]->(t) | {id: t.id, name: t.name, type: t.type}] AS targets
LIMIT 500
"""


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


async def initialize_cognee() -> None:
    """
    Initialize Cognee's relational and vector databases.
    Must be called before any other operations. Idempotent.

    Raises:
        DatasetInitError: If setup fails.
    """
    try:
        await setup()
        logger.info("Cognee database setup complete.")
    except Exception as exc:
        raise DatasetInitError(str(exc)) from exc


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
        # Nested cache: { user_email: { "nodeset_context": UserMemoryContext, "knowledge_context": List[UserContextRecord] } }
        self._user_context_cache: dict[str, dict[str, Any]] = {}

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

        await get_or_create_all_sector_nodesets()

        self._initialized = True
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

        user_cache = self._user_context_cache.get(normalized)
        if user_cache and USER_NODESET_CONTEXT in user_cache:
            return user_cache[USER_NODESET_CONTEXT]

        nodeset_name, user_nodeset = await get_or_create_user_nodeset(normalized)
        global_nodeset = self._global_nodeset or await get_or_create_global_nodeset()

        ctx = UserMemoryContext(
            user_email=normalized,
            nodeset_name=nodeset_name,
            user_nodeset=user_nodeset,
            global_nodeset=global_nodeset,
        )

        if normalized not in self._user_context_cache:
            self._user_context_cache[normalized] = {}
        self._user_context_cache[normalized][USER_NODESET_CONTEXT] = ctx

        logger.info(
            "User context resolved for '%s' → NodeSet '%s'.", normalized, nodeset_name
        )
        return ctx

    async def get_user_knowledge_context(
        self,
        user_email: str,
        top_k: int = 25,
    ) -> List[UserContextRecord]:
        """
        Retrieve all user-specific interest nodes for a given user, ranked
        by status priority then recency.

        Internally reuses :meth:`get_user_context` so the nodeset lookup is
        served from the in-process cache — no redundant DB calls.

        The graph store is queried directly with a parameterized Cypher query
        so that the nodeset name filter is safe and driver-optimised.

        Ranking
        -------
        Primary key — status priority (ascending):
            ``Bought`` → ``Interested`` → ``Understood`` → ``Confused``
            → ``Sold`` → ``Avoids`` → ``Not Interested`` → unknown
        Secondary key — ``updated_at`` timestamp (descending, most recent first).
        Nodes without a timestamp rank last within their status tier.

        Parameters
        ----------
        user_email:
            The authenticated user's email address (the user ID used
            throughout the AlphaMesh codebase).
        top_k:
            Maximum number of records to return.  Defaults to 25.

        Returns
        -------
        List of :class:`UserContextRecord` ordered by status priority then recency.

        Raises
        ------
        QueryError: If the graph query fails.
        """
        ctx = await self.get_user_context(user_email)
        normalized = ctx.user_email

        # Check if the knowledge context is already cached
        user_cache = self._user_context_cache.get(normalized, {})
        if USER_KNOWLEDGE_CONTEXT in user_cache:
            cached_records = user_cache[USER_KNOWLEDGE_CONTEXT]
            logger.info(
                "get_user_knowledge_context: Cache hit for '%s' → %d records.",
                normalized,
                len(cached_records),
            )
            return cached_records[:top_k]

        nodeset_name = ctx.nodeset_name
        graph_engine = await get_graph_engine()

        try:
            # Run both queries in parallel
            investment_task = graph_engine.query(
                _USER_INVESTMENT_CONTEXT_CYPHER,
                {"nodeset_name": nodeset_name},
            )
            learning_task = graph_engine.query(
                _USER_LEARNING_CONTEXT_CYPHER,
                {"nodeset_name": nodeset_name},
            )

            investment_rows, learning_rows = await asyncio.gather(
                investment_task, learning_task
            )
        except Exception as exc:
            logger.error(
                "get_user_knowledge_context: parallel graph queries failed for '%s' (nodeset '%s'): %s",
                user_email,
                nodeset_name,
                exc,
            )
            raise QueryError(str(exc)) from exc

        records: List[UserContextRecord] = []

        # Parse investment rows (includes targets, supporting_events, threatening_events)
        for row in investment_rows:
            records.append(self._parse_user_context_row(row))

        # Parse learning rows (includes targets only)
        for row in learning_rows:
            records.append(self._parse_user_context_row(row))

        records.sort(
            key=lambda r: (
                _STATUS_RANK.get(r.status, 99),
                -(r.updated_at.timestamp() if r.updated_at else 0),
            )
        )

        # Cache the full list of records
        if normalized not in self._user_context_cache:
            self._user_context_cache[normalized] = {}
        self._user_context_cache[normalized][USER_KNOWLEDGE_CONTEXT] = records

        logger.info(
            "get_user_knowledge_context: %d records for '%s' → top-%d returned (cached).",
            len(records),
            normalized,
            top_k,
        )
        return records[:top_k]

    def _parse_user_context_row(self, row: Dict[str, Any]) -> UserContextRecord:
        """Helper to parse a single Cypher row into a UserContextRecord."""
        raw_ts = row.get("updated_at")
        parsed_ts: Optional[datetime] = None
        if isinstance(raw_ts, datetime):
            parsed_ts = raw_ts
        elif isinstance(raw_ts, (int, float)):
            try:
                # Cognee timestamps are often in milliseconds
                parsed_ts = datetime.utcfromtimestamp(raw_ts / 1000)
            except Exception:
                parsed_ts = None
        elif isinstance(raw_ts, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed_ts = datetime.strptime(raw_ts, fmt)
                    break
                except ValueError:
                    continue

        def clean_list(lst):
            if not lst:
                return []
            return [i for i in lst if i and i.get("id") is not None]

        return UserContextRecord(
            node_id=str(row.get("node_id") or ""),
            node_type=str(row.get("node_type") or ""),
            status=str(row.get("status") or ""),
            reason=str(row.get("reason") or ""),
            updated_at=parsed_ts,
            targets=clean_list(row.get("targets")),
            supporting_events=clean_list(row.get("supporting_events")),
            threatening_events=clean_list(row.get("threatening_events")),
            raw_properties=row.get("props") or {},
        )

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

    async def ingest_document_lean(
        self,
        ticker: str,
        report_type: str,
        content: str,
        period: Optional[str] = None,
        include_summaries: bool = True,
        is_global: bool = True,
    ) -> Any:
        """
        Ingest a financial document using the lean pipeline (no graph extraction).

        REPLACES the pattern of: ingest_financial_report() → cognify()
        for standard document ingestion use cases.

        The lean pipeline:
          - Chunks the document
          - Optionally generates terse financial summaries (80 tokens/chunk)
          - Embeds and indexes chunks for vector retrieval
          - Does NOT extract entities or build graph edges

        Graph construction happens lazily via conversation write-back as
        users query the system and the synthesiser extracts relationships.

        Args:
            ticker:            Stock ticker (e.g. "AAPL").
            report_type:       "10-K", "10-Q", "8-K", "annual", "quarterly".
            content:           Full text of the report.
            period:            Reporting period string (e.g. "Q3 2024"). Optional.
            include_summaries: If True, runs lean per-chunk summarisation.
            is_global:         True for public SEC filings (default).

        Returns:
            PipelineRunInfo from run_custom_pipeline.

        Raises:
            IngestionError:   If cognee.add() fails.
            MemorySystemError: If the pipeline fails.
        """
        self._require_initialized()

        if not ticker or not content:
            raise ValueError("ticker and content are required.")

        header = f"FINANCIAL REPORT\nTICKER: {ticker.upper()}\nTYPE: {report_type}\n"
        if period:
            header += f"PERIOD: {period}\n"
        text = header + "\n" + content

        node_set = [GLOBAL_NODESET_NAME] if is_global else None
        await self._add_to_cognee(text, node_set=node_set)

        logger.info(
            "Running lean pipeline for %s %s (%d chars, summaries=%s).",
            report_type,
            ticker.upper(),
            len(text),
            include_summaries,
        )

        tasks = await build_lean_document_pipeline(include_summaries=include_summaries)

        try:
            result = await run_custom_pipeline(
                tasks=tasks,
                dataset=DATASET_NAME,
                pipeline_name="lean_document_pipeline",
                incremental_loading=True,
            )
            logger.info(
                "Lean document pipeline completed for %s %s.", report_type, ticker
            )
            return result
        except Exception as exc:
            raise MemorySystemError(f"Lean document pipeline failed: {exc}") from exc

    async def ingest_conversation(
        self,
        user_email: str,
        messages: List[dict],
    ) -> None:
        """
        DEPRECATED: Conversation insights now flow into the knowledge graph via
        OrchestratorAgent._synthesize_node → run_conversation_writeback.

        Kept for backward compatibility. In the new architecture this method
        is a no-op — calling it will log a warning and return immediately.

        To write conversation entities to the graph, call run_conversation_writeback()
        from the synthesiser after each conversation turn.
        """
        logger.warning(
            "ingest_conversation() is deprecated. Conversation insights are now "
            "written to the graph via run_conversation_writeback() from the synthesiser. "
            "This call has no effect."
        )
        return

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
        chunk_size: Optional[int] = 500,
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
        query_scope: QueryScope = QueryScope.MARKET,
        entity_name: Optional[str] = None,
        top_k: int = 10,
        system_prompt_override: Optional[str] = None,
        session_id: Optional[str] = None,
        only_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph with strict NodeSet isolation.

        Internally delegates to `FinancialGraphRetriever` which fans out to up
        to 3 parallel sub-searches depending on `query_scope`, then reranks
        results by recency and attaches source citations to every answer.

        Under no circumstances will this query include another user's NodeSet.

        Parameters
        ----------
        user_email:
            Authenticated user's email address.
        query_text:
            Natural language query string.
        query_scope:
            `QueryScope` enum that determines the search fan-out strategy.
            Controlled entirely by the caller — typically the orchestration
            layer or a future agent preprocessor:

            * ``QueryScope.COMPANY``
                Fire 3 parallel searches: company anchor + its sector + market.
                Set ``entity_name`` to the ticker or company name.
            * ``QueryScope.SECTOR``
                Fire 2 parallel searches: the named sector + market.
                Set ``entity_name`` to the sector name.
            * ``QueryScope.MARKET``
                Single market-wide search.  ``entity_name`` is ignored.

        entity_name:
            For COMPANY scope: ticker symbol or full company name.
            For SECTOR  scope: sector name (must match a NodeSet in the graph).
            Ignored for MARKET scope.  Leave ``None`` when not applicable.
        top_k:
            Maximum number of reranked edges to return to the LLM.
        system_prompt_override:
            Optional runtime system prompt string.  When supplied it fully
            replaces ``FINANCIAL_SEARCH_SYSTEM_PROMPT``.  Intended for future
            orchestration agents that resolve user preferences (tone, verbosity)
            before the query reaches this layer.
        session_id:
            Optional identifier for session-based conversation caching.

        Returns
        -------
        List of result dicts, each with shape::

            {
                "answer":    <str>,
                "citations": [
                    {"source_url": ..., "chunk_text": ..., "node_id": ...},
                    ...
                ]
            }

        Raises
        ------
        QueryError: If retrieval or completion fails.
        ValueError: If query_text is empty.
        """
        self._require_initialized()

        if not query_text or not query_text.strip():
            raise ValueError("query_text must not be empty.")

        # Deterministically derive the two authorised nodeset names — no DB call needed
        nodeset_names = get_user_nodeset_names(user_email)

        system_prompt = get_search_system_prompt(system_prompt_override)

        logger.info(
            "Query for '%s': scope=%s, entity=%s, nodesets=%s, text='%.100s'",
            user_email,
            query_scope.value,
            entity_name,
            nodeset_names,
            query_text,
        )

        try:
            retriever = FinancialGraphRetriever(
                user_nodeset_names=nodeset_names,
                query_scope=query_scope,
                entity_name=entity_name,
                system_prompt=system_prompt,
                top_k=top_k,
                session_id=session_id,
            )
            result_list = await retriever.get_completion(
                query_text, only_context=only_context
            )
            result_list = result_list or []
            logger.info(
                "Query returned %d results for '%s'.", len(result_list), user_email
            )
            return result_list

        except Exception as exc:
            raise QueryError(str(exc)) from exc
