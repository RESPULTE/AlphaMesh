"""
core/memory/financial_retriever.py

Custom Cognee retriever for the AlphaMesh financial knowledge graph.

Key capabilities
----------------
1. **Scoped parallel search** — up to 3 concurrent `brute_force_triplet_search`
   calls, each targeting a different NodeSet scope (company, sector, market).
   The scope is determined entirely by the *caller* (see `QueryScope` and
   `FinancialGraphRetriever.__init__`); no heuristic detection happens here.
   This is intentional: a future orchestration agent will resolve query intent
   upstream and pass the resolved scope and entity name directly.

2. **Recency reranking** — after merging deduplicated edges from all sub-searches
   the edges are sorted by the timestamp field named in `RERANK_TIMESTAMP_FIELD`.
   Change that constant to switch to a different timestamp attribute without
   touching any other code.

3. **Source citations** — every result object is augmented with a `citations`
   list built from node attributes (`url`, `text`) so that answers are
   fully traceable back to source documents.

4. **User isolation** — every sub-search always includes the caller's own
   `user_nodeset_names`; the Market / sector nodesets are *additive*.  No
   cross-user NodeSet name can ever appear in the `node_name` filter.

Privacy contract
----------------
`user_nodeset_names` is always ["Market", "USER_<hash>"].  Sub-search helpers
append sector / company names on top — they never replace the user slice.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
)

from core.memory.graph.models import GLOBAL_NODESET_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# The DataPoint attribute used for recency reranking.
# Change this single value if the timestamp field name changes in the schema.
RERANK_TIMESTAMP_FIELD: str = "updated_at"

# Fallback timestamp used when a node has no timestamp — sorts to the bottom.
_EPOCH = datetime.min


# ---------------------------------------------------------------------------
# Query scope
# ---------------------------------------------------------------------------


class QueryScope(str, Enum):
    """
    Defines how broadly the retriever fans out its parallel sub-searches.

    IMPORTANT: The caller is responsible for determining the correct scope
    based on the user's query intent. This class is a *parameter*, not an
    internal heuristic. A future orchestration agent will resolve the scope
    upstream and pass it in directly.

    Values
    ------
    COMPANY
        Fires three sub-searches: company anchor → its sector → the full
        market.  Use when the user is asking about a specific company.

    SECTOR
        Fires two sub-searches: the named sector → the full market.
        Use when the user is asking a sector-level question.

    MARKET
        Fires a single sub-search against the market nodeset plus the user's
        private nodeset.  Use for broad market-wide questions.
    """

    COMPANY = "company"
    SECTOR = "sector"
    MARKET = "market"


# ---------------------------------------------------------------------------
# Citation & context output types
# ---------------------------------------------------------------------------


class Citation(NamedTuple):
    """Traceable source reference extracted from a graph node's attributes."""

    source_url: Optional[str]
    chunk_text: Optional[str]
    node_id: Optional[str]


class RetrievalContext(NamedTuple):
    """Combined output of `get_context_from_objects`."""

    context_text: str
    citations: List[Citation]


# ---------------------------------------------------------------------------
# Reranking — standalone function for easy extension / replacement
# ---------------------------------------------------------------------------


def rerank_by_recency(edges: List[Edge], top_k: int) -> List[Edge]:
    """
    Sort a list of graph edges by the recency of their endpoint nodes and
    return the top-k entries.

    The recency score is the *maximum* of the `RERANK_TIMESTAMP_FIELD`
    value found on either endpoint node.  If neither node carries a timestamp
    the edge is treated as the oldest possible (``datetime.min``) and sinks
    to the bottom of the ranking.

    Modifying `RERANK_TIMESTAMP_FIELD` (module-level constant) is the only
    change needed to switch to a different timestamp attribute.

    Args:
        edges: Flat list of `Edge` objects to rerank. May include duplicates;
               deduplication should happen *before* calling this function.
        top_k: Number of top-ranked edges to return.

    Returns:
        Up to ``top_k`` edges ordered from most-recent to least-recent.
    """

    def _best_ts(edge: Edge) -> datetime:
        ts = _EPOCH
        for node in (edge.node1, edge.node2):
            if node is None:
                continue
            attrs = getattr(node, "attributes", {}) or {}
            raw = attrs.get(RERANK_TIMESTAMP_FIELD)
            if isinstance(raw, datetime) and raw > ts:
                ts = raw
        return ts

    ranked = sorted(edges, key=_best_ts, reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------


class FinancialGraphRetriever(GraphCompletionRetriever):
    """
    Graph-completion retriever with multi-scope parallel search, recency
    reranking, and source citation extraction.

    Inherits all LLM completion logic from `GraphCompletionRetriever` and
    overrides only the retrieval and context-building stages.

    Scope fan-out (controlled by `query_scope`)
    -------------------------------------------
    COMPANY  → 3 concurrent searches: company anchor + sector + market
    SECTOR   → 2 concurrent searches: sector + market
    MARKET   → 1 search:              market + user private nodeset

    Privacy guarantee
    -----------------
    ``user_nodeset_names`` (["Market", "USER_<hash>"]) is always included
    in every sub-search's ``node_name`` filter.  Sector / company names are
    *appended*; they never substitute the user slice.  No other user's
    NodeSet name can reach this retriever.

    Parameters
    ----------
    user_nodeset_names:
        The two nodeset names the user is authorised to query, i.e.
        ["Market", "USER_<hash>"].  Derived by ``get_user_nodeset_names()``.
    query_scope:
        `QueryScope` enum value.  Determined entirely by the caller —
        typically the orchestration layer or `FinancialMemorySystem.query()`.
    entity_name:
        For COMPANY scope: the ticker or company name used to narrow the
        company-anchor sub-search.
        For SECTOR scope: the sector name used to anchor the sector sub-search.
        Ignored for MARKET scope.  May be ``None``.
    system_prompt:
        Optional runtime override for the LLM system prompt.  When supplied
        it replaces the default ``FINANCIAL_SEARCH_SYSTEM_PROMPT``.
    top_k:
        Maximum number of reranked edges to return (and feed into the LLM).
    All other keyword arguments are forwarded to `GraphCompletionRetriever`.
    """

    def __init__(
        self,
        user_nodeset_names: List[str],
        query_scope: QueryScope = QueryScope.MARKET,
        entity_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        top_k: int = 10,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        node_type: Optional[Type] = NodeSet,
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 3.5,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ) -> None:
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=user_nodeset_names,  # base filter; overridden per sub-search
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            session_id=session_id,
            response_model=response_model,
        )
        self._user_nodeset_names: List[str] = list(user_nodeset_names)
        self.query_scope: QueryScope = query_scope
        self.entity_name: Optional[str] = entity_name

    # ------------------------------------------------------------------
    # Private sub-search helpers
    # ------------------------------------------------------------------

    def _build_node_names(self, *extra: str) -> List[str]:
        """
        Compose a node_name filter list that always includes the user's
        authorised nodesets and appends any additional scope names.

        Privacy note: extra names are appended, never substituted.  This
        means the user's private nodeset is always in the filter.
        """
        combined = list(self._user_nodeset_names)
        for name in extra:
            if name and name not in combined:
                combined.append(name)
        return combined

    async def _triplet_search(self, node_names: List[str], query: str) -> List[Edge]:
        """
        Run a single `brute_force_triplet_search` call with the given nodeset
        filter and return the raw edges (or empty list on failure).
        """
        collections = self._get_vector_index_collections() or None
        try:
            results = await brute_force_triplet_search(
                query=query,
                top_k=self.top_k,
                collections=collections,
                node_type=self.node_type,
                node_name=node_names,
                wide_search_top_k=self.wide_search_top_k,
                triplet_distance_penalty=self.triplet_distance_penalty,
            )
            return results if isinstance(results, list) else []
        except Exception as exc:
            logger.warning("Sub-search failed for node_names=%s: %s", node_names, exc)
            return []

    async def _search_market_scope(self, query: str) -> List[Edge]:
        """
        Single sub-search anchored to the Market nodeset plus the user's
        private nodeset.

        Used directly for MARKET queries and as the final stage for COMPANY
        and SECTOR queries to capture broad market-level influence.
        """
        node_names = self._build_node_names(GLOBAL_NODESET_NAME)
        return await self._triplet_search(node_names, query)

    async def _search_sector_scope(self, query: str, sector_name: str) -> List[Edge]:
        """
        Sub-search anchored to a specific sector nodeset, plus Market and
        the user's private nodeset.

        Args:
            sector_name: The exact sector NodeSet name (e.g. "Information Technology").
        """
        node_names = self._build_node_names(sector_name, GLOBAL_NODESET_NAME)
        return await self._triplet_search(node_names, query)

    async def _search_company_scope(self, query: str, company_name: str) -> List[Edge]:
        """
        Sub-search anchored to the company's entity name (ticker or full name),
        plus the user's private nodeset.

        The company sub-search does NOT pin to a sector nodeset here — sector
        and market are handled by separate concurrent calls in
        `_retrieve_for_company()`.  The company node is reachable from the
        user nodeset via `belongs_to_set` traversal.

        Args:
            company_name: Ticker symbol or company full name.
        """
        node_names = self._build_node_names()
        # We pass entity_name as a secondary text hint inside the query itself
        # so the vector search picks up the right embeddings; the actual
        # NodeSet anchor is the user + market nodesets (company nodes hang off them).
        augmented_query = f"{company_name}: {query}"
        return await self._triplet_search(node_names, augmented_query)

    # ------------------------------------------------------------------
    # Fan-out coordinators per scope
    # ------------------------------------------------------------------

    async def _retrieve_for_company(self, query: str) -> List[Edge]:
        """
        Fan out to 3 concurrent sub-searches:
          1. Company-anchored (narrowed vector search via augmented query)
          2. Sector-anchored  (resolved via entity_name → known sector name)
          3. Market-wide

        The sector name from `entity_name` is used directly as a NodeSet
        anchor. At ingestion time every Company node is attached to its
        sector NodeSet via `belongs_to_set`, so using the sector name here
        pulls in that sub-graph along with market-level events.
        """
        sector_name = self.entity_name or GLOBAL_NODESET_NAME
        company_name = self.entity_name or ""

        results = await asyncio.gather(
            self._search_company_scope(query, company_name),
            self._search_sector_scope(query, sector_name),
            self._search_market_scope(query),
        )
        return self._merge_and_deduplicate(
            list(results[0]) + list(results[1]) + list(results[2])
        )

    async def _retrieve_for_sector(self, query: str) -> List[Edge]:
        """
        Fan out to 2 concurrent sub-searches:
          1. Sector-anchored
          2. Market-wide
        """
        sector_name = self.entity_name or GLOBAL_NODESET_NAME

        results = await asyncio.gather(
            self._search_sector_scope(query, sector_name),
            self._search_market_scope(query),
        )
        return self._merge_and_deduplicate(list(results[0]) + list(results[1]))

    async def _retrieve_for_market(self, query: str) -> List[Edge]:
        """Single market-wide sub-search (no fan-out needed)."""
        edges = await self._search_market_scope(query)
        return self._merge_and_deduplicate(edges)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_and_deduplicate(edges: List[Edge]) -> List[Edge]:
        """
        Remove duplicate edges from the merged result set.

        An edge is considered a duplicate if both its node IDs and relationship
        name are identical.  The first occurrence is kept.
        """
        seen: set[Tuple] = set()
        unique: List[Edge] = []
        for edge in edges:
            n1_id = getattr(edge.node1, "id", None)
            n2_id = getattr(edge.node2, "id", None)
            rel = getattr(edge, "relationship_name", None) or (
                edge.attributes.get("relationship_name") if edge.attributes else None
            )
            key = (str(n1_id), str(n2_id), str(rel))
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        return unique

    # ------------------------------------------------------------------
    # BaseRetriever overrides
    # ------------------------------------------------------------------

    async def get_retrieved_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
    ) -> List[Edge]:
        """
        Execute the scoped parallel sub-searches, merge, deduplicate, and
        rerank by recency.

        The number of concurrent sub-searches is determined by `query_scope`:
          - COMPANY → 3 concurrent
          - SECTOR  → 2 concurrent
          - MARKET  → 1 (no fan-out)

        Args:
            query:       Single query string (mutually exclusive with query_batch).
            query_batch: Batch mode is not supported by this retriever; raises
                         ``NotImplementedError`` if supplied.

        Returns:
            Up to ``top_k`` Edge objects, ordered by recency (most recent first).
        """
        if query_batch:
            raise NotImplementedError(
                "FinancialGraphRetriever does not support batch queries."
            )
        if not query:
            return []

        logger.info(
            "FinancialGraphRetriever: scope=%s, entity=%s, query='%.80s'",
            self.query_scope,
            self.entity_name,
            query,
        )

        if self.query_scope == QueryScope.COMPANY:
            edges = await self._retrieve_for_company(query)
        elif self.query_scope == QueryScope.SECTOR:
            edges = await self._retrieve_for_sector(query)
        else:
            edges = await self._retrieve_for_market(query)

        reranked = rerank_by_recency(edges, self.top_k)
        logger.info(
            "FinancialGraphRetriever: %d unique edges → %d after rerank",
            len(edges),
            len(reranked),
        )
        return reranked

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
    ) -> RetrievalContext:
        """
        Convert retrieved edges to a text context string and extract source
        citations from node attributes.

        Citations are built from each unique node encountered across all edges.
        A citation is emitted when a node carries a ``url`` or ``text``
        attribute (populated during ingestion from the original document
        metadata and chunk content respectively).

        Args:
            query:            The original user query (unused here, passed for API compat).
            query_batch:      Not supported; ignored.
            retrieved_objects: The list of Edge objects from `get_retrieved_objects`.

        Returns:
            `RetrievalContext` namedtuple with:
              - ``context_text``: Human-readable graph context for the LLM.
              - ``citations``:    List of `Citation` objects for traceability.
        """
        edges: List[Edge] = retrieved_objects or []

        if not edges:
            return RetrievalContext(context_text="", citations=[])

        context_text = await self.resolve_edges_to_text(edges)
        citations = self._extract_citations(edges)
        return RetrievalContext(context_text=context_text, citations=citations)

    @staticmethod
    def _extract_citations(edges: List[Edge]) -> List[Citation]:
        """
        Walk all unique nodes in the edge list and produce one `Citation`
        per node that carries traceable document metadata.

        Fields sourced from node attributes:
          - ``url``  → Citation.source_url
          - ``text`` → Citation.chunk_text (the exact ingested text chunk)
          - ``id``   → Citation.node_id

        Only nodes with at least one of ``url`` or ``text`` are included.
        """
        seen_ids: set[str] = set()
        citations: List[Citation] = []

        for edge in edges:
            for node in (edge.node1, edge.node2):
                if node is None:
                    continue
                node_id = str(getattr(node, "id", "") or "")
                if node_id in seen_ids:
                    continue
                seen_ids.add(node_id)

                attrs: Dict[str, Any] = getattr(node, "attributes", {}) or {}
                url = attrs.get("url") or attrs.get("source_url")
                text = attrs.get("text") or attrs.get("chunk_text")

                if url or text:
                    citations.append(
                        Citation(
                            source_url=url,
                            chunk_text=text,
                            node_id=node_id or None,
                        )
                    )

        return citations

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate an LLM completion and attach source citations to the result.

        Accepts a `RetrievalContext` (from `get_context_from_objects`) or a
        plain string (fallback for compatibility).  Citations are appended as
        structured metadata on every returned result object.

        Args:
            query:            The original user query.
            query_batch:      Not supported.
            retrieved_objects: Raw edges (forwarded to parent for session logic).
            context:          `RetrievalContext` namedtuple or plain str.

        Returns:
            List of result dicts, each with shape::

                {
                    "answer":    <str>,          # LLM completion text
                    "citations": [               # traceable sources
                        {
                            "source_url": <str|None>,
                            "chunk_text":  <str|None>,
                            "node_id":     <str|None>,
                        },
                        ...
                    ]
                }
        """
        if isinstance(context, RetrievalContext):
            context_text = context.context_text
            citations = context.citations
        else:
            context_text = context or ""
            citations = []

        # Delegate to parent for session-aware LLM completion
        raw_completions = await super().get_completion_from_context(
            query=query,
            query_batch=query_batch,
            retrieved_objects=retrieved_objects,
            context=context_text,
        )

        # Wrap each completion with citation metadata
        results: List[Dict[str, Any]] = []
        for completion in raw_completions:
            results.append(
                {
                    "answer": completion,
                    "citations": [c._asdict() for c in citations],
                }
            )
        return results

    async def get_completion(
        self,
        query: Optional[str] = None,
        only_context: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Full pipeline: retrieve → build context + citations → LLM completion.

        Returns a list of result dicts (see `get_completion_from_context`).
        """
        retrieved_objects = await self.get_retrieved_objects(query=query)
        context = await self.get_context_from_objects(
            query=query, retrieved_objects=retrieved_objects
        )
        if only_context:
            return {
                "context_text": context.context_text,
                "citations": [c._asdict() for c in context.citations],
            }
        return await self.get_completion_from_context(
            query=query,
            retrieved_objects=retrieved_objects,
            context=context,
        )
