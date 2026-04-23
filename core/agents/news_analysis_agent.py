"""
core/agents/news_analysis_agent.py

News analysis agent with dual-store ingestion and chunk-level extraction.

Changes
-------
- Added `_rewrite_queries_node` as the first node in the LangGraph workflow.
  This node makes a lightweight structured-output LLM call to expand the
  orchestrator's already-tailored query into three domain-specific retrieval
  strings (company / sector / market) and immediately fires the memory
  retrieval as a background asyncio task stored on the state.
  The task is awaited later in `_rendezvous_node`, exactly as before.

- `run()` no longer reads `input_data.memory_task` � the news agent now
  self-manages its memory retrieval lifecycle entirely.  BaseAgentInput no
  longer carries a `memory_task` field.

- `_fetch_news_node` and `_ingest_articles_node` run concurrently with the
  memory retrieval task thanks to the background task created in
  `_rewrite_queries_node`.

- `_ingest_articles_node` now fires both Chroma retrieval queries (new chunks
  and existing chunks) in parallel via asyncio.gather, eliminating a
  sequential round-trip.

Graph topology:
  rewrite_queries ? fetch_news ? ingest_articles ? rendezvous ? analyse_news
                 ? (memory_task fires in background) ?
"""

from __future__ import annotations

import asyncio
import re as _re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.agents.base_agent import AbstractAgent
from core.agents.models.base_agent_models import AgentSentiment, BaseAgentInput
from core.agents.models.news_agent_models import (
    CitedSource,
    NewsAgentOutput,
    NewsAgentState,
)
from core.agents.news_fetcher import build_news_query, fetch_articles
from core.agents.prompts.news_agent_prompts import (
    NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT,
    NEWS_ANALYSIS_USER_PROMPT,
    NEWS_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
    NEWS_MEMORY_QUERY_REWRITE_SYSTEM_PROMPT,
)
from core.config import settings
from core.event_queue import publish_progress, publish_success
from core.logger import get_logger
from core.memory.graph.graph_queue import make_extraction_task
from core.memory.retrieval.models import MemoryContext, RetrievedChunk, RewrittenQueries
from core.services import service_manager

logger = get_logger(__name__)
_ENTITY_CACHE: Dict[str, Dict[Tuple[str, str], Tuple[str, str]]] = {}
_ENTITY_CACHE_TS: Dict[str, float] = {}


class NewsAnalysisStructuredOutput(BaseModel):
    analysis: str
    sentiment: AgentSentiment = Field(default_factory=AgentSentiment)


def _get_default_date_range(
    start_date: datetime | None,
    end_date: datetime | None,
    default_days_back: int = 30,
) -> Tuple[datetime, datetime]:
    """
    Fill in missing start/end dates with sensible defaults.

    Args:
        start_date: Requested start date (or None for default)
        end_date: Requested end date (or None for default)
        default_days_back: How many days back to default start_date (default: 1 month)

    Returns:
        Tuple of (start_date, end_date) with defaults applied
    """
    now_utc = datetime.now(timezone.utc)

    if end_date is None:
        end_date = now_utc
    if start_date is None:
        start_date = now_utc - timedelta(days=default_days_back)

    return start_date, end_date


def _constrain_date_range(
    start_date: date,
    end_date: date,
    api_limit_days: int = 28,
) -> Tuple[date, date]:
    """
    Constrain date range to valid API query bounds.

    - end_date cannot be in the future (capped at now)
    - start_date cannot exceed the API limit window (default 28 days)

    Args:
        start_date: Requested start date
        end_date: Requested end date
        api_limit_days: Maximum number of days in the past for API queries

    Returns:
        Tuple of (constrained_start, constrained_end) as date objects in YYYY-MM-DD format
    """
    # Use consistent timezone awareness for comparison
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    now = datetime.now().date()

    api_limit_date = now - timedelta(days=api_limit_days)

    start_date_only = start_date
    end_date_only = end_date

    if end_date_only > now:
        end_date_only = now
    if start_date_only < api_limit_date:
        start_date_only = api_limit_date

    return start_date_only, end_date_only


def _normalize_entity_tuple(
    name: Any,
    entity_type: Any,
) -> Tuple[str, str] | None:
    normalized_name = str(name or "").strip()
    normalized_type = str(entity_type or "").strip()
    if not normalized_name or not normalized_type:
        return None
    return normalized_name, normalized_type


def _coerce_entity_tuple(entity: Any) -> Tuple[str, str] | None:
    if isinstance(entity, dict):
        return _normalize_entity_tuple(
            entity.get("name") or entity.get("entity_name"),
            entity.get("entity_type"),
        )
    if isinstance(entity, (tuple, list)) and len(entity) >= 2:
        return _normalize_entity_tuple(entity[0], entity[1])
    return _normalize_entity_tuple(
        getattr(entity, "name", None),
        getattr(entity, "entity_type", None),
    )


def _get_cached_entities(conversation_id: str) -> List[Tuple[str, str]]:
    if not conversation_id:
        return []
    now = time.time()
    ts = _ENTITY_CACHE_TS.get(conversation_id)
    if ts is None or now - ts > settings.SUBGRAPH_TTL_SECONDS:
        _ENTITY_CACHE.pop(conversation_id, None)
        _ENTITY_CACHE_TS.pop(conversation_id, None)
        return []
    cache = _ENTITY_CACHE.setdefault(conversation_id, {})
    normalized_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for raw in list(cache.values()):
        normalized = _coerce_entity_tuple(raw)
        if normalized is None:
            continue
        normalized_cache[normalized] = normalized
    _ENTITY_CACHE[conversation_id] = normalized_cache
    return list(normalized_cache.values())


def _merge_cached_entities(
    conversation_id: str,
    entities: List[Any],
) -> None:
    if not conversation_id:
        return
    cache = _ENTITY_CACHE.setdefault(conversation_id, {})
    for entity in entities:
        normalized = _coerce_entity_tuple(entity)
        if normalized is None:
            continue
        cache[normalized] = normalized
    _ENTITY_CACHE_TS[conversation_id] = time.time()


def _build_deduplicated_sources(
    chunks: List[RetrievedChunk],
) -> Tuple[List[CitedSource], Dict[int, int]]:
    """
    Deduplicate chunks by article (title + url) and return:
      - sources: one CitedSource per unique article, numbered 1..N
      - chunk_to_source_id: maps the original chunk index (0-based) to its
        canonical source_id so context blocks can reference the right number.

    Multiple chunks from the same article get the same source_id.
    CitedSource.page_content accumulates all unique chunk texts for that
    article so the tooltip remains informative.
    """
    article_map: Dict[Tuple[str, str], Tuple[int, List[str]]] = {}
    chunk_to_source_id: Dict[int, int] = {}
    next_id = 1

    for chunk_idx, chunk in enumerate(chunks):
        metadata = chunk.metadata or {}
        title = metadata.get("article_title") or chunk.article_title or "Unknown Title"
        url = metadata.get("source_url") or chunk.source_url or ""
        key = (title, url)

        if key not in article_map:
            article_map[key] = (next_id, [chunk.text])
            next_id += 1
        else:
            sid, texts = article_map[key]
            if chunk.text not in texts:
                texts.append(chunk.text)

        chunk_to_source_id[chunk_idx] = article_map[key][0]

    sources: List[CitedSource] = []
    for (title, url), (source_id, texts) in article_map.items():
        sources.append(
            CitedSource(
                source_id=source_id,
                title=title,
                url=url,
                page_content="\n\n".join(texts),
            )
        )

    return sources, chunk_to_source_id


class NewsAnalysisAgent(AbstractAgent):
    """LangGraph-based news analysis agent with self-managed memory retrieval."""

    def __init__(self) -> None:
        super().__init__()
        self._llm = service_manager.get_agent()
        self._graph = self._build_graph()
        self._ingestor = service_manager.get_ingestor()

    @staticmethod
    def name() -> str:
        return "news_agent"

    @staticmethod
    def description() -> str:
        return (
            "Fetches and ingests recent news, retrieves relevant stored memory, "
            "and synthesizes a grounded financial analysis with cited sources. "
            "Best for: recent events, earnings, analyst ratings, macro news, "
            "company announcements, sector developments."
        )

    @staticmethod
    def get_output_schema_class() -> Type[BaseModel]:
        return NewsAgentOutput

    async def run(self, input_data: BaseAgentInput) -> NewsAgentOutput:
        """Run the agent end-to-end with the provided input."""
        start_date, end_date = _get_default_date_range(
            input_data.start_date, input_data.end_date
        )
        start_date, end_date = _constrain_date_range(start_date, end_date)

        initial_state = NewsAgentState(
            query=input_data.query,
            ticker=input_data.ticker or "",
            start_date=start_date,
            end_date=end_date,
            conversation_id=input_data.conversation_id,
            company_context=input_data.company_context,
        )

        state_payload = initial_state.model_dump()
        # memory_task is excluded from model_dump() (exclude=True) but must be
        # carried in the state dict so LangGraph can thread it between nodes.
        # It starts as None; _rewrite_queries_node sets it.
        state_payload["memory_task"] = None
        final_state = await self._graph.ainvoke(state_payload)
        return NewsAgentOutput(**final_state)

    # -- Graph construction ----------------------------------------------------

    def _build_graph(self):
        """Compile the linear LangGraph workflow."""
        workflow = StateGraph(NewsAgentState, output_schema=NewsAgentOutput)

        workflow.add_node("rewrite_queries", self._rewrite_queries_node)
        workflow.add_node("fetch_news", self._fetch_news_node)
        workflow.add_node("ingest_articles", self._ingest_articles_node)
        workflow.add_node("rendezvous", self._rendezvous_node)
        workflow.add_node("analyse_news", self._analyse_news_node)

        workflow.add_edge(START, "rewrite_queries")
        workflow.add_edge("rewrite_queries", "fetch_news")
        workflow.add_edge("fetch_news", "ingest_articles")
        workflow.add_edge("ingest_articles", "rendezvous")
        workflow.add_edge("rendezvous", "analyse_news")
        workflow.add_edge("analyse_news", END)

        return workflow.compile()

    # -- Node: rewrite_queries -------------------------------------------------

    async def _rewrite_queries_node(self, state: NewsAgentState) -> dict:
        """
        Expand the orchestrator's rewritten query into three domain-specific
        memory retrieval strings (company / sector / market) and immediately
        fire the memory retrieval as a background asyncio task.

        original_query is stamped onto the RewrittenQueries object here so
        that TwoStageReranker.rank() has the right query string available when
        _rendezvous_node calls it on the final combined chunk pool.

        Falls back gracefully: if the LLM call or task creation fails,
        `memory_task` remains None and the rendezvous node skips memory.
        """
        rewritten_queries: RewrittenQueries | None = None
        try:
            publish_progress(
                "news_agent", "Expanding query for multi-domain memory retrieval…"
            )
            structured_llm = self._llm.with_structured_output(RewrittenQueries)
            rewritten_queries = await structured_llm.ainvoke(
                [
                    SystemMessage(content=NEWS_MEMORY_QUERY_REWRITE_SYSTEM_PROMPT),
                    HumanMessage(content=state.query),
                ]
            )
            # Stamp the agent-level query so the final Jina reranker call in
            # _rendezvous_node has a focused, orchestrator-rewritten query string.
            if rewritten_queries is not None:
                rewritten_queries.original_query = state.query
            logger.info(
                "_rewrite_queries_node: domains=%s for query='%.80s'",
                rewritten_queries.active_domains if rewritten_queries else [],
                state.query,
            )
        except Exception:
            logger.exception(
                "_rewrite_queries_node: query rewrite LLM call failed — "
                "memory retrieval will be skipped"
            )

        memory_task = None
        if rewritten_queries and rewritten_queries.active_domains:
            try:
                svc = service_manager.get_retriever()

                async def _retrieve() -> MemoryContext:
                    try:
                        return await svc.comprehensive_retrieve(rewritten_queries)
                    except Exception as exc:
                        logger.error(
                            "_rewrite_queries_node: memory retrieval failed: %s", exc
                        )
                        return MemoryContext(
                            chunks=[], rewritten_queries=rewritten_queries
                        )

                memory_task = asyncio.create_task(_retrieve())
            except Exception:
                logger.exception(
                    "_rewrite_queries_node: failed to create memory retrieval task"
                )

        return {"memory_task": memory_task}

    # -- Node: fetch_news ------------------------------------------------------

    async def _fetch_news_node(self, state: NewsAgentState) -> dict:
        """
        Fetch news articles from NewsAPI and enrich them with full content
        scraped by trafilatura.

        � build_news_query() constructs a boolean NewsAPI query that combines
          the ticker symbol with optional company name / keywords.
        � Results are filtered to a curated list of trusted financial domains.
        � trafilatura replaces the truncated NewsAPI `content` field (~200 chars)
          with the full article body.
        � Falls back gracefully to the NewsAPI snippet when scraping fails.
        """
        logger.info(
            "_fetch_news_node: fetching news for ticker='%s' (%s ? %s)",
            state.ticker,
            state.start_date,
            state.end_date,
        )

        publish_progress("news_agent", f"Fetching recent news for '{state.ticker}'…")

        q = build_news_query(ticker=state.ticker)
        logger.info("NewsAPI query: %s", q)

        try:
            articles = await fetch_articles(
                q=q,
                from_date=state.start_date.isoformat(),
                to_date=state.end_date.isoformat(),
            )
            publish_success("news_agent", f"Fetched {len(articles)} article(s)")
        except Exception as exc:
            logger.error("News fetch pipeline failed: %s", exc)
            raise

        logger.info(
            "Fetched %d articles for ticker '%s' (%s ? %s).",
            len(articles),
            state.ticker,
            state.start_date,
            state.end_date,
        )
        return {"raw_articles": articles}

    # -- Node: ingest_articles -------------------------------------------------

    async def _ingest_articles_node(self, state: NewsAgentState) -> dict:
        """
        Ingest articles into Neo4j and ChromaDB, then retrieve the most
        relevant chunks for the current query via scored similarity search.

        ingest_articles() returns involved_chunks in memory (no round-trip),
        but those chunks carry no relevance scores. Two parallel ChromaDB
        similarity queries are still needed to score new and existing chunks
        against the query before reranking. The in-memory return value is
        intentionally ignored for this reason.

        The two queries are fired in parallel - they are independent and each
        involves an embedding round-trip.
        existing_chunk_ids are intentionally queried as well because those
        chunks may not appear in comprehensive_retrieve() top-k results.
        """
        if not state.raw_articles:
            return {"chunk_ids": [], "retrieved_chunks": []}

        publish_progress(
            "news_agent",
            f"Ingesting {len(state.raw_articles)} article(s) into dual-store memory…",
        )

        try:
            new_chunk_ids, existing_chunk_ids, _ = (
                await service_manager.get_ingestor().ingest_articles(state.raw_articles)
            )
            publish_success(
                "news_agent",
                f"Ingestion done — {len(new_chunk_ids)} new chunks, {len(existing_chunk_ids)} existing",
            )
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            raise

        query = state.query

        async def _query(chunk_ids: List[str], domain: str) -> List[RetrievedChunk]:
            if not chunk_ids:
                return []
            docs_with_scores = await service_manager.get_chroma_adapter().query(
                query_text=query,
                n_results=settings.RETRIEVER_SEED_TOP_K,
                where={"chunk_id": {"$in": chunk_ids}},
            )
            return [
                RetrievedChunk.from_document(
                    doc, score=score, source="vector", domain=domain
                )
                for doc, score in docs_with_scores
            ]

        new_chunks, existing_chunks = await asyncio.gather(
            _query(new_chunk_ids, "new"),
            _query(existing_chunk_ids, "existing"),
        )
        retrieved_chunks = new_chunks + existing_chunks

        logger.info(
            "Ingested articles into memory. #New: %s, #Existing: %s",
            len(new_chunk_ids),
            len(existing_chunk_ids),
        )
        # chunk_ids retained in state for schema compatibility but no longer
        # consumed downstream � entity extraction uses final_ranked IDs in
        # _rendezvous_node instead.
        return {
            "chunk_ids": new_chunk_ids + existing_chunk_ids,
            "retrieved_chunks": retrieved_chunks,
        }

    # -- Node: rendezvous ------------------------------------------------------

    async def _rendezvous_node(self, state: NewsAgentState) -> dict:
        memory_context: MemoryContext | None = None
        if state.memory_task is not None:
            try:
                memory_context = await state.memory_task
                logger.info(
                    "_rendezvous_node: memory returned %d chunks",
                    len(memory_context.chunks) if memory_context else 0,
                )
            except Exception as exc:
                logger.error("Memory retrieval task failed: %s", exc)

        final_chunks = list(state.retrieved_chunks)
        if memory_context is not None:
            final_chunks = final_chunks + memory_context.chunks

        total = len(final_chunks)
        publish_progress("news_agent", f"Reranking {total} candidate chunk(s)…")

        # TwoStageReranker: composite prefilter → Jina cross-encoder.
        # query comes from state.query (the agent-level, orchestrator-rewritten
        # query — already focused for news analysis).
        final_ranked = await service_manager.get_reranker().rank(
            state.query, final_chunks
        )

        merged_entity_tuples: List[Tuple[str, str]] = []
        if memory_context is not None and memory_context.entity_tuples:
            merged_entity_tuples.extend(memory_context.entity_tuples)

        final_ranked_chunk_ids = list(
            dict.fromkeys(chunk.chunk_id for chunk in final_ranked if chunk.chunk_id)
        )
        if final_ranked_chunk_ids:
            try:
                rows = (
                    await service_manager.get_neo4j_adapter().get_entities_for_chunks(
                        final_ranked_chunk_ids
                    )
                )
                for row in rows:
                    parsed = _normalize_entity_tuple(
                        row.get("entity_name"),
                        row.get("entity_type"),
                    )
                    if parsed is not None:
                        merged_entity_tuples.append(parsed)
            except Exception:
                logger.exception(
                    "_rendezvous_node: failed to fetch entity tuples for final-ranked chunks"
                )

        if state.conversation_id and merged_entity_tuples:
            _merge_cached_entities(state.conversation_id, merged_entity_tuples)

        # Extract entities only from the reranked set — chunks that actually
        # contributed to this query. Fire-and-forget: extraction enriches the
        # graph for future retrieval but does not affect the current analysis.
        pending_chunk_ids = [
            chunk.chunk_id
            for chunk in final_ranked
            if chunk.chunk_id and chunk.extraction_status == "PENDING"
        ]
        pending_chunk_ids = list(dict.fromkeys(pending_chunk_ids))
        if pending_chunk_ids and state.conversation_id:
            turn_id = getattr(state, "turn_id", None) or state.conversation_id
            try:
                task = make_extraction_task(
                    turn_id=turn_id,
                    conversation_id=state.conversation_id,
                    source_agent=self.name(),
                    extraction_text=None,
                    system_prompt=None,
                    immediate=settings.EXTRACTION_IMMEDIATE,
                    task_kind="chunk_entities",
                    chunk_ids=pending_chunk_ids,
                )
                await service_manager.get_graph_queue_manager().enqueue(task)
            except Exception:
                logger.exception(
                    "_rendezvous_node: failed to enqueue chunk entity extraction"
                )

        return {"final_chunks": final_ranked}

    # -- Node: analyse_news ----------------------------------------------------

    async def _analyse_news_node(self, state: NewsAgentState) -> dict:
        """Generate a grounded financial analysis from retrieved chunks."""

        chunks = state.final_chunks
        if not chunks:
            return {
                "analysis": "No relevant news data was found for this query.",
                "sources": [],
                "entities_enriched": [],
            }
        publish_progress(
            "news_agent", f"Generating grounded news analysis ({len(chunks)} chunk(s))…"
        )

        sources, chunk_to_source_id = _build_deduplicated_sources(chunks)

        context_lines = []
        for idx, chunk in enumerate(chunks):
            sid = chunk_to_source_id.get(idx, "?")
            context_lines.append(f"[{sid}] {chunk.text}")
        context_block = "\n\n".join(context_lines)

        conversation_id = state.conversation_id or ""
        cached_entities = _get_cached_entities(conversation_id)
        entities_section = ""
        if cached_entities:
            entity_lines = [
                f"  - {name} ({entity_type})" for name, entity_type in cached_entities
            ]
            entities_section = (
                "Known entities from prior turns:\n" + "\n".join(entity_lines) + "\n\n"
            )

        company_context_section = (
            f"Company Context:\n{state.company_context}\n\n"
            if state.company_context
            else ""
        )

        messages = [
            SystemMessage(content=NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=NEWS_ANALYSIS_USER_PROMPT.format(
                    query=state.query,
                    entities_section=company_context_section + entities_section,
                    context=context_block,
                )
            ),
        ]

        relationships_extracted = False
        sentiment = None
        try:
            structured_llm = self._llm.with_structured_output(
                NewsAnalysisStructuredOutput
            )
            response = await structured_llm.ainvoke(messages)
            analysis_text = (response.analysis or "").strip()
            sentiment = response.sentiment
            if not analysis_text:
                analysis_text = (
                    "Analysis could not be generated due to an internal error."
                )
            else:
                publish_success("news_agent", "News analysis complete.")
        except Exception as exc:
            logger.error("_analyse_news_node: analysis LLM call failed: %s", exc)
            analysis_text = "Analysis could not be generated due to an internal error."

        # Citation filtering (unchanged)
        cited_ids = set(int(m) for m in _re.findall(r"\[(\d+)\]", analysis_text))
        _cited_in_order = sorted(cited_ids)
        _old_to_new: dict = {
            old_id: new_id for new_id, old_id in enumerate(_cited_in_order, start=1)
        }

        def _remap(m: _re.Match) -> str:
            sid = int(m.group(1))
            return f"[{_old_to_new[sid]}]" if sid in _old_to_new else m.group(0)

        analysis_text = _re.sub(r"\[(\d+)\]", _remap, analysis_text)

        _sources_by_old_id = {s.source_id: s for s in sources}
        sources = [
            CitedSource(
                source_id=new_id,
                title=_sources_by_old_id[old_id].title,
                url=_sources_by_old_id[old_id].url,
                page_content=_sources_by_old_id[old_id].page_content,
            )
            for old_id, new_id in _old_to_new.items()
            if old_id in _sources_by_old_id
        ]

        task_id = None
        if state.conversation_id and analysis_text:
            turn_id = getattr(state, "turn_id", None) or state.conversation_id
            try:
                task = make_extraction_task(
                    turn_id=turn_id,
                    conversation_id=state.conversation_id,
                    source_agent=self.name(),
                    extraction_text=analysis_text,
                    system_prompt=NEWS_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
                    llm_config={"temperature": getattr(self._llm, "temperature", 0.7)},
                )
                task_id = await service_manager.get_graph_queue_manager().enqueue(task)
            except Exception:
                logger.exception("_analyse_news_node: failed to enqueue graph task")

        return {
            "analysis": analysis_text,
            "sources": sources,
            "subgraph_id": task_id,
            "relationships_extracted": relationships_extracted,
            "sentiment": sentiment,
        }
