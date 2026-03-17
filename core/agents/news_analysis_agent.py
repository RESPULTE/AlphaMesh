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

- `run()` no longer reads `input_data.memory_task` — the news agent now
  self-manages its memory retrieval lifecycle entirely.  BaseAgentInput no
  longer carries a `memory_task` field.

- `_fetch_news_node` and `_ingest_articles_node` run concurrently with the
  memory retrieval task thanks to the background task created in
  `_rewrite_queries_node`.

Graph topology:
  rewrite_queries → fetch_news → ingest_articles → rendezvous → analyse_news
                 ↘ (memory_task fires in background) ↗
"""

from __future__ import annotations

import asyncio
import re as _re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core.agents.base_agent import AbstractAgent
from core.agents.models import (
    BaseAgentInput,
    CitedSource,
    NewsAgentOutput,
    NewsAgentState,
)
from core.agents.news_fetcher import build_news_query, fetch_articles
from core.agents.prompts import (
    NEWS_ANALYSIS_USER_PROMPT,
    NEWS_MEMORY_QUERY_REWRITE_SYSTEM_PROMPT,
)
from core.config import settings
from core.logger import get_logger
from core.memory.graph.extraction_prompts import (
    ANALYSIS_ONLY_RELATIONSHIP_PROMPT,
    COMBINED_ANALYSIS_RELATIONSHIP_PROMPT,
)
from core.memory.graph.relationship_extractor import extract_with_retry
from core.memory.graph.subgraph_builder import InMemorySubgraphBuilder
from core.memory.retrieval.models import MemoryContext, RetrievedChunk, RewrittenQueries
from core.services import service_manager

logger = get_logger(__name__)
_ENTITY_CACHE: Dict[str, Dict[str, dict]] = {}
_ENTITY_CACHE_TS: Dict[str, float] = {}


def _get_cached_entities(conversation_id: str) -> List[dict]:
    if not conversation_id:
        return []
    now = time.time()
    ts = _ENTITY_CACHE_TS.get(conversation_id)
    if ts is None or now - ts > settings.SUBGRAPH_TTL_SECONDS:
        _ENTITY_CACHE.pop(conversation_id, None)
        _ENTITY_CACHE_TS.pop(conversation_id, None)
        return []
    return list(_ENTITY_CACHE.get(conversation_id, {}).values())


def _merge_cached_entities(conversation_id: str, entities: List[dict]) -> None:
    if not conversation_id:
        return
    cache = _ENTITY_CACHE.setdefault(conversation_id, {})
    for entity in entities:
        entity_id = getattr(entity, "id", None)
        entity_name = getattr(entity, "name", None)
        entity_type = getattr(entity, "entity_type", None)
        if not entity_id or not entity_name or not entity_type:
            continue
        cache[entity_id] = {
            "id": entity_id,
            "name": entity_name,
            "entity_type": entity_type,
        }
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
        self._subgraph_builder = InMemorySubgraphBuilder()

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
        start_date = input_data.start_date or (
            datetime.now(timezone.utc) - timedelta(days=7)
        )
        end_date = input_data.end_date or datetime.now(timezone.utc)

        initial_state = NewsAgentState(
            query=input_data.query,
            ticker=input_data.ticker or "",
            start_date=start_date,
            end_date=end_date,
            conversation_id=input_data.conversation_id,
        )

        state_payload = initial_state.model_dump()
        # memory_task is excluded from model_dump() (exclude=True) but must be
        # carried in the state dict so LangGraph can thread it between nodes.
        # It starts as None; _rewrite_queries_node sets it.
        state_payload["memory_task"] = None
        final_state = await self._graph.ainvoke(state_payload)
        return NewsAgentOutput(**final_state)

    # ── Graph construction ────────────────────────────────────────────────────

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

    # ── Node: rewrite_queries ─────────────────────────────────────────────────

    async def _rewrite_queries_node(self, state: NewsAgentState) -> dict:
        """
        Expand the orchestrator's rewritten query into three domain-specific
        memory retrieval strings (company / sector / market) and immediately
        fire the memory retrieval as a background asyncio task.

        The task is stored on `memory_task` and awaited later in
        `_rendezvous_node`, which runs after news ingestion completes —
        effectively making memory retrieval concurrent with news fetching.

        Falls back gracefully: if the LLM call or task creation fails,
        `memory_task` remains None and the rendezvous node skips memory.
        """
        rewritten_queries: RewrittenQueries | None = None
        try:
            structured_llm = self._llm.with_structured_output(RewrittenQueries)
            rewritten_queries = await structured_llm.ainvoke(
                [
                    SystemMessage(content=NEWS_MEMORY_QUERY_REWRITE_SYSTEM_PROMPT),
                    HumanMessage(content=state.query),
                ]
            )
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
                    return await svc.retrieve(rewritten_queries)

                memory_task = asyncio.ensure_future(_retrieve())
            except Exception:
                logger.exception(
                    "_rewrite_queries_node: failed to create memory retrieval task"
                )

        # memory_task is excluded from Pydantic serialisation but LangGraph
        # carries it through the state dict; return it explicitly here.
        return {"memory_task": memory_task}

    # ── Node: fetch_news ──────────────────────────────────────────────────────

    async def _fetch_news_node(self, state: NewsAgentState) -> dict:
        """
        Fetch news articles from NewsAPI and enrich them with full content
        scraped by trafilatura.

        • build_news_query() constructs a boolean NewsAPI query that combines
          the ticker symbol with optional company name / keywords.
        • Results are filtered to a curated list of trusted financial domains.
        • trafilatura replaces the truncated NewsAPI `content` field (~200 chars)
          with the full article body.
        • Falls back gracefully to the NewsAPI snippet when scraping fails.
        """
        now = datetime.now()
        api_limit_date = now - timedelta(days=28)

        start = state.start_date
        end = state.end_date

        if end > now:
            end = now
        if start < api_limit_date:
            start = api_limit_date

        q = build_news_query(ticker=state.ticker)
        logger.info("NewsAPI query: %s", q)

        try:
            articles = await fetch_articles(
                q=q,
                from_date=start.date().isoformat(),
                to_date=end.date().isoformat(),
                language="en",
                sort_by="relevancy",
                page=1,
                page_size=50,
            )
        except Exception as exc:
            logger.error("News fetch pipeline failed: %s", exc)
            raise

        logger.info(
            "Fetched %d articles for ticker '%s' (%s → %s).",
            len(articles),
            state.ticker,
            start.date(),
            end.date(),
        )
        return {"raw_articles": articles}

    # ── Node: ingest_articles ─────────────────────────────────────────────────

    async def _ingest_articles_node(self, state: NewsAgentState) -> dict:
        """Ingest articles into Neo4j and ChromaDB."""
        if not state.raw_articles:
            return {"chunk_ids": []}

        try:
            new_chunk_ids, existing_chunk_ids, _ = (
                await service_manager.get_ingestor().ingest_articles(state.raw_articles)
            )
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            raise

        retrieved_chunks: List[RetrievedChunk] = []
        query = state.query

        if new_chunk_ids:
            docs_with_scores = await service_manager.get_chroma_adapter().query(
                query_text=query,
                n_results=10,
                where={"chunk_id": {"$in": new_chunk_ids}},
            )
            retrieved_chunks += [
                RetrievedChunk.from_document(
                    doc, score=score, source="vector", domain="new"
                )
                for doc, score in docs_with_scores
            ]

        if existing_chunk_ids:
            docs_with_scores = await service_manager.get_chroma_adapter().query(
                query_text=query,
                n_results=10,
                where={"chunk_id": {"$in": existing_chunk_ids}},
            )
            retrieved_chunks += [
                RetrievedChunk.from_document(
                    doc, score=score, source="vector", domain="existing"
                )
                for doc, score in docs_with_scores
            ]

        all_chunk_ids = new_chunk_ids + existing_chunk_ids
        logger.info(
            "Ingested articles into memory. #New: %s, #Existing: %s",
            len(new_chunk_ids),
            len(existing_chunk_ids),
        )
        return {"chunk_ids": all_chunk_ids, "retrieved_chunks": retrieved_chunks}

    # ── Node: rendezvous ──────────────────────────────────────────────────────

    async def _rendezvous_node(self, state: NewsAgentState) -> dict:
        """
        Await the memory retrieval task (fired in _rewrite_queries_node) and
        merge its results with the freshly ingested article chunks.
        """
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
                memory_context = None

        new_chunks = list(state.retrieved_chunks)

        if memory_context is None:
            final_ranked = service_manager.get_reranker().rank(new_chunks)
            return {"final_chunks": final_ranked}

        combined = new_chunks + memory_context.chunks
        final_ranked = service_manager.get_reranker().rank(combined)
        return {"final_chunks": final_ranked}

    # ── Node: analyse_news ────────────────────────────────────────────────────

    async def _analyse_news_node(self, state: NewsAgentState) -> dict:
        """Generate a grounded financial analysis from retrieved chunks."""
        chunks = state.final_chunks
        if not chunks:
            logger.warning("_analyse_news_node: no chunks available")
            return {
                "analysis": "No relevant news data was found for this query.",
                "sources": [],
                "entities_enriched": [],
            }

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
                f"  - {e['name']} ({e['entity_type']})" for e in cached_entities
            ]
            entities_section = (
                "Known entities from prior turns:\n" + "\n".join(entity_lines) + "\n\n"
            )

        messages = [
            SystemMessage(content=COMBINED_ANALYSIS_RELATIONSHIP_PROMPT),
            HumanMessage(
                content=NEWS_ANALYSIS_USER_PROMPT.format(
                    query=state.query,
                    entities_section=entities_section,
                    context=context_block,
                )
            ),
        ]

        try:
            response = await extract_with_retry(self._llm, messages)
            analysis_text = response.analysis
            relationships = response.relationships
            relationships_extracted = response.parse_success
        except Exception as exc:
            logger.error("_analyse_news_node: extraction failed: %s", exc)
            # Fallback: plain LLM call without relationship extraction
            fallback_response = await self._llm.ainvoke(
                [
                    SystemMessage(content=ANALYSIS_ONLY_RELATIONSHIP_PROMPT),
                    HumanMessage(
                        content=NEWS_ANALYSIS_USER_PROMPT.format(
                            query=state.query,
                            entities_section=entities_section,
                            context=context_block,
                        )
                    ),
                ]
            )
            analysis_text = fallback_response.content if fallback_response else ""
            relationships = []
            relationships_extracted = False

        # ── Update entity cache ───────────────────────────────────────────────
        if conversation_id and relationships:
            _merge_cached_entities(conversation_id, relationships)

        # ── Citation filtering: keep only sources cited in the analysis ───────
        cited_ids = set(int(m) for m in _re.findall(r"\[(\d+)\]", analysis_text))
        _cited_in_order = sorted(cited_ids)
        _old_to_new: dict[int, int] = {
            old_id: new_id for new_id, old_id in enumerate(_cited_in_order, start=1)
        }

        def _remap(m: _re.Match) -> str:
            sid = int(m.group(1))
            return f"[{_old_to_new[sid]}]" if sid in _old_to_new else m.group(0)

        analysis_text = _re.sub(r"\[(\d+)\]", _remap, analysis_text)

        _sources_by_old_id: dict[int, CitedSource] = {s.source_id: s for s in sources}
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
        # ── End citation filtering ─────────────────────────────────────────────

        subgraph_id = await self._subgraph_builder.schedule_subgraph_extraction(
            agent_name=self.name(),
            conversation_id=state.conversation_id or "",
            analysis_text=analysis_text,
            relationships=relationships,
            relationships_extracted=relationships_extracted,
            llm=self._llm,
        )

        return {
            "analysis": analysis_text,
            "sources": sources,
            "subgraph_id": subgraph_id,
            "relationships_extracted": relationships_extracted,
        }
