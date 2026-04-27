"""News analysis agent graph and node logic."""

from __future__ import annotations

import asyncio
import re as _re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Type
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.agents.base_agent import AbstractAgent
from core.agents.models.base_agent_models import AgentSentiment, BaseAgentInput
from core.agents.models.news_agent_models import (
    CitedSource,
    NewsAgentOutput,
    NewsAgentState,
    ResearchStepLog,
    ResearchStepPlan,
)
from core.agents.news_fetcher import (
    build_news_query,
    fetch_articles,
    fetch_articles_from_tavily,
)
from core.agents.prompts.news_agent_prompts import (
    NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT,
    NEWS_ANALYSIS_USER_PROMPT,
    NEWS_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
    NEWS_MEMORY_QUERY_REWRITE_SYSTEM_PROMPT,
    NEWS_RESEARCH_PLANNER_SYSTEM_PROMPT,
)
from core.agents.utils import (
    extract_first_sentence,
    persist_agent_memory_summary,
    resolve_agent_memory_context,
    trim_text,
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
    """Fill in missing start/end dates with defaults."""
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
    """Constrain date range to NewsAPI bounds."""
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
    """Deduplicate chunks by article and map each chunk to a source id."""
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


def _build_context_block(chunks: List[RetrievedChunk], mapping: Dict[int, int]) -> str:
    return "\n\n".join(
        f"[{mapping.get(idx, '?')}] {chunk.text}" for idx, chunk in enumerate(chunks)
    )


def _build_context_prefix(
    *,
    company_context: str | None,
    agent_memory_context: str | None,
    cached_entities: List[Tuple[str, str]],
) -> str:
    sections: List[str] = []
    if company_context:
        sections.append(f"Company Context:\n{company_context}")
    if agent_memory_context:
        sections.append(f"Agent Memory Context:\n{agent_memory_context}")
    if cached_entities:
        entity_lines = [f"  - {name} ({entity_type})" for name, entity_type in cached_entities]
        sections.append("Known entities from prior turns:\n" + "\n".join(entity_lines))
    if not sections:
        return ""
    return "\n\n".join(sections) + "\n\n"


def _remap_citations_and_sources(
    analysis_text: str,
    sources: List[CitedSource],
) -> Tuple[str, List[CitedSource]]:
    cited_ids = sorted(set(int(match) for match in _re.findall(r"\[(\d+)\]", analysis_text)))
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(cited_ids, start=1)}

    def _remap(match: _re.Match[str]) -> str:
        source_id = int(match.group(1))
        return f"[{old_to_new[source_id]}]" if source_id in old_to_new else match.group(0)

    remapped_text = _re.sub(r"\[(\d+)\]", _remap, analysis_text)
    if not old_to_new:
        return remapped_text, []

    by_old_id = {source.source_id: source for source in sources}
    remapped_sources: List[CitedSource] = []
    for old_id, new_id in old_to_new.items():
        source = by_old_id.get(old_id)
        if source is None:
            continue
        remapped_sources.append(
            CitedSource(
                source_id=new_id,
                title=source.title,
                url=source.url,
                page_content=source.page_content,
            )
        )
    return remapped_text, remapped_sources


def _list_unique_actions(logs: List[ResearchStepLog]) -> List[str]:
    actions: List[str] = []
    for row in logs:
        action = str(getattr(row, "action", "") or "").strip()
        if not action or action == "proceed" or action in actions:
            continue
        actions.append(action)
    return actions


class NewsAnalysisAgent(AbstractAgent):
    """LangGraph-based news analysis agent with self-managed memory retrieval."""

    def __init__(self) -> None:
        super().__init__()
        self._llm = service_manager.get_agent()
        self._graph = self._build_graph()
        self._memory_context_by_conversation: Dict[str, str] = {}

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

    @staticmethod
    def render_memory_summary(memory_summary: Dict[str, Any]) -> str:
        if not memory_summary:
            return ""
        actions = (
            memory_summary.get("research_actions")
            or memory_summary.get("tools_used")
            or []
        )
        if not isinstance(actions, list):
            actions = []
        sentiment = memory_summary.get("sentiment") or {}
        sentiment_label = ""
        if isinstance(sentiment, dict):
            sentiment_label = str(sentiment.get("label") or "").strip()
        source_count = int(memory_summary.get("source_count") or 0)
        catalyst = trim_text(memory_summary.get("main_catalyst") or "", max_chars=200)
        return (
            f"actions={','.join(str(a) for a in actions[:4]) or 'none'}; "
            f"sources={source_count}; "
            f"sentiment={sentiment_label or 'N/A'}; "
            f"catalyst={catalyst or 'N/A'}"
        )

    async def run(self, input_data: BaseAgentInput) -> NewsAgentOutput:
        """Run the agent end-to-end with the provided input."""
        start_date, end_date = _get_default_date_range(
            input_data.start_date, input_data.end_date
        )
        start_date, end_date = _constrain_date_range(start_date, end_date)
        conversation_id, effective_memory_context = resolve_agent_memory_context(
            conversation_id=input_data.conversation_id,
            incoming_memory_context=input_data.agent_memory_context,
            memory_context_cache=self._memory_context_by_conversation,
        )

        initial_state = NewsAgentState(
            query=input_data.query,
            ticker=input_data.ticker or "",
            start_date=start_date,
            end_date=end_date,
            conversation_id=conversation_id or None,
            turn_id=input_data.turn_id,
            agent_memory_context=effective_memory_context,
            company_context=input_data.company_context,
            max_research_iterations=settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS,
        )

        state_payload = initial_state.model_dump()
        state_payload["memory_task"] = None
        final_state = await self._graph.ainvoke(state_payload)
        output = NewsAgentOutput(**final_state)
        persist_agent_memory_summary(
            conversation_id=conversation_id,
            rendered_summary=self.render_memory_summary(output.memory_summary),
            memory_context_cache=self._memory_context_by_conversation,
        )

        return output

    # -- Graph construction ----------------------------------------------------

    def _build_graph(self):
        """Compile the iterative LangGraph workflow."""
        workflow = StateGraph(NewsAgentState, output_schema=NewsAgentOutput)

        workflow.add_node("rewrite_queries", self._rewrite_queries_node)
        workflow.add_node("plan_research", self._plan_research_node)
        workflow.add_node("fetch_news", self._fetch_news_node)
        workflow.add_node("ingest_articles", self._ingest_articles_node)
        workflow.add_node("rendezvous", self._rendezvous_node)
        workflow.add_node("analyse_news", self._analyse_news_node)

        workflow.add_edge(START, "rewrite_queries")
        workflow.add_edge("rewrite_queries", "plan_research")
        workflow.add_conditional_edges(
            "plan_research",
            self._route_after_research_plan,
            {"fetch_news": "fetch_news", "rendezvous": "rendezvous"},
        )
        workflow.add_edge("fetch_news", "ingest_articles")
        workflow.add_edge("ingest_articles", "plan_research")
        workflow.add_edge("rendezvous", "analyse_news")
        workflow.add_edge("analyse_news", END)

        return workflow.compile()

    @staticmethod
    def _route_after_research_plan(state: NewsAgentState) -> str:
        plan = state.research_plan
        if plan is None:
            return "rendezvous"
        if state.research_iteration >= state.max_research_iterations:
            return "rendezvous"
        if plan.action == "proceed":
            return "rendezvous"
        return "fetch_news"

    # -- Node: rewrite_queries -------------------------------------------------

    async def _rewrite_queries_node(self, state: NewsAgentState) -> dict:
        """Rewrite memory queries and start background retrieval."""
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
                            chunks=[],
                            rewritten_queries=rewritten_queries,
                            entity_tuples=[],
                        )

                memory_task = asyncio.create_task(_retrieve())
            except Exception:
                logger.exception(
                    "_rewrite_queries_node: failed to create memory retrieval task"
                )

        return {"memory_task": memory_task}

    # -- Node: plan_research --------------------------------------------------

    @staticmethod
    def _count_unique_source_urls(articles: List[dict]) -> int:
        urls = {
            str(article.get("url") or "").strip()
            for article in articles
            if str(article.get("url") or "").strip()
        }
        return len(urls)

    @staticmethod
    def _history_block(logs: List[ResearchStepLog], limit: int = 6) -> str:
        if not logs:
            return "(none)"
        lines: List[str] = []
        for row in logs[-limit:]:
            lines.append(
                f"- iter={row.iteration} action={row.action} query='{row.query}' "
                f"fetched={row.fetched_articles} newly_added={row.newly_added_articles}"
            )
        return "\n".join(lines)

    async def _plan_research_node(self, state: NewsAgentState) -> dict:
        unique_sources = self._count_unique_source_urls(state.raw_articles)
        chunk_count = len(state.retrieved_chunks)
        threshold_met = (
            unique_sources >= settings.NEWS_AGENT_MIN_SOURCES_FOR_SUFFICIENCY
            and chunk_count >= settings.NEWS_AGENT_MIN_CHUNKS_FOR_SUFFICIENCY
        )
        at_limit = state.research_iteration >= state.max_research_iterations

        if at_limit or threshold_met:
            if at_limit:
                reason = (
                    f"Reached research iteration limit ({state.max_research_iterations})."
                )
            else:
                reason = (
                    "Sufficiency threshold reached "
                    f"(sources={unique_sources}, chunks={chunk_count})."
                )
            plan = ResearchStepPlan(
                action="proceed",
                query="",
                rationale=reason,
                max_results=1,
            )
            publish_progress("news_agent", f"Research planning: {reason}")
            return {"research_plan": plan, "is_information_sufficient": True}

        planner_prompt = (
            f"Query: {state.query}\n"
            f"Ticker: {state.ticker or 'N/A'}\n"
            f"Iteration: {state.research_iteration} / {state.max_research_iterations}\n"
            f"Current unique sources: {unique_sources}\n"
            f"Current chunk count: {chunk_count}\n"
            f"Sufficiency thresholds: "
            f"sources>={settings.NEWS_AGENT_MIN_SOURCES_FOR_SUFFICIENCY}, "
            f"chunks>={settings.NEWS_AGENT_MIN_CHUNKS_FOR_SUFFICIENCY}\n"
            f"Recent research history:\n{self._history_block(state.research_logs)}\n\n"
            "Return only ResearchStepPlan."
        )
        if state.agent_memory_context:
            planner_prompt += (
                "\n\nAgent Memory Context (from prior turns):\n"
                f"{state.agent_memory_context}"
            )
        publish_progress("news_agent", "Planning next research step...")
        try:
            structured_llm = self._llm.with_structured_output(ResearchStepPlan)
            plan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=NEWS_RESEARCH_PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=planner_prompt),
                ]
            )
        except Exception:
            logger.exception("_plan_research_node: planner call failed; using fallback")
            if state.research_iteration == 0:
                plan = ResearchStepPlan(
                    action="newsapi",
                    query=state.query,
                    rationale="Fallback to NewsAPI on planner failure.",
                    max_results=settings.NEWS_FETCH_MAX_ARTICLES,
                )
            elif settings.TAVILY_API_KEY:
                plan = ResearchStepPlan(
                    action="web_search",
                    query=state.query,
                    rationale="Fallback targeted web search after planner failure.",
                    max_results=settings.TAVILY_SEARCH_MAX_RESULTS,
                )
            else:
                plan = ResearchStepPlan(
                    action="proceed",
                    query="",
                    rationale="Planner failure and no Tavily key configured.",
                    max_results=1,
                )

        if not plan.query and plan.action != "proceed":
            plan.query = state.query
        if plan.action == "web_search" and not settings.TAVILY_API_KEY:
            plan = ResearchStepPlan(
                action="newsapi",
                query=plan.query or state.query,
                rationale=(
                    "Tavily key not configured; falling back to NewsAPI for this step."
                ),
                max_results=plan.max_results,
            )
        if plan.action == "newsapi":
            plan.max_results = max(
                1, min(plan.max_results, settings.NEWS_FETCH_MAX_ARTICLES)
            )
        else:
            plan.max_results = max(1, min(plan.max_results, 20))

        return {
            "research_plan": plan,
            "is_information_sufficient": plan.action == "proceed",
        }

    # -- Node: fetch_news ------------------------------------------------------

    async def _fetch_news_node(self, state: NewsAgentState) -> dict:
        """Execute the planner-selected research tool and deduplicate by URL."""
        plan = state.research_plan
        if plan is None or plan.action == "proceed":
            return {
                "latest_articles": [],
                "research_iteration": state.research_iteration,
                "research_logs": state.research_logs,
            }

        logger.info(
            "_fetch_news_node: action=%s iteration=%d query='%.120s'",
            plan.action,
            state.research_iteration,
            plan.query,
        )
        query_text = (plan.query or state.query).strip()
        articles: List[dict] = []

        if plan.action == "newsapi":
            publish_progress(
                "news_agent",
                f"Research iter {state.research_iteration + 1}: fetching from NewsAPI...",
            )
            if not query_text:
                query_text = build_news_query(ticker=state.ticker)
            elif state.ticker:
                query_text = f"({state.ticker}) AND ({query_text})"
            try:
                articles = await fetch_articles(
                    q=query_text,
                    from_date=state.start_date.isoformat(),
                    to_date=state.end_date.isoformat(),
                    page_size=plan.max_results,
                )
            except Exception:
                logger.exception("_fetch_news_node: NewsAPI fetch failed")
                articles = []
        elif plan.action == "web_search":
            publish_progress(
                "news_agent",
                f"Research iter {state.research_iteration + 1}: targeted Tavily web search...",
            )
            articles = await fetch_articles_from_tavily(
                query=query_text,
                max_results=plan.max_results,
                include_domains=plan.include_domains,
                exclude_domains=plan.exclude_domains,
            )

        seen_urls = set(state.seen_urls or [])
        latest_articles: List[dict] = []
        for article in articles:
            url = str(article.get("url") or "").strip()
            if not url:
                continue
            if url in seen_urls:
                continue
            latest_articles.append(article)
            seen_urls.add(url)

        log_row = ResearchStepLog(
            iteration=state.research_iteration + 1,
            action=plan.action,
            query=query_text,
            rationale=plan.rationale,
            fetched_articles=len(articles),
            newly_added_articles=len(latest_articles),
        )
        publish_success(
            "news_agent",
            f"Research iter {state.research_iteration + 1}: "
            f"{plan.action} fetched={len(articles)} new={len(latest_articles)}",
        )

        return {
            "latest_articles": latest_articles,
            "raw_articles": list(state.raw_articles) + latest_articles,
            "seen_urls": list(seen_urls),
            "research_logs": list(state.research_logs) + [log_row],
            "research_iteration": state.research_iteration + 1,
        }

    # -- Node: ingest_articles -------------------------------------------------

    async def _ingest_articles_node(self, state: NewsAgentState) -> dict:
        """Ingest this iteration's articles and append scored chunks."""
        if not state.latest_articles:
            logger.info("_ingest_articles_node: no new articles in this iteration")
            return {"retrieved_chunks": state.retrieved_chunks}

        publish_progress(
            "news_agent",
            f"Ingesting {len(state.latest_articles)} new article(s) into memory...",
        )

        try:
            new_chunk_ids, existing_chunk_ids, _ = (
                await service_manager.get_ingestor().ingest_articles(state.latest_articles)
            )
            publish_success(
                "news_agent",
                f"Ingestion done: {len(new_chunk_ids)} new chunks, {len(existing_chunk_ids)} existing",
            )
        except Exception:
            logger.exception("_ingest_articles_node: ingestion failed")
            return {"retrieved_chunks": state.retrieved_chunks}

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
        merged_chunks = list(state.retrieved_chunks)
        chunk_map: Dict[str, RetrievedChunk] = {
            chunk.chunk_id: chunk for chunk in merged_chunks if chunk.chunk_id
        }
        for chunk in new_chunks + existing_chunks:
            if chunk.chunk_id and chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
                merged_chunks.append(chunk)

        logger.info(
            "_ingest_articles_node: cumulative retrieved chunks=%d",
            len(merged_chunks),
        )
        return {"retrieved_chunks": merged_chunks}

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

        final_ranked = await service_manager.get_reranker().rank(
            state.query, final_chunks
        )

        merged_entity_tuples: List[Tuple[str, str]] = []
        if memory_context is not None:
            merged_entity_tuples.extend(memory_context.entity_tuples or [])
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

        pending_chunk_ids = [
            chunk.chunk_id
            for chunk in final_ranked
            if chunk.chunk_id and chunk.extraction_status == "PENDING"
        ]
        pending_chunk_ids = list(dict.fromkeys(pending_chunk_ids))
        if pending_chunk_ids and state.conversation_id:
            turn_id = (getattr(state, "turn_id", None) or "").strip() or str(uuid4())
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
        context_block = _build_context_block(chunks, chunk_to_source_id)

        conversation_id = state.conversation_id or ""
        cached_entities = _get_cached_entities(conversation_id)
        context_prefix = _build_context_prefix(
            company_context=state.company_context,
            agent_memory_context=state.agent_memory_context,
            cached_entities=cached_entities,
        )

        messages = [
            SystemMessage(content=NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=NEWS_ANALYSIS_USER_PROMPT.format(
                    query=state.query,
                    entities_section=context_prefix,
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

        analysis_text, sources = _remap_citations_and_sources(analysis_text, sources)

        task_id = None
        if state.conversation_id and analysis_text:
            turn_id = (getattr(state, "turn_id", None) or "").strip() or str(uuid4())
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

        tools_used = _list_unique_actions(state.research_logs)

        top_references: List[dict] = []
        for src in sources[:3]:
            top_references.append(
                {
                    "source_id": int(src.source_id),
                    "title": src.title,
                    "url": src.url,
                }
            )

        memory_summary = {
            "research_actions": tools_used,
            "tools_used": tools_used,
            "source_count": len(sources),
            "top_references": top_references,
            "sentiment": sentiment.model_dump() if sentiment is not None else {},
            "main_catalyst": extract_first_sentence(analysis_text),
        }

        return {
            "analysis": analysis_text,
            "sources": sources,
            "subgraph_id": task_id,
            "relationships_extracted": relationships_extracted,
            "sentiment": sentiment,
            "memory_summary": memory_summary,
        }
