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
    DomainQuery,
    NewsAgentOutput,
    NewsAgentState,
    QueryRewritePlan,
    ResearchStepLog,
    ResearchStepPlan,
)
from core.agents.news_fetcher import fetch_news
from langgraph.types import Send

from core.agents.prompts.news_agent_prompts import (
    NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT,
    NEWS_ANALYSIS_USER_PROMPT,
    NEWS_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
    NEWS_QUERY_REWRITE_SYSTEM_PROMPT,
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
from core.memory.retrieval.models import RetrievedChunk, RewrittenQueries
from core.services import service_manager

logger = get_logger(__name__)
_ENTITY_CACHE: Dict[str, Dict[Tuple[str, str], Tuple[str, str]]] = {}
_ENTITY_CACHE_TS: Dict[str, float] = {}

# Maximum number of RetrievedChunk objects retained per conversation in working memory.
_WORKING_MEMORY_MAX_CHUNKS: int = 100


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
        entity_lines = [
            f"  - {name} ({entity_type})" for name, entity_type in cached_entities
        ]
        sections.append("Known entities from prior turns:\n" + "\n".join(entity_lines))
    if not sections:
        return ""
    return "\n\n".join(sections) + "\n\n"


def _remap_citations_and_sources(
    analysis_text: str,
    sources: List[CitedSource],
) -> Tuple[str, List[CitedSource]]:
    cited_ids = sorted(
        set(int(match) for match in _re.findall(r"\[(\d+)\]", analysis_text))
    )
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(cited_ids, start=1)}

    def _remap(match: _re.Match[str]) -> str:
        source_id = int(match.group(1))
        return (
            f"[{old_to_new[source_id]}]" if source_id in old_to_new else match.group(0)
        )

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
        # Working memory: stores the reranked RetrievedChunk objects accessed during
        # each turn, keyed by conversation_id.  Chunks are deduplicated by chunk_id
        # and capped at _WORKING_MEMORY_MAX_CHUNKS entries per conversation.
        self._working_memory_by_conversation: Dict[str, List[RetrievedChunk]] = {}

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

    # -- Working memory helpers ------------------------------------------------

    def _get_working_memory_chunks(self, conversation_id: str) -> List[RetrievedChunk]:
        """Return a snapshot of previously accessed chunks for this conversation."""
        if not conversation_id:
            return []
        return list(self._working_memory_by_conversation.get(conversation_id, []))

    def _update_working_memory_chunks(
        self,
        conversation_id: str,
        new_chunks: List[RetrievedChunk],
    ) -> None:
        """Merge new_chunks into this conversation's working memory.

        Deduplicates by chunk_id (new chunks take precedence) and caps the store
        at _WORKING_MEMORY_MAX_CHUNKS, retaining the most-recently merged tail.
        Chunks without a chunk_id are silently skipped.
        """
        if not conversation_id or not new_chunks:
            return
        existing = self._working_memory_by_conversation.get(conversation_id, [])
        chunk_map: Dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in existing if c.chunk_id
        }
        for chunk in new_chunks:
            if chunk.chunk_id:
                chunk_map[chunk.chunk_id] = chunk
        merged = list(chunk_map.values())
        if len(merged) > _WORKING_MEMORY_MAX_CHUNKS:
            merged = merged[-_WORKING_MEMORY_MAX_CHUNKS:]
        self._working_memory_by_conversation[conversation_id] = merged
        logger.debug(
            "_update_working_memory_chunks: conversation=%s stored=%d chunks",
            conversation_id,
            len(merged),
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
        final_state = await self._graph.ainvoke(state_payload)
        output = NewsAgentOutput(**final_state)

        # Persist the reranked chunks from this turn into per-conversation working memory.
        final_chunks: List[RetrievedChunk] = final_state.get("final_chunks") or []
        self._update_working_memory_chunks(conversation_id, final_chunks)

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

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("rewrite_queries", self._rewrite_queries_node)
        workflow.add_node("retrieve_memory", self._retrieve_memory_node)
        workflow.add_node("fetch_and_ingest", self._fetch_and_ingest_node)
        workflow.add_node("rendezvous", self._rendezvous_node)
        workflow.add_node("analyse_news", self._analyse_news_node)

        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self._route_after_planner,
            ["rewrite_queries", "analyse_news"],
        )
        workflow.add_conditional_edges(
            "rewrite_queries",
            self._route_after_rewrite,
            ["retrieve_memory", "fetch_and_ingest"],
        )
        workflow.add_edge("fetch_and_ingest", "rendezvous")
        workflow.add_edge("retrieve_memory", "rendezvous")
        workflow.add_edge("rendezvous", "planner")
        workflow.add_edge("analyse_news", END)

        return workflow.compile()

    @staticmethod
    def _route_after_planner(state: NewsAgentState) -> str:
        """Go to query rewriting when fetching, or directly to analysis."""
        plan = state.research_plan
        if plan is None or plan.action == "proceed":
            return "analyse_news"
        return "rewrite_queries"

    @staticmethod
    def _route_after_rewrite(state: NewsAgentState) -> list:
        """Fan out both parallel branches after query rewriting."""
        return [
            Send("retrieve_memory", state),
            Send("fetch_and_ingest", state),
        ]

    # -- Node: planner (entry point) ------------------------------------------

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

    async def _planner_node(self, state: NewsAgentState) -> dict:
        """Planner: rewrites queries into domains and assesses information sufficiency."""
        at_limit = state.research_iteration >= state.max_research_iterations
        if at_limit:
            reason = f"Reached research iteration limit ({state.max_research_iterations})."
            plan = ResearchStepPlan(action="proceed", rationale=reason)
            publish_progress("news_agent", f"Research planning: {reason}")
            return {"research_plan": plan, "is_information_sufficient": True}

        # Build working memory summary for the LLM prompt.
        conversation_id = state.conversation_id or ""
        wm_chunks = self._get_working_memory_chunks(conversation_id)
        if wm_chunks:
            wm_lines = [f"  {len(wm_chunks)} chunk(s) available from prior turns:"]
            for chunk in wm_chunks[:5]:
                title = (
                    chunk.article_title
                    or (chunk.metadata or {}).get("article_title")
                    or "Unknown"
                )
                snippet = (chunk.text or "")[:120].replace("\n", " ")
                wm_lines.append(f"    - [{title}] {snippet}…")
            if len(wm_chunks) > 5:
                wm_lines.append(f"    … and {len(wm_chunks) - 5} more.")
            wm_block = "\n".join(wm_lines)
        else:
            wm_block = "  Empty (no prior turns in this conversation)."

        source_count = len(state.sources)
        chunk_count = len(state.final_chunks)
        planner_prompt = (
            f"Query: {state.query}\n"
            f"Ticker: {state.ticker or 'N/A'}\n"
            f"Iteration: {state.research_iteration} / {state.max_research_iterations}\n"
            f"Accumulated sources: {source_count}"
            f" (threshold: {settings.NEWS_AGENT_MIN_SOURCES_FOR_SUFFICIENCY})\n"
            f"Accumulated ranked chunks: {chunk_count}"
            f" (threshold: {settings.NEWS_AGENT_MIN_CHUNKS_FOR_SUFFICIENCY})\n"
            f"\nWorking memory (chunks from prior turns of this conversation):\n{wm_block}\n"
            f"\nRecent research history:\n{self._history_block(state.research_logs)}\n"
            "\nReturn only ResearchStepPlan."
        )
        if state.agent_memory_context:
            planner_prompt += (
                "\n\nAgent memory context (summarised prior conversations):\n"
                f"{state.agent_memory_context}"
            )

        publish_progress(
            "news_agent",
            f"Planning research (iteration {state.research_iteration})…",
        )
        try:
            structured_llm = self._llm.with_structured_output(ResearchStepPlan)
            plan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=NEWS_RESEARCH_PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=planner_prompt),
                ]
            )
        except Exception:
            logger.exception("_planner_node: LLM call failed; using fallback")
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
                    rationale="Fallback web search after planner failure.",
                    max_results=settings.TAVILY_SEARCH_MAX_RESULTS,
                )
            else:
                plan = ResearchStepPlan(
                    action="proceed",
                    rationale="Planner failure and no Tavily key configured.",
                )

        # Guard: fall back to newsapi when Tavily key is absent.
        if plan.action == "web_search" and not settings.TAVILY_API_KEY:
            plan = ResearchStepPlan(
                action="newsapi",
                query=plan.query or state.query,
                rationale="Tavily key not configured; falling back to NewsAPI.",
                max_results=plan.max_results,
            )

        if not plan.query and plan.action != "proceed":
            plan.query = state.query
        if plan.action == "newsapi":
            plan.max_results = max(1, min(plan.max_results, settings.NEWS_FETCH_MAX_ARTICLES))
        else:
            plan.max_results = max(1, min(plan.max_results, 20))

        logger.info(
            "_planner_node: iter=%d action=%s query='%.80s'",
            state.research_iteration,
            plan.action,
            plan.query,
        )
        return {
            "research_plan": plan,
            "is_information_sufficient": plan.action == "proceed",
        }

    # -- Node: rewrite_queries -------------------------------------------------

    async def _rewrite_queries_node(self, state: NewsAgentState) -> dict:
        """Rewrite the planner's base query into domain-specific retrieval strings."""
        plan = state.research_plan
        if plan is None:
            return {"rewrite_plan": None}

        # Build the history block so the rewriter can avoid repeating prior queries.
        prior_queries: List[str] = [
            f"  iter={log.iteration} domain=online query='{log.query}'"
            for log in state.research_logs
            if log.query
        ]
        if state.rewrite_plan:
            for dq in state.rewrite_plan.queries:
                prior_queries.append(f"  iter={state.research_iteration} domain={dq.domain} query='{dq.query}'")

        prior_block = "\n".join(prior_queries) if prior_queries else "  (none)"

        rewrite_prompt = (
            f"Original query: {state.query}\n"
            f"Ticker: {state.ticker or 'N/A'}\n"
            f"Iteration: {state.research_iteration}\n"
            f"Planner base query: {plan.query}\n"
            f"Planner action: {plan.action}\n"
            f"\nPrevious retrieval queries across all iterations:\n{prior_block}\n"
            "\nReturn only QueryRewritePlan."
        )

        publish_progress(
            "news_agent",
            f"Rewriting queries (iteration {state.research_iteration})…",
        )
        try:
            structured_llm = self._llm.with_structured_output(QueryRewritePlan)
            rewrite_plan: QueryRewritePlan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=NEWS_QUERY_REWRITE_SYSTEM_PROMPT),
                    HumanMessage(content=rewrite_prompt),
                ]
            )
        except Exception:
            logger.exception("_rewrite_queries_node: LLM call failed; using fallback")
            rewrite_plan = QueryRewritePlan(
                queries=[DomainQuery(domain="company", query=plan.query or state.query)],
                rationale="Fallback: single company query due to rewriter failure.",
            )

        logger.info(
            "_rewrite_queries_node: iter=%d produced %d domain queries: %s",
            state.research_iteration,
            len(rewrite_plan.queries),
            [(dq.domain, dq.query[:60]) for dq in rewrite_plan.queries],
        )
        return {"rewrite_plan": rewrite_plan}

    # -- Node: retrieve_memory -------------------------------------------------

    async def _retrieve_memory_node(self, state: NewsAgentState) -> dict:
        """Retrieve relevant chunks from semantic memory using the rewritten domain queries."""
        rewrite = state.rewrite_plan
        plan = state.research_plan
        if rewrite is None or plan is None:
            return {"memory_chunks": []}

        # Map domain queries from rewrite plan into RewrittenQueries fields.
        domain_queries: Dict[str, str] = {dq.domain: dq.query for dq in rewrite.queries}
        active_domains = list(domain_queries.keys())

        # Guarantee at least one domain.
        if not active_domains:
            domain_queries["company"] = plan.query or state.query
            active_domains = ["company"]

        rewritten = RewrittenQueries(
            company_query=domain_queries.get("company"),
            sector_query=domain_queries.get("sector"),
            market_query=domain_queries.get("market"),
            knowledge_query=domain_queries.get("knowledge"),
            active_domains=active_domains,
            original_query=state.query,
        )

        publish_progress(
            "news_agent",
            f"Retrieving memory ({', '.join(active_domains)})…",
        )
        try:
            svc = service_manager.get_retriever()
            context = await svc.comprehensive_retrieve(rewritten)
            logger.info(
                "_retrieve_memory_node: retrieved %d chunks from domains=%s",
                len(context.chunks),
                active_domains,
            )
            return {"memory_chunks": context.chunks}
        except Exception as exc:
            logger.error("_retrieve_memory_node: retrieval failed: %s", exc)
            return {"memory_chunks": []}

    # -- Node: fetch_and_ingest ------------------------------------------------

    async def _fetch_and_ingest_node(self, state: NewsAgentState) -> dict:
        """Fetch articles for all rewritten queries in parallel, then ingest and score."""
        plan = state.research_plan
        rewrite = state.rewrite_plan
        if plan is None or plan.action == "proceed" or rewrite is None:
            return {"retrieved_chunks": state.retrieved_chunks}

        queries = rewrite.queries or [DomainQuery(domain="company", query=plan.query or state.query)]
        action = plan.action
        iter_label = state.research_iteration + 1

        publish_progress(
            "news_agent",
            f"Research iter {iter_label}: fetching {len(queries)} query/queries via {action}…",
        )

        async def _fetch_one(dq: DomainQuery) -> List[dict]:
            try:
                return await fetch_news(
                    action,
                    dq.query,
                    from_date=state.start_date.isoformat() if action == "newsapi" else None,
                    to_date=state.end_date.isoformat() if action == "newsapi" else None,
                    max_results=plan.max_results,
                    include_domains=plan.include_domains,
                    exclude_domains=plan.exclude_domains,
                )
            except Exception:
                logger.exception(
                    "_fetch_and_ingest_node: fetch failed for domain=%s query='%.80s'",
                    dq.domain,
                    dq.query,
                )
                return []

        results_per_query: List[List[dict]] = await asyncio.gather(
            *(_fetch_one(dq) for dq in queries)
        )
        all_fetched = [a for batch in results_per_query for a in batch]

        # Deduplicate by URL across all fetched articles and against prior seen URLs.
        seen_urls = set(state.seen_urls or [])
        new_articles: List[dict] = []
        for article in all_fetched:
            url = str(article.get("url") or "").strip()
            if url and url not in seen_urls:
                new_articles.append(article)
                seen_urls.add(url)

        query_summary = "; ".join(f"{dq.domain}:{dq.query[:40]}" for dq in queries)
        log_row = ResearchStepLog(
            iteration=iter_label,
            action=action,
            query=query_summary,
            rationale=plan.rationale,
            fetched_articles=len(all_fetched),
            newly_added_articles=len(new_articles),
        )
        publish_success(
            "news_agent",
            f"Research iter {iter_label}: {action} "
            f"fetched={len(all_fetched)} new={len(new_articles)} "
            f"across {len(queries)} domain query/queries",
        )

        if not new_articles:
            logger.info("_fetch_and_ingest_node: no new articles this iteration")
            return {
                "seen_urls": list(seen_urls),
                "research_logs": list(state.research_logs) + [log_row],
                "retrieved_chunks": state.retrieved_chunks,
            }

        publish_progress(
            "news_agent",
            f"Ingesting {len(new_articles)} new article(s) into memory…",
        )
        try:
            new_chunk_ids, existing_chunk_ids, _ = (
                await service_manager.get_ingestor().ingest_articles(new_articles)
            )
            publish_success(
                "news_agent",
                f"Ingestion done: {len(new_chunk_ids)} new chunks, "
                f"{len(existing_chunk_ids)} existing",
            )
        except Exception:
            logger.exception("_fetch_and_ingest_node: ingestion failed")
            return {
                "seen_urls": list(seen_urls),
                "research_logs": list(state.research_logs) + [log_row],
                "retrieved_chunks": state.retrieved_chunks,
            }

        async def _score_chunks(chunk_ids: List[str], domain: str) -> List[RetrievedChunk]:
            if not chunk_ids:
                return []
            docs_with_scores = await service_manager.get_chroma_adapter().query(
                query_text=state.query,
                n_results=settings.RETRIEVER_SEED_TOP_K,
                where={"chunk_id": {"$in": chunk_ids}},
            )
            return [
                RetrievedChunk.from_document(doc, score=score, source="vector", domain=domain)
                for doc, score in docs_with_scores
            ]

        new_chunks, existing_chunks = await asyncio.gather(
            _score_chunks(new_chunk_ids, "new"),
            _score_chunks(existing_chunk_ids, "existing"),
        )

        # Merge into cumulative retrieved_chunks, deduplicating by chunk_id.
        chunk_map: Dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in state.retrieved_chunks if c.chunk_id
        }
        for chunk in new_chunks + existing_chunks:
            if chunk.chunk_id and chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
        merged_chunks = list(chunk_map.values())

        logger.info(
            "_fetch_and_ingest_node: cumulative retrieved chunks=%d", len(merged_chunks)
        )
        return {
            "seen_urls": list(seen_urls),
            "research_logs": list(state.research_logs) + [log_row],
            "retrieved_chunks": merged_chunks,
        }


    # -- Node: rendezvous ------------------------------------------------------

    async def _rendezvous_node(self, state: NewsAgentState) -> dict:
        """Merge parallel branches, rerank, build sources, and loop back to planner."""
        # Merge chunks from the online ingest branch and the memory retrieval branch.
        all_candidates = list(state.retrieved_chunks) + list(state.memory_chunks)
        total = len(all_candidates)
        publish_progress("news_agent", f"Reranking {total} candidate chunk(s)…")

        final_ranked = await service_manager.get_reranker().rank(
            state.query, all_candidates
        )

        # Build deduplicated sources so the planner can use len(state.sources)
        # as a sufficiency signal on the next iteration.
        sources, _ = _build_deduplicated_sources(final_ranked)

        # Entity enrichment from graph store.
        merged_entity_tuples: List[Tuple[str, str]] = []
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

        # Enqueue chunk-entity extraction for any PENDING chunks.
        pending_chunk_ids = list(
            dict.fromkeys(
                chunk.chunk_id
                for chunk in final_ranked
                if chunk.chunk_id and chunk.extraction_status == "PENDING"
            )
        )
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

        logger.info(
            "_rendezvous_node: iter=%d final_ranked=%d sources=%d",
            state.research_iteration,
            len(final_ranked),
            len(sources),
        )
        return {
            "final_chunks": final_ranked,
            "sources": sources,
            # Reset memory_chunks so the next iteration doesn't double-count them.
            "memory_chunks": [],
            # Increment the shared iteration counter once per full fetch cycle.
            "research_iteration": state.research_iteration + 1,
        }

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
