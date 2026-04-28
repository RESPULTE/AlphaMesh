"""News analysis agent graph and node logic."""

from __future__ import annotations

import asyncio
import re as _re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Type
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from core.agents.base_agent import AbstractAgent
from core.agents.models.base_agent_models import AgentSentiment, BaseAgentInput
from core.agents.models.news_agent_models import (
    CitedSource,
    DomainQuery,
    NewsAgentOutput,
    NewsAgentState,
    PlannerDecision,
    ResearchStepLog,
)
from core.agents.news_fetcher import search_web
from core.agents.prompts.news_agent_prompts import (
    NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT,
    NEWS_ANALYSIS_USER_PROMPT,
    NEWS_DEFERRED_ALLOWED_ENTITY_TYPES,
    NEWS_DEFERRED_ALLOWED_RELATIONSHIP_TYPES,
    NEWS_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
    NEWS_PLANNER_SYSTEM_PROMPT,
)
from core.agents.utils import extract_first_sentence
from core.agents.working_memory.news_working_memory import NewsWorkingMemoryManager
from core.config import settings
from core.event_queue import publish_progress, publish_success
from core.logger import get_logger
from core.memory.graph.graph_queue import make_extraction_task
from core.memory.retrieval.models import RetrievedChunk, RewrittenQueries
from core.services import service_manager

logger = get_logger(__name__)
_ENTITY_CACHE: Dict[str, Dict[Tuple[str, str], Tuple[str, str]]] = {}
_ENTITY_CACHE_TS: Dict[str, float] = {}

_MIN_RELEVANT_DISTINCT_SOURCES: int = 2


class NewsAnalysisStructuredOutput(BaseModel):
    analysis: str
    sentiment: Optional[AgentSentiment] = None


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


def _normalize_entity_tuple(name: Any, entity_type: Any) -> Tuple[str, str] | None:
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


def _merge_cached_entities(conversation_id: str, entities: List[Any]) -> None:
    if not conversation_id:
        return
    cache = _ENTITY_CACHE.setdefault(conversation_id, {})
    for entity in entities:
        normalized = _coerce_entity_tuple(entity)
        if normalized is None:
            continue
        cache[normalized] = normalized
    _ENTITY_CACHE_TS[conversation_id] = time.time()


def _build_context_block(
    chunks: List[RetrievedChunk],
    mapping: Dict[int, int],
    rationale_by_chunk_id: Dict[str, str] | None = None,
) -> str:
    rationale_by_chunk_id = rationale_by_chunk_id or {}
    lines: List[str] = []
    for idx, chunk in enumerate(chunks):
        source_id = mapping.get(idx, "?")
        rationale = rationale_by_chunk_id.get(chunk.chunk_id, "").strip()
        if not rationale:
            rationale = "Selected by planner as relevant to the query."
        lines.append(
            f"[{source_id}] Planner relevance rationale: {rationale}\n{chunk.text}"
        )
    return "\n\n".join(lines)


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


class NewsAnalysisAgent(AbstractAgent):
    """LangGraph-based news analysis agent with iterative research planning."""

    def __init__(self) -> None:
        super().__init__()
        self._llm = service_manager.get_agent()
        self._graph = self._build_graph()
        self._working_memory = NewsWorkingMemoryManager()

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
        return NewsWorkingMemoryManager.render_memory_summary(memory_summary)

    @classmethod
    def build_memory_context_from_history(
        cls,
        history_turns: List[dict],
        window: int = 8,
    ) -> str:
        return NewsWorkingMemoryManager.build_context_from_history_summaries(
            history_turns, window=window
        )

    def _resolve_agent_memory_context(
        self,
        *,
        conversation_id: str,
        incoming_memory_context: str | None,
    ) -> str:
        return self._working_memory.resolve_agent_memory_context(
            conversation_id=conversation_id,
            incoming_memory_context=incoming_memory_context,
        )

    def _persist_agent_memory_summary(
        self, conversation_id: str, rendered_summary: str
    ) -> None:
        self._working_memory.persist_agent_memory_summary(
            conversation_id=conversation_id,
            rendered_summary=rendered_summary,
        )

    def _get_working_memory_chunks(self, conversation_id: str) -> List[RetrievedChunk]:
        return self._working_memory.get_working_memory_chunks(conversation_id)

    def _persist_finalized_working_memory(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        query: str,
        chunks: List[RetrievedChunk],
        score_unavailable: bool,
    ) -> None:
        self._working_memory.persist_finalized_turn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            query=query,
            chunks=chunks,
            score_unavailable=score_unavailable,
            source_key_fn=RetrievedChunk._source_key,
        )

    async def run(self, input_data: BaseAgentInput) -> NewsAgentOutput:
        """Run the agent end-to-end with the provided input."""
        start_date, end_date = _get_default_date_range(
            input_data.start_date, input_data.end_date
        )
        start_date, end_date = _constrain_date_range(start_date, end_date)

        conversation_id = (input_data.conversation_id or "").strip()
        effective_memory_context = self._resolve_agent_memory_context(
            conversation_id=conversation_id,
            incoming_memory_context=input_data.agent_memory_context,
        )

        initial_state = NewsAgentState(
            query=input_data.query,
            goal=input_data.goal,
            ticker=input_data.ticker or "",
            start_date=start_date,
            end_date=end_date,
            conversation_id=conversation_id or None,
            turn_id=input_data.turn_id,
            agent_memory_context=effective_memory_context,
            company_context=input_data.company_context,
        )

        final_state = await self._graph.ainvoke(initial_state.model_dump())
        output = NewsAgentOutput(**final_state)

        final_chunks: List[RetrievedChunk] = final_state.get("final_chunks") or []
        turn_id = (input_data.turn_id or "").strip() or str(uuid4())
        self._persist_finalized_working_memory(
            conversation_id=conversation_id,
            turn_id=turn_id,
            query=input_data.goal,
            chunks=final_chunks,
            score_unavailable=bool(
                final_state.get("rendezvous_score_unavailable", False)
            ),
        )
        self._persist_agent_memory_summary(
            conversation_id=conversation_id,
            rendered_summary=self.render_memory_summary(output.memory_summary),
        )
        return output

    def _build_graph(self):
        """Compile the iterative LangGraph workflow."""
        workflow = StateGraph(NewsAgentState, output_schema=NewsAgentOutput)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retrieve_memory", self._retrieve_memory_node)
        workflow.add_node("fetch_and_ingest", self._fetch_and_ingest_node)
        workflow.add_node("rendezvous", self._rendezvous_node)
        workflow.add_node("analyse_news", self._analyse_news_node)

        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self._route_after_planner,
            ["analyse_news", "retrieve_memory", "fetch_and_ingest"],
        )
        workflow.add_edge("fetch_and_ingest", "rendezvous")
        workflow.add_edge("retrieve_memory", "rendezvous")
        workflow.add_edge("rendezvous", "planner")
        workflow.add_edge("analyse_news", END)
        return workflow.compile()

    @staticmethod
    def _route_after_planner(state: NewsAgentState):
        decision = state.planner_decision
        if (
            decision is None
            or decision.proceed_to_analysis
            or decision.action == "proceed"
        ):
            return "analyse_news"
        return [
            Send("retrieve_memory", state),
            Send("fetch_and_ingest", state),
        ]

    async def _planner_node(self, state: NewsAgentState) -> dict:
        """Planner node: decides proceed/fetch and writes per-domain queries."""
        at_limit = (
            state.research_iteration >= settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS
        )
        goal = state.goal
        if at_limit:
            reason = f"Reached research iteration limit ({settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS})."
            decision = PlannerDecision(
                action="proceed", proceed_to_analysis=True, rationale=reason
            )
            publish_progress("news_agent", f"Research planning: {reason}")
            return {
                "planner_decision": decision,
                "is_information_sufficient": True,
            }

        # Enforce rendezvous source gate in code before planner LLM.
        if (
            not state.rendezvous_score_unavailable
            and state.final_chunks
            and not state.rendezvous_has_minimum_sources
        ):
            decision = PlannerDecision(
                action="web_search",
                proceed_to_analysis=False,
                queries=[DomainQuery(domain="company", query=goal)],
                rationale=(
                    f"Cannot proceed yet: fewer than {_MIN_RELEVANT_DISTINCT_SOURCES} "
                    "distinct relevant sources above threshold."
                ),
                max_results=settings.TAVILY_SEARCH_MAX_RESULTS,
            )
            return {
                "planner_decision": decision,
                "is_information_sufficient": False,
                "final_chunks": state.final_chunks,
            }

        conversation_id = state.conversation_id or ""
        planner_prompt = (
            f"Goal: {goal}\n"
            f"Ticker: {state.ticker or 'N/A'}\n"
            f"Iteration index: {state.research_iteration} (max={settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS})\n"
            "\n"
            f"Iteration history:\n{ResearchStepLog._history_block(state.research_logs)}\n\n"
            f"Current candidate chunks:\n{RetrievedChunk._chunks_block(state.final_chunks)}\n\n"
            f"Working memory (prior finalized turns):\n"
            f"{self._working_memory.render_working_memory_block(conversation_id, turn_limit=4)}\n"
        )
        if state.agent_memory_context:
            planner_prompt += (
                "\nAgent memory context (summarized prior turns):\n"
                f"{state.agent_memory_context}\n"
            )
        planner_prompt += "\nReturn only PlannerDecision."

        publish_progress(
            "news_agent",
            f"Planning research (iteration {state.research_iteration})...",
        )

        try:
            structured_llm = self._llm.with_structured_output(PlannerDecision)
            decision: PlannerDecision = await structured_llm.ainvoke(
                [
                    SystemMessage(content=NEWS_PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=planner_prompt),
                ]
            )
        except Exception:
            logger.exception("_planner_node: LLM call failed; using fallback")
            if state.research_iteration == 0:
                decision = PlannerDecision(
                    action="newsapi",
                    proceed_to_analysis=False,
                    queries=[DomainQuery(domain="company", query=goal)],
                    rationale="Fallback to NewsAPI on planner failure.",
                    max_results=settings.NEWS_FETCH_MAX_ARTICLES,
                )
            else:
                decision = PlannerDecision(
                    action="web_search",
                    proceed_to_analysis=False,
                    queries=[DomainQuery(domain="company", query=goal)],
                    rationale="Fallback web search after planner failure.",
                    max_results=settings.TAVILY_SEARCH_MAX_RESULTS,
                )

        if decision.action != "proceed" and not decision.queries:
            decision = decision.model_copy(
                update={"queries": [DomainQuery(domain="company", query=goal)]}
            )

        if decision.action == "proceed":
            decision = decision.model_copy(update={"proceed_to_analysis": True})

        if decision.action == "newsapi":
            max_results = max(
                1, min(decision.max_results, settings.NEWS_FETCH_MAX_ARTICLES)
            )
        else:
            max_results = max(1, min(decision.max_results, 20))
        decision = decision.model_copy(update={"max_results": max_results})

        selected_ids = {s.chunk_id for s in decision.relevant_chunks if s.chunk_id}
        filtered_chunks = [
            chunk for chunk in state.final_chunks if chunk.chunk_id in selected_ids
        ]
        if not filtered_chunks:
            filtered_chunks = state.final_chunks

        logger.info(
            "_planner_node: iter=%d action=%s proceed=%s queries=%d",
            state.research_iteration,
            decision.action,
            decision.proceed_to_analysis,
            len(decision.queries),
        )
        return {
            "planner_decision": decision,
            "is_information_sufficient": decision.proceed_to_analysis,
            "final_chunks": filtered_chunks,
        }

    async def _retrieve_memory_node(self, state: NewsAgentState) -> dict:
        """Retrieve relevant chunks from semantic memory using planner queries."""
        decision = state.planner_decision
        if decision is None or decision.action == "proceed":
            return {"memory_chunks": []}

        domain_queries: Dict[str, str] = {
            q.domain: q.query for q in decision.queries if q.query
        }
        active_domains = list(domain_queries.keys())
        goal = state.goal
        if not active_domains:
            domain_queries["company"] = goal
            active_domains = ["company"]

        rewritten = RewrittenQueries(
            company_query=domain_queries.get("company"),
            sector_query=domain_queries.get("sector"),
            market_query=domain_queries.get("market"),
            knowledge_query=domain_queries.get("knowledge"),
            active_domains=active_domains,
            original_query=goal,
        )

        publish_progress(
            "news_agent", f"Retrieving memory ({', '.join(active_domains)})..."
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

    async def _fetch_and_ingest_node(self, state: NewsAgentState) -> dict:
        """Fetch articles for planner queries in parallel, then ingest and score."""
        decision = state.planner_decision
        if decision is None or decision.action == "proceed":
            return {"retrieved_chunks": state.retrieved_chunks}

        goal = state.goal
        queries = decision.queries or [DomainQuery(domain="company", query=goal)]
        action = decision.action
        iter_label = state.research_iteration + 1
        publish_progress(
            "news_agent",
            f"Research iter {iter_label}: fetching {len(queries)} query/queries via {action}...",
        )

        async def _fetch_one(dq: DomainQuery) -> List[dict]:
            try:
                return await search_web(
                    action,
                    dq.query,
                    from_date=(
                        state.start_date.isoformat() if action == "newsapi" else None
                    ),
                    to_date=state.end_date.isoformat() if action == "newsapi" else None,
                )
            except Exception:
                logger.exception(
                    "_fetch_and_ingest_node: fetch failed for domain=%s query='%.80s'",
                    dq.domain,
                    dq.query,
                )
                return []

        results_per_query: List[List[dict]] = await asyncio.gather(
            *(_fetch_one(q) for q in queries)
        )
        all_fetched = [article for batch in results_per_query for article in batch]

        seen_urls = set(state.seen_urls or [])
        new_articles: List[dict] = []
        for article in all_fetched:
            url = str(article.get("url") or "").strip()
            if url and url not in seen_urls:
                new_articles.append(article)
                seen_urls.add(url)

        log_row = ResearchStepLog(
            iteration=iter_label,
            action=action,
            query="; ".join(f"{q.domain}:{q.query[:40]}" for q in queries),
            queries=queries,
            rationale=decision.rationale,
            total_fetched_articles=len(all_fetched),
            newly_fetched_articles=len(new_articles),
        )
        publish_success(
            "news_agent",
            f"Research iter {iter_label}: {action} fetched={len(all_fetched)} "
            f"new={len(new_articles)} across {len(queries)} domain query/queries",
        )

        if not new_articles:
            return {
                "seen_urls": list(seen_urls),
                "research_logs": list(state.research_logs) + [log_row],
                "retrieved_chunks": state.retrieved_chunks,
            }

        publish_progress(
            "news_agent", f"Ingesting {len(new_articles)} new article(s) into memory..."
        )
        try:
            new_chunk_ids, existing_chunk_ids, _ = (
                await service_manager.get_ingestor().ingest_articles(new_articles)
            )
            publish_success(
                "news_agent",
                f"Ingestion done: {len(new_chunk_ids)} new chunks, {len(existing_chunk_ids)} existing",
            )
        except Exception:
            logger.exception("_fetch_and_ingest_node: ingestion failed")
            return {
                "seen_urls": list(seen_urls),
                "research_logs": list(state.research_logs) + [log_row],
                "retrieved_chunks": state.retrieved_chunks,
            }

        async def _score_chunks(
            chunk_ids: List[str], domain: str
        ) -> List[RetrievedChunk]:
            if not chunk_ids:
                return []
            docs_with_scores = await service_manager.get_chroma_adapter().query(
                query_text=goal,
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
            _score_chunks(new_chunk_ids, "new"),
            _score_chunks(existing_chunk_ids, "existing"),
        )

        chunk_map: Dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in state.retrieved_chunks if c.chunk_id
        }
        for chunk in new_chunks + existing_chunks:
            if chunk.chunk_id and chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        return {
            "seen_urls": list(seen_urls),
            "research_logs": list(state.research_logs) + [log_row],
            "retrieved_chunks": list(chunk_map.values()),
        }

    async def _rendezvous_node(self, state: NewsAgentState) -> dict:
        """Merge parallel branches, rerank, apply threshold gate, then loop to planner."""
        conversation_id = state.conversation_id or ""
        all_candidates = (
            list(state.retrieved_chunks)
            + list(state.memory_chunks)
            + self._get_working_memory_chunks(conversation_id)
        )
        publish_progress(
            "news_agent", f"Reranking {len(all_candidates)} candidate chunk(s)..."
        )

        goal = state.goal
        final_ranked = await service_manager.get_reranker().rank(goal, all_candidates)
        score_available = any(
            chunk.reranker_relevance_score is not None for chunk in final_ranked
        )
        threshold = settings.NEWS_AGENT_MIN_RELEVANCE_SCORE
        if score_available:
            relevant_chunks = [
                chunk
                for chunk in final_ranked
                if chunk.reranker_relevance_score is not None
                and chunk.reranker_relevance_score >= threshold
            ]
        else:
            relevant_chunks = list(final_ranked)

        relevant_sources, _ = RetrievedChunk._build_deduplicated_sources(
            relevant_chunks
        )
        distinct_source_keys = {
            f"{src.title}|{src.url}"
            for src in relevant_sources
            if (src.title or "").strip() or (src.url or "").strip()
        }
        has_minimum_sources = (
            score_available
            and len(distinct_source_keys) >= _MIN_RELEVANT_DISTINCT_SOURCES
            and len(relevant_chunks) > 0
        )

        # Entity enrichment from graph store.
        merged_entity_tuples: List[Tuple[str, str]] = []
        final_ranked_chunk_ids = list(
            dict.fromkeys(chunk.chunk_id for chunk in relevant_chunks if chunk.chunk_id)
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
                        row.get("entity_name"), row.get("entity_type")
                    )
                    if parsed is not None:
                        merged_entity_tuples.append(parsed)
            except Exception:
                logger.exception(
                    "_rendezvous_node: failed to fetch entity tuples for relevant chunks"
                )

        if conversation_id and merged_entity_tuples:
            _merge_cached_entities(conversation_id, merged_entity_tuples)

        # Enqueue chunk-entity extraction for any PENDING chunks.
        pending_chunk_ids = list(
            dict.fromkeys(
                chunk.chunk_id
                for chunk in relevant_chunks
                if chunk.chunk_id and chunk.extraction_status == "PENDING"
            )
        )
        if pending_chunk_ids and conversation_id:
            turn_id = (getattr(state, "turn_id", None) or "").strip() or str(uuid4())
            try:
                task = make_extraction_task(
                    turn_id=turn_id,
                    conversation_id=conversation_id,
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

        updated_logs = list(state.research_logs)
        no_relevant_note = ""
        if score_available and len(relevant_chunks) == 0:
            no_relevant_note = f"No chunks met relevance threshold {threshold:.2f} from this iteration's results."
        elif (
            score_available
            and len(distinct_source_keys) < _MIN_RELEVANT_DISTINCT_SOURCES
        ):
            no_relevant_note = (
                f"Relevant chunks found but only {len(distinct_source_keys)} distinct source(s); "
                f"need at least {_MIN_RELEVANT_DISTINCT_SOURCES}."
            )

        if updated_logs and updated_logs[-1].action in {"newsapi", "web_search"}:
            last = updated_logs[-1]
            updated_logs[-1] = last.model_copy(
                update={
                    "relevant_chunk_count": len(relevant_chunks),
                    "relevant_source_count": len(distinct_source_keys),
                    "score_unavailable": not score_available,
                    "no_relevant_note": no_relevant_note,
                }
            )

        logger.info(
            "_rendezvous_node: iter=%d total_chunks=%d ranked=%d relevant=%d sources=%d score_available=%s",
            state.research_iteration,
            len(all_candidates),
            len(final_ranked),
            len(relevant_chunks),
            len(distinct_source_keys),
            score_available,
        )
        return {
            "final_chunks": relevant_chunks,
            "sources": relevant_sources,
            "memory_chunks": [],
            "research_logs": updated_logs,
            "rendezvous_has_minimum_sources": has_minimum_sources,
            "rendezvous_score_unavailable": not score_available,
            "rendezvous_relevant_chunk_count": len(relevant_chunks),
            "rendezvous_relevant_source_count": len(distinct_source_keys),
            "research_iteration": state.research_iteration + 1,
        }

    async def _analyse_news_node(self, state: NewsAgentState) -> dict:
        """Generate a grounded financial analysis from retrieved chunks."""
        chunks = state.final_chunks
        if not chunks:
            return {
                "analysis": "No relevant news data was found for this goal.",
                "sources": [],
                "entities_enriched": [],
            }

        publish_progress(
            "news_agent",
            f"Generating grounded news analysis ({len(chunks)} chunk(s))...",
        )
        sources, chunk_to_source_id = RetrievedChunk._build_deduplicated_sources(chunks)
        planner_reasons: Dict[str, str] = {}
        decision = state.planner_decision
        if decision is not None:
            for selected in decision.relevant_chunks:
                chunk_id = (selected.chunk_id or "").strip()
                if not chunk_id:
                    continue
                planner_reasons[chunk_id] = (selected.reason or "").strip()
        context_block = _build_context_block(
            chunks,
            chunk_to_source_id,
            planner_reasons,
        )

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
                    goal=state.goal,
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
                    allowed_entity_types=list(NEWS_DEFERRED_ALLOWED_ENTITY_TYPES),
                    allowed_relationship_types=list(
                        NEWS_DEFERRED_ALLOWED_RELATIONSHIP_TYPES
                    ),
                    llm_config={"temperature": getattr(self._llm, "temperature", 0.7)},
                )
                task_id = await service_manager.get_graph_queue_manager().enqueue(task)
            except Exception:
                logger.exception("_analyse_news_node: failed to enqueue graph task")

        tools_used = ResearchStepLog._list_unique_actions(state.research_logs)
        top_references = [
            {"source_id": int(src.source_id), "title": src.title, "url": src.url}
            for src in sources[:3]
        ]
        memory_summary = {
            "research_actions": tools_used,
            "tools_used": tools_used,
            "source_count": len(sources),
            "top_references": top_references,
            "sentiment": sentiment.model_dump() if sentiment is not None else {},
            "main_catalyst": extract_first_sentence(analysis_text),
        }
        result = {
            "analysis": analysis_text,
            "sources": sources,
            "subgraph_id": task_id,
            "relationships_extracted": relationships_extracted,
            "sentiment": sentiment,
            "memory_summary": memory_summary,
        }

        if sentiment is not None:
            result["sentiment"] = sentiment
        return result
