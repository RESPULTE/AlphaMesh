"""News analysis agent graph and node logic."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple, Type
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite
from pydantic import BaseModel, ConfigDict, Field

from core.agents.base_agent import AbstractAgent
from core.agents.models.base_agent_models import AgentSentiment, BaseAgentInput
from core.agents.models.news_agent_models import (
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
from core.agents.utils import (
    build_analysis_context_prefix,
    constrain_date_range,
    get_default_date_range,
)
from core.agents.working_memory.news_working_memory import NewsWorkingMemoryManager
from core.config import settings
from core.event_queue import publish_progress, publish_success
from core.logger import get_logger
from core.memory.graph.graph_queue import make_extraction_task
from core.memory.retrieval.models import RetrievedChunk, RewrittenQueries
from core.services import service_manager

logger = get_logger(__name__)

_TARGET_MIN_CHUNKS_PER_SOURCE = 3
_PLANNER_DOMAIN_ORDER = ("company", "sector", "market", "knowledge")


def _history_block(logs: List[ResearchStepLog], limit: int = 6) -> str:
    if not logs:
        return "(none)"
    lines: List[str] = []
    for row in logs[-limit:]:
        query_lines = ", ".join(f"{q.domain}:{q.query}" for q in row.queries)
        lines.append(
            f"Iteration {row.iteration}\n"
            f"  action: {row.action}\n"
            f"  queries: {query_lines or '(none)'}\n"
            f"  total fetched articles: {row.total_fetched_articles}\n"
            f"  newly fetched articles: {row.newly_fetched_articles}\n"
            f"  merged chunks: {row.merged_chunk_count}"
        )
    return "\n\n".join(lines)


def _list_unique_actions(logs: List[ResearchStepLog]) -> List[str]:
    actions: List[str] = []
    for row in logs:
        action = str(row.action or "").strip()
        if action and action not in actions:
            actions.append(action)
    return actions


class NewsAnalysisStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_context_sufficient: bool = False
    analysis: str = ""
    missing_information_goal: str = ""
    persist_chunk_ids: List[int | str] = Field(default_factory=list)
    source_chunk_ids: List[int | str] = Field(default_factory=list)
    sentiment: Optional[AgentSentiment] = None


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
    def render_memory_summary(memory_summary: Dict[str, object]) -> str:
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

    @classmethod
    def _build_missing_goal_output(cls) -> NewsAgentOutput:
        reason = (
            "news_agent requires a non-empty goal. "
            "Goal is now mandatory and query-only execution is deprecated."
        )
        return NewsAgentOutput(
            analysis="News analysis skipped: missing execution goal.",
            sources=[],
            entities_enriched=[],
            memory_summary={
                "bypassed": True,
                "reason": reason,
                "research_actions": [],
                "tools_used": [],
            },
        )

    @staticmethod
    def _resolve_selected_chunks(
        chunks: List[RetrievedChunk],
        selected_chunk_ids: List[str],
    ) -> List[RetrievedChunk]:
        selected_ids = {
            str(chunk_id).strip() for chunk_id in (selected_chunk_ids or [])
        }
        if not selected_ids:
            return chunks
        filtered = [chunk for chunk in chunks if chunk.chunk_id in selected_ids]
        return filtered or chunks

    @staticmethod
    def _dedupe_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        by_id: Dict[str, RetrievedChunk] = {}
        fallback_chunks: List[RetrievedChunk] = []
        for chunk in chunks:
            chunk_id = (chunk.chunk_id or "").strip()
            if not chunk_id:
                fallback_chunks.append(chunk)
                continue
            existing = by_id.get(chunk_id)
            if existing is None:
                by_id[chunk_id] = chunk
                continue
            incoming_score = chunk.relevance_score
            existing_score = existing.relevance_score
            if incoming_score is not None and (
                existing_score is None or incoming_score > existing_score
            ):
                by_id[chunk_id] = chunk
        return list(by_id.values()) + fallback_chunks

    @staticmethod
    def _is_tavily_chunk(chunk: RetrievedChunk) -> bool:
        return str(chunk.relevance_source or "").strip() == "tavily"

    @classmethod
    def _merge_and_rank_chunks(
        cls, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        deduped = cls._dedupe_chunks(chunks)
        tavily_chunks = [chunk for chunk in deduped if cls._is_tavily_chunk(chunk)]
        threshold = settings.NEWS_AGENT_MIN_RELEVANCE_SCORE
        non_tavily_chunks = [
            chunk
            for chunk in deduped
            if not cls._is_tavily_chunk(chunk)
            if chunk.relevance_score is not None and chunk.relevance_score >= threshold
        ]
        ranked_non_tavily = sorted(
            non_tavily_chunks,
            key=lambda chunk: chunk.relevance_score or 0.0,
            reverse=True,
        )
        return cls._dedupe_chunks(tavily_chunks + ranked_non_tavily)

    @staticmethod
    def _build_instructional_missing_goal(
        *,
        base_goal: str,
        missing_goal: str,
    ) -> str:
        resolved_base = (base_goal or "").strip()
        resolved_missing = (missing_goal or "").strip() or resolved_base
        return (
            "Retrieval objective:\n"
            f"- Missing information to retrieve: {resolved_missing}\n"
            "- Construct domain-specific search queries that directly target this missing information.\n"
            "- Prefer high-signal, evidence-bearing terms (entities, events, metrics, timeframe).\n"
        )

    @staticmethod
    def _ordered_planner_domains(queries: List[DomainQuery]) -> List[str]:
        domains = [q.domain for q in queries if q.domain in _PLANNER_DOMAIN_ORDER]
        if not domains:
            return list(_PLANNER_DOMAIN_ORDER)
        return list(dict.fromkeys(domains))

    @staticmethod
    def _build_grouped_query_context_block(
        *,
        planner_domains: List[str],
        final_chunks: List[RetrievedChunk],
        fetched_chunk_ids: List[str],
        memory_chunk_ids: List[str],
    ) -> Tuple[str, Dict[str, str]]:
        chunk_id_to_alias, _ = RetrievedChunk._build_chunk_alias_maps(final_chunks)
        fetched_ids = {chunk_id for chunk_id in fetched_chunk_ids if chunk_id}
        memory_ids = {chunk_id for chunk_id in memory_chunk_ids if chunk_id}
        grouped: Dict[str, List[RetrievedChunk]] = {
            domain: [] for domain in planner_domains
        }

        for chunk in final_chunks:
            chunk_id = (chunk.chunk_id or "").strip()
            if not chunk_id:
                continue

            if chunk_id in fetched_ids:
                metadata = chunk.metadata or {}
                raw_domains = metadata.get("planner_domains") or []
                if isinstance(raw_domains, str):
                    raw_domains = [raw_domains]
                for domain in raw_domains:
                    if domain in grouped:
                        grouped[domain].append(chunk)

            if chunk_id in memory_ids:
                domain = (chunk.domain or "").strip()
                if domain in grouped:
                    grouped[domain].append(chunk)

        lines: List[str] = []
        for domain in planner_domains:
            chunks = grouped.get(domain, [])
            if not chunks:
                continue
            deduped = NewsAnalysisAgent._dedupe_chunks(chunks)
            lines.append(f"[{domain}]")
            for chunk in deduped:
                chunk_alias = chunk_id_to_alias.get((chunk.chunk_id or "").strip(), "")
                if not chunk_alias:
                    continue
                relevance = (
                    "N/A"
                    if chunk.relevance_score is None
                    else f"{float(chunk.relevance_score):.4f}"
                )
                chunk_text = str(chunk.text or "").strip() or "(empty)"
                lines.append(
                    f"- chunk_id={chunk_alias} | date={chunk.date_tag or 'N/A'} | relevance_score={relevance}\n"
                    f"  text={chunk_text}"
                )
            lines.append("")
        return ("\n".join(lines).strip() or "(none)", chunk_id_to_alias)

    async def run(self, input_data: BaseAgentInput) -> NewsAgentOutput:
        """Run the agent end-to-end with the provided input."""
        if not input_data.goal:
            logger.warning("run: missing goal; skipping news execution.")
            return self._build_missing_goal_output()

        start_date, end_date = get_default_date_range(
            input_data.start_date, input_data.end_date
        )
        start_date, end_date = constrain_date_range(start_date, end_date)

        conversation_id = (input_data.conversation_id or "").strip()
        effective_memory_context = self._working_memory.resolve_agent_memory_context(
            conversation_id=conversation_id,
            incoming_memory_context=input_data.agent_memory_context,
        )
        working_memory_chunks = self._working_memory.get_working_memory_chunks(
            conversation_id
        )

        initial_state = NewsAgentState(
            query="",
            goal=input_data.goal,
            ticker=input_data.ticker or "",
            start_date=start_date,
            end_date=end_date,
            conversation_id=conversation_id or None,
            turn_id=input_data.turn_id,
            agent_memory_context=effective_memory_context,
            company_context=input_data.company_context,
            final_chunks=working_memory_chunks,
        )

        final_state = await self._graph.ainvoke(initial_state.model_dump())
        output = NewsAgentOutput(**final_state)
        self._working_memory.persist_agent_memory_summary(
            conversation_id=conversation_id,
            rendered_summary=self.render_memory_summary(output.memory_summary),
        )

        final_chunks: List[RetrievedChunk] = final_state.get("final_chunks") or []
        existing_memory = self._working_memory.get_existing_conversation_memory(
            conversation_id
        )
        turn_index = (
            len(existing_memory.turn_records) if existing_memory is not None else 0
        ) + 1
        self._working_memory.persist_finalized_turn(
            conversation_id=conversation_id,
            turn_index=turn_index,
            chunks=final_chunks,
            source_key_fn=RetrievedChunk._source_key,
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

        workflow.add_conditional_edges(
            START,
            self._route_from_start,
            ["analyse_news", "planner"],
        )
        workflow.add_edge("planner", "retrieve_memory")
        workflow.add_edge("planner", "fetch_and_ingest")

        workflow.add_edge("fetch_and_ingest", "rendezvous")
        workflow.add_edge("retrieve_memory", "rendezvous")
        workflow.add_edge("rendezvous", "analyse_news")
        workflow.add_conditional_edges(
            "analyse_news",
            self._route_after_analysis,
            ["planner", END],
        )
        return workflow.compile()

    @staticmethod
    def _route_from_start(state: NewsAgentState) -> str:
        return "analyse_news" if state.final_chunks else "planner"

    @staticmethod
    def _route_after_analysis(state: NewsAgentState):
        if state.is_context_sufficient:
            return END
        return "planner"

    async def _planner_node(self, state: NewsAgentState) -> dict:
        """Planner node: chooses fetch action and writes per-domain queries."""
        goal = (state.missing_information_goal or state.goal or "").strip()
        planner_prompt = (
            f"Instructional retrieval goal:\n{goal}\n"
            f"Ticker: {state.ticker or 'N/A'}\n"
            f"Iteration index: {state.research_iteration} (max={settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS})\n"
            "\n"
            f"Current turn tool-call history:\n{_history_block(state.research_logs)}\n"
            "\n"
            "Return only PlannerDecision."
        )

        publish_progress(
            "news_agent",
            f"Planning retrieval strategy (iteration {state.research_iteration})...",
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
            decision = PlannerDecision(
                action="newsapi",
                queries=[DomainQuery(domain="company", query=goal)],
            )

        if not decision.queries:
            decision = decision.model_copy(
                update={"queries": [DomainQuery(domain="company", query=goal)]}
            )

        logger.info(
            "_planner_node: iter=%d action=%s queries=%d",
            state.research_iteration,
            decision.action,
            len(decision.queries),
        )
        return {
            "planner_decision": decision,
        }

    async def _retrieve_memory_node(self, state: NewsAgentState) -> dict:
        """Retrieve relevant chunks from semantic memory using planner queries."""
        decision = state.planner_decision
        if decision is None:
            return {"memory_chunks": []}

        domain_queries: Dict[str, str] = {
            q.domain: q.query for q in decision.queries if q.query
        }
        active_domains = list(domain_queries.keys())
        goal = (state.missing_information_goal or state.goal or "").strip()
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
        if decision is None:
            return {"retrieved_chunks": []}

        goal = (state.missing_information_goal or state.goal or "").strip()
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

        url_to_domains: Dict[str, List[str]] = {}
        for domain_query, batch in zip(queries, results_per_query, strict=False):
            for article in batch:
                url = str(article.get("url") or "").strip()
                if not url:
                    continue
                domains = url_to_domains.setdefault(url, [])
                if domain_query.domain not in domains:
                    domains.append(domain_query.domain)

        all_fetched: List[dict] = []
        per_query_target = max(
            _TARGET_MIN_CHUNKS_PER_SOURCE,
            (
                settings.NEWS_FETCH_MAX_ARTICLES
                if action == "newsapi"
                else settings.TAVILY_SEARCH_MAX_RESULTS
            ),
        )
        for batch in results_per_query:
            all_fetched.extend(batch[:per_query_target])

        seen_urls = set(state.seen_urls or [])
        newly_seen_urls: List[str] = []
        new_articles: List[dict] = []
        for article in all_fetched:
            url = str(article.get("url") or "").strip()
            if url and url not in seen_urls:
                new_articles.append(article)
                seen_urls.add(url)
                newly_seen_urls.append(url)

        log_row = ResearchStepLog(
            iteration=iter_label,
            action=action,
            queries=queries,
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
                "seen_urls": newly_seen_urls,
                "research_logs": [log_row],
                "retrieved_chunks": [],
            }

        publish_progress(
            "news_agent", f"Ingesting {len(new_articles)} new article(s) into memory..."
        )
        try:
            new_chunk_ids, existing_chunk_ids, involved_chunks = (
                await service_manager.get_ingestor().ingest_articles(new_articles)
            )
            publish_success(
                "news_agent",
                f"Ingestion done: {len(new_chunk_ids)} new chunks, {len(existing_chunk_ids)} existing",
            )
        except Exception:
            logger.exception("_fetch_and_ingest_node: ingestion failed")
            return {
                "seen_urls": newly_seen_urls,
                "research_logs": [log_row],
                "retrieved_chunks": [],
            }

        chunk_by_id: Dict[str, RetrievedChunk] = {
            chunk.chunk_id: chunk for chunk in involved_chunks if chunk.chunk_id
        }
        ordered_ids = list(dict.fromkeys(new_chunk_ids + existing_chunk_ids))
        ordered_chunks: List[RetrievedChunk] = [
            chunk_by_id[chunk_id] for chunk_id in ordered_ids if chunk_id in chunk_by_id
        ]
        if not ordered_chunks:
            ordered_chunks = list(involved_chunks)

        scored_raw_chunks: List[RetrievedChunk] = []
        if action == "newsapi":
            chunk_ids = [chunk_id for chunk_id in ordered_ids if chunk_id]
            docs_with_scores: List[tuple] = []
            if chunk_ids:
                try:
                    docs_with_scores = await service_manager.get_chroma_adapter().query(
                        query_text=goal,
                        n_results=settings.RETRIEVER_SEED_TOP_K,
                        where={"chunk_id": {"$in": chunk_ids}},
                    )
                except Exception:
                    logger.exception(
                        "_fetch_and_ingest_node: failed to score ingested NewsAPI chunks from vector store"
                    )
            for doc, vector_score in docs_with_scores:
                chunk = RetrievedChunk.from_document(
                    doc,
                    score=vector_score,
                    source="vector",
                    domain="new",
                    relevance_source="vector",
                )
                source_url = str(
                    chunk.source_url or (chunk.metadata or {}).get("source_url") or ""
                ).strip()
                planner_domains = url_to_domains.get(source_url, [])
                if planner_domains:
                    scored_raw_chunks.append(
                        chunk.model_copy(
                            update={
                                "metadata": {
                                    **(chunk.metadata or {}),
                                    "planner_domains": planner_domains,
                                }
                            }
                        )
                    )
                else:
                    scored_raw_chunks.append(chunk)
            if not scored_raw_chunks:
                scored_raw_chunks = [
                    chunk.model_copy(
                        update={
                            "domain": "new",
                            "relevance_source": "vector",
                        }
                    )
                    for chunk in ordered_chunks
                ]
        else:
            tavily_score_by_url: Dict[str, float] = {}
            for article in new_articles:
                raw_score = article.get("tavily_relevance_score")
                url = str(article.get("url") or "").strip()
                if url and isinstance(raw_score, (int, float)):
                    tavily_score_by_url[url] = float(raw_score)
            for chunk in ordered_chunks:
                source_url = str(
                    chunk.source_url or (chunk.metadata or {}).get("source_url") or ""
                ).strip()
                planner_domains = url_to_domains.get(source_url, [])
                scored_raw_chunks.append(
                    chunk.model_copy(
                        update={
                            "relevance_score": float(
                                tavily_score_by_url.get(source_url, 1.0)
                            ),
                            "relevance_source": "tavily",
                            "domain": "new",
                            "metadata": {
                                **(chunk.metadata or {}),
                                "planner_domains": planner_domains,
                            },
                        }
                    )
                )

        existing_ids = {c.chunk_id for c in state.retrieved_chunks if c.chunk_id}
        newly_scored_chunks: List[RetrievedChunk] = []
        for chunk in scored_raw_chunks:
            if chunk.chunk_id and chunk.chunk_id not in existing_ids:
                existing_ids.add(chunk.chunk_id)
                newly_scored_chunks.append(chunk)

        return {
            "seen_urls": newly_seen_urls,
            "research_logs": [log_row],
            "retrieved_chunks": newly_scored_chunks,
        }

    async def _rendezvous_node(self, state: NewsAgentState) -> dict:
        """Merge tool results, memory retrieval, and working memory chunks."""
        planner_domains = self._ordered_planner_domains(
            state.planner_decision.queries if state.planner_decision else []
        )
        working_memory_chunk_ids = [
            (chunk.chunk_id or "").strip()
            for chunk in state.final_chunks
            if chunk.chunk_id
        ]
        fetched_chunk_ids = [
            (chunk.chunk_id or "").strip()
            for chunk in state.retrieved_chunks
            if chunk.chunk_id
        ]
        memory_chunk_ids = [
            (chunk.chunk_id or "").strip()
            for chunk in state.memory_chunks
            if chunk.chunk_id
        ]

        merged = self._dedupe_chunks(
            list(state.final_chunks)
            + list(state.retrieved_chunks)
            + list(state.memory_chunks)
        )
        ranked = self._merge_and_rank_chunks(merged)
        grouped_query_context_block, chunk_id_to_alias = (
            self._build_grouped_query_context_block(
                planner_domains=planner_domains,
                final_chunks=ranked,
                fetched_chunk_ids=fetched_chunk_ids,
                memory_chunk_ids=memory_chunk_ids,
            )
        )
        working_memory_id_set = set(working_memory_chunk_ids)
        ranked_working_memory_chunks = [
            chunk
            for chunk in ranked
            if (chunk.chunk_id or "").strip() in working_memory_id_set
        ]
        working_memory_context_block = (
            RetrievedChunk._render_candidate_chunks(
                ranked_working_memory_chunks,
                chunk_id_to_alias,
            )
            if ranked_working_memory_chunks
            else "(none)"
        )

        if state.conversation_id:
            self._working_memory.merge_working_chunks(
                conversation_id=state.conversation_id,
                chunks=ranked,
            )

        updated_logs = list(state.research_logs)
        if updated_logs:
            last = updated_logs[-1]
            updated_logs[-1] = last.model_copy(
                update={"merged_chunk_count": len(ranked)}
            )

        return {
            "final_chunks": ranked,
            "memory_chunks": Overwrite(value=[]),
            "retrieved_chunks": Overwrite(value=[]),
            "research_logs": Overwrite(value=updated_logs),
            "research_iteration": state.research_iteration + 1,
            "grouped_query_context_block": grouped_query_context_block,
            "working_memory_context_block": working_memory_context_block,
        }

    async def _analyse_news_node(self, state: NewsAgentState) -> dict:
        """Check context sufficiency and generate grounded analysis when possible."""
        chunks = self._dedupe_chunks(state.final_chunks)
        forced_final_pass = (
            state.research_iteration >= settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS
        )

        if not chunks:
            if forced_final_pass:
                return {
                    "analysis": "Insufficient context after maximum retrieval iterations. No relevant news data was found.",
                    "sources": [],
                    "is_context_sufficient": True,
                    "memory_summary": {
                        "research_actions": _list_unique_actions(state.research_logs),
                        "tools_used": _list_unique_actions(state.research_logs),
                        "source_count": 0,
                        "missing_information_goal": state.missing_information_goal
                        or state.goal,
                    },
                    "final_chunks": [],
                }
            return {
                "is_context_sufficient": False,
                "missing_information_goal": self._build_instructional_missing_goal(
                    base_goal=state.goal,
                    missing_goal=state.missing_information_goal or state.goal,
                ),
                "persist_chunk_ids": [],
                "final_chunks": [],
            }

        publish_progress(
            "news_agent",
            f"Evaluating context sufficiency ({len(chunks)} chunk(s))...",
        )

        chunk_id_to_alias, alias_to_chunk_id = RetrievedChunk._build_chunk_alias_maps(
            chunks
        )
        grouped_context_block = (
            state.grouped_query_context_block or ""
        ).strip() or "(none)"
        working_memory_context_block = (
            state.working_memory_context_block or ""
        ).strip() or RetrievedChunk._render_candidate_chunks(chunks, chunk_id_to_alias)

        context_prefix = build_analysis_context_prefix(
            company_context=state.company_context,
            agent_memory_context=state.agent_memory_context,
            cached_entities=[],
        )

        messages = [
            SystemMessage(content=NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=NEWS_ANALYSIS_USER_PROMPT.format(
                    goal=state.missing_information_goal or state.goal,
                    iteration=state.research_iteration,
                    max_iterations=settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS,
                    forced_final_pass=str(forced_final_pass).lower(),
                    entities_section=context_prefix,
                    grouped_context=grouped_context_block,
                    working_memory_context=working_memory_context_block,
                )
            ),
        ]

        relationships_extracted = False
        try:
            structured_llm = self._llm.with_structured_output(
                NewsAnalysisStructuredOutput
            )
            response: NewsAnalysisStructuredOutput = await structured_llm.ainvoke(
                messages
            )
        except Exception as exc:
            logger.error("_analyse_news_node: analysis LLM call failed: %s", exc)
            response = NewsAnalysisStructuredOutput(
                is_context_sufficient=forced_final_pass,
                analysis=(
                    "Insufficient context to answer comprehensively."
                    if not forced_final_pass
                    else "Best-effort analysis could not be generated due to an internal error."
                ),
                missing_information_goal=state.missing_information_goal or state.goal,
                persist_chunk_ids=[],
                sentiment=None,
            )

        selected_chunk_ids = [
            alias_to_chunk_id.get(str(item).strip(), str(item).strip())
            for item in (response.persist_chunk_ids or [])
        ]
        persisted_chunks = self._resolve_selected_chunks(chunks, selected_chunk_ids)

        if not response.is_context_sufficient and not forced_final_pass:
            missing_goal = (response.missing_information_goal or "").strip()
            if not missing_goal:
                missing_goal = state.missing_information_goal or state.goal
            missing_goal = self._build_instructional_missing_goal(
                base_goal=state.goal,
                missing_goal=missing_goal,
            )
            return {
                "is_context_sufficient": False,
                "missing_information_goal": missing_goal,
                "persist_chunk_ids": selected_chunk_ids,
                "final_chunks": persisted_chunks,
                "analysis": "",
                "sources": [],
            }

        selected_source_chunk_ids = [
            alias_to_chunk_id.get(str(item).strip(), str(item).strip())
            for item in (response.source_chunk_ids or response.persist_chunk_ids or [])
        ]
        if not selected_chunk_ids and selected_source_chunk_ids:
            selected_chunk_ids = list(selected_source_chunk_ids)
            persisted_chunks = self._resolve_selected_chunks(chunks, selected_chunk_ids)
        selected_source_chunks = self._resolve_selected_chunks(
            chunks,
            selected_source_chunk_ids,
        )
        sources, _ = RetrievedChunk._build_deduplicated_sources(selected_source_chunks)
        analysis_text = response.analysis

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

        tools_used = _list_unique_actions(state.research_logs)
        top_references = [
            {"source_id": int(src.source_id), "title": src.title, "url": src.url}
            for src in sources[:3]
        ]
        memory_summary = {
            "research_actions": tools_used,
            "tools_used": tools_used,
            "source_count": len(sources),
            "top_references": top_references,
            "sentiment": response.sentiment.model_dump() if response.sentiment else {},
            "missing_information_goal": (
                response.missing_information_goal or ""
            ).strip(),
            "persist_chunk_ids": selected_chunk_ids,
        }

        result = {
            "analysis": analysis_text,
            "sources": sources,
            "subgraph_id": task_id,
            "relationships_extracted": relationships_extracted,
            "sentiment": response.sentiment,
            "memory_summary": memory_summary,
            "is_context_sufficient": True,
            "persist_chunk_ids": selected_chunk_ids,
            "final_chunks": persisted_chunks,
        }
        if response.sentiment is not None:
            result["sentiment"] = response.sentiment

        publish_success("news_agent", "News analysis complete.")
        return result
