"""News analysis agent graph and node logic."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Type
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
    NEWS_PLANNER_SYSTEM_PROMPT,
    build_news_deferred_relationship_system_prompt,
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


class NewsAnalysisStructuredOutputForced(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_context_sufficient: bool = True
    analysis: str = ""
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

    @classmethod
    def _rank_chunks(cls, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        def _is_tavily_chunk(chunk: RetrievedChunk) -> bool:
            return str(chunk.relevance_source or "").strip() == "tavily"

        deduped_chunks = RetrievedChunk._dedupe_chunks_by_article_text(
            RetrievedChunk._dedupe_chunks(chunks)
        )
        tavily_chunks = [chunk for chunk in deduped_chunks if _is_tavily_chunk(chunk)]
        non_tavily_chunks = [
            chunk
            for chunk in deduped_chunks
            if not _is_tavily_chunk(chunk)
            if chunk.relevance_score is not None
            and chunk.relevance_score >= settings.NEWS_AGENT_MIN_RELEVANCE_SCORE
        ]
        ranked_non_tavily = sorted(
            non_tavily_chunks,
            key=lambda chunk: chunk.relevance_score or 0.0,
            reverse=True,
        )
        return RetrievedChunk._dedupe_chunks_by_article_text(
            RetrievedChunk._dedupe_chunks(tavily_chunks + ranked_non_tavily)
        )

    async def _rank_chunks_with_reranker(
        self,
        *,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        baseline_ranked = self._rank_chunks(chunks)
        rerank_query = str(query or "").strip()
        if not baseline_ranked or not rerank_query:
            return baseline_ranked

        normalized_chunks = [
            RetrievedChunk.normalize_for_reranking(chunk, "new")
            for chunk in baseline_ranked
        ]
        try:
            reranked = await service_manager.get_reranker().rank(
                rerank_query, normalized_chunks
            )
            logger.info(
                "_rank_chunks_with_reranker: reranked %d chunks into %d",
                len(normalized_chunks),
                len(reranked),
            )
        except Exception:
            logger.exception(
                "_rank_chunks_with_reranker: failed to rerank; using baseline rank"
            )
            return baseline_ranked

        if not reranked:
            return baseline_ranked
        return self._rank_chunks(reranked)

    @staticmethod
    def _source_url_key(chunk: RetrievedChunk) -> str:
        source_url = str(
            chunk.source_url or (chunk.metadata or {}).get("source_url") or ""
        ).strip()
        return RetrievedChunk._canonicalize_source_url_key(source_url)

    @classmethod
    def _with_chunk_metadata(
        cls,
        chunk: RetrievedChunk,
        *,
        relevance_score: float | None,
        relevance_source: str,
        planner_domains: List[str],
    ) -> RetrievedChunk:
        update_payload = {
            "domain": "new",
            "relevance_source": relevance_source,
            "metadata": {
                **(chunk.metadata or {}),
                "planner_domains": planner_domains,
            },
        }
        if relevance_score is not None:
            update_payload["relevance_score"] = relevance_score
        return chunk.model_copy(update=update_payload)

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

    @classmethod
    def _build_article_context_block(
        cls,
        *,
        final_chunks: List[RetrievedChunk],
    ) -> tuple[str, Dict[str, str]]:
        chunk_id_to_alias, _ = RetrievedChunk._build_chunk_alias_maps(final_chunks)
        grouped: Dict[str, List[RetrievedChunk]] = {}
        group_labels: Dict[str, str] = {}
        group_order: List[str] = []

        for chunk in final_chunks:
            chunk_id = (chunk.chunk_id or "").strip()
            if not chunk_id:
                continue
            metadata = chunk.metadata or {}
            article_title = (
                str(metadata.get("article_title") or chunk.article_title or "").strip()
                or "Unknown Title"
            )
            source_url_key = cls._source_url_key(chunk)
            group_key = source_url_key or f"title:{article_title.lower()}"
            if group_key not in grouped:
                grouped[group_key] = []
                group_labels[group_key] = article_title
                group_order.append(group_key)
            grouped[group_key].append(chunk)

        lines: List[str] = []
        for i, group_key in enumerate(group_order, 1):
            chunks = grouped.get(group_key, [])
            if not chunks:
                continue
            deduped = RetrievedChunk._dedupe_chunks_by_article_text(
                RetrievedChunk._dedupe_chunks(chunks)
            )
            lines.append(
                f"Article {i}: [{group_labels.get(group_key, 'Unknown Title')}]"
            )
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
        seen_url_keys = self._working_memory.get_seen_url_keys(conversation_id)
        seen_chunk_ids = self._working_memory.get_seen_chunk_ids(conversation_id)

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
            seen_url_keys=seen_url_keys,
            seen_chunk_ids=seen_chunk_ids,
        )

        final_state = await self._graph.ainvoke(initial_state.model_dump())
        output = NewsAgentOutput(**final_state)
        self._working_memory.persist_agent_memory_summary(
            conversation_id=conversation_id,
            rendered_summary=self.render_memory_summary(output.memory_summary),
        )
        self._working_memory.merge_seen_history(
            conversation_id=conversation_id,
            url_keys=[
                str(key).strip() for key in (final_state.get("seen_url_keys") or [])
            ],
            chunk_ids=[
                str(chunk_id).strip()
                for chunk_id in (final_state.get("seen_chunk_ids") or [])
            ],
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
            seen_url_keys=self._working_memory.get_seen_url_keys(conversation_id),
            seen_chunk_ids=self._working_memory.get_seen_chunk_ids(conversation_id),
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

        url_key_to_domains: Dict[str, List[str]] = {}
        seen_url_keys = set(state.seen_url_keys or [])
        newly_seen_url_keys: List[str] = []
        new_articles: List[dict] = []
        all_fetched_count = 0
        for domain_query, batch in zip(queries, results_per_query, strict=False):
            all_fetched_count += len(batch)
            for article in batch:
                url = str(article.get("url") or "").strip()
                url_key = NewsWorkingMemoryManager.canonicalize_url_key(url)
                if not url_key:
                    continue

                domains = url_key_to_domains.setdefault(url_key, [])
                if domain_query.domain not in domains:
                    domains.append(domain_query.domain)

                if url_key in seen_url_keys:
                    continue
                new_articles.append(article)
                seen_url_keys.add(url_key)
                newly_seen_url_keys.append(url_key)

        log_row = ResearchStepLog(
            iteration=iter_label,
            action=action,
            queries=queries,
            total_fetched_articles=all_fetched_count,
            newly_fetched_articles=len(new_articles),
        )
        publish_success(
            "news_agent",
            f"Research iter {iter_label}: {action} fetched={all_fetched_count} "
            f"new={len(new_articles)} across {len(queries)} domain query/queries",
        )

        if not new_articles:
            return {
                "seen_url_keys": newly_seen_url_keys,
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
                "seen_url_keys": newly_seen_url_keys,
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
                planner_domains = url_key_to_domains.get(
                    self._source_url_key(chunk), []
                )
                scored_raw_chunks.append(
                    self._with_chunk_metadata(
                        chunk,
                        relevance_score=chunk.relevance_score,
                        relevance_source="vector",
                        planner_domains=planner_domains,
                    )
                )
            if not scored_raw_chunks:
                scored_raw_chunks = [
                    self._with_chunk_metadata(
                        chunk,
                        relevance_score=chunk.relevance_score,
                        relevance_source="vector",
                        planner_domains=url_key_to_domains.get(
                            self._source_url_key(chunk), []
                        ),
                    )
                    for chunk in ordered_chunks
                ]
        else:
            tavily_score_by_url_key: Dict[str, float] = {}
            for article in new_articles:
                raw_score = article.get("tavily_relevance_score")
                url = str(article.get("url") or "").strip()
                url_key = NewsWorkingMemoryManager.canonicalize_url_key(url)
                if url_key and isinstance(raw_score, (int, float)):
                    tavily_score_by_url_key[url_key] = float(raw_score)
            for chunk in ordered_chunks:
                source_url_key = self._source_url_key(chunk)
                planner_domains = url_key_to_domains.get(source_url_key, [])
                scored_raw_chunks.append(
                    self._with_chunk_metadata(
                        chunk,
                        relevance_score=float(
                            tavily_score_by_url_key.get(source_url_key, 1.0)
                        ),
                        relevance_source="tavily",
                        planner_domains=planner_domains,
                    )
                )

        seen_chunk_ids = {
            chunk_id
            for chunk_id in (
                list(state.seen_chunk_ids)
                + [chunk.chunk_id for chunk in state.final_chunks if chunk.chunk_id]
            )
            if chunk_id
        }
        newly_scored_chunks: List[RetrievedChunk] = []
        newly_seen_chunk_ids: List[str] = []
        for chunk in scored_raw_chunks:
            chunk_id = str(chunk.chunk_id or "").strip()
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            newly_seen_chunk_ids.append(chunk_id)
            newly_scored_chunks.append(chunk)

        return {
            "seen_url_keys": newly_seen_url_keys,
            "seen_chunk_ids": newly_seen_chunk_ids,
            "research_logs": [log_row],
            "retrieved_chunks": newly_scored_chunks,
        }

    async def _rendezvous_node(self, state: NewsAgentState) -> dict:
        """Merge tool results, memory retrieval, and working memory chunks."""
        rerank_query = ", ".join([q.query for q in state.planner_decision.queries])
        if rerank_query.lower().startswith("retrieval objective:"):
            rerank_query = str(state.goal or "").strip()

        ranked = await self._rank_chunks_with_reranker(
            query=rerank_query,
            chunks=(
                list(state.final_chunks)
                + list(state.retrieved_chunks)
                + list(state.memory_chunks)
            ),
        )
        logger.info(
            "_rendezvous_node: merged %d working memory + %d retrieved + %d planner chunks into %d ranked",
            len(state.final_chunks),
            len(state.retrieved_chunks),
            len(state.memory_chunks),
            len(ranked),
        )
        article_context_block, _ = self._build_article_context_block(
            final_chunks=ranked,
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
            "article_context_block": article_context_block,
        }

    async def _analyse_news_node(self, state: NewsAgentState) -> dict:
        """Check context sufficiency and generate grounded analysis when possible."""
        chunks = self._rank_chunks(state.final_chunks)
        forced_final_pass = (
            state.research_iteration
        ) >= settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS

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

        _, alias_to_chunk_id = RetrievedChunk._build_chunk_alias_maps(chunks)
        context_prefix = build_analysis_context_prefix(
            company_context=state.company_context,
            agent_memory_context=state.agent_memory_context,
        )

        messages = [
            SystemMessage(content=NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=NEWS_ANALYSIS_USER_PROMPT.format(
                    goal=state.goal,
                    iteration=state.research_iteration,
                    max_iterations=settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS,
                    forced_final_pass=str(forced_final_pass).lower(),
                    entities_section=context_prefix,
                    article_context=state.article_context_block,
                )
            ),
        ]

        relationships_extracted = False
        response_model = (
            NewsAnalysisStructuredOutputForced
            if forced_final_pass
            else NewsAnalysisStructuredOutput
        )
        try:
            structured_llm = self._llm.with_structured_output(response_model)
            response = await structured_llm.ainvoke(messages)
        except Exception as exc:
            logger.error("_analyse_news_node: analysis LLM call failed: %s", exc)
            if forced_final_pass:
                response = NewsAnalysisStructuredOutputForced(
                    is_context_sufficient=True,
                    analysis=(
                        "Best-effort analysis could not be generated due to an internal error."
                    ),
                    source_chunk_ids=[],
                    sentiment=None,
                )
            else:
                response = NewsAnalysisStructuredOutput(
                    is_context_sufficient=False,
                    analysis="Insufficient context to answer comprehensively.",
                    missing_information_goal=state.missing_information_goal
                    or state.goal,
                    persist_chunk_ids=[],
                    sentiment=None,
                )

        final_stage_chunk_ids = list(
            dict.fromkeys(
                str(chunk.chunk_id).strip()
                for chunk in chunks
                if str(chunk.chunk_id).strip()
            )
        )
        available_chunk_ids = set(final_stage_chunk_ids)

        def _normalize_selected_chunk_ids(raw_ids: List[int | str]) -> List[str]:
            mapped_ids = [
                alias_to_chunk_id.get(str(item).strip(), str(item).strip())
                for item in (raw_ids or [])
                if str(item).strip()
            ]
            return [
                chunk_id for chunk_id in mapped_ids if chunk_id in available_chunk_ids
            ]

        response_missing_goal = str(getattr(response, "missing_information_goal", ""))
        response_persist_chunk_ids = list(getattr(response, "persist_chunk_ids", []))
        selected_chunk_ids = _normalize_selected_chunk_ids(response_persist_chunk_ids)

        if not response.is_context_sufficient and not forced_final_pass:
            missing_goal = response_missing_goal.strip()
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
                "final_chunks": chunks,
                "analysis": "",
                "sources": [],
            }

        selected_source_chunk_ids = _normalize_selected_chunk_ids(
            response.source_chunk_ids or response_persist_chunk_ids or []
        )
        if not selected_chunk_ids and selected_source_chunk_ids:
            selected_chunk_ids = list(selected_source_chunk_ids)
        selected_source_chunks = self._resolve_selected_chunks(
            chunks,
            selected_source_chunk_ids,
        )
        sources, _ = RetrievedChunk._build_deduplicated_sources(selected_source_chunks)
        analysis_text = response.analysis

        task_id = None
        if state.conversation_id and analysis_text:
            turn_id = (getattr(state, "turn_id", None) or "").strip() or str(uuid4())
            allowed_entity_types = list(NEWS_DEFERRED_ALLOWED_ENTITY_TYPES)
            allowed_relationship_types = list(NEWS_DEFERRED_ALLOWED_RELATIONSHIP_TYPES)
            relationship_system_prompt = build_news_deferred_relationship_system_prompt(
                allowed_entity_types=allowed_entity_types,
                allowed_relationship_types=allowed_relationship_types,
            )
            try:
                task = make_extraction_task(
                    turn_id=turn_id,
                    conversation_id=state.conversation_id,
                    source_agent=self.name(),
                    extraction_text=analysis_text,
                    chunk_ids=final_stage_chunk_ids,
                    system_prompt=relationship_system_prompt,
                    allowed_entity_types=allowed_entity_types,
                    allowed_relationship_types=allowed_relationship_types,
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
            "missing_information_goal": response_missing_goal.strip(),
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
            "final_chunks": chunks,
        }
        if response.sentiment is not None:
            result["sentiment"] = response.sentiment

        publish_success("news_agent", "News analysis complete.")
        return result
