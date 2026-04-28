"""News analysis agent graph and node logic."""

from __future__ import annotations

import asyncio
import re as _re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Type
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from core.agents.base_agent import AbstractAgent
from core.agents.models.base_agent_models import AgentSentiment, BaseAgentInput
from core.agents.models.news_agent_models import (
    CitedSource,
    DomainQuery,
    NewsAgentOutput,
    NewsAgentState,
    PlannerDecision,
    RelevantChunkSelection,
    ResearchStepLog,
)
from core.agents.news_fetcher import search_web
from core.agents.prompts.news_agent_prompts import (
    NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT,
    NEWS_ANALYSIS_USER_PROMPT,
    NEWS_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT,
    NEWS_PLANNER_SYSTEM_PROMPT,
)
from core.agents.utils import extract_first_sentence, trim_text
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
_WORKING_MEMORY_MAX_TURNS: int = 20
_MIN_RELEVANT_DISTINCT_SOURCES: int = 2


class NewsAnalysisStructuredOutput(BaseModel):
    analysis: str
    sentiment: AgentSentiment = Field(default_factory=AgentSentiment)


@dataclass
class TurnRelevantMemory:
    turn_id: str
    query: str
    chunk_ids: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    score_unavailable: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConversationWorkingMemory:
    agent_memory_context: str = ""
    working_chunks: List[RetrievedChunk] = field(default_factory=list)
    turn_records: List[TurnRelevantMemory] = field(default_factory=list)


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


def _list_unique_actions(logs: List[ResearchStepLog]) -> List[str]:
    actions: List[str] = []
    for row in logs:
        action = str(getattr(row, "action", "") or "").strip()
        if not action or action in {"proceed", "none"} or action in actions:
            continue
        actions.append(action)
    return actions


def _source_key(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata or {}
    return str(metadata.get("source_url") or chunk.source_url or "").strip()


class NewsAnalysisAgent(AbstractAgent):
    """LangGraph-based news analysis agent with iterative research planning."""

    def __init__(self) -> None:
        super().__init__()
        self._llm = service_manager.get_agent()
        self._graph = self._build_graph()
        self._conversation_memory: Dict[str, ConversationWorkingMemory] = {}

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
            f"sources={source_count}; sentiment={sentiment_label or 'N/A'}; "
            f"catalyst={catalyst or 'N/A'}"
        )

    def _get_conversation_memory(
        self, conversation_id: str
    ) -> ConversationWorkingMemory:
        return self._conversation_memory.setdefault(
            conversation_id, ConversationWorkingMemory()
        )

    def _resolve_agent_memory_context(
        self,
        *,
        conversation_id: str,
        incoming_memory_context: str | None,
    ) -> str:
        memory = self._get_conversation_memory(conversation_id)
        incoming = (incoming_memory_context or "").strip()
        if incoming and incoming != memory.agent_memory_context:
            memory.agent_memory_context = incoming
        return incoming or memory.agent_memory_context

    def _persist_agent_memory_summary(
        self, conversation_id: str, rendered_summary: str
    ) -> None:
        if not conversation_id:
            return
        summary = (rendered_summary or "").strip()
        if not summary:
            return
        memory = self._get_conversation_memory(conversation_id)
        memory.agent_memory_context = summary

    def _get_working_memory_chunks(self, conversation_id: str) -> List[RetrievedChunk]:
        if not conversation_id:
            return []
        memory = self._get_conversation_memory(conversation_id)
        return list(memory.working_chunks)

    def _persist_finalized_working_memory(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        query: str,
        chunks: List[RetrievedChunk],
        score_unavailable: bool,
    ) -> None:
        if not conversation_id:
            return
        memory = self._get_conversation_memory(conversation_id)
        existing = memory.working_chunks
        chunk_map: Dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in existing if c.chunk_id
        }
        for chunk in chunks:
            if chunk.chunk_id:
                chunk_map[chunk.chunk_id] = chunk
        merged = list(chunk_map.values())
        if len(merged) > _WORKING_MEMORY_MAX_CHUNKS:
            merged = merged[-_WORKING_MEMORY_MAX_CHUNKS:]
        memory.working_chunks = merged

        source_urls = list(
            dict.fromkeys(_source_key(chunk) for chunk in chunks if _source_key(chunk))
        )
        record = TurnRelevantMemory(
            turn_id=turn_id,
            query=query,
            chunk_ids=[chunk.chunk_id for chunk in chunks if chunk.chunk_id],
            source_urls=source_urls,
            score_unavailable=score_unavailable,
        )
        memory.turn_records.append(record)
        if len(memory.turn_records) > _WORKING_MEMORY_MAX_TURNS:
            memory.turn_records = memory.turn_records[-_WORKING_MEMORY_MAX_TURNS:]

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
            query=input_data.query,
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

    @staticmethod
    def _history_block(logs: List[ResearchStepLog], limit: int = 6) -> str:
        if not logs:
            return "(none)"
        lines: List[str] = []
        for row in logs[-limit:]:
            query_lines = (
                ", ".join(f"{q.domain}:{q.query}" for q in row.queries)
                or row.query
                or "(none)"
            )
            lines.append(
                f"Iteration {row.iteration}\n"
                f"  action: {row.action}\n"
                f"  queries: {query_lines}\n"
                f"  Total fetched articles: {row.total_fetched_articles}\n"
                f"  newly fetched articles: {row.newly_fetched_articles}\n"
                f"  relevant chunks: {row.relevant_chunk_count}\n"
                f"  relevant sources: {row.relevant_source_count}\n"
                f"  score unavailable: {row.score_unavailable}\n"
                f"  note: {row.no_relevant_note or '(none)'}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _chunks_block(chunks: List[RetrievedChunk], *, limit: int = 12) -> str:
        if not chunks:
            return "(none)"
        lines: List[str] = []
        for chunk in chunks[:limit]:
            title = (
                chunk.article_title
                or (chunk.metadata or {}).get("article_title")
                or "Unknown"
            )
            url = _source_key(chunk) or "no-url"
            relevance = chunk.reranker_relevance_score
            relevance_text = "N/A" if relevance is None else f"{relevance:.4f}"
            preview = (chunk.text or "").replace("\n", " ")
            preview = trim_text(preview, max_chars=160)
            lines.append(
                f"- chunk_id={chunk.chunk_id} | title={title} | source={url} | "
                f"relevance_score={relevance_text}\n  text={preview}"
            )
        if len(chunks) > limit:
            lines.append(f"... and {len(chunks) - limit} more chunk(s)")
        return "\n".join(lines)

    def _working_memory_block(
        self, conversation_id: str, *, turn_limit: int = 4
    ) -> str:
        if not conversation_id:
            return "(none)"
        memory = self._conversation_memory.get(conversation_id)
        if memory is None or not memory.turn_records:
            return "(none)"
        lines: List[str] = []
        for row in memory.turn_records[-turn_limit:]:
            ts = row.created_at.isoformat()
            lines.append(
                f"- turn={row.turn_id} at={ts}\n"
                f"  query={row.query}\n"
                f"  relevant_chunks={len(row.chunk_ids)}\n"
                f"  relevant_sources={len(row.source_urls)}\n"
                f"  score_unavailable={row.score_unavailable}"
            )
        return "\n".join(lines)

    @staticmethod
    def _apply_relevant_chunk_selection(
        chunks: List[RetrievedChunk],
        selections: List[RelevantChunkSelection],
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        selected_ids = {s.chunk_id for s in selections if s.chunk_id}
        if not selected_ids:
            return chunks
        filtered = [chunk for chunk in chunks if chunk.chunk_id in selected_ids]
        return filtered or chunks

    async def _planner_node(self, state: NewsAgentState) -> dict:
        """Planner node: decides proceed/fetch and writes per-domain queries."""
        at_limit = (
            state.research_iteration >= settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS
        )
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

        conversation_id = state.conversation_id or ""
        planner_prompt = (
            f"Query: {state.query}\n"
            f"Ticker: {state.ticker or 'N/A'}\n"
            f"Iteration index: {state.research_iteration} (max={settings.NEWS_AGENT_MAX_RESEARCH_ITERATIONS})\n"
            f"Rendezvous gate passed (>= {_MIN_RELEVANT_DISTINCT_SOURCES} distinct relevant sources): "
            f"{state.rendezvous_has_minimum_sources}\n"
            f"Rendezvous score unavailable: {state.rendezvous_score_unavailable}\n"
            f"Relevant chunk count: {state.rendezvous_relevant_chunk_count}\n"
            f"Relevant source count: {state.rendezvous_relevant_source_count}\n"
            f"Relevance threshold: {settings.NEWS_AGENT_MIN_RELEVANCE_SCORE:.2f}\n\n"
            f"Iteration history:\n{self._history_block(state.research_logs)}\n\n"
            f"Current candidate chunks:\n{self._chunks_block(state.final_chunks)}\n\n"
            f"Working memory (prior finalized turns):\n"
            f"{self._working_memory_block(conversation_id)}\n"
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
                    queries=[DomainQuery(domain="company", query=state.query)],
                    rationale="Fallback to NewsAPI on planner failure.",
                    max_results=settings.NEWS_FETCH_MAX_ARTICLES,
                )
            elif settings.TAVILY_API_KEY:
                decision = PlannerDecision(
                    action="web_search",
                    proceed_to_analysis=False,
                    queries=[DomainQuery(domain="company", query=state.query)],
                    rationale="Fallback web search after planner failure.",
                    max_results=settings.TAVILY_SEARCH_MAX_RESULTS,
                )
            else:
                decision = PlannerDecision(
                    action="proceed",
                    proceed_to_analysis=True,
                    rationale="Planner failure and no Tavily key configured.",
                )

        if decision.action == "web_search" and not settings.TAVILY_API_KEY:
            decision = decision.model_copy(
                update={
                    "action": "newsapi",
                    "proceed_to_analysis": False,
                    "rationale": "Tavily key not configured; falling back to NewsAPI.",
                }
            )

        if decision.action != "proceed" and not decision.queries:
            decision = decision.model_copy(
                update={"queries": [DomainQuery(domain="company", query=state.query)]}
            )

        if decision.action == "proceed":
            decision = decision.model_copy(update={"proceed_to_analysis": True})

        # Hard guard: first iteration cannot proceed with no evidence.
        if (
            state.research_iteration == 0
            and not state.final_chunks
            and decision.proceed_to_analysis
        ):
            decision = decision.model_copy(
                update={
                    "action": "newsapi",
                    "proceed_to_analysis": False,
                    "queries": [DomainQuery(domain="company", query=state.query)],
                    "rationale": "Need to fetch at least once before proceeding.",
                }
            )

        # Hard guard: when Jina relevance is available, require rendezvous gate before proceeding.
        if (
            not at_limit
            and not state.rendezvous_score_unavailable
            and state.final_chunks
            and not state.rendezvous_has_minimum_sources
            and decision.proceed_to_analysis
        ):
            fallback_action = "web_search" if settings.TAVILY_API_KEY else "newsapi"
            decision = decision.model_copy(
                update={
                    "action": fallback_action,
                    "proceed_to_analysis": False,
                    "queries": decision.queries
                    or [DomainQuery(domain="company", query=state.query)],
                    "rationale": (
                        f"Cannot proceed yet: fewer than {_MIN_RELEVANT_DISTINCT_SOURCES} "
                        "distinct relevant sources above threshold."
                    ),
                }
            )

        if decision.action == "newsapi":
            max_results = max(
                1, min(decision.max_results, settings.NEWS_FETCH_MAX_ARTICLES)
            )
        else:
            max_results = max(1, min(decision.max_results, 20))
        decision = decision.model_copy(update={"max_results": max_results})

        filtered_chunks = self._apply_relevant_chunk_selection(
            state.final_chunks, decision.relevant_chunks
        )

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
        if not active_domains:
            domain_queries["company"] = state.query
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

        queries = decision.queries or [DomainQuery(domain="company", query=state.query)]
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
                query_text=state.query,
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

        final_ranked = await service_manager.get_reranker().rank(
            state.query, all_candidates
        )
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

        relevant_sources, _ = _build_deduplicated_sources(relevant_chunks)
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
                "analysis": "No relevant news data was found for this query.",
                "sources": [],
                "entities_enriched": [],
            }

        publish_progress(
            "news_agent",
            f"Generating grounded news analysis ({len(chunks)} chunk(s))...",
        )
        sources, chunk_to_source_id = _build_deduplicated_sources(chunks)
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
        return {
            "analysis": analysis_text,
            "sources": sources,
            "subgraph_id": task_id,
            "relationships_extracted": relationships_extracted,
            "sentiment": sentiment,
            "memory_summary": memory_summary,
        }
