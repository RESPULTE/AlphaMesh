"""News analysis agent with dual-store ingestion and chunk-level extraction."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Type

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
from core.config import settings
from core.logger import get_logger
from core.memory.graph.extraction_prompts import (
    ANALYSIS_ONLY_RELATIONSHIP_PROMPT,
    COMBINED_ANALYSIS_RELATIONSHIP_PROMPT,
)
from core.memory.graph.relationship_extractor import (
    extract_with_retry,
    retry_relationships_only,
)
from core.memory.graph.subgraph_builder import InMemorySubgraphBuilder
from core.memory.retrieval.models import RetrievedChunk
from core.memory.stores.subgraph_store import SubgraphStore
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
        entity_id = entity.get("entity_id")
        entity_name = entity.get("entity_name")
        entity_type = entity.get("entity_type")
        if not entity_id or not entity_name or not entity_type:
            continue

        cache[entity_id] = {
            "id": entity_id,
            "name": entity_name,
            "entity_type": entity_type,
        }
    _ENTITY_CACHE_TS[conversation_id] = time.time()


class NewsAnalysisAgent(AbstractAgent):
    """LangGraph-based news analysis agent."""

    def __init__(self) -> None:
        """Initialize the agent and compile the graph."""
        super().__init__()
        self._llm = service_manager.get_agent()
        self._graph = self._build_graph()

    @staticmethod
    def name() -> str:
        """Return the agent name."""
        return "news_agent"

    @staticmethod
    def description() -> str:
        """Return the agent description."""
        return "Ingests news into dual stores, schedules background extraction, and synthesizes analysis."

    @staticmethod
    def get_output_schema_class() -> Type[BaseModel]:
        """Return the output schema class."""
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
            memory_task=input_data.memory_task,
            conversation_id=input_data.conversation_id,
        )

        state_payload = initial_state.model_dump()
        state_payload["memory_task"] = input_data.memory_task
        final_state = await self._graph.ainvoke(state_payload)
        return NewsAgentOutput(**final_state)

    def _build_graph(self):
        """Compile the linear LangGraph workflow."""
        workflow = StateGraph(NewsAgentState, output_schema=NewsAgentOutput)

        workflow.add_node("fetch_news", self._fetch_news_node)
        workflow.add_node("ingest_articles", self._ingest_articles_node)
        workflow.add_node("rendezvous", self._rendezvous_node)
        workflow.add_node("analyse_news", self._analyse_news_node)

        workflow.add_edge(START, "fetch_news")
        workflow.add_edge("fetch_news", "ingest_articles")
        workflow.add_edge("ingest_articles", "rendezvous")
        workflow.add_edge("rendezvous", "analyse_news")
        workflow.add_edge("analyse_news", END)

        return workflow.compile()

    async def _fetch_news_node(self, state: NewsAgentState) -> dict:
        """
        Fetch news articles from NewsAPI and enrich them with full content
        scraped by trafilatura.

        Changes vs. the original node
        ──────────────────────────────
        • build_news_query() constructs a boolean NewsAPI query that combines
          the ticker symbol with optional company name / keywords.
        • Results are filtered to a curated list of trusted financial domains.
        • trafilatura replaces the truncated NewsAPI `content` field (~200 chars)
          with the full article body, which the downstream chunker/embedder can
          use without losing context.
        • Falls back gracefully to the NewsAPI snippet when scraping fails
          (e.g. paywalled pages).
        """
        now = datetime.now()
        api_limit_date = now - timedelta(days=28)

        # Clamp dates to NewsAPI's allowed window
        start = state.start_date
        end = state.end_date

        if end > now:
            end = now
        if start < api_limit_date:
            start = api_limit_date

        # ── Build an advanced boolean query ─────────────────────────────────
        # Modify must_include / must_exclude / any_of to taste, or expose them
        # as fields on NewsAgentState for per-request customisation.
        q = build_news_query(
            ticker=state.ticker,
            # company_name="Apple Inc",      # optional: improves recall
            # must_include=["earnings"],     # optional: narrow to a topic
            # must_exclude=["crypto"],       # optional: filter noise
            # any_of=["revenue", "profit"],  # optional: OR group
            # exact_phrase="quarterly results",  # optional: exact match
        )
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

    async def _ingest_articles_node(self, state: NewsAgentState) -> dict:
        """Ingest articles into Neo4j and ChromaDB."""
        if not state.raw_articles:
            return {"chunk_ids": []}

        try:
            chunk_ids, _ = await service_manager.get_ingestor().ingest_articles(
                state.raw_articles
            )
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            raise

        retrieved_chunks: List[RetrievedChunk] = []
        if chunk_ids:
            query = state.query
            docs_with_scores = await service_manager.get_chroma_adapter().query(
                query_text=query, n_results=10, where={"chunk_id": {"$in": chunk_ids}}
            )
            retrieved_chunks = [
                RetrievedChunk.from_document(
                    doc, score=score, source="vector", domain="new"
                )
                for doc, score in docs_with_scores
            ]

        logger.info("Ingested articles into memory. #Chunk IDs: %s", len(chunk_ids))
        return {"chunk_ids": chunk_ids, "retrieved_chunks": retrieved_chunks}

    async def _rendezvous_node(self, state: NewsAgentState) -> dict:
        """Merge memory retrieval results with freshly ingested chunks."""
        memory_context = None
        if state.memory_task is not None:
            try:
                memory_context = await state.memory_task
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

    async def _analyse_news_node(self, state: NewsAgentState) -> dict:
        """Generate a grounded financial analysis from retrieved chunks."""
        if not state.final_chunks:
            return {"analysis": "No relevant news articles found.", "sources": []}

        extracted_entities: List[object] = []
        chunk_ids = [chunk.chunk_id for chunk in state.final_chunks if chunk.chunk_id]
        if chunk_ids:
            extracted_entities = (
                await service_manager.get_ingestor().extract_entities_for_chunks(
                    chunk_ids
                )
            )
        if state.conversation_id:
            _merge_cached_entities(state.conversation_id, extracted_entities)
            cached_entities = _get_cached_entities(state.conversation_id)
        else:
            cached_entities = []

        sources: List[CitedSource] = []
        for idx, chunk in enumerate(state.final_chunks, start=1):
            metadata = chunk.metadata or {}
            sources.append(
                CitedSource(
                    source_id=idx,
                    title=metadata.get("article_title", "Unknown Title"),
                    url=metadata.get("source_url", ""),
                    page_content=chunk.text,
                )
            )

        context_blocks = []
        for chunk, source in zip(state.final_chunks, sources):
            label = "NEW" if chunk.domain == "new" else f"MEMORY:{chunk.domain}"
            context_blocks.append(
                f"[{label} | score={chunk.composite_score:.2f}] {source.title}\n"
                f"{source.page_content}"
            )
        context = "\n\n".join(context_blocks)

        system_prompt = COMBINED_ANALYSIS_RELATIONSHIP_PROMPT
        known_entities_block = ""
        if cached_entities:
            known_entities_lines = [
                f"{entity['entity_type']}: {entity['name']}"
                for entity in cached_entities
            ]
            known_entities_block = (
                "Known entities (from extracted chunks):\n"
                + "\n".join(known_entities_lines)
            )

        prompt_parts = [f"Question: {state.query}"]
        if known_entities_block:
            prompt_parts.append(known_entities_block)
        prompt_parts.append(f"Context:\n{context}")
        prompt_parts.append(
            "Provide a concise, evidence-based analysis. "
            "When extracting relationships, you may use the known entities list; "
            "do not invent new entities."
        )
        user_prompt = "\n\n".join(prompt_parts)

        try:
            result = await extract_with_retry(
                self._llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
            analysis_text = result.analysis
            relationships = result.relationships
            relationships_extracted = result.parse_success
        except Exception as exc:
            logger.error("Analysis generation failed: %s", exc)
            raise

        subgraph_id = None
        if settings.EXTRACTION_ENABLED and state.conversation_id:
            builder = InMemorySubgraphBuilder(
                embedding_func=service_manager.get_embedding_func(),
                fuzzy_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
                semantic_threshold=settings.EXTRACTION_SEMANTIC_THRESHOLD,
            )
            store = service_manager.get_subgraph_store()
            subgraph_id = SubgraphStore.make_key(self.name(), state.conversation_id)

            async def _build_and_store():
                graph = await builder.build(relationships, source_agent=self.name())
                await store.save(subgraph_id, graph)

            if relationships_extracted:
                task = asyncio.create_task(_build_and_store())
            else:
                task = asyncio.create_task(
                    retry_relationships_only(
                        self._llm,
                        analysis_text,
                        self.name(),
                        state.conversation_id,
                        builder,
                        store,
                        subgraph_id,
                        ANALYSIS_ONLY_RELATIONSHIP_PROMPT,
                    )
                )

            if settings.EXTRACTION_IMMEDIATE:
                await task

        return {
            "analysis": analysis_text,
            "sources": sources,
            "subgraph_id": subgraph_id,
            "relationships_extracted": relationships_extracted,
        }
