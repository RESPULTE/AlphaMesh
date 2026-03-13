"""News analysis agent with dual-store ingestion and chunk-level extraction."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core.agents.base_agent import AbstractAgent
from core.agents.models import (
    BaseAgentInput,
    ChunkResult,
    CitedSource,
    NewsAgentOutput,
    NewsAgentState,
)
from core.logger import get_logger
from core.services import service_manager

logger = get_logger(__name__)


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
        return (
            "Ingests news into dual stores, schedules background extraction, and synthesizes analysis."
        )

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
        )

        final_state = await self._graph.ainvoke(initial_state.model_dump())
        return NewsAgentOutput(**final_state)

    def _build_graph(self):
        """Compile the linear LangGraph workflow."""
        workflow = StateGraph(NewsAgentState, output_schema=NewsAgentOutput)

        workflow.add_node("fetch_news", self._fetch_news_node)
        workflow.add_node("ingest_articles", self._ingest_articles_node)
        workflow.add_node("retrieve_chunks", self._retrieve_chunks_node)
        workflow.add_node("analyse_news", self._analyse_news_node)

        workflow.add_edge(START, "fetch_news")
        workflow.add_edge("fetch_news", "ingest_articles")
        workflow.add_edge("ingest_articles", "retrieve_chunks")
        workflow.add_edge("retrieve_chunks", "analyse_news")
        workflow.add_edge("analyse_news", END)

        return workflow.compile()

    def _extract_companies(self, articles: List[dict], ticker: str) -> List[str]:
        """Derive a companies-involved list from article metadata."""
        companies = set()
        if ticker:
            companies.add(ticker.upper())
        for article in articles:
            source = article.get("source") or {}
            name = source.get("name")
            if name:
                companies.add(name)
        return sorted(companies)

    async def _fetch_news_node(self, state: NewsAgentState) -> dict:
        """Fetch raw news articles from NewsAPI."""
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: service_manager.get_news_api().get_everything(
                    q=state.ticker,
                    from_param=state.start_date.date().isoformat(),
                    to=state.end_date.date().isoformat(),
                    language="en",
                    sort_by="relevancy",
                    page=1,
                    page_size=50,
                ),
            )
        except Exception as exc:
            logger.error("NewsAPI request failed: %s", exc)
            raise

        if response.get("status") != "ok":
            raise RuntimeError(response.get("message", "NewsAPI error"))

        articles = [a for a in response.get("articles", []) if a.get("content")]
        logger.info("Fetched %d articles from NewsAPI.", len(articles))
        _ = self._extract_companies(articles, state.ticker)
        return {"raw_articles": articles}

    async def _ingest_articles_node(self, state: NewsAgentState) -> dict:
        """Ingest articles into Neo4j and ChromaDB."""
        if not state.raw_articles:
            return {"chunk_ids": []}

        companies_involved = self._extract_companies(state.raw_articles, state.ticker)
        try:
            chunk_ids = await service_manager.get_ingestor().ingest_articles(
                state.raw_articles, companies_involved
            )
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            raise
        logger.info("Ingested articles into memory. #Chunk IDs: %s", len(chunk_ids))
        return {"chunk_ids": chunk_ids}

    async def _retrieve_chunks_node(self, state: NewsAgentState) -> dict:
        """Retrieve relevant chunks from ChromaDB."""
        embedding_func = service_manager.get_embedding_func()
        chroma_adapter = service_manager.get_chroma_adapter()

        try:
            query_embedding = await embedding_func.aembed_query(state.query)
        except Exception as exc:
            logger.error("Embedding query failed: %s", exc)
            raise

        try:
            result = await chroma_adapter.query(
                query_embedding, n_results=20, where=None
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            raise

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: List[ChunkResult] = []
        for idx, chunk_id in enumerate(ids):
            retrieved.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    text=documents[idx] if idx < len(documents) else "",
                    metadata=metadatas[idx] if idx < len(metadatas) else {},
                    score=distances[idx] if idx < len(distances) else 0.0,
                )
            )
        logger.info("Retrieved %d chunks from ChromaDB.", len(retrieved))
        return {"retrieved_chunks": retrieved}

    async def _analyse_news_node(self, state: NewsAgentState) -> dict:
        """Generate a grounded financial analysis from retrieved chunks."""
        if not state.retrieved_chunks:
            return {"analysis": "No relevant news articles found.", "sources": []}

        sources: List[CitedSource] = []
        for idx, chunk in enumerate(state.retrieved_chunks, start=1):
            metadata = chunk.metadata or {}
            sources.append(
                CitedSource(
                    source_id=idx,
                    title=metadata.get("article_title", "Unknown Title"),
                    url=metadata.get("source_url", ""),
                    page_content=chunk.text,
                )
            )

        context = "\n\n".join(
            [f"[{s.source_id}] {s.title}\n{s.page_content}" for s in sources]
        )

        system_prompt = (
            "You are a professional financial analyst. "
            "Use only the provided context to answer the question."
        )
        user_prompt = (
            f"Question: {state.query}\n\n"
            f"Context:\n{context}\n\n"
            "Provide a concise, evidence-based analysis."
        )

        try:
            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            analysis_text = response.content
        except Exception as exc:
            logger.error("Analysis generation failed: %s", exc)
            raise

        return {"analysis": analysis_text, "sources": sources}
