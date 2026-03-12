"""News analysis agent with dual-store ingestion and chunk-level extraction."""
from __future__ import annotations

import asyncio
import uuid
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
from core.graph.extraction_prompts import build_extraction_prompt
from core.graph.models import ChunkExtractionResult, ENTITY_NAMESPACE, EntityNode
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
        return "Ingests news into dual stores, extracts entities per chunk, and synthesizes analysis."

    @staticmethod
    def get_output_schema_class() -> Type[BaseModel]:
        """Return the output schema class."""
        return NewsAgentOutput

    async def run(self, input_data: BaseAgentInput) -> NewsAgentOutput:
        """Run the agent end-to-end with the provided input."""
        start_date = input_data.start_date or (datetime.now(timezone.utc) - timedelta(days=7))
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
        workflow.add_node("identify_unextracted", self._identify_unextracted_node)
        workflow.add_node("extract_entities", self._extract_entities_node)
        workflow.add_node("analyse_news", self._analyse_news_node)

        workflow.add_edge(START, "fetch_news")
        workflow.add_edge("fetch_news", "ingest_articles")
        workflow.add_edge("ingest_articles", "retrieve_chunks")
        workflow.add_edge("retrieve_chunks", "identify_unextracted")
        workflow.add_edge("identify_unextracted", "extract_entities")
        workflow.add_edge("extract_entities", "analyse_news")
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

        where = None
        if state.ticker:
            where = {"companies_involved": state.ticker}

        try:
            result = await chroma_adapter.query(query_embedding, n_results=20, where=where)
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

        return {"retrieved_chunks": retrieved}

    async def _identify_unextracted_node(self, state: NewsAgentState) -> dict:
        """Identify chunks that still require extraction."""
        if not state.retrieved_chunks:
            return {"unextracted_chunk_ids": []}

        chunk_ids = [chunk.chunk_id for chunk in state.retrieved_chunks]
        try:
            status_map = await service_manager.get_neo4j_adapter().get_chunk_extraction_status(
                chunk_ids
            )
        except Exception as exc:
            logger.error("Failed to fetch extraction status: %s", exc)
            raise

        unextracted = [
            chunk_id for chunk_id in chunk_ids if status_map.get(chunk_id) == "PENDING"
        ]
        return {"unextracted_chunk_ids": unextracted}

    async def _extract_entities_node(self, state: NewsAgentState) -> dict:
        """Extract entities and relationships for unextracted chunks."""
        if not state.unextracted_chunk_ids:
            return {"extraction_results": [], "entities_enriched": []}

        chunk_lookup = {c.chunk_id: c for c in state.retrieved_chunks}
        prompt = build_extraction_prompt()
        extraction_chain = prompt | self._llm.with_structured_output(ChunkExtractionResult)

        async def _extract_chunk(chunk: ChunkResult) -> ChunkExtractionResult:
            companies = chunk.metadata.get("companies_involved", [])
            if isinstance(companies, str):
                companies = [c.strip() for c in companies.split(",") if c.strip()]

            try:
                result: ChunkExtractionResult = await extraction_chain.ainvoke(
                    {"chunk_text": chunk.text, "companies": ", ".join(companies)}
                )
            except Exception as exc:
                logger.error("Extraction failed for chunk %s: %s", chunk.chunk_id, exc)
                raise

            result.chunk_id = chunk.chunk_id
            return result

        tasks = [
            _extract_chunk(chunk_lookup[cid]) for cid in state.unextracted_chunk_ids if cid in chunk_lookup
        ]
        results = await asyncio.gather(*tasks)

        entities_enriched: List[EntityNode] = []
        neo4j_adapter = service_manager.get_neo4j_adapter()
        chroma_adapter = service_manager.get_chroma_adapter()

        for result in results:
            local_id_map = {}
            for entity in result.entities:
                canonical_id = str(
                    uuid.uuid5(
                        ENTITY_NAMESPACE,
                        f"{entity.name.lower()}::{entity.entity_type.lower()}",
                    )
                )
                local_key = entity.local_id or canonical_id
                entity.id = canonical_id
                local_id_map[local_key] = entity
                entities_enriched.append(entity)
                await neo4j_adapter.merge_entity_node(entity)
                await neo4j_adapter.merge_relationship(
                    result.chunk_id,
                    entity.id,
                    "MENTIONS_ENTITY",
                    {"confidence": 1.0},
                )

            for rel in result.relationships:
                source_entity = local_id_map.get(rel.source_entity_local_id)
                target_entity = local_id_map.get(rel.target_entity_local_id)
                if not source_entity or not target_entity:
                    continue
                await neo4j_adapter.merge_relationship(
                    source_entity.id,
                    target_entity.id,
                    "RELATED_TO",
                    {
                        "relationship_type": rel.relationship_type,
                        "source_chunk_id": result.chunk_id,
                        "confidence": rel.confidence,
                    },
                )

            await neo4j_adapter.update_chunk_extraction_status(result.chunk_id, "EXTRACTED")

            chunk = chunk_lookup.get(result.chunk_id)
            if chunk:
                updated_metadata = dict(chunk.metadata)
                updated_metadata["extraction_status"] = "EXTRACTED"
                await chroma_adapter.update_metadata([result.chunk_id], [updated_metadata])

        return {"extraction_results": results, "entities_enriched": entities_enriched}

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
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            analysis_text = response.content
        except Exception as exc:
            logger.error("Analysis generation failed: %s", exc)
            raise

        return {"analysis": analysis_text, "sources": sources}
