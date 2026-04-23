from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from core.agents.models.news_agent_models import NewsAgentState
from core.agents.news_analysis_agent import NewsAnalysisAgent, _get_cached_entities
from core.memory.retrieval.dual_store_retriever import DualStoreRetriever
from core.memory.retrieval.models import MemoryContext, RetrievedChunk, RewrittenQueries
from core.memory.retrieval.reranker import CompositePrefilter


def _make_chunk(
    chunk_id: str,
    text: str,
    *,
    title: str,
    url: str,
    extraction_status: str = "EXTRACTED",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="vector",
        metadata={"article_title": title, "source_url": url},
        article_title=title,
        source_url=url,
        extraction_status=extraction_status,
    )


@dataclass
class _FakeGraphQueueManager:
    enqueued: list

    async def enqueue(self, task):
        self.enqueued.append(task)
        return "subgraph-1"


class _FakeStructuredLLM:
    def __init__(self, response) -> None:
        self._response = response

    async def ainvoke(self, _messages):
        return self._response


class _FakeLLM:
    def __init__(self, payload: dict, temperature: float = 0.23) -> None:
        self._payload = payload
        self.temperature = temperature
        self.schema = None
        self.structured_calls = 0

    def with_structured_output(self, schema):
        self.schema = schema
        self.structured_calls += 1
        return _FakeStructuredLLM(schema.model_validate(self._payload))


def test_analyse_news_node_uses_structured_output_and_defers_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    news_module._ENTITY_CACHE.clear()
    news_module._ENTITY_CACHE_TS.clear()

    queue = _FakeGraphQueueManager(enqueued=[])
    monkeypatch.setattr(
        news_module.service_manager,
        "get_graph_queue_manager",
        lambda: queue,
    )

    captured_task_kwargs = {}

    def _fake_make_extraction_task(**kwargs):
        captured_task_kwargs.update(kwargs)
        return {"kind": "deferred", **kwargs}

    monkeypatch.setattr(news_module, "make_extraction_task", _fake_make_extraction_task)

    llm = _FakeLLM(
        payload={
            "analysis": "Primary takeaway from second source [2].",
            "sentiment": {
                "score": 68,
                "label": "BUY",
                "rationale": "Guidance and momentum are improving.",
            },
        }
    )
    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._llm = llm

    state = NewsAgentState(
        query="AAPL near-term setup",
        conversation_id="conv-analysis",
        final_chunks=[
            _make_chunk(
                "chunk-1",
                "first chunk",
                title="Article One",
                url="https://example.com/one",
            ),
            _make_chunk(
                "chunk-2",
                "second chunk",
                title="Article Two",
                url="https://example.com/two",
            ),
        ],
    )

    result = asyncio.run(agent._analyse_news_node(state))

    assert llm.structured_calls == 1
    assert result["relationships_extracted"] is False
    assert result["analysis"] == "Primary takeaway from second source [1]."
    assert result["sentiment"].label == "BUY"
    assert len(result["sources"]) == 1
    assert result["sources"][0].source_id == 1
    assert result["sources"][0].title == "Article Two"
    assert queue.enqueued
    assert captured_task_kwargs["extraction_text"] == result["analysis"]
    assert captured_task_kwargs["conversation_id"] == "conv-analysis"


def test_rendezvous_node_merges_entity_tuples_into_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    news_module._ENTITY_CACHE.clear()
    news_module._ENTITY_CACHE_TS.clear()

    class _FakeReranker:
        async def rank(self, _query, chunks):
            return chunks

    class _FakeNeo4jAdapter:
        async def get_entities_for_chunks(self, chunk_ids):
            assert set(chunk_ids) == {"chunk-r1", "chunk-r2"}
            return [
                {"entity_name": "Apple", "entity_type": "Company"},
                {"entity_name": "Gross Margin", "entity_type": "FinancialConcept"},
                {"entity_name": "Apple", "entity_type": "Company"},
            ]

    monkeypatch.setattr(
        news_module.service_manager,
        "get_reranker",
        lambda: _FakeReranker(),
    )
    monkeypatch.setattr(
        news_module.service_manager,
        "get_neo4j_adapter",
        lambda: _FakeNeo4jAdapter(),
    )

    async def _run():
        memory_context = MemoryContext(
            chunks=[
                _make_chunk(
                    "chunk-r2",
                    "from memory",
                    title="Memory Article",
                    url="https://example.com/memory",
                )
            ],
            rewritten_queries=RewrittenQueries(
                company_query="apple",
                active_domains=["company"],
            ),
            entity_tuples=[("Apple", "Company"), ("NASDAQ", "Market")],
        )
        memory_task: asyncio.Future[MemoryContext] = (
            asyncio.get_running_loop().create_future()
        )
        memory_task.set_result(memory_context)

        state = NewsAgentState(
            query="AAPL context",
            conversation_id="conv-rendezvous",
            retrieved_chunks=[
                _make_chunk(
                    "chunk-r1",
                    "from ingest",
                    title="Ingest Article",
                    url="https://example.com/ingest",
                )
            ],
            memory_task=memory_task,
        )

        agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
        return await agent._rendezvous_node(state)

    result = asyncio.run(_run())

    assert [chunk.chunk_id for chunk in result["final_chunks"]] == ["chunk-r1", "chunk-r2"]
    cached = _get_cached_entities("conv-rendezvous")
    assert set(cached) == {
        ("Apple", "Company"),
        ("NASDAQ", "Market"),
        ("Gross Margin", "FinancialConcept"),
    }


def test_ingest_articles_node_queries_new_and_existing_chunk_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    class _FakeIngestor:
        async def ingest_articles(self, _articles):
            return ["chunk-new"], ["chunk-existing"], []

    class _FakeChromaAdapter:
        def __init__(self) -> None:
            self.calls = []

        async def query(self, *, query_text, n_results, where):
            _ = (query_text, n_results)
            self.calls.append(where["chunk_id"]["$in"])
            chunk_id = where["chunk_id"]["$in"][0]
            return [
                (
                    Document(
                        page_content=f"text for {chunk_id}",
                        metadata={"chunk_id": chunk_id},
                        id=chunk_id,
                    ),
                    0.91,
                )
            ]

    chroma = _FakeChromaAdapter()
    monkeypatch.setattr(news_module.service_manager, "get_ingestor", lambda: _FakeIngestor())
    monkeypatch.setattr(
        news_module.service_manager,
        "get_chroma_adapter",
        lambda: chroma,
    )

    state = NewsAgentState(
        query="AAPL latest catalysts",
        raw_articles=[{"url": "https://example.com/article"}],
    )
    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    result = asyncio.run(agent._ingest_articles_node(state))

    assert chroma.calls == [["chunk-new"], ["chunk-existing"]]
    assert [chunk.chunk_id for chunk in result["retrieved_chunks"]] == [
        "chunk-new",
        "chunk-existing",
    ]


def test_comprehensive_retrieve_returns_deduped_entity_tuples() -> None:
    class _FakeChromaAdapter:
        async def query(self, *, query_text: str, **_kwargs):
            if query_text != "company query":
                return []
            return [
                (
                    Document(
                        page_content="seed chunk",
                        metadata={"chunk_id": "chunk-a"},
                        id="chunk-a",
                    ),
                    0.95,
                )
            ]

    class _FakeNeo4jAdapter:
        async def get_entities_for_chunks(self, chunk_ids):
            rows = []
            for chunk_id in chunk_ids:
                rows.append(
                    {
                        "entity_id": f"entity-{chunk_id}",
                        "entity_name": "Apple",
                        "entity_type": "Company",
                        "source_chunk_id": chunk_id,
                    }
                )
                rows.append(
                    {
                        "entity_id": f"entity-{chunk_id}",
                        "entity_name": "Apple",
                        "entity_type": "Company",
                        "source_chunk_id": chunk_id,
                    }
                )
            return rows

        async def get_entity_neighbors(self, _entity_ids, _exclude_ids):
            return []

        async def get_chunks_for_entities(self, _entity_ids, _exclude_chunk_ids):
            return []

    retriever = DualStoreRetriever(
        neo4j_adapter=_FakeNeo4jAdapter(),
        chroma_adapter=_FakeChromaAdapter(),
        prefilter=CompositePrefilter(alpha=0.8, beta=0.2, prefilter_k=5),
    )

    context = asyncio.run(
        retriever.comprehensive_retrieve(
            RewrittenQueries(
                company_query="company query",
                active_domains=["company"],
            )
        )
    )

    assert [chunk.chunk_id for chunk in context.chunks] == ["chunk-a"]
    assert context.entity_tuples == [("Apple", "Company")]
