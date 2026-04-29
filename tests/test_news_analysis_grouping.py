from __future__ import annotations

import asyncio
from datetime import date
from typing import List

import pytest
from langchain_core.documents import Document

from core.agents.models.base_agent_models import BaseAgentInput
from core.agents.models.news_agent_models import DomainQuery, NewsAgentState, PlannerDecision
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.prompts.news_agent_prompts import (
    build_news_deferred_relationship_system_prompt,
)
from core.agents.working_memory.news_working_memory import NewsWorkingMemoryManager
from core.memory.retrieval.models import RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    url: str,
    title: str,
    relevance: float,
    planner_domains: list[str] | None = None,
    domain: str | None = None,
    relevance_source: str | None = None,
) -> RetrievedChunk:
    metadata = {"source_url": url, "article_title": title}
    if planner_domains:
        metadata["planner_domains"] = planner_domains
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="vector",
        metadata=metadata,
        source_url=url,
        article_title=title,
        relevance_score=relevance,
        domain=domain,
        relevance_source=relevance_source,
    )


class _FakeStructuredLLM:
    def __init__(self, response, owner) -> None:
        self._response = response
        self._owner = owner

    async def ainvoke(self, messages):
        self._owner.last_messages = messages
        return self._response


class _FakeLLM:
    def __init__(self, payload: dict):
        self._payload = payload
        self.last_messages = None

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(schema.model_validate(self._payload), self)


class _FakeWorkingMemory:
    def __init__(self) -> None:
        self.merged: List[RetrievedChunk] = []

    def merge_working_chunks(self, *, conversation_id: str, chunks: List[RetrievedChunk]) -> None:
        self.merged = list(chunks)


def test_rendezvous_builds_article_context_with_deduped_working_and_retrieved_chunks() -> None:
    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._working_memory = _FakeWorkingMemory()

    state = NewsAgentState(
        goal="Assess implications",
        conversation_id="conv-1",
        planner_decision=PlannerDecision(
            action="web_search",
            queries=[
                DomainQuery(domain="company", query="AAPL earnings"),
                DomainQuery(domain="sector", query="semiconductor demand"),
                DomainQuery(domain="market", query="rates and equities"),
            ],
        ),
        final_chunks=[
            _chunk(
                "wm-1",
                "working memory insight",
                url="https://wm.example.com/1",
                title="WM",
                relevance=0.9,
            )
        ],
        retrieved_chunks=[
            _chunk(
                "f-1",
                "fetched chunk",
                url="https://fetch.example.com/1",
                title="Fetched",
                relevance=0.8,
                planner_domains=["company", "market"],
            ),
            _chunk(
                "wm-1",
                "working memory duplicate",
                url="https://wm.example.com/1",
                title="WM",
                relevance=0.1,
                planner_domains=["company"],
            ),
        ],
        memory_chunks=[
            _chunk(
                "m-1",
                "memory chunk",
                url="https://mem.example.com/1",
                title="Memory",
                relevance=0.7,
                domain="sector",
            ),
            _chunk(
                "m-low",
                "low relevance memory chunk",
                url="https://mem.example.com/2",
                title="Memory low",
                relevance=0.2,
                domain="market",
            ),
            _chunk(
                "f-1",
                "fetched chunk duplicate in memory",
                url="https://fetch.example.com/1",
                title="Fetched",
                relevance=0.3,
                domain="sector",
            ),
        ],
    )

    result = asyncio.run(agent._rendezvous_node(state))
    article_block = result["article_context_block"]

    assert "[WM]" in article_block
    assert "[Fetched]" in article_block
    assert "[Memory]" in article_block
    assert "working memory insight" in article_block
    assert "memory chunk" in article_block
    assert "fetched chunk" in article_block
    assert "low relevance memory chunk" not in article_block
    assert "working memory duplicate" not in article_block
    assert "fetched chunk duplicate in memory" not in article_block
    assert article_block.count("fetched chunk") == 1
    assert "chunk_id=?" not in article_block

    assert len(agent._working_memory.merged) == len(result["final_chunks"])
    assert all(chunk.relevance_score is not None and chunk.relevance_score >= 0.4 for chunk in agent._working_memory.merged)


def test_analyse_news_prompt_contains_article_grouped_section() -> None:
    llm = _FakeLLM(
        {
            "is_context_sufficient": False,
            "analysis": "",
            "missing_information_goal": "Need recent guidance revision details",
            "persist_chunk_ids": ["1"],
            "source_chunk_ids": [],
            "sentiment": None,
        }
    )

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._llm = llm

    state = NewsAgentState(
        goal="Evaluate AAPL qualitative outlook",
        planner_decision=PlannerDecision(
            action="web_search",
            queries=[DomainQuery(domain="company", query="AAPL updates")],
        ),
        final_chunks=[
            _chunk(
                "f-1",
                "context chunk",
                url="https://f.example.com/1",
                title="F1",
                relevance=0.9,
            )
        ],
        article_context_block="[F1]\n- chunk_id=1 | date=01-01-2026 | relevance_score=0.9000\n  text=context chunk",
    )

    _ = asyncio.run(agent._analyse_news_node(state))

    assert llm.last_messages is not None
    human_prompt = llm.last_messages[1].content
    assert "Article-grouped evidence" in human_prompt
    assert "[F1]" in human_prompt
    assert "Domain-grouped evidence" not in human_prompt
    assert "Working-memory evidence" not in human_prompt


def test_rendezvous_keeps_low_score_tavily_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.agents import news_analysis_agent as news_module

    monkeypatch.setattr(news_module.settings, "NEWS_AGENT_MIN_RELEVANCE_SCORE", 0.6)

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._working_memory = _FakeWorkingMemory()

    state = NewsAgentState(
        goal="Assess implications",
        conversation_id="conv-tavily",
        planner_decision=PlannerDecision(
            action="web_search",
            queries=[DomainQuery(domain="company", query="AAPL earnings")],
        ),
        retrieved_chunks=[
            _chunk(
                "t-low",
                "tavily low score chunk",
                url="https://fetch.example.com/t-low",
                title="Tavily low",
                relevance=0.1,
                planner_domains=["company"],
                relevance_source="tavily",
            )
        ],
        memory_chunks=[
            _chunk(
                "m-low",
                "memory low score chunk",
                url="https://mem.example.com/m-low",
                title="Memory low",
                relevance=0.2,
                domain="company",
            )
        ],
    )

    result = asyncio.run(agent._rendezvous_node(state))
    final_ids = [chunk.chunk_id for chunk in result["final_chunks"]]
    article_block = result["article_context_block"]

    assert "t-low" in final_ids
    assert "tavily low score chunk" in article_block
    assert "memory low score chunk" not in article_block


def test_fetch_and_ingest_newsapi_requeries_vector_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    query_args = {}

    async def _fake_search_web(action, query, *, from_date=None, to_date=None):
        assert action == "newsapi"
        assert query == "AAPL earnings"
        return [
            {
                "url": "https://example.com/1",
                "title": "One",
                "content": "content one",
            },
            {
                "url": "https://example.com/2",
                "title": "Two",
                "content": "content two",
            },
        ]

    class _FakeIngestor:
        async def ingest_articles(self, _articles):
            involved = [
                _chunk(
                    "c1",
                    "chunk one",
                    url="https://example.com/1",
                    title="One",
                    relevance=0.0,
                ),
                _chunk(
                    "c2",
                    "chunk two",
                    url="https://example.com/2",
                    title="Two",
                    relevance=0.0,
                ),
                _chunk(
                    "c3",
                    "chunk three",
                    url="https://example.com/3",
                    title="Three",
                    relevance=0.0,
                ),
            ]
            return ["c1"], ["c2", "c3"], involved

    class _FakeChromaAdapter:
        async def query(self, **kwargs):
            query_args.update(kwargs)
            return [
                (
                    Document(
                        id="c2",
                        page_content="chunk two",
                        metadata={
                            "chunk_id": "c2",
                            "source_url": "https://example.com/2",
                            "article_title": "Two",
                            "published_at": "2026-04-01T00:00:00Z",
                        },
                    ),
                    0.91,
                ),
                (
                    Document(
                        id="c1",
                        page_content="chunk one",
                        metadata={
                            "chunk_id": "c1",
                            "source_url": "https://example.com/1",
                            "article_title": "One",
                            "published_at": "2026-04-01T00:00:00Z",
                        },
                    ),
                    0.72,
                ),
            ]

    monkeypatch.setattr(news_module, "search_web", _fake_search_web)
    monkeypatch.setattr(
        news_module.service_manager, "get_ingestor", lambda: _FakeIngestor()
    )
    monkeypatch.setattr(
        news_module.service_manager, "get_chroma_adapter", lambda: _FakeChromaAdapter()
    )
    monkeypatch.setattr(news_module.settings, "RETRIEVER_SEED_TOP_K", 2)

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    state = NewsAgentState(
        goal="Assess earnings quality",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        planner_decision=PlannerDecision(
            action="newsapi",
            queries=[DomainQuery(domain="company", query="AAPL earnings")],
        ),
    )

    result = asyncio.run(agent._fetch_and_ingest_node(state))
    retrieved = result["retrieved_chunks"]

    assert query_args["query_text"] == "Assess earnings quality"
    assert query_args["n_results"] == 2
    assert query_args["where"] == {"chunk_id": {"$in": ["c1", "c2", "c3"]}}
    assert [chunk.chunk_id for chunk in retrieved] == ["c2", "c1"]
    assert all(chunk.relevance_source == "vector" for chunk in retrieved)


def test_fetch_and_ingest_dedupes_query_variant_urls_and_seen_chunk_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    ingested_batches = []

    async def _fake_search_web(action, query, *, from_date=None, to_date=None):
        assert action == "web_search"
        return [
            {
                "url": "https://example.com/article?id=1&utm_source=feed",
                "title": "One",
                "content": "content one",
                "tavily_relevance_score": 0.8,
            },
            {
                "url": "https://example.com/article?id=2",
                "title": "One duplicate base path",
                "content": "content duplicate",
                "tavily_relevance_score": 0.7,
            },
        ]

    class _FakeIngestor:
        async def ingest_articles(self, articles):
            ingested_batches.append(articles)
            return (
                ["c1"],
                [],
                [
                    _chunk(
                        "c1",
                        "chunk one",
                        url="https://example.com/article?id=1",
                        title="One",
                        relevance=0.0,
                    )
                ],
            )

    monkeypatch.setattr(news_module, "search_web", _fake_search_web)
    monkeypatch.setattr(
        news_module.service_manager, "get_ingestor", lambda: _FakeIngestor()
    )

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    state = NewsAgentState(
        goal="Assess web news",
        planner_decision=PlannerDecision(
            action="web_search",
            queries=[DomainQuery(domain="company", query="AAPL news")],
        ),
        seen_chunk_ids=["c1"],
    )

    result = asyncio.run(agent._fetch_and_ingest_node(state))

    assert len(ingested_batches) == 1
    assert len(ingested_batches[0]) == 1
    assert result["seen_url_keys"] == ["https://example.com/article"]
    assert result["retrieved_chunks"] == []
    assert result["seen_chunk_ids"] == []


def test_news_relationship_prompt_factory_scopes_schema_enums() -> None:
    prompt = build_news_deferred_relationship_system_prompt(
        allowed_entity_types=["Company", "Market"],
        allowed_relationship_types=["AFFECTS", "RELATED_TO"],
    )

    assert '"Company"' in prompt
    assert '"Market"' in prompt
    assert '"AFFECTS"' in prompt
    assert '"RELATED_TO"' in prompt
    assert "<relationships>" in prompt


def test_run_seeds_and_persists_seen_history_across_turns() -> None:
    captured_state = {}

    class _FakeGraph:
        async def ainvoke(self, state):
            captured_state.update(state)
            return {
                "analysis": "done",
                "sources": [],
                "memory_summary": {},
                "final_chunks": [
                    _chunk(
                        "chunk-new",
                        "new chunk",
                        url="https://example.com/new",
                        title="New",
                        relevance=0.9,
                    )
                ],
                "seen_url_keys": list(state.get("seen_url_keys") or [])
                + ["https://example.com/new"],
                "seen_chunk_ids": list(state.get("seen_chunk_ids") or [])
                + ["chunk-new"],
                "is_context_sufficient": True,
            }

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._graph = _FakeGraph()
    agent._working_memory = NewsWorkingMemoryManager(
        max_chunks=30,
        max_turns=20,
        seen_url_cap=10,
        seen_chunk_cap=30,
    )
    agent._working_memory.merge_seen_history(
        conversation_id="conv-seen",
        url_keys=["https://example.com/old"],
        chunk_ids=["chunk-old"],
    )

    _ = asyncio.run(
        agent.run(
            BaseAgentInput(
                goal="test run",
                conversation_id="conv-seen",
            )
        )
    )

    assert captured_state["seen_url_keys"] == ["https://example.com/old"]
    assert captured_state["seen_chunk_ids"] == ["chunk-old"]
    assert agent._working_memory.get_seen_url_keys("conv-seen") == [
        "https://example.com/old",
        "https://example.com/new",
    ]
    assert agent._working_memory.get_seen_chunk_ids("conv-seen") == [
        "chunk-old",
        "chunk-new",
    ]
