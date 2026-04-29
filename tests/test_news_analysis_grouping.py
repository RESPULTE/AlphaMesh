from __future__ import annotations

import asyncio
from datetime import date
from typing import List

import pytest
from langchain_core.documents import Document

from core.agents.models.news_agent_models import DomainQuery, NewsAgentState, PlannerDecision
from core.agents.news_analysis_agent import NewsAnalysisAgent
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


def test_rendezvous_builds_grouped_context_with_multi_domain_and_memory_chunks() -> None:
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
            )
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
            )
        ],
    )

    result = asyncio.run(agent._rendezvous_node(state))
    grouped_block = result["grouped_query_context_block"]
    working_block = result["working_memory_context_block"]

    assert "[company]" in grouped_block
    assert "[sector]" in grouped_block
    assert "[market]" in grouped_block
    assert "fetched chunk" in grouped_block
    assert "memory chunk" in grouped_block
    assert "low relevance memory chunk" not in grouped_block
    # multi-domain fetched chunk appears in both company and market sections
    assert grouped_block.count("fetched chunk") >= 2
    assert "chunk_id=?" not in grouped_block

    assert "working memory insight" in working_block
    assert "working memory insight" not in grouped_block
    assert len(agent._working_memory.merged) == len(result["final_chunks"])
    assert all(chunk.relevance_score is not None and chunk.relevance_score >= 0.4 for chunk in agent._working_memory.merged)


def test_analyse_news_prompt_contains_grouped_and_working_memory_sections() -> None:
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
        grouped_query_context_block="[company]\n- chunk_id=1 | date=01-01-2026 | relevance_score=0.9000\n  text=context chunk",
        working_memory_context_block="- chunk_id=wm1 | date=01-01-2026 | relevance_score=0.7000\n  text=working memory chunk",
    )

    _ = asyncio.run(agent._analyse_news_node(state))

    assert llm.last_messages is not None
    human_prompt = llm.last_messages[1].content
    assert "Domain-grouped evidence" in human_prompt
    assert "[company]" in human_prompt
    assert "Working-memory evidence" in human_prompt
    assert "working memory chunk" in human_prompt


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
    grouped_block = result["grouped_query_context_block"]

    assert "t-low" in final_ids
    assert "tavily low score chunk" in grouped_block
    assert "memory low score chunk" not in grouped_block


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
