from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from core.agents.models.news_agent_models import (
    DomainQuery,
    NewsAgentState,
    PlannerDecision,
    ResearchStepLog,
)
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.memory.retrieval.models import RetrievedChunk


def _make_chunk(
    chunk_id: str,
    text: str,
    *,
    title: str,
    url: str,
    relevance_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="vector",
        metadata={"article_title": title, "source_url": url},
        article_title=title,
        source_url=url,
        relevance_score=relevance_score,
        relevance_source="vector",
        extraction_status="EXTRACTED",
    )


@dataclass
class _FakeGraphQueueManager:
    enqueued: list

    async def enqueue(self, task):
        self.enqueued.append(task)
        return "subgraph-1"


class _FakeStructuredLLM:
    def __init__(self, response, owner=None) -> None:
        self._response = response
        self._owner = owner

    async def ainvoke(self, messages):
        if self._owner is not None:
            self._owner.last_messages = messages
        return self._response


class _FakeLLM:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        narrative_text: str = "",
        temperature: float = 0.23,
    ) -> None:
        self._payload = payload
        self._narrative_text = narrative_text
        self.temperature = temperature
        self.last_messages = None

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(schema.model_validate(self._payload), owner=self)

    async def ainvoke(self, messages):
        self.last_messages = messages
        return SimpleNamespace(text=self._narrative_text)


def test_rendezvous_node_does_not_enqueue_chunk_entity_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    queue = _FakeGraphQueueManager(enqueued=[])
    monkeypatch.setattr(
        news_module.service_manager, "get_graph_queue_manager", lambda: queue
    )

    ranked = [
        _make_chunk("chunk-1", "first", title="A", url="https://example.com/a"),
        _make_chunk("chunk-2", "second", title="B", url="https://example.com/b"),
    ]

    async def _fake_rank(*, query, chunks):
        _ = (query, chunks)
        return ranked

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._working_memory = news_module.NewsWorkingMemoryManager()
    agent._rank_chunks_with_reranker = _fake_rank

    state = NewsAgentState(
        goal="queue behavior",
        conversation_id="conv-rendezvous",
        planner_decision=PlannerDecision(
            action="newsapi",
            queries=[DomainQuery(domain="company", query="AAPL")],
        ),
        research_logs=[
            ResearchStepLog(
                iteration=1,
                action="newsapi",
                queries=[DomainQuery(domain="company", query="AAPL")],
            )
        ],
    )

    result = asyncio.run(agent._rendezvous_node(state))
    assert [chunk.chunk_id for chunk in result["final_chunks"]] == ["chunk-1", "chunk-2"]
    assert queue.enqueued == []


def test_analyse_news_node_enqueues_deferred_relationship_extraction_with_final_stage_chunk_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module
    monkeypatch.setattr(
        news_module.settings, "ENABLE_ANALYSIS_TOKEN_STREAMING", False
    )

    queue = _FakeGraphQueueManager(enqueued=[])
    monkeypatch.setattr(
        news_module.service_manager, "get_graph_queue_manager", lambda: queue
    )

    captured_task_kwargs = {}

    def _fake_make_extraction_task(**kwargs):
        captured_task_kwargs.update(kwargs)
        return {"kind": "deferred", **kwargs}

    monkeypatch.setattr(news_module, "make_extraction_task", _fake_make_extraction_task)

    structured_llm = _FakeLLM(
        payload={
            "is_context_sufficient": True,
            "source_chunk_ids": [2],
        }
    )
    narrative_llm = _FakeLLM(
        narrative_text="Primary takeaway from second source."
    )
    monkeypatch.setattr(
        news_module.service_manager,
        "get_agent",
        lambda temperature=0.0: (
            structured_llm if float(temperature) == 0.0 else narrative_llm
        ),
    )

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._working_memory = news_module.NewsWorkingMemoryManager()

    state = NewsAgentState(
        query="AAPL near-term setup",
        goal="Assess current AAPL setup",
        conversation_id="conv-analysis",
        final_chunks=[
            _make_chunk(
                "chunk-1",
                "first chunk",
                title="Article One",
                url="https://example.com/one",
                relevance_score=0.9,
            ),
            _make_chunk(
                "chunk-2",
                "second chunk",
                title="Article Two",
                url="https://example.com/two",
                relevance_score=0.8,
            ),
        ],
    )

    result = asyncio.run(agent._analyse_news_node(state))

    assert result["analysis"] == "Primary takeaway from second source."
    assert structured_llm.last_messages is not None
    assert narrative_llm.last_messages is not None
    assert queue.enqueued
    assert captured_task_kwargs["extraction_text"] == result["analysis"]
    assert captured_task_kwargs["conversation_id"] == "conv-analysis"
    assert captured_task_kwargs["task_kind"] == news_module.TASK_KIND_SCOPED_EXTRACTION
    assert captured_task_kwargs["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert str(captured_task_kwargs["chunk_system_prompt"] or "").strip()
    assert (
        captured_task_kwargs["chunk_system_prompt"]
        != captured_task_kwargs["system_prompt"]
    )
    assert captured_task_kwargs["allowed_entity_types"]
    assert captured_task_kwargs["allowed_relationship_types"]


def test_analyse_news_node_uses_chunk_level_citation_context_and_remaps_sparse_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module
    monkeypatch.setattr(
        news_module.settings, "ENABLE_ANALYSIS_TOKEN_STREAMING", False
    )

    queue = _FakeGraphQueueManager(enqueued=[])
    monkeypatch.setattr(
        news_module.service_manager, "get_graph_queue_manager", lambda: queue
    )

    structured_llm = _FakeLLM(
        payload={
            "is_context_sufficient": True,
            "source_chunk_ids": [1, 2, 3, 4],
        }
    )
    narrative_llm = _FakeLLM(
        narrative_text=(
            "Revenue momentum improved versus prior guidance [2]. "
            "Execution risk remains due to supply constraints [4]."
        )
    )
    monkeypatch.setattr(
        news_module.service_manager,
        "get_agent",
        lambda temperature=0.0: (
            structured_llm if float(temperature) == 0.0 else narrative_llm
        ),
    )

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._working_memory = news_module.NewsWorkingMemoryManager()

    state = NewsAgentState(
        goal="Assess current setup",
        conversation_id="conv-analysis-remap",
        final_chunks=[
            _make_chunk(
                "chunk-1",
                "first chunk text",
                title="Same Article",
                url="https://example.com/same",
                relevance_score=0.9,
            ),
            _make_chunk(
                "chunk-2",
                "second chunk text",
                title="Same Article",
                url="https://example.com/same",
                relevance_score=0.8,
            ),
            _make_chunk(
                "chunk-3",
                "third chunk text",
                title="Same Article",
                url="https://example.com/same",
                relevance_score=0.7,
            ),
            _make_chunk(
                "chunk-4",
                "fourth chunk text",
                title="Same Article",
                url="https://example.com/same",
                relevance_score=0.6,
            ),
        ],
    )

    result = asyncio.run(agent._analyse_news_node(state))

    assert narrative_llm.last_messages is not None
    narrative_prompt = narrative_llm.last_messages[1].content
    assert "Chunk-level citation evidence" in narrative_prompt
    assert "[1] title=Same Article" in narrative_prompt
    assert "[2] title=Same Article" in narrative_prompt
    assert "[4] title=Same Article" in narrative_prompt

    assert "[1]" in result["analysis"]
    assert "[2]" in result["analysis"]
    assert "[4]" not in result["analysis"]
    assert [src.source_id for src in result["sources"]] == [1, 2]
    assert result["sources"][0].title == "Same Article"
    assert result["sources"][1].title == "Same Article"


def test_analyse_news_node_without_citations_keeps_chunk_level_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module
    monkeypatch.setattr(
        news_module.settings, "ENABLE_ANALYSIS_TOKEN_STREAMING", False
    )

    queue = _FakeGraphQueueManager(enqueued=[])
    monkeypatch.setattr(
        news_module.service_manager, "get_graph_queue_manager", lambda: queue
    )

    structured_llm = _FakeLLM(
        payload={
            "is_context_sufficient": True,
            "source_chunk_ids": [1, 2],
        }
    )
    narrative_text = "Catalysts are improving while key execution risks remain."
    narrative_llm = _FakeLLM(narrative_text=narrative_text)
    monkeypatch.setattr(
        news_module.service_manager,
        "get_agent",
        lambda temperature=0.0: (
            structured_llm if float(temperature) == 0.0 else narrative_llm
        ),
    )

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._working_memory = news_module.NewsWorkingMemoryManager()

    state = NewsAgentState(
        goal="Assess current setup",
        conversation_id="conv-analysis-fallback",
        final_chunks=[
            _make_chunk(
                "chunk-1",
                "first chunk text",
                title="Article One",
                url="https://example.com/one",
                relevance_score=0.9,
            ),
            _make_chunk(
                "chunk-2",
                "second chunk text",
                title="Article Two",
                url="https://example.com/two",
                relevance_score=0.8,
            ),
        ],
    )

    result = asyncio.run(agent._analyse_news_node(state))

    assert result["analysis"] == narrative_text
    assert [src.source_id for src in result["sources"]] == [1, 2]
    assert result["sources"][0].title == "Article One"
    assert result["sources"][1].title == "Article Two"
