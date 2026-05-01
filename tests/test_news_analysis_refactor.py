from __future__ import annotations

import asyncio
from dataclasses import dataclass

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
    def __init__(self, payload: dict, temperature: float = 0.23) -> None:
        self._payload = payload
        self.temperature = temperature
        self.last_messages = None

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(schema.model_validate(self._payload), owner=self)


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

    queue = _FakeGraphQueueManager(enqueued=[])
    monkeypatch.setattr(
        news_module.service_manager, "get_graph_queue_manager", lambda: queue
    )

    captured_task_kwargs = {}

    def _fake_make_extraction_task(**kwargs):
        captured_task_kwargs.update(kwargs)
        return {"kind": "deferred", **kwargs}

    monkeypatch.setattr(news_module, "make_extraction_task", _fake_make_extraction_task)

    llm = _FakeLLM(
        payload={
            "is_context_sufficient": True,
            "analysis": "Primary takeaway from second source.",
            "source_chunk_ids": [2],
        }
    )
    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._llm = llm
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
    assert queue.enqueued
    assert captured_task_kwargs["extraction_text"] == result["analysis"]
    assert captured_task_kwargs["conversation_id"] == "conv-analysis"
    assert captured_task_kwargs["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert captured_task_kwargs["allowed_entity_types"]
    assert captured_task_kwargs["allowed_relationship_types"]
