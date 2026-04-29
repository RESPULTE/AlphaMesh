from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from core.agents.models.base_agent_models import BaseAgentInput
from core.agents.models.news_agent_models import (
    NewsAgentState,
    PlannerDecision,
    ResearchStepLog,
)
from core.agents.news_analysis_agent import NewsAnalysisAgent, _get_cached_entities
from core.memory.retrieval.models import RetrievedChunk


def _make_chunk(
    chunk_id: str,
    text: str,
    *,
    title: str,
    url: str,
    relevance_score: float | None = None,
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
        reranker_relevance_score=relevance_score,
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
        self.schema = None
        self.structured_calls = 0
        self.last_messages = None

    def with_structured_output(self, schema):
        self.schema = schema
        self.structured_calls += 1
        return _FakeStructuredLLM(schema.model_validate(self._payload), owner=self)


def test_analyse_news_node_uses_structured_output_and_defers_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    news_module._ENTITY_CACHE.clear()
    news_module._ENTITY_CACHE_TS.clear()

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
    agent._working_memory = news_module.NewsWorkingMemoryManager()

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
    assert captured_task_kwargs["allowed_entity_types"]
    assert captured_task_kwargs["allowed_relationship_types"]


def test_analyse_news_node_includes_planner_chunk_rationales_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    queue = _FakeGraphQueueManager(enqueued=[])
    monkeypatch.setattr(
        news_module.service_manager, "get_graph_queue_manager", lambda: queue
    )

    llm = _FakeLLM(
        payload={
            "analysis": "Summary with references [1].",
            "sentiment": {"score": 55, "label": "NEUTRAL", "rationale": "Mixed."},
        }
    )
    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._llm = llm
    agent._working_memory = news_module.NewsWorkingMemoryManager()

    state = NewsAgentState(
        query="AAPL setup",
        final_chunks=[
            _make_chunk(
                "chunk-a",
                "Chunk A text",
                title="Article A",
                url="https://example.com/a",
            ),
            _make_chunk(
                "chunk-b",
                "Chunk B text",
                title="Article B",
                url="https://example.com/b",
            ),
        ],
        planner_decision=PlannerDecision(
            action="proceed",
            relevant_chunks=[1, "2"],
        ),
    )

    _ = asyncio.run(agent._analyse_news_node(state))

    assert llm.last_messages is not None
    human_prompt = llm.last_messages[1].content
    assert "Goal:" in human_prompt
    assert "Chunk A text" in human_prompt
    assert "Chunk B text" in human_prompt


def test_rendezvous_node_applies_threshold_and_sets_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    news_module._ENTITY_CACHE.clear()
    news_module._ENTITY_CACHE_TS.clear()

    class _FakeReranker:
        async def rank(self, _query, _chunks):
            return [
                _make_chunk(
                    "chunk-r1",
                    "from ingest",
                    title="Ingest Article",
                    url="https://example.com/ingest",
                    relevance_score=0.92,
                ),
                _make_chunk(
                    "chunk-r2",
                    "from memory",
                    title="Memory Article",
                    url="https://example.com/memory",
                    relevance_score=0.81,
                ),
                _make_chunk(
                    "chunk-r3",
                    "low score",
                    title="Low Score Article",
                    url="https://example.com/low",
                    relevance_score=0.40,
                ),
            ]

    class _FakeNeo4jAdapter:
        async def get_entities_for_chunks(self, chunk_ids):
            assert set(chunk_ids) == {"chunk-r1", "chunk-r2"}
            return [
                {"entity_name": "Apple", "entity_type": "Company"},
                {"entity_name": "Gross Margin", "entity_type": "FinancialConcept"},
            ]

    monkeypatch.setattr(
        news_module.service_manager, "get_reranker", lambda: _FakeReranker()
    )
    monkeypatch.setattr(
        news_module.service_manager, "get_neo4j_adapter", lambda: _FakeNeo4jAdapter()
    )
    monkeypatch.setattr(news_module.settings, "NEWS_AGENT_MIN_RELEVANCE_SCORE", 0.60)

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._working_memory = news_module.NewsWorkingMemoryManager()

    state = NewsAgentState(
        query="AAPL context",
        conversation_id="conv-rendezvous",
        retrieved_chunks=[
            _make_chunk(
                "seed", "seed text", title="Seed", url="https://example.com/seed"
            )
        ],
        research_logs=[
            ResearchStepLog(
                iteration=1,
                action="newsapi",
                query="company:AAPL",
                queries=[],
                total_fetched_articles=3,
                newly_fetched_articles=2,
            )
        ],
    )

    result = asyncio.run(agent._rendezvous_node(state))
    assert [chunk.chunk_id for chunk in result["final_chunks"]] == [
        "chunk-r1",
        "chunk-r2",
    ]
    assert result["rendezvous_score_unavailable"] is False
    assert result["rendezvous_has_minimum_sources"] is True
    assert result["rendezvous_relevant_source_count"] == 2
    updated_logs = result["research_logs"].value
    assert updated_logs[-1].relevant_chunk_count == 2
    cached = _get_cached_entities("conv-rendezvous")
    assert ("Apple", "Company") in cached


def test_rendezvous_node_returns_all_when_score_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import news_analysis_agent as news_module

    class _FakeReranker:
        async def rank(self, _query, chunks):
            return chunks

    class _FakeNeo4jAdapter:
        async def get_entities_for_chunks(self, _chunk_ids):
            return []

    monkeypatch.setattr(
        news_module.service_manager, "get_reranker", lambda: _FakeReranker()
    )
    monkeypatch.setattr(
        news_module.service_manager, "get_neo4j_adapter", lambda: _FakeNeo4jAdapter()
    )

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._working_memory = news_module.NewsWorkingMemoryManager()

    state = NewsAgentState(
        query="AAPL context",
        conversation_id="conv-no-score",
        retrieved_chunks=[
            _make_chunk("chunk-r1", "a", title="A", url="https://a.example.com"),
            _make_chunk("chunk-r2", "b", title="B", url="https://b.example.com"),
        ],
        research_logs=[
            ResearchStepLog(
                iteration=1,
                action="web_search",
                query="company:AAPL",
                queries=[],
            )
        ],
    )
    result = asyncio.run(agent._rendezvous_node(state))
    assert [chunk.chunk_id for chunk in result["final_chunks"]] == [
        "chunk-r1",
        "chunk-r2",
    ]
    assert result["rendezvous_score_unavailable"] is True
    assert result["rendezvous_has_minimum_sources"] is False


def test_planner_node_filters_selected_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner_payload = {
        "action": "proceed",
        "queries": [],
        "relevant_chunks": [2],
    }

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._llm = _FakeLLM(planner_payload)
    from core.agents import news_analysis_agent as news_module

    agent._working_memory = news_module.NewsWorkingMemoryManager()

    state = NewsAgentState(
        query="AAPL catalysts",
        conversation_id="conv-plan",
        research_iteration=1,
        final_chunks=[
            _make_chunk("chunk-a", "indirect", title="A", url="https://example.com/a"),
            _make_chunk("chunk-b", "direct", title="B", url="https://example.com/b"),
        ],
        rendezvous_score_unavailable=True,
    )
    result = asyncio.run(agent._planner_node(state))
    assert result["is_information_sufficient"] is True
    assert result["planner_decision"].action == "proceed"
    assert [chunk.chunk_id for chunk in result["final_chunks"]] == ["chunk-b"]


def test_run_reuses_cached_agent_memory_context() -> None:
    from core.agents import news_analysis_agent as news_module

    captured_states: list[dict] = []

    class _FakeGraph:
        async def ainvoke(self, state):
            captured_states.append(state)
            return {
                "analysis": "Qualitative analysis output.",
                "sources": [],
                "final_chunks": [
                    _make_chunk(
                        "chunk-final",
                        "final text",
                        title="Final",
                        url="https://example.com/final",
                    )
                ],
                "rendezvous_score_unavailable": False,
                "memory_summary": {
                    "research_actions": ["newsapi"],
                    "tools_used": ["newsapi"],
                    "source_count": 2,
                    "top_references": [],
                    "sentiment": {"label": "NEUTRAL", "score": 50},
                    "main_catalyst": "Guidance was maintained.",
                },
            }

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._graph = _FakeGraph()
    agent._working_memory = news_module.NewsWorkingMemoryManager()

    first = asyncio.run(
        agent.run(
            BaseAgentInput(
                query="First turn",
                goal="Assess first turn",
                conversation_id="conv-memory",
                agent_memory_context="- [older] sources=1; catalyst=prior update",
            )
        )
    )
    second = asyncio.run(
        agent.run(
            BaseAgentInput(
                query="Second turn",
                goal="Assess second turn",
                conversation_id="conv-memory",
            )
        )
    )

    assert first.memory_summary["source_count"] == 2
    assert captured_states[0]["agent_memory_context"].startswith("- [older]")
    assert captured_states[1]["agent_memory_context"].startswith("actions=newsapi")
    assert second.memory_summary["main_catalyst"] == "Guidance was maintained."


def test_render_memory_summary_delegates_to_manager() -> None:
    summary = {
        "tools_used": ["newsapi"],
        "source_count": 3,
        "main_catalyst": "Catalyst",
    }
    rendered = NewsAnalysisAgent.render_memory_summary(summary)
    assert rendered.startswith("actions=newsapi")
    assert "sources=3" in rendered


def test_planner_decision_accepts_numeric_relevant_chunks_and_missing_findings_summary() -> None:
    decision = PlannerDecision.model_validate(
        {
            "action": "proceed",
            "queries": [],
            "relevant_chunks": [1, "2"],
        }
    )
    normalized = decision._normalize_planner_selection_ids({"1": "chunk-a", "2": "chunk-b"})
    assert normalized.relevant_chunks == ["chunk-a", "chunk-b"]
    assert normalized.findings_summary == ""


def test_run_requires_goal_and_skips_graph() -> None:
    class _FailGraph:
        async def ainvoke(self, _state):
            raise AssertionError("Graph should not run when goal is empty")

    agent = NewsAnalysisAgent.__new__(NewsAnalysisAgent)
    agent._graph = _FailGraph()
    from core.agents import news_analysis_agent as news_module

    agent._working_memory = news_module.NewsWorkingMemoryManager()

    output = asyncio.run(
        agent.run(
            BaseAgentInput(
                query="legacy only",
                goal="",
            )
        )
    )
    assert output.analysis.startswith("News analysis skipped")
    assert output.memory_summary.get("bypassed") is True


def test_parallel_fanin_updates_are_merged_for_chunk_fields() -> None:
    def _fanout_router(state: NewsAgentState):
        return [Send("branch_a", state), Send("branch_b", state)]

    def _branch_a(_state: NewsAgentState):
        return {
            "seen_urls": ["https://example.com/a"],
            "research_logs": [
                ResearchStepLog(iteration=1, action="web_search", query="a")
            ],
            "retrieved_chunks": [
                _make_chunk(
                    "chunk-a", "a text", title="A", url="https://example.com/a"
                )
            ],
            "memory_chunks": [
                _make_chunk(
                    "chunk-ma",
                    "ma text",
                    title="MA",
                    url="https://example.com/ma",
                )
            ],
        }

    def _branch_b(_state: NewsAgentState):
        return {
            "seen_urls": ["https://example.com/b"],
            "research_logs": [
                ResearchStepLog(iteration=1, action="newsapi", query="b")
            ],
            "retrieved_chunks": [
                _make_chunk(
                    "chunk-b", "b text", title="B", url="https://example.com/b"
                )
            ],
            "memory_chunks": [
                _make_chunk(
                    "chunk-mb",
                    "mb text",
                    title="MB",
                    url="https://example.com/mb",
                )
            ],
        }

    workflow = StateGraph(NewsAgentState)
    workflow.add_node("fanout", lambda _state: {})
    workflow.add_node("branch_a", _branch_a)
    workflow.add_node("branch_b", _branch_b)
    workflow.add_edge(START, "fanout")
    workflow.add_conditional_edges("fanout", _fanout_router, ["branch_a", "branch_b"])
    workflow.add_edge("branch_a", END)
    workflow.add_edge("branch_b", END)
    graph = workflow.compile()

    state = NewsAgentState(goal="fanin-test")
    final_state = graph.invoke(state.model_dump())
    assert sorted(final_state["seen_urls"]) == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert len(final_state["research_logs"]) == 2
    assert {chunk.chunk_id for chunk in final_state["retrieved_chunks"]} == {
        "chunk-a",
        "chunk-b",
    }
    assert {chunk.chunk_id for chunk in final_state["memory_chunks"]} == {
        "chunk-ma",
        "chunk-mb",
    }
