from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from core.agents.financial_tools import ToolResult
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models.base_agent_models import BaseAgentInput
from core.agents.models.fundamental_agent_models import (
    ExecutorBatchLog,
    ExecutorToolLog,
    FundamentalTaskSummary,
    IterativeToolPlan,
    ToolCallBatch,
    ToolCallSpec,
    _AgentState,
)


class _FakeStructuredLLM:
    def __init__(self, response, owner=None) -> None:
        self._response = response
        self._owner = owner

    async def ainvoke(self, messages):
        if self._owner is not None:
            self._owner.last_messages = messages
        return self._response


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.schema = None
        self.structured_calls = 0
        self.last_messages = None

    def with_structured_output(self, schema):
        self.schema = schema
        self.structured_calls += 1
        return _FakeStructuredLLM(
            schema.model_validate(self._payload), owner=self
        )


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        data=[
            [100.0, 110.0, 120.0],
            [10.0, 11.0, 9.5],
            [40.0, 45.0, 47.0],
            [0.40, 0.41, 0.39],
        ],
        index=["Revenues", "NetIncomeLoss", "GrossProfit", "gross_margin"],
        columns=["2022-12-31", "2023-12-31", "2024-12-31"],
    )


def _valuation_dependency_df() -> pd.DataFrame:
    return pd.DataFrame(
        data=[
            [100.0, 110.0],  # stock_price
            [10.0, 10.0],  # shares
            [50.0, 60.0],  # LongTermDebt
            [20.0, 25.0],  # CashAndMarketableSecurities
        ],
        index=[
            "stock_price",
            "Common_stock_and_capital_stock__shares_authorized_in_shares",
            "LongTermDebt",
            "CashAndMarketableSecurities",
        ],
        columns=["2024-12-31", "2025-12-31"],
    )


def test_completion_review_requests_single_replan_and_dedupes_chart_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    llm = _FakeLLM(
        payload={
            "task_completed": False,
            "task_completion_reason": "Margin trend coverage is incomplete.",
            "replan_guidance": "Compute profitability trends before final analysis.",
            "reviewer_notes": "Prefer trend visuals over scalar-only rows.",
            "charts": [
                {
                    "chart_type": "line",
                    "title": "Core Trends",
                    "row_labels": ["Revenues", "NetIncomeLoss", "Revenues"],
                    "group_rows": True,
                    "rationale": "Primary operating performance.",
                },
                {
                    "chart_type": "bar",
                    "title": "Should avoid duplicate Revenues",
                    "row_labels": ["Revenues", "GrossProfit"],
                    "group_rows": True,
                    "rationale": "Comparison chart.",
                },
            ],
            "raw_row_labels": ["Revenues", "NetIncomeLoss", "GrossProfit", "Unknown"],
        }
    )
    monkeypatch.setattr(module.service_manager, "get_agent", lambda temperature=0: llm)

    state = _AgentState(
        query="How strong are profitability trends for AAPL?",
        ticker="AAPL",
        financial_data=_sample_df(),
        tool_plan=IterativeToolPlan(batches=[], data_summary=""),
        iteration_count=2,
        completion_review_replan_used=False,
        executor_logs=[
            ExecutorBatchLog(
                batch_index=0,
                batch_reasoning="Compute baseline ratios.",
                calls=[
                    ExecutorToolLog(
                        tool_name="profitability_ratios",
                        parameters={},
                        success=True,
                        summary="Computed gross_margin.",
                        output_row_labels=["gross_margin"],
                    )
                ],
            )
        ],
    )

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    result = asyncio.run(agent._completion_review_node(state))

    assert llm.structured_calls == 1
    assert result["completion_review_should_replan"] is True
    assert result["completion_review_replan_used"] is True
    assert "profitability trends" in result["completion_replan_guidance"].lower()

    viz_plan = result["visualization_plan"]
    all_chart_rows = [row for chart in viz_plan.charts for row in chart.row_labels]
    assert len(all_chart_rows) == len(set(all_chart_rows))
    assert set(all_chart_rows).issubset(set(_sample_df().index))
    assert set(result["raw_display_data"].index).issubset(set(_sample_df().index))


def test_completion_review_does_not_replan_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    llm = _FakeLLM(
        payload={
            "task_completed": False,
            "task_completion_reason": "Still missing follow-up metrics.",
            "replan_guidance": "Run one more derived metric pass.",
            "charts": [],
            "raw_row_labels": [],
        }
    )
    monkeypatch.setattr(module.service_manager, "get_agent", lambda temperature=0: llm)

    state = _AgentState(
        query="Review completion behavior",
        ticker="AAPL",
        financial_data=_sample_df(),
        tool_plan=IterativeToolPlan(batches=[], data_summary=""),
        iteration_count=3,
        completion_review_replan_used=True,
    )
    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    result = asyncio.run(agent._completion_review_node(state))

    assert result["task_completed"] is False
    assert result["completion_review_should_replan"] is False
    assert result["completion_replan_guidance"] == ""


def test_completion_review_enforces_visualisation_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    monkeypatch.setattr(module.settings, "FUNDAMENTAL_VIZ_MAX_ROWS_PER_CHART", 1)
    monkeypatch.setattr(module.settings, "FUNDAMENTAL_RAW_DISPLAY_MAX_ROWS", 2)

    llm = _FakeLLM(
        payload={
            "task_completed": True,
            "task_completion_reason": "Complete.",
            "charts": [
                {
                    "chart_type": "area",
                    "title": "Too many rows",
                    "row_labels": ["Revenues", "NetIncomeLoss", "GrossProfit"],
                    "group_rows": True,
                }
            ],
            "raw_row_labels": [
                "Revenues",
                "NetIncomeLoss",
                "GrossProfit",
                "gross_margin",
            ],
        }
    )
    monkeypatch.setattr(module.service_manager, "get_agent", lambda temperature=0: llm)

    state = _AgentState(
        query="Threshold enforcement",
        ticker="AAPL",
        financial_data=_sample_df(),
        tool_plan=IterativeToolPlan(batches=[], data_summary=""),
        iteration_count=1,
    )
    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    result = asyncio.run(agent._completion_review_node(state))

    viz_plan = result["visualization_plan"]
    assert len(viz_plan.charts) == 1
    assert len(viz_plan.charts[0].row_labels) == 1
    assert len(viz_plan.raw_row_labels) == 2
    assert len(result["raw_display_data"]) == 2


def test_completion_review_normalises_chart_types_and_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    llm = _FakeLLM(
        payload={
            "task_completed": True,
            "task_completion_reason": "Complete.",
            "charts": [
                {
                    "chart_type": "pie",
                    "data_mode": "timeseries",
                    "title": "Composition",
                    "row_labels": ["Revenues", "NetIncomeLoss"],
                    "group_rows": True,
                },
                {
                    "chart_type": "unknown_chart",
                    "data_mode": "snapshot",
                    "title": "Fallback",
                    "row_labels": ["GrossProfit"],
                    "group_rows": True,
                },
            ],
            "raw_row_labels": ["Revenues"],
        }
    )
    monkeypatch.setattr(module.service_manager, "get_agent", lambda temperature=0: llm)

    state = _AgentState(
        query="Chart normalisation behavior",
        ticker="AAPL",
        financial_data=_sample_df(),
        tool_plan=IterativeToolPlan(batches=[], data_summary=""),
        iteration_count=1,
    )
    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    result = asyncio.run(agent._completion_review_node(state))

    charts = result["visualization_plan"].charts
    assert len(charts) == 2
    assert charts[0].chart_type == "pie"
    assert charts[0].data_mode == "snapshot"
    assert charts[1].chart_type == "bar"
    assert charts[1].data_mode == "snapshot"


def test_run_reuses_cached_agent_memory_context() -> None:
    from core.agents import fundamental_analysis_agent as module

    captured_payloads: list[dict] = []

    class _FakeDb:
        async def initialize(self):
            return None

    class _FakeGraph:
        async def ainvoke(self, payload, config=None):
            _ = config
            captured_payloads.append(payload)
            return {
                "financial_data": pd.DataFrame(),
                "analysis": "Computed analysis",
                "tool_results": [],
                "entities_enriched": [],
                "subgraph_id": None,
                "relationships_extracted": False,
                "memory_summary": {
                    "tools_used": ["profitability_ratios"],
                    "key_rows": ["Revenues"],
                    "computed_rows": ["gross_margin"],
                    "task_completed": True,
                    "task_completion_reason": "",
                    "main_conclusion": "Margins are stable.",
                },
                "task_completed": True,
                "task_completion_reason": "",
                "visualization_plan": None,
                "raw_display_data": None,
            }

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    agent.db = _FakeDb()
    agent._graph = _FakeGraph()
    agent._working_memory = module.FundamentalWorkingMemoryManager()

    first = asyncio.run(
        agent.run(
            BaseAgentInput(
                query="First",
                conversation_id="conv-fund-memory",
                agent_memory_context="- [older] tools=cagr; rows=Revenues",
            )
        )
    )
    second = asyncio.run(
        agent.run(
            BaseAgentInput(
                query="Second",
                conversation_id="conv-fund-memory",
            )
        )
    )

    assert first.memory_summary["tools_used"] == ["profitability_ratios"]
    assert captured_payloads[0]["agent_memory_context"].startswith("- [older]")
    assert captured_payloads[1]["agent_memory_context"].startswith("tools=profitability_ratios")
    assert second.memory_summary["main_conclusion"] == "Margins are stable."


def test_data_prep_uses_cached_dataframe_when_coverage_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    cfg = SimpleNamespace(
        start_dt=datetime(2022, 1, 1),
        end_dt=datetime(2024, 12, 31),
        periods=[2022, 2023, 2024],
    )
    monkeypatch.setattr(module, "_resolve_date_range", lambda _state: cfg)

    fetch_calls = {"count": 0}

    async def _unexpected_fetch(*_args, **_kwargs):
        fetch_calls["count"] += 1
        raise AssertionError("_fetch_raw_data should not be called on cache hit")

    monkeypatch.setattr(module, "_fetch_raw_data", _unexpected_fetch)

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    agent._working_memory = module.FundamentalWorkingMemoryManager()
    agent.db = object()

    cached_df = _sample_df()
    agent._working_memory.upsert_cached_financial_data(
        conversation_id="conv-cache-hit",
        ticker="aapl",
        granularity="yearly",
        financial_data=cached_df,
    )

    state = _AgentState(
        ticker="AAPL",
        granularity="yearly",
        conversation_id="conv-cache-hit",
    )
    result = asyncio.run(agent._data_prep_node(state))

    assert fetch_calls["count"] == 0
    assert result["financial_data"].equals(cached_df)
    assert result["available_concepts"] == list(cached_df.index)


def test_data_prep_cache_miss_when_requested_range_exceeds_cached_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    cfg = SimpleNamespace(
        start_dt=datetime(2021, 1, 1),
        end_dt=datetime(2024, 12, 31),
        periods=[2021, 2022, 2023, 2024],
    )
    monkeypatch.setattr(module, "_resolve_date_range", lambda _state: cfg)

    fetch_calls = {"count": 0}
    fetched_df = pd.DataFrame(
        [[80.0, 90.0, 100.0, 110.0]],
        index=["Revenues"],
        columns=["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"],
    )

    async def _fake_fetch(*_args, **_kwargs):
        fetch_calls["count"] += 1
        return fetched_df.copy(deep=True), pd.DataFrame()

    monkeypatch.setattr(module, "_fetch_raw_data", _fake_fetch)

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    agent._working_memory = module.FundamentalWorkingMemoryManager()
    agent.db = object()
    agent._working_memory.upsert_cached_financial_data(
        conversation_id="conv-cache-miss",
        ticker="AAPL",
        granularity="yearly",
        financial_data=pd.DataFrame(
            [[90.0, 100.0, 110.0]],
            index=["Revenues"],
            columns=["2022-12-31", "2023-12-31", "2024-12-31"],
        ),
    )

    state = _AgentState(
        ticker="AAPL",
        granularity="yearly",
        conversation_id="conv-cache-miss",
    )
    result = asyncio.run(agent._data_prep_node(state))

    assert fetch_calls["count"] == 1
    assert result["financial_data"].equals(fetched_df)


def test_data_prep_cache_miss_when_cached_data_is_price_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module
    from core.agents.working_memory.fundamental_working_memory import (
        FundamentalTickerDataFrameCacheEntry,
    )

    cfg = SimpleNamespace(
        start_dt=datetime(2022, 1, 1),
        end_dt=datetime(2024, 12, 31),
        periods=[2022, 2023, 2024],
    )
    monkeypatch.setattr(module, "_resolve_date_range", lambda _state: cfg)

    fetch_calls = {"count": 0}
    fetched_df = pd.DataFrame(
        [[80.0, 90.0, 100.0]],
        index=["Revenues"],
        columns=["2022-12-31", "2023-12-31", "2024-12-31"],
    )

    async def _fake_fetch(*_args, **_kwargs):
        fetch_calls["count"] += 1
        return fetched_df.copy(deep=True), pd.DataFrame()

    monkeypatch.setattr(module, "_fetch_raw_data", _fake_fetch)

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    agent._working_memory = module.FundamentalWorkingMemoryManager()
    agent.db = object()

    memory = agent._working_memory.get_conversation_memory("conv-price-only-cache")
    memory.financial_df_cache_by_ticker["AAPL"] = FundamentalTickerDataFrameCacheEntry(
        ticker_key="AAPL",
        granularity="yearly",
        financial_data=pd.DataFrame(
            [[150.0, 180.0, 210.0]],
            index=["stock_price"],
            columns=["2022-12-31", "2023-12-31", "2024-12-31"],
        ),
    )

    state = _AgentState(
        ticker="AAPL",
        granularity="yearly",
        conversation_id="conv-price-only-cache",
    )
    result = asyncio.run(agent._data_prep_node(state))

    assert fetch_calls["count"] == 1
    assert result["financial_data"].equals(fetched_df)


def test_tool_executor_populates_structured_result_detail_fields() -> None:
    state = _AgentState(
        query="Compute margin trend",
        ticker="AAPL",
        financial_data=_sample_df(),
        tool_plan=IterativeToolPlan(
            batches=[
                ToolCallBatch(
                    calls=[
                        ToolCallSpec(
                            tool_name="cagr",
                            parameters={"metric": "Revenues"},
                            reasoning="Measure long-term growth.",
                        )
                    ],
                    batch_reasoning="Run CAGR on revenue baseline.",
                )
            ],
            data_summary="Compute one growth metric.",
        ),
        current_batch_index=0,
    )
    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    result = asyncio.run(agent._tool_executor_node(state))

    logs = result["executor_logs"]
    assert logs
    call_log = logs[0].calls[0]
    assert call_log.added_row_count >= 0
    assert isinstance(call_log.series_values, dict)
    assert hasattr(call_log, "scalar_value")


def test_tool_executor_updates_working_memory_dataframe_cache() -> None:
    from core.agents import fundamental_analysis_agent as module

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    agent._working_memory = module.FundamentalWorkingMemoryManager()

    state = _AgentState(
        query="Compute margin trend",
        ticker="AAPL",
        conversation_id="conv-tool-cache",
        granularity="yearly",
        financial_data=_sample_df(),
        tool_plan=IterativeToolPlan(
            batches=[
                ToolCallBatch(
                    calls=[
                        ToolCallSpec(
                            tool_name="cagr",
                            parameters={"metric": "Revenues"},
                            reasoning="Measure long-term growth.",
                        )
                    ],
                    batch_reasoning="Run CAGR on revenue baseline.",
                )
            ],
            data_summary="Compute one growth metric.",
        ),
        current_batch_index=0,
    )

    result = asyncio.run(agent._tool_executor_node(state))

    cached_df = agent._working_memory.resolve_cached_financial_data(
        conversation_id="conv-tool-cache",
        ticker="AAPL",
        granularity="yearly",
    )
    assert cached_df is not None
    assert cached_df.equals(result["financial_data"])
    for row_label in result["computed_row_labels"]:
        assert row_label in cached_df.index


def test_tool_planner_includes_prior_working_memory_block_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    planner_payload = {
        "batches": [],
        "data_summary": "No tools needed.",
        "selected_row_labels": ["Revenues"],
    }
    llm = _FakeLLM(payload=planner_payload)
    monkeypatch.setattr(module.service_manager, "get_agent", lambda temperature=0: llm)

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    agent._working_memory = module.FundamentalWorkingMemoryManager()
    agent._working_memory.persist_finalized_turn(
        conversation_id="conv-fund-plan",
        turn_id="turn-1",
        query="Prior turn",
        task_completed=True,
        task_completion_reason="",
        computed_row_labels=["gross_margin"],
        executor_logs=[
            ExecutorBatchLog(
                batch_index=0,
                batch_reasoning="Prior batch",
                calls=[
                    ExecutorToolLog(
                        tool_name="profitability_ratios",
                        parameters={},
                        success=True,
                        summary="Computed gross_margin",
                        output_row_labels=["gross_margin"],
                        added_row_count=1,
                    )
                ],
            )
        ],
    )

    state = _AgentState(
        query="Current turn",
        ticker="AAPL",
        conversation_id="conv-fund-plan",
        available_concepts=["Revenues", "NetIncomeLoss"],
        tool_results=[],
        computed_row_labels=[],
        iteration_count=0,
    )
    _ = asyncio.run(agent._tool_planner_node(state))

    assert llm.schema is IterativeToolPlan
    assert llm.last_messages is not None
    planner_user_prompt = llm.last_messages[1].content
    assert "Goal:" in planner_user_prompt
    assert "Tasklist Cap:" in planner_user_prompt
    assert "Prior fundamentals working memory" in planner_user_prompt
    assert "call=profitability_ratios" in planner_user_prompt


def test_tool_planner_truncates_batches_to_tasklist_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    monkeypatch.setattr(module.settings, "FUNDAMENTAL_AGENT_TASKLIST_MAX_ITEMS", 1)

    llm = _FakeLLM(
        payload={
            "batches": [
                {
                    "batch_reasoning": "First",
                    "calls": [
                        {
                            "tool_name": "cagr",
                            "parameters": {"metric": "Revenues"},
                            "reasoning": "Growth",
                        }
                    ],
                },
                {
                    "batch_reasoning": "Second",
                    "calls": [
                        {
                            "tool_name": "cagr",
                            "parameters": {"metric": "NetIncomeLoss"},
                            "reasoning": "Income growth",
                        }
                    ],
                },
            ],
            "data_summary": "Two-step plan.",
            "selected_row_labels": [],
        }
    )
    monkeypatch.setattr(module.service_manager, "get_agent", lambda temperature=0: llm)

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    agent._working_memory = module.FundamentalWorkingMemoryManager()
    state = _AgentState(
        goal="Assess profitability and growth.",
        ticker="AAPL",
        available_concepts=["Revenues", "NetIncomeLoss"],
        iteration_count=0,
    )
    result = asyncio.run(agent._tool_planner_node(state))
    assert result["tool_plan"].batch_count() == 1


def test_task_summary_node_records_batch_summary() -> None:
    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    state = _AgentState(
        active_task_id="fund-task-1",
        active_task_completed=True,
        executor_logs=[
            ExecutorBatchLog(
                batch_index=0,
                batch_reasoning="Compute trend metrics.",
                calls=[
                    ExecutorToolLog(
                        tool_name="cagr",
                        parameters={"metric": "Revenues"},
                        success=True,
                        output_row_labels=["Revenues CAGR"],
                    )
                ],
            )
        ],
        task_summaries=[],
    )
    result = asyncio.run(agent._task_summary_node(state))
    assert len(result["task_summaries"]) == 1
    summary: FundamentalTaskSummary = result["task_summaries"][0]
    assert summary.task_id == "fund-task-1"
    assert summary.output_row_labels == ["Revenues CAGR"]


def test_task_summary_node_marks_failure_from_executor_logs() -> None:
    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    state = _AgentState(
        active_task_id="fund-task-2",
        active_task_completed=True,  # should be ignored by summary logic now
        last_batch_failed=True,
        executor_logs=[
            ExecutorBatchLog(
                batch_index=1,
                batch_reasoning="Compute derived metric.",
                calls=[
                    ExecutorToolLog(
                        tool_name="custom_formula",
                        parameters={"formula": "A-B"},
                        success=False,
                        error="Missing row B",
                        output_row_labels=[],
                    )
                ],
            )
        ],
        task_summaries=[],
    )
    result = asyncio.run(agent._task_summary_node(state))
    summary: FundamentalTaskSummary = result["task_summaries"][0]
    assert summary.task_id == "fund-task-2"
    assert summary.success is False


def test_task_summary_node_no_active_task_is_noop() -> None:
    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    state = _AgentState(
        active_task_id="",
        executor_logs=[
            ExecutorBatchLog(batch_index=0, batch_reasoning="", calls=[])
        ],
        task_summaries=[],
    )
    result = asyncio.run(agent._task_summary_node(state))
    assert result == {}


def test_should_continue_router_parity() -> None:
    plan = IterativeToolPlan(
        batches=[
            ToolCallBatch(
                calls=[
                    ToolCallSpec(
                        tool_name="cagr",
                        parameters={"metric": "Revenues"},
                        reasoning="Batch 0",
                    )
                ],
                batch_reasoning="Batch 0",
            ),
            ToolCallBatch(
                calls=[
                    ToolCallSpec(
                        tool_name="cagr",
                        parameters={"metric": "NetIncomeLoss"},
                        reasoning="Batch 1",
                    )
                ],
                batch_reasoning="Batch 1",
            ),
        ],
        data_summary="Two-step plan",
    )

    fail_state = _AgentState(
        tool_plan=plan,
        current_batch_index=1,
        iteration_count=1,
        last_batch_failed=True,
        tool_results=[ToolResult(tool_name="cagr", success=False, error="boom")],
    )
    assert FundamentalAnalysisAgent._should_continue(fail_state) == "replan"

    next_state = _AgentState(
        tool_plan=plan,
        current_batch_index=1,
        iteration_count=1,
        last_batch_failed=False,
        tool_results=[ToolResult(tool_name="cagr", success=True, summary="ok")],
    )
    assert FundamentalAnalysisAgent._should_continue(next_state) == "next_batch"

    done_state = _AgentState(
        tool_plan=IterativeToolPlan(
            batches=[plan.batches[0]],
            data_summary="Single batch",
        ),
        current_batch_index=1,
        iteration_count=1,
        last_batch_failed=False,
        tool_results=[ToolResult(tool_name="cagr", success=True, summary="ok")],
    )
    assert FundamentalAnalysisAgent._should_continue(done_state) == "done"


def test_build_graph_routes_executor_directly_to_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agents import fundamental_analysis_agent as module

    class _FakeStateGraph:
        last_instance = None

        def __init__(self, *_args, **_kwargs):
            self.nodes = []
            self.edges = []
            self.conditional = []
            _FakeStateGraph.last_instance = self

        def add_node(self, name, fn):
            self.nodes.append((name, fn))

        def add_edge(self, src, dst):
            self.edges.append((src, dst))

        def add_conditional_edges(self, src, router, mapping):
            self.conditional.append((src, router, mapping))

        def compile(self):
            return self

    monkeypatch.setattr(module, "StateGraph", _FakeStateGraph)
    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    _ = agent._build_graph()
    fake_graph = _FakeStateGraph.last_instance
    node_names = [name for name, _fn in fake_graph.nodes]

    assert "task_check" not in node_names
    assert ("tool_executor", "task_summary") in fake_graph.edges
    assert ("tool_executor", "task_check") not in fake_graph.edges


def test_render_memory_summary_delegates_to_manager() -> None:
    summary = {
        "tools_used": ["cagr"],
        "key_rows": ["Revenues"],
        "task_completed": True,
        "main_conclusion": "Revenue trend is stable.",
    }
    rendered = FundamentalAnalysisAgent.render_memory_summary(summary)
    assert rendered.startswith("tools=cagr")
    assert "rows=Revenues" in rendered


def test_analyst_node_handles_list_content_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.agents import fundamental_analysis_agent as module

    class _FakeResponse:
        def __init__(self, content):
            self.content = content

    class _FakeAnalystLLM:
        async def ainvoke(self, _messages):
            return _FakeResponse(
                [
                    {"text": "Revenue trend remains positive.\n"},
                    {
                        "text": (
                            "<sentiment>{\"score\":72,\"label\":\"BUY\","
                            "\"rationale\":\"Growth and margins are stable.\"}"
                            "</sentiment>"
                        )
                    },
                ]
            )

    monkeypatch.setattr(module.settings, "ENABLE_ANALYSIS_TOKEN_STREAMING", False)
    monkeypatch.setattr(
        module.service_manager, "get_agent", lambda temperature=0.7: _FakeAnalystLLM()
    )

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    state = _AgentState(
        query="Assess financial health",
        ticker="AAPL",
        financial_data=_sample_df(),
        task_completed=True,
        task_completion_reason="",
    )

    result = asyncio.run(agent._analyst_node(state))

    assert "Revenue trend remains positive." in result["analysis"]
    assert "<sentiment>" not in result["analysis"]
    assert result["sentiment"] is not None
    assert result["sentiment"].label == "BUY"


def test_tool_executor_preserves_derived_dependency_across_batches() -> None:
    plan = IterativeToolPlan(
        batches=[
            ToolCallBatch(
                calls=[
                    ToolCallSpec(
                        tool_name="custom_formula",
                        parameters={
                            "metric_name": "MarketCap",
                            "expression": (
                                "stock_price * "
                                "Common_stock_and_capital_stock__shares_authorized_in_shares"
                            ),
                            "dependencies": [
                                "stock_price",
                                "Common_stock_and_capital_stock__shares_authorized_in_shares",
                            ],
                            "description": "Market capitalization.",
                        },
                        reasoning="Derive MarketCap first.",
                    )
                ],
                batch_reasoning="Batch 0 derives MarketCap.",
            ),
            ToolCallBatch(
                calls=[
                    ToolCallSpec(
                        tool_name="custom_formula",
                        parameters={
                            "metric_name": "EnterpriseValue",
                            "expression": (
                                "MarketCap + LongTermDebt - CashAndMarketableSecurities"
                            ),
                            "dependencies": [
                                "MarketCap",
                                "LongTermDebt",
                                "CashAndMarketableSecurities",
                            ],
                            "description": "EV from market cap, debt, and cash.",
                        },
                        reasoning="Use derived MarketCap from batch 0.",
                    )
                ],
                batch_reasoning="Batch 1 derives EnterpriseValue from MarketCap.",
            ),
        ],
        data_summary="Derive valuation inputs in sequence.",
    )

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    base_state = _AgentState(
        ticker="AAPL",
        financial_data=_valuation_dependency_df(),
        tool_plan=plan,
        current_batch_index=0,
        tool_results=[],
    )
    first = asyncio.run(agent._tool_executor_node(base_state))

    second_state = _AgentState(
        ticker="AAPL",
        financial_data=first["financial_data"],
        tool_plan=plan,
        current_batch_index=first["current_batch_index"],
        tool_results=base_state.tool_results + first["tool_results"],
    )
    second = asyncio.run(agent._tool_executor_node(second_state))

    assert first["tool_results"][0].success is True
    assert second["tool_results"][0].success is True
    assert "MarketCap" in second["financial_data"].index
    assert "EnterpriseValue" in second["financial_data"].index


def test_tool_executor_returns_per_batch_tool_result_delta_only() -> None:
    plan = IterativeToolPlan(
        batches=[
            ToolCallBatch(
                calls=[
                    ToolCallSpec(
                        tool_name="custom_formula",
                        parameters={
                            "metric_name": "MarketCap",
                            "expression": (
                                "stock_price * "
                                "Common_stock_and_capital_stock__shares_authorized_in_shares"
                            ),
                            "dependencies": [
                                "stock_price",
                                "Common_stock_and_capital_stock__shares_authorized_in_shares",
                            ],
                            "description": "Market capitalization.",
                        },
                        reasoning="First batch.",
                    )
                ],
            ),
            ToolCallBatch(
                calls=[
                    ToolCallSpec(
                        tool_name="custom_formula",
                        parameters={
                            "metric_name": "EnterpriseValue",
                            "expression": (
                                "MarketCap + LongTermDebt - CashAndMarketableSecurities"
                            ),
                            "dependencies": [
                                "MarketCap",
                                "LongTermDebt",
                                "CashAndMarketableSecurities",
                            ],
                            "description": "Enterprise value.",
                        },
                        reasoning="Second batch.",
                    )
                ],
            ),
        ],
        data_summary="Two sequential calls.",
    )

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    base_state = _AgentState(
        ticker="AAPL",
        financial_data=_valuation_dependency_df(),
        tool_plan=plan,
        current_batch_index=0,
        tool_results=[],
    )
    first = asyncio.run(agent._tool_executor_node(base_state))

    accumulated_after_first = base_state.tool_results + first["tool_results"]
    assert len(first["tool_results"]) == 1
    assert len(accumulated_after_first) == 1

    second_state = _AgentState(
        ticker="AAPL",
        financial_data=first["financial_data"],
        tool_plan=plan,
        current_batch_index=first["current_batch_index"],
        tool_results=accumulated_after_first,
    )
    second = asyncio.run(agent._tool_executor_node(second_state))
    accumulated_after_second = second_state.tool_results + second["tool_results"]

    assert len(second["tool_results"]) == 1
    assert len(accumulated_after_second) == 2


def test_tool_executor_recomputed_row_overwrites_previous_values() -> None:
    plan = IterativeToolPlan(
        batches=[
            ToolCallBatch(
                calls=[
                    ToolCallSpec(
                        tool_name="custom_formula",
                        parameters={
                            "metric_name": "MarketCap",
                            "expression": (
                                "stock_price * "
                                "Common_stock_and_capital_stock__shares_authorized_in_shares"
                            ),
                            "dependencies": [
                                "stock_price",
                                "Common_stock_and_capital_stock__shares_authorized_in_shares",
                            ],
                            "description": "Initial market cap.",
                        },
                        reasoning="Compute baseline.",
                    )
                ],
            ),
            ToolCallBatch(
                calls=[
                    ToolCallSpec(
                        tool_name="custom_formula",
                        parameters={
                            "metric_name": "MarketCap",
                            "expression": (
                                "stock_price * "
                                "Common_stock_and_capital_stock__shares_authorized_in_shares * 2"
                            ),
                            "dependencies": [
                                "stock_price",
                                "Common_stock_and_capital_stock__shares_authorized_in_shares",
                            ],
                            "description": "Overwrite with updated assumption.",
                        },
                        reasoning="Recompute MarketCap with multiplier.",
                    )
                ],
            ),
        ],
        data_summary="Recompute same derived row across batches.",
    )

    agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
    first_state = _AgentState(
        ticker="AAPL",
        financial_data=_valuation_dependency_df(),
        tool_plan=plan,
        current_batch_index=0,
    )
    first = asyncio.run(agent._tool_executor_node(first_state))

    second_state = _AgentState(
        ticker="AAPL",
        financial_data=first["financial_data"],
        tool_plan=plan,
        current_batch_index=first["current_batch_index"],
        tool_results=first["tool_results"],
    )
    second = asyncio.run(agent._tool_executor_node(second_state))

    result_df = second["financial_data"]
    market_cap = result_df.loc["MarketCap"]
    assert float(market_cap["2024-12-31"]) == 2000.0
    assert float(market_cap["2025-12-31"]) == 2200.0
