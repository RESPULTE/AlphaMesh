from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models.base_agent_models import BaseAgentInput
from core.agents.models.fundamental_agent_models import (
    ExecutorBatchLog,
    ExecutorToolLog,
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
    assert "Prior fundamentals working memory" in planner_user_prompt
    assert "call=profitability_ratios" in planner_user_prompt
