from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models.fundamental_agent_models import (
    ExecutorBatchLog,
    ExecutorToolLog,
    IterativeToolPlan,
    _AgentState,
)


class _FakeStructuredLLM:
    def __init__(self, response) -> None:
        self._response = response

    async def ainvoke(self, _messages):
        return self._response


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.schema = None
        self.structured_calls = 0

    def with_structured_output(self, schema):
        self.schema = schema
        self.structured_calls += 1
        return _FakeStructuredLLM(schema.model_validate(self._payload))


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
