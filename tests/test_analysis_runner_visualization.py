from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from api.models.responses import (
    FundamentalsVisualizationPayload,
    StreamEvent,
    TickerResult,
)
from api.services.analysis_runner import (
    _build_final_result,
    _build_fundamentals_visualization_payload,
    _build_turn_payload,
)
from core.agents.models.fundamental_agent_models import ChartSpec, VisualizationPlan


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        data=[[100.0, 120.0], [10.0, 9.5]],
        index=["Revenues", "NetIncomeLoss"],
        columns=["2023-12-31", "2024-12-31"],
    )


def test_visualization_payload_is_optional_for_legacy_response() -> None:
    legacy = SimpleNamespace()
    payload = _build_fundamentals_visualization_payload(legacy)
    assert payload is None


def test_visualization_payload_sanitises_chart_contract() -> None:
    df = _sample_df()
    response = SimpleNamespace(
        fundamentals_visualization=VisualizationPlan(
            charts=[
                ChartSpec(
                    chart_type="unsupported_type",
                    data_mode="snapshot",
                    title="Fallback chart",
                    row_labels=["Revenues"],
                    group_rows=True,
                ),
                ChartSpec(
                    chart_type="pie",
                    data_mode="timeseries",
                    title="Composition",
                    row_labels=["Revenues", "NetIncomeLoss"],
                    group_rows=True,
                ),
            ],
            raw_row_labels=["Revenues"],
            reviewer_notes="Show key rows.",
        ),
        fundamentals_raw_display_data=df.loc[["Revenues"]],
        fundamentals_task_completed=False,
        fundamentals_task_completion_reason="Need more checks",
    )

    payload = _build_fundamentals_visualization_payload(response)
    assert payload is not None
    assert payload.charts[0].chart_type == "bar"
    assert payload.charts[0].data_mode == "snapshot"
    assert payload.charts[1].chart_type == "pie"
    assert payload.charts[1].data_mode == "snapshot"
    assert payload.raw_data is not None
    assert payload.task_completed is False


def test_final_result_embeds_fundamentals_visualization_payload() -> None:
    df = _sample_df()
    response = SimpleNamespace(
        tickers=["AAPL"],
        agent_analyses={"fundamentals_agent": "Fundamental analysis"},
        sources=[],
        fundamental_data=df,
        summary="Synthesis",
        fundamentals_visualization=VisualizationPlan(
            charts=[
                ChartSpec(
                    chart_type="line",
                    data_mode="timeseries",
                    title="Trend",
                    row_labels=["Revenues"],
                    group_rows=True,
                )
            ],
            raw_row_labels=["Revenues"],
        ),
        fundamentals_raw_display_data=df.loc[["Revenues"]],
        fundamentals_task_completed=True,
        fundamentals_task_completion_reason="",
    )
    result = _build_final_result(
        request_id="req-1",
        conversation_id="conv-1",
        final_response=response,
        duration_ms=45.6,
    )

    assert result.ticker_results
    assert result.ticker_results[0].fundamentals_visualization is not None
    assert result.ticker_results[0].fundamentals_visualization.charts[0].chart_type == "line"


def test_stream_event_and_ticker_result_are_backward_compatible() -> None:
    ticker_result = TickerResult(ticker="AAPL", analysis_text="ok", sources=[])
    assert ticker_result.fundamentals_visualization is None

    viz_payload = FundamentalsVisualizationPayload(charts=[], raw_row_labels=[])
    event = StreamEvent(
        event_type="fundamentals_visualization",
        request_id="req-1",
        fundamentals_visualization=viz_payload,
    )
    assert event.event_type == "fundamentals_visualization"


def test_turn_payload_includes_agent_memory_summaries_and_turn_id() -> None:
    result = _build_final_result(
        request_id="req-2",
        conversation_id="conv-9",
        final_response=SimpleNamespace(
            tickers=["MSFT"],
            agent_analyses={"news_agent": "News view"},
            sources=[],
            fundamental_data=None,
            summary="Combined synthesis",
            fundamentals_visualization=None,
            fundamentals_raw_display_data=None,
            fundamentals_task_completed=True,
            fundamentals_task_completion_reason="",
        ),
        duration_ms=12.4,
    )

    payload = _build_turn_payload(
        request_id="req-2",
        conversation_id="conv-9",
        user_id="demo@alphamesh.local",
        session_id="sess-1",
        turn_id="turn-abc",
        user_message="What changed this quarter?",
        final_result=result,
        agent_memory_summaries={"news_agent": {"source_count": 3}},
    )

    assert payload["turn_id"] == "turn-abc"
    assert payload["agent_memory_summaries"]["news_agent"]["source_count"] == 3
