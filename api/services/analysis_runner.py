"""
api/services/analysis_runner.py

Orchestrates one analysis turn:
  1. Registers per-request SSE queue and EventQueue sink.
  2. Sets the ContextVar so agents publish to the right response group.
  3. Calls OrchestratorAgent.run() in the background.
  4. Builds FinalResult and enqueues the `complete` SSE event.
  5. Tears down the sink and schedules queue cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from langchain_core.messages import HumanMessage

from api.models.requests import ChatRequest
from api.models.responses import (
    DataFramePayload,
    FinalResult,
    FundamentalsChartSpecPayload,
    FundamentalsVisualizationPayload,
    SourceItem,
    TickerResult,
)
from api.services.conversation_service import ConversationStore
from api.services.event_broadcaster import EventBroadcaster
from api.services.session_service import SessionService
from api.sinks.sse_sink import SSESink
from core.agents.orchestrator_agent import OrchestratorAgent
from core.services import service_manager

logger = logging.getLogger(__name__)
_SUPPORTED_CHART_TYPES = {
    "line",
    "bar",
    "area",
    "scatter",
    "stacked_bar",
    "stacked_area",
    "pie",
}
_SNAPSHOT_UNSUPPORTED_TYPES = {"line", "area", "scatter", "stacked_area"}


def _extract_row_semantics(final_response) -> dict[str, dict[str, Any]]:
    raw = getattr(final_response, "fundamentals_row_semantics", {}) or {}
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for row_label, meta in raw.items():
        meta_dict: dict[str, Any] | None = None
        if isinstance(meta, dict):
            meta_dict = dict(meta)
        elif hasattr(meta, "model_dump"):
            try:
                dumped = meta.model_dump()
                if isinstance(dumped, dict):
                    meta_dict = dumped
            except Exception:
                meta_dict = None
        if not meta_dict:
            continue
        key = str(row_label or "").strip()
        if not key:
            continue
        normalized[key] = meta_dict
    return normalized


def _build_metrics_payload(final_response) -> Optional[DataFramePayload]:
    fundamental_df = getattr(final_response, "fundamental_data", None)
    if fundamental_df is None or getattr(fundamental_df, "empty", True):
        return None
    try:
        return DataFramePayload.from_dataframe(
            fundamental_df,
            row_semantics=_extract_row_semantics(final_response),
        )
    except Exception:
        logger.warning(
            "_build_metrics_payload: DataFrame serialisation failed; skipping."
        )
        return None


def _normalise_chart_type(chart_type: str, data_mode: str) -> str:
    normalised_type = (chart_type or "").strip().lower()
    if normalised_type not in _SUPPORTED_CHART_TYPES:
        return "bar" if data_mode == "snapshot" else "line"
    if normalised_type == "pie":
        return "pie"
    if data_mode == "snapshot" and normalised_type in _SNAPSHOT_UNSUPPORTED_TYPES:
        return "bar"
    return normalised_type


def _normalise_data_mode(data_mode: str) -> str:
    normalised = (data_mode or "").strip().lower()
    if normalised in {"timeseries", "snapshot"}:
        return normalised
    return "timeseries"


def _build_fundamentals_visualization_payload(
    final_response,
) -> Optional[FundamentalsVisualizationPayload]:
    viz_plan = getattr(final_response, "fundamentals_visualization", None)
    if viz_plan is None:
        return None

    charts: list[FundamentalsChartSpecPayload] = []
    for chart in getattr(viz_plan, "charts", []) or []:
        raw_mode = _normalise_data_mode(getattr(chart, "data_mode", "timeseries"))
        chart_type = _normalise_chart_type(getattr(chart, "chart_type", "line"), raw_mode)
        data_mode = "snapshot" if chart_type == "pie" else raw_mode
        if data_mode == "snapshot" and chart_type in _SNAPSHOT_UNSUPPORTED_TYPES:
            chart_type = "bar"

        row_labels = [
            row for row in (getattr(chart, "row_labels", []) or []) if isinstance(row, str)
        ]
        if not row_labels:
            continue

        charts.append(
            FundamentalsChartSpecPayload(
                chart_type=chart_type,
                data_mode=data_mode,
                snapshot_period=(getattr(chart, "snapshot_period", "latest") or "latest"),
                title=(getattr(chart, "title", "") or "").strip(),
                row_labels=row_labels,
                group_rows=bool(getattr(chart, "group_rows", True)),
                rationale=getattr(chart, "rationale", "") or "",
            )
        )

    raw_data_payload: Optional[DataFramePayload] = None
    raw_df = getattr(final_response, "fundamentals_raw_display_data", None)
    if raw_df is not None and not getattr(raw_df, "empty", True):
        try:
            raw_data_payload = DataFramePayload.from_dataframe(
                raw_df,
                row_semantics=_extract_row_semantics(final_response),
            )
        except Exception:
            logger.warning(
                "_build_fundamentals_visualization_payload: raw_data serialisation failed."
            )

    raw_row_labels = [
        row
        for row in (getattr(viz_plan, "raw_row_labels", []) or [])
        if isinstance(row, str)
    ]

    payload = FundamentalsVisualizationPayload(
        charts=charts,
        raw_row_labels=raw_row_labels,
        raw_data=raw_data_payload,
        reviewer_notes=getattr(viz_plan, "reviewer_notes", "") or "",
        task_completed=bool(getattr(final_response, "fundamentals_task_completed", True)),
        task_completion_reason=getattr(
            final_response, "fundamentals_task_completion_reason", ""
        )
        or "",
    )

    if not payload.charts and not payload.raw_row_labels and payload.raw_data is None:
        return None
    return payload


class AnalysisRunner:
    """
    Wires OrchestratorAgent ? EventQueue ? SSE for a single analysis request.
    """

    def __init__(
        self,
        broadcaster: EventBroadcaster,
        store: ConversationStore,
        session_service: SessionService,
        orchestrator: OrchestratorAgent,
    ) -> None:
        self._broadcaster = broadcaster
        self._store = store
        self._session_service = session_service
        self._orchestrator = orchestrator

    async def launch(
        self,
        request_id: str,
        chat_request: ChatRequest,
        *,
        user_id: str,
        session_id: str,
    ) -> str:
        """
        Prepare conversation ownership state, then fire-and-forget the analysis task.
        Returns the conversation_id immediately so POST /chat can ACK the client.
        """
        conversation_id = chat_request.conversation_id or str(uuid4())
        await self._store.ensure_conversation(conversation_id, user_id)
        await self._session_service.link_conversation(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )
        self._broadcaster.create(request_id)
        asyncio.create_task(
            self._run(
                request_id,
                conversation_id,
                chat_request,
                user_id=user_id,
                session_id=session_id,
            ),
            name=f"analysis_{request_id[:8]}",
        )
        return conversation_id

    async def _emit_market_data(
        self,
        event_queue: asyncio.Queue,
        request_id: str,
        ticker: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        market_svc = service_manager.get_market_data_service()
        try:
            quote, chart = await asyncio.wait_for(
                asyncio.gather(
                    market_svc.get_quote(ticker),
                    market_svc.get_intraday(ticker),
                ),
                timeout=10.0,
            )
            chart_payload = chart if isinstance(chart, list) else []
            await event_queue.put(
                {"event_type": "init", "request_id": request_id, "quote": quote}
            )
            await event_queue.put(
                {"event_type": "chart", "request_id": request_id, "chart": chart_payload}
            )
            quote_payload = quote if isinstance(quote, dict) else {"ticker": ticker, "companyName": ticker}
            return quote_payload, chart_payload
        except asyncio.TimeoutError:
            fallback_quote = {"ticker": ticker, "companyName": ticker}
            await event_queue.put(
                {
                    "event_type": "init",
                    "request_id": request_id,
                    "quote": fallback_quote,
                }
            )
            return fallback_quote, []
        except Exception as exc:
            logger.warning("Market data fetch failed: %s", exc)
            fallback_quote = {"ticker": ticker, "companyName": ticker}
            await event_queue.put(
                {
                    "event_type": "init",
                    "request_id": request_id,
                    "quote": fallback_quote,
                }
            )
            return fallback_quote, []

    async def _refresh_conversation_memory_index(
        self,
        *,
        conversation_id: str,
        user_email: str,
        turns: list[dict],
    ) -> None:
        try:
            await service_manager.get_conversation_memory_service().ensure_index(
                conversation_id=conversation_id,
                user_email=user_email,
                turns=turns,
            )
        except Exception:
            logger.exception(
                "_refresh_conversation_memory_index: background index refresh failed for '%s'",
                conversation_id,
            )

    # -- Private --------------------------------------------------------------

    async def _run(
        self,
        request_id: str,
        conversation_id: str,
        chat_request: ChatRequest,
        *,
        user_id: str,
        session_id: str,
    ) -> None:
        """Background task: runs the full agent pipeline for one user turn."""
        from core.event_queue import _current_response_label, get_queue

        queue = get_queue()
        event_queue = self._broadcaster.get(request_id)
        if event_queue is None:
            logger.error(
                "AnalysisRunner: no broadcaster queue for request %s", request_id
            )
            return

        loop = asyncio.get_running_loop()
        t_start = time.monotonic()
        market_data_emitted = False
        market_data_task: Optional[asyncio.Task[None]] = None
        market_quote: Optional[dict[str, Any]] = None
        market_chart: list[dict[str, Any]] = []

        # -- 1. Register response group + SSE sink -----------------------------
        response_label = queue.start_response("orchestrator")
        response_group = queue.get_response(response_label)
        response_id = response_group.response_id  # stable UUID for this run

        sink = SSESink(
            request_id=request_id,
            response_id=response_id,
            event_queue=event_queue,
            loop=loop,
        )
        queue.add_sink(sink)

        # -- 2. Bind ContextVar so agents publish to the right group ------------
        token = _current_response_label.set(response_label)

        # -- 2a. Market data fetch is driven by validated ticker events ---------
        async def _emit_and_capture_market_data(ticker: str) -> None:
            nonlocal market_quote, market_chart
            quote_payload, chart_payload = await self._emit_market_data(
                event_queue, request_id, ticker
            )
            market_quote = quote_payload
            if chart_payload:
                market_chart = chart_payload

        def _schedule_market_data(ticker: str) -> None:
            nonlocal market_data_emitted, market_data_task
            if market_data_emitted:
                return
            market_data_emitted = True
            market_data_task = asyncio.create_task(
                _emit_and_capture_market_data(ticker),
                name=f"market_{request_id[:8]}",
            )

        class _MarketDataSink:
            def on_event(self, event) -> None:
                if event.response_id != response_id:
                    return
                data = getattr(event, "data", None) or {}
                if data.get("event_type") != "ticker_resolved":
                    return
                ticker = data.get("ticker") or (data.get("tickers") or [None])[0]
                if not ticker:
                    return
                try:
                    loop.call_soon_threadsafe(_schedule_market_data, str(ticker))
                except RuntimeError:
                    _schedule_market_data(str(ticker))

            def on_group_opened(self, group) -> None:
                pass

            def on_group_closed(self, group) -> None:
                pass

        market_sink = _MarketDataSink()
        queue.add_sink(market_sink)

        try:
            # -- 3. Load conversation state ------------------------------------
            history = await self._store.get_langchain_messages(
                conversation_id,
                user_email=user_id,
            )
            history_turns = await self._store.get_turns(
                conversation_id,
                user_email=user_id,
            )
            messages = history + [HumanMessage(content=chat_request.message)]
            conversation_memory_block = "(none)"
            conversation_memory_hits: list[dict] = []

            try:
                (
                    conversation_memory_block,
                    conversation_memory_hits,
                ) = await service_manager.get_conversation_memory_service().ensure_index_and_retrieve(
                    conversation_id=conversation_id,
                    user_email=user_id,
                    turns=history_turns,
                    query=chat_request.message,
                )
            except Exception:
                logger.exception(
                    "_run: failed to prepare private conversation memory for '%s'",
                    conversation_id,
                )
                conversation_memory_block = "(none)"
                conversation_memory_hits = []

            # -- 4. Run the orchestrator ---------------------------------------
            final_response = await self._orchestrator.run(
                messages=messages,
                conversation_id=conversation_id,
                user_email=user_id,
                history_turns=history_turns,
                conversation_memory_block=conversation_memory_block,
                conversation_memory_hits=conversation_memory_hits,
            )

            final_ticker = (getattr(final_response, "tickers", []) or [None])[0]
            if final_ticker and not market_data_emitted:
                market_data_emitted = True
                await _emit_and_capture_market_data(final_ticker)
            elif market_data_task is not None and not market_data_task.done():
                await market_data_task

            # -- 5. Emit metrics payload (if available) ------------------------
            visualization_payload = _build_fundamentals_visualization_payload(
                final_response
            )
            if visualization_payload is not None:
                await event_queue.put(
                    {
                        "event_type": "fundamentals_visualization",
                        "request_id": request_id,
                        "fundamentals_visualization": visualization_payload.model_dump(),
                    }
                )

            metrics_payload = _build_metrics_payload(final_response)
            if metrics_payload is not None:
                await event_queue.put(
                    {
                        "event_type": "metrics",
                        "request_id": request_id,
                        "financial_data": metrics_payload.model_dump(),
                    }
                )

            # -- 6. Build wire-format result -----------------------------------
            duration_ms = (time.monotonic() - t_start) * 1000
            final_result = _build_final_result(
                request_id=request_id,
                conversation_id=conversation_id,
                final_response=final_response,
                duration_ms=duration_ms,
                fundamentals_visualization_payload=visualization_payload,
                market_quote=market_quote,
                market_chart=market_chart,
            )

            # -- 7. Persist rich conversation turn -----------------------------
            turn_payload = _build_turn_payload(
                request_id=request_id,
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id,
                turn_id=getattr(final_response, "turn_id", "") or f"{conversation_id}:{request_id}",
                user_message=chat_request.message,
                final_result=final_result,
                agent_memory_summaries=(
                    getattr(final_response, "agent_memory_summaries", {}) or {}
                ),
            )
            await self._store.append_turn(
                conversation_id=conversation_id,
                user_email=user_id,
                turn=turn_payload,
            )
            asyncio.create_task(
                self._refresh_conversation_memory_index(
                    conversation_id=conversation_id,
                    user_email=user_id,
                    turns=[*history_turns, turn_payload],
                ),
                name=f"convmem_index_{conversation_id[:8]}",
            )

            # -- 8. Deliver completion event -----------------------------------
            await sink.drain()
            await event_queue.put(
                {
                    "event_type": "complete",
                    "request_id": request_id,
                    "result": final_result.model_dump(),
                }
            )

        except Exception as exc:
            logger.exception(
                "AnalysisRunner: unhandled exception for request %s", request_id
            )
            try:
                await sink.drain()
                await event_queue.put(
                    {
                        "event_type": "error",
                        "request_id": request_id,
                        "error": str(exc),
                    }
                )
            except Exception:
                pass

        finally:
            # -- Teardown: remove sink, reset ContextVar, close response group --
            queue.remove_sink(sink)
            queue.remove_sink(market_sink)
            _current_response_label.reset(token)
            # Close the response group without touching _active_group (race-safe)
            if response_group.is_open:
                response_group.close()

            # Give the SSE consumer time to drain before evicting the queue
            self._broadcaster.schedule_cleanup(request_id)


# -- Result builder (pure function � easy to unit-test) -----------------------


def _build_final_result(
    request_id: str,
    conversation_id: str,
    final_response,  # core.agents.models.orchestrator_models.FinalResponse
    duration_ms: float,
    fundamentals_visualization_payload: Optional[FundamentalsVisualizationPayload] = None,
    market_quote: Optional[dict[str, Any]] = None,
    market_chart: Optional[list[dict[str, Any]]] = None,
) -> FinalResult:
    """
    Map OrchestratorAgent's FinalResponse ? API-layer FinalResult.
    """
    tickers = getattr(final_response, "tickers", [])
    agent_analyses: dict = getattr(final_response, "agent_analyses", {}) or {}
    sources_raw = getattr(final_response, "sources", []) or []
    fundamental_df = getattr(final_response, "fundamental_data", None)

    # Primary analysis text (prefer fundamentals, fall back to news, then synthesis)
    primary_analysis = (
        agent_analyses.get("fundamentals_agent")
        or agent_analyses.get("news_agent")
        or final_response.summary
        or ""
    )

    # Serialise the financial DataFrame
    financial_payload: Optional[DataFramePayload] = None
    if fundamental_df is not None and not fundamental_df.empty:
        try:
            financial_payload = DataFramePayload.from_dataframe(
                fundamental_df,
                row_semantics=_extract_row_semantics(final_response),
            )
        except Exception:
            logger.warning(
                "_build_final_result: DataFrame serialisation failed; skipping."
            )

    # Map CitedSource ? SourceItem
    sources = [
        SourceItem(
            source_id=s.source_id,
            title=s.title,
            url=s.url,
            page_content=s.page_content,
        )
        for s in sources_raw
    ]

    ticker_result = TickerResult(
        ticker=tickers[0] if tickers else "",
        analysis_text=primary_analysis,
        market_quote=market_quote,
        market_chart=market_chart or [],
        financial_data=financial_payload,
        fundamentals_visualization=(
            fundamentals_visualization_payload
            if fundamentals_visualization_payload is not None
            else _build_fundamentals_visualization_payload(final_response)
        ),
        sources=sources,
    )

    return FinalResult(
        request_id=request_id,
        conversation_id=conversation_id,
        synthesis=final_response.summary or "",
        ticker_results=[ticker_result],
        agent_analyses=agent_analyses,
        duration_ms=round(duration_ms, 1),
    )


def _build_turn_payload(
    *,
    request_id: str,
    conversation_id: str,
    user_id: str,
    session_id: str,
    turn_id: str,
    user_message: str,
    final_result: FinalResult,
    agent_memory_summaries: dict,
) -> dict:
    """Build one persisted rich turn record from final response payload."""
    created_at = datetime.now(timezone.utc).isoformat()
    tickers = [r.ticker for r in final_result.ticker_results if r.ticker]

    return {
        "turn_id": turn_id,
        "request_id": request_id,
        "conversation_id": conversation_id,
        "user_email": user_id,
        "session_id": session_id,
        "created_at": created_at,
        "duration_ms": final_result.duration_ms,
        "user_message": user_message,
        "assistant_synthesis": final_result.synthesis,
        "agent_analyses": final_result.agent_analyses,
        "agent_memory_summaries": agent_memory_summaries,
        "ticker_results": [r.model_dump() for r in final_result.ticker_results],
        "tickers": tickers,
    }
