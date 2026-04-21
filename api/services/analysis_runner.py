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
from typing import Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from api.models.requests import ChatRequest
from api.models.responses import DataFramePayload, FinalResult, SourceItem, TickerResult
from api.services.event_broadcaster import EventBroadcaster
from api.sinks.sse_sink import SSESink
from core.memory.conversation.store import ConversationStore
from core.memory.sessions.session_service import SessionService
from core.services import service_manager

logger = logging.getLogger(__name__)

def _build_metrics_payload(final_response) -> Optional[DataFramePayload]:
    fundamental_df = getattr(final_response, "fundamental_data", None)
    if fundamental_df is None or getattr(fundamental_df, "empty", True):
        return None
    try:
        return DataFramePayload.from_dataframe(fundamental_df)
    except Exception:
        logger.warning(
            "_build_metrics_payload: DataFrame serialisation failed; skipping."
        )
        return None


class AnalysisRunner:
    """
    Wires OrchestratorAgent ? EventQueue ? SSE for a single analysis request.
    """

    def __init__(
        self,
        broadcaster: EventBroadcaster,
        store: ConversationStore,
        session_service: SessionService,
    ) -> None:
        self._broadcaster = broadcaster
        self._store = store
        self._session_service = session_service

    def launch(
        self,
        request_id: str,
        chat_request: ChatRequest,
        *,
        user_id: str,
        session_id: str,
    ) -> str:
        """
        Create the asyncio.Queue, then fire-and-forget the analysis task.
        Returns the conversation_id immediately so POST /chat can ACK the client.
        """
        conversation_id = chat_request.conversation_id or str(uuid4())
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
    ) -> None:
        market_svc = service_manager.get_market_data_service()
        try:
            quote, chart = await asyncio.wait_for(
                asyncio.gather(
                    market_svc.get_quote(ticker),
                    market_svc.get_intraday(ticker),
                ),
                timeout=10.0,
            )
            await event_queue.put(
                {"event_type": "init", "request_id": request_id, "quote": quote}
            )
            await event_queue.put(
                {"event_type": "chart", "request_id": request_id, "chart": chart}
            )
        except asyncio.TimeoutError:
            await event_queue.put(
                {
                    "event_type": "init",
                    "request_id": request_id,
                    "quote": {"ticker": ticker, "companyName": ticker},
                }
            )
        except Exception as exc:
            logger.warning("Market data fetch failed: %s", exc)
            await event_queue.put(
                {
                    "event_type": "init",
                    "request_id": request_id,
                    "quote": {"ticker": ticker, "companyName": ticker},
                }
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
        # Import here to avoid circular imports at module load time
        from core.agents.orchestrator_agent import OrchestratorAgent
        from core.event_queue import _current_response_label, get_queue

        orchestrator = OrchestratorAgent()
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
        def _schedule_market_data(ticker: str) -> None:
            nonlocal market_data_emitted
            if market_data_emitted:
                return
            market_data_emitted = True
            asyncio.create_task(
                self._emit_market_data(event_queue, request_id, ticker),
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
            # -- 3. Prepare conversation ---------------------------------------
            await self._store.ensure_conversation(
                conversation_id, user_id
            )
            await self._session_service.link_conversation(
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id,
            )
            history = await self._store.get_langchain_messages(conversation_id)
            messages = history + [HumanMessage(content=chat_request.message)]

            # -- 4. Run the orchestrator ---------------------------------------
            final_response = await orchestrator.run(
                messages=messages,
                conversation_id=conversation_id,
                user_email=user_id,
            )

            final_ticker = (getattr(final_response, "tickers", []) or [None])[0]
            if final_ticker and not market_data_emitted:
                await self._emit_market_data(event_queue, request_id, final_ticker)

            # -- 5. Persist conversation turn ----------------------------------
            await self._store.add_messages(
                conversation_id,
                [
                    HumanMessage(content=chat_request.message),
                    AIMessage(content=final_response.summary or ""),
                ],
            )

            # -- 6. Emit metrics payload (if available) ------------------------
            metrics_payload = _build_metrics_payload(final_response)
            if metrics_payload is not None:
                await event_queue.put(
                    {
                        "event_type": "metrics",
                        "request_id": request_id,
                        "financial_data": metrics_payload.model_dump(),
                    }
                )

            # -- 7. Build wire-format result -----------------------------------
            duration_ms = (time.monotonic() - t_start) * 1000
            final_result = _build_final_result(
                request_id=request_id,
                conversation_id=conversation_id,
                final_response=final_response,
                duration_ms=duration_ms,
            )

            # -- 8. Deliver completion event -----------------------------------
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
            financial_payload = DataFramePayload.from_dataframe(fundamental_df)
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
        financial_data=financial_payload,
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
