"""
api/services/analysis_runner.py

Orchestrates one analysis turn:
  1. Registers per-request SSE queue and EventQueue sink.
  2. Sets the ContextVar so agents publish to the right response group.
  3. Calls OrchestratorAgent.run() in the background.
  4. Builds FinalResult and enqueues the `complete` SSE event.
  5. Tears down the sink and schedules queue cleanup.

Separation of concerns
───────────────────────
• OrchestratorAgent knows nothing about SSE or request_id.
• EventQueue knows nothing about asyncio.Queue or SSE.
• AnalysisRunner is the only place that wires them together.

Concurrency
───────────
Each call to `launch()` creates an independent asyncio.Task.  Because
OrchestratorAgent.run() is stateless and all service singletons (LLM, Neo4j,
Chroma) are async-safe, concurrent tasks do not interfere.

The ContextVar `_current_response_label` is inherited by the new task (Python
asyncio tasks copy the ContextVar context from their creator).  Within the
task, the ContextVar is overwritten with `token = _cv.set(label)` and reset
in the finally block, so the creator's context is unaffected.
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
from api.services.conversation_store import ConversationStore
from api.services.event_broadcaster import EventBroadcaster
from api.sinks.sse_sink import SSESink

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """
    Wires OrchestratorAgent ↔ EventQueue ↔ SSE for a single analysis request.
    """

    def __init__(
        self,
        broadcaster: EventBroadcaster,
        store: ConversationStore,
    ) -> None:
        self._broadcaster = broadcaster
        self._store = store

    def launch(self, request_id: str, chat_request: ChatRequest) -> str:
        """
        Create the asyncio.Queue, then fire-and-forget the analysis task.
        Returns the conversation_id immediately so POST /chat can ACK the client.
        """
        conversation_id = chat_request.conversation_id or str(uuid4())
        self._broadcaster.create(request_id)
        asyncio.create_task(
            self._run(request_id, conversation_id, chat_request),
            name=f"analysis_{request_id[:8]}",
        )
        return conversation_id

    # ── Private ───────────────────────────────────────────────────────────────

    async def _run(
        self,
        request_id: str,
        conversation_id: str,
        chat_request: ChatRequest,
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

        # ── 1. Register response group + SSE sink ─────────────────────────────
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

        # ── 2. Bind ContextVar so agents publish to the right group ───────────
        token = _current_response_label.set(response_label)

        try:
            # ── 3. Prepare conversation ────────────────────────────────────────
            await self._store.ensure_conversation(
                conversation_id, chat_request.user_email
            )
            history = await self._store.get_langchain_messages(conversation_id)
            messages = history + [HumanMessage(content=chat_request.message)]

            # ── 4. Run the orchestrator ────────────────────────────────────────
            final_response = await orchestrator.run(
                messages=messages,
                conversation_id=conversation_id,
                user_email=chat_request.user_email,
            )

            # ── 5. Persist conversation turn ──────────────────────────────────
            await self._store.add_messages(
                conversation_id,
                [
                    HumanMessage(content=chat_request.message),
                    AIMessage(content=final_response.summary or ""),
                ],
            )

            # ── 6. Build wire-format result ───────────────────────────────────
            duration_ms = (time.monotonic() - t_start) * 1000
            final_result = _build_final_result(
                request_id=request_id,
                conversation_id=conversation_id,
                final_response=final_response,
                duration_ms=duration_ms,
            )

            # ── 7. Deliver completion event ───────────────────────────────────
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
            # ── Teardown: remove sink, reset ContextVar, close response group ──
            queue.remove_sink(sink)
            _current_response_label.reset(token)
            # Close the response group without touching _active_group (race-safe)
            if response_group.is_open:
                response_group.close()

            # Give the SSE consumer time to drain before evicting the queue
            self._broadcaster.schedule_cleanup(request_id)


# ─────────────────────────────────────────────────────────────────────────────
# Result builder (pure function — easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────


def _build_final_result(
    request_id: str,
    conversation_id: str,
    final_response,  # core.agents.models.orchestrator_models.FinalResponse
    duration_ms: float,
) -> FinalResult:
    """
    Map OrchestratorAgent's FinalResponse → API-layer FinalResult.

    TickerResult strategy (current: single ticker)
    ───────────────────────────────────────────────
    • ticker          — first entry in final_response.tickers (added to
                        FinalResponse by the orchestrator patch)
    • analysis_text   — fundamentals analysis if available, else news analysis
    • financial_data  — serialised fundamental DataFrame (may be None)
    • sources         — cited news sources

    When multi-ticker orchestration lands, this function grows a loop over
    final_response.tickers with per-ticker data sourced from per-agent outputs.
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

    # Map CitedSource → SourceItem
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
