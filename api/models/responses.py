"""
api/models/responses.py

Outbound response schemas for the AlphaMesh FastAPI layer.

Design notes
────────────
• DataFramePayload uses row-major format (index/columns/data) which is compact,
  preserves the metric-rows / period-columns shape that the backend produces,
  and maps directly to a 2-D array that any chart library can consume.
  NaN values are encoded as JSON null.

• TickerResult is a List even though the current agent processes one ticker at
  a time.  This design is intentional: when multi-ticker support lands in the
  orchestrator, the frontend contract is already correct.

• StreamEvent is the only type sent over the SSE channel.  The frontend can
  switch on `event_type` and render accordingly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


class DataFramePayload(BaseModel):
    """
    Serialised representation of a pandas DataFrame.

    Shape: rows = financial metrics, columns = reporting period dates.

    Example
    ───────
    {
      "index":   ["Revenues", "NetIncomeLoss", "gross_margin"],
      "columns": ["2021-12-31", "2022-12-31", "2023-12-31"],
      "data":    [[394328.0, 365817.0, 274515.0],
                  [99803.0,  94680.0,  57411.0],
                  [0.43,     0.43,     0.39]]
    }

    NaN → null  (JSON null is preserved through Pydantic serialisation).
    """

    index: List[str] = Field(description="Row labels (metric names).")
    columns: List[str] = Field(description="Column labels (period-end dates).")
    data: List[List[Optional[float]]] = Field(
        description="Row-major numeric values; NaN represented as null."
    )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "DataFramePayload":
        """Construct a DataFramePayload from a pandas DataFrame."""

        def _safe(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                f = float(v)
                return None if math.isnan(f) or math.isinf(f) else f
            except (TypeError, ValueError):
                return None

        return cls(
            index=[str(idx) for idx in df.index],
            columns=[str(col) for col in df.columns],
            data=[
                [_safe(df.iat[r, c]) for c in range(len(df.columns))]
                for r in range(len(df.index))
            ],
        )


class SourceItem(BaseModel):
    """One cited news article."""

    source_id: int
    title: str
    url: str
    page_content: str = Field(description="Aggregated chunk text from this article.")


class TickerResult(BaseModel):
    """
    All output specific to a single ticker symbol.

    Currently one TickerResult is produced per run.  When multi-ticker
    orchestration lands, the list in FinalResult grows accordingly.
    """

    ticker: str
    analysis_text: str = Field(description="Per-agent or combined narrative analysis.")
    financial_data: Optional[DataFramePayload] = Field(
        default=None,
        description="Serialised financial DataFrame; null when no EDGAR data was fetched.",
    )
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="News sources cited in the analysis.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level results
# ─────────────────────────────────────────────────────────────────────────────


class FinalResult(BaseModel):
    """
    Delivered as the payload of the final `complete` SSE event.
    Contains everything the frontend needs to render the full response.
    """

    request_id: str
    conversation_id: str
    synthesis: str = Field(description="Orchestrator's cross-agent narrative.")
    ticker_results: List[TickerResult] = Field(
        default_factory=list,
        description="Per-ticker structured output (one entry per company analysed).",
    )
    agent_analyses: Dict[str, str] = Field(
        default_factory=dict,
        description="Raw analysis text keyed by agent name, e.g. 'news_agent'.",
    )
    duration_ms: float = Field(description="Total wall-clock time for this request.")


# ─────────────────────────────────────────────────────────────────────────────
# Acknowledgement (returned synchronously by POST /chat)
# ─────────────────────────────────────────────────────────────────────────────


class ChatAck(BaseModel):
    """
    Immediate response from POST /api/v1/chat.

    The client should open GET /api/v1/stream/{request_id} immediately after
    receiving this to begin receiving SSE progress events.
    """

    request_id: str
    conversation_id: str


# ─────────────────────────────────────────────────────────────────────────────
# SSE event envelope
# ─────────────────────────────────────────────────────────────────────────────


class StreamEvent(BaseModel):
    """
    Every SSE message from GET /api/v1/stream/{request_id} is a StreamEvent.

    event_type discriminates the payload:
      progress — incremental status update while the agents are running
      complete  — final structured result; stream ends after this
      error     — unrecoverable failure; stream ends after this
    """

    event_type: Literal[
        "progress",
        "complete",
        "error",
        "init",
        "chart",
        "metrics",
        "ticker_resolved",
    ]
    request_id: str

    # ── present when event_type == "progress" ─────────────────────────────
    source: Optional[str] = None
    level: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None

    # ── present when event_type == "complete" ─────────────────────────────
    result: Optional[FinalResult] = None

    # ── present when event_type == "error" ────────────────────────────────
    error: Optional[str] = None

    # -- present when event_type == "init" (market quote) ----------------------
    quote: Optional[Dict[str, Any]] = None

    # -- present when event_type == "chart" (intraday series) ------------------
    chart: Optional[List[Dict[str, Any]]] = None

    # -- present when event_type == "metrics" (fundamentals DataFrame) ---------
    financial_data: Optional[DataFramePayload] = None

    # -- present when event_type == "ticker_resolved" --------------------------
    ticker: Optional[str] = None
    tickers: Optional[List[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Conversation list / history
# ─────────────────────────────────────────────────────────────────────────────


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str


class ConversationSummary(BaseModel):
    conversation_id: str
    created_at: str
    last_message_at: str
    message_count: int


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[ConversationMessage]



