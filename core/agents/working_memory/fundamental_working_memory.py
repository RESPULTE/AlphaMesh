from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import pandas as pd
from core.agents.working_memory.base import (
    ConversationWorkingMemoryBase,
    ConversationWorkingMemoryManagerBase,
    TurnRelevantMemoryBase,
)
from core.agents.utils import trim_text
from core.agents.models.fundamental_agent_models import ExecutorBatchLog


@dataclass
class FundamentalTurnCallRecord:
    tool_name: str
    parameters: dict
    success: bool
    error: str | None = None
    summary: str = ""
    reasoning: str | None = None
    output_row_labels: List[str] = field(default_factory=list)
    scalar_value: float | None = None
    series_values: dict[str, float] = field(default_factory=dict)
    added_row_count: int = 0


@dataclass
class FundamentalTurnBatchRecord:
    batch_index: int
    batch_reasoning: str = ""
    calls: List[FundamentalTurnCallRecord] = field(default_factory=list)


@dataclass
class FundamentalTurnRelevantMemory(TurnRelevantMemoryBase):
    task_completed: bool = True
    task_completion_reason: str = ""
    computed_row_labels: List[str] = field(default_factory=list)
    batch_records: List[FundamentalTurnBatchRecord] = field(default_factory=list)
    tool_call_count: int = 0
    successful_tool_call_count: int = 0
    failed_tool_call_count: int = 0


@dataclass
class FundamentalTickerDataFrameCacheEntry:
    ticker_key: str
    granularity: str
    financial_data: pd.DataFrame
    min_period: pd.Timestamp | None = None
    max_period: pd.Timestamp | None = None


@dataclass
class FundamentalConversationWorkingMemory(
    ConversationWorkingMemoryBase[FundamentalTurnRelevantMemory]
):
    turn_records: List[FundamentalTurnRelevantMemory] = field(default_factory=list)
    financial_df_cache_by_ticker: dict[str, FundamentalTickerDataFrameCacheEntry] = (
        field(default_factory=dict)
    )


class FundamentalWorkingMemoryManager(
    ConversationWorkingMemoryManagerBase[
        FundamentalTurnRelevantMemory, FundamentalConversationWorkingMemory
    ]
):
    AGENT_NAME = "fundamentals_agent"

    def __init__(self, *, max_turns: int = 20) -> None:
        super().__init__(
            max_chunks=1,
            max_turns=max_turns,
            conversation_factory=FundamentalConversationWorkingMemory,
        )

    @staticmethod
    def normalize_ticker_key(ticker: str | None) -> str:
        return str(ticker or "").strip().upper()

    @staticmethod
    def _normalize_granularity(granularity: str | None) -> str:
        value = str(granularity or "").strip().lower()
        return value if value in {"yearly", "quarterly"} else "yearly"

    @staticmethod
    def _extract_period_bounds(
        financial_data: pd.DataFrame,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if financial_data.empty:
            return None, None
        parsed = pd.to_datetime(financial_data.columns, errors="coerce")
        parsed = parsed[~parsed.isna()]
        if len(parsed) == 0:
            return None, None
        return parsed.min(), parsed.max()

    @staticmethod
    def _to_timestamp(value: datetime | pd.Timestamp | None) -> pd.Timestamp | None:
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
        except Exception:
            return None
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        return ts

    @staticmethod
    def _has_non_price_fundamental_rows(financial_data: pd.DataFrame | None) -> bool:
        if financial_data is None or financial_data.empty:
            return False
        for label in financial_data.index:
            normalized = str(label or "").strip().lower()
            if normalized and not normalized.startswith("stock_price"):
                return True
        return False

    def upsert_cached_financial_data(
        self,
        *,
        conversation_id: str,
        ticker: str | None,
        granularity: str | None,
        financial_data: pd.DataFrame | None,
    ) -> None:
        if not conversation_id or financial_data is None or financial_data.empty:
            return
        if not self._has_non_price_fundamental_rows(financial_data):
            return
        ticker_key = self.normalize_ticker_key(ticker)
        if not ticker_key:
            return
        normalized_granularity = self._normalize_granularity(granularity)
        cached_df = financial_data.copy(deep=True)
        min_period, max_period = self._extract_period_bounds(cached_df)
        memory = self.get_conversation_memory(conversation_id)
        memory.financial_df_cache_by_ticker[ticker_key] = (
            FundamentalTickerDataFrameCacheEntry(
                ticker_key=ticker_key,
                granularity=normalized_granularity,
                financial_data=cached_df,
                min_period=min_period,
                max_period=max_period,
            )
        )

    def resolve_cached_financial_data(
        self,
        *,
        conversation_id: str,
        ticker: str | None,
        granularity: str | None,
        start_dt: datetime | pd.Timestamp | None = None,
        end_dt: datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame | None:
        if not conversation_id:
            return None
        ticker_key = self.normalize_ticker_key(ticker)
        if not ticker_key:
            return None
        memory = self.get_existing_conversation_memory(conversation_id)
        if memory is None:
            return None
        entry = memory.financial_df_cache_by_ticker.get(ticker_key)
        if entry is None:
            return None
        if entry.granularity != self._normalize_granularity(granularity):
            return None

        cached_df = entry.financial_data
        if cached_df is None or cached_df.empty:
            return None
        if not self._has_non_price_fundamental_rows(cached_df):
            memory.financial_df_cache_by_ticker.pop(ticker_key, None)
            return None

        requested_start = self._to_timestamp(start_dt)
        requested_end = self._to_timestamp(end_dt)
        if requested_start is not None or requested_end is not None:
            min_period = entry.min_period
            max_period = entry.max_period
            if min_period is None or max_period is None:
                min_period, max_period = self._extract_period_bounds(cached_df)
            if min_period is None or max_period is None:
                return None
            granularity_key = self._normalize_granularity(granularity)
            if granularity_key == "yearly":
                if (
                    requested_start is not None
                    and requested_start.year < min_period.year
                ):
                    return None
                if requested_end is not None and requested_end.year > max_period.year:
                    return None
            elif granularity_key == "quarterly":
                min_quarter = min_period.to_period("Q")
                max_quarter = max_period.to_period("Q")
                if (
                    requested_start is not None
                    and requested_start.to_period("Q") < min_quarter
                ):
                    return None
                if (
                    requested_end is not None
                    and requested_end.to_period("Q") > max_quarter
                ):
                    return None
            else:
                if requested_start is not None and requested_start < min_period:
                    return None
                if requested_end is not None and requested_end > max_period:
                    return None

        return cached_df.copy(deep=True)

    @staticmethod
    def _convert_batch_records(
        executor_logs: List[ExecutorBatchLog],
    ) -> List[FundamentalTurnBatchRecord]:
        records: List[FundamentalTurnBatchRecord] = []
        for batch in executor_logs:
            call_records = [
                FundamentalTurnCallRecord(
                    tool_name=call.tool_name,
                    parameters=dict(call.parameters or {}),
                    success=bool(call.success),
                    error=call.error,
                    summary=call.summary or "",
                    reasoning=call.reasoning,
                    output_row_labels=list(call.output_row_labels or []),
                    scalar_value=call.scalar_value,
                    series_values=dict(call.series_values or {}),
                    added_row_count=int(call.added_row_count or 0),
                )
                for call in batch.calls
            ]
            records.append(
                FundamentalTurnBatchRecord(
                    batch_index=batch.batch_index,
                    batch_reasoning=batch.batch_reasoning or "",
                    calls=call_records,
                )
            )
        return records

    def persist_finalized_turn(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        query: str,
        task_completed: bool,
        task_completion_reason: str,
        computed_row_labels: List[str],
        executor_logs: List[ExecutorBatchLog],
    ) -> None:
        if not conversation_id:
            return
        batch_records = self._convert_batch_records(executor_logs)
        all_calls = [call for batch in batch_records for call in batch.calls]
        success_count = sum(1 for call in all_calls if call.success)
        failed_count = len(all_calls) - success_count

        self.append_turn_record(
            conversation_id=conversation_id,
            record=FundamentalTurnRelevantMemory(
                turn_id=turn_id,
                query=query,
                task_completed=task_completed,
                task_completion_reason=task_completion_reason,
                computed_row_labels=list(computed_row_labels or []),
                batch_records=batch_records,
                tool_call_count=len(all_calls),
                successful_tool_call_count=success_count,
                failed_tool_call_count=failed_count,
            ),
        )

    def render_planner_working_memory_block(
        self,
        conversation_id: str,
        *,
        turn_limit: int = 4,
        max_calls_per_turn: int = 12,
    ) -> str:
        if not conversation_id:
            return "(none)"
        memory = self.get_existing_conversation_memory(conversation_id)
        if memory is None or not memory.turn_records:
            return "(none)"

        lines: List[str] = []
        for row in memory.turn_records[-turn_limit:]:
            lines.append(
                f"- turn={row.turn_id} query={row.query}\n"
                f"  task_completed={row.task_completed}\n"
                f"  task_completion_reason={row.task_completion_reason or '(none)'}\n"
                f"  tool_calls={row.tool_call_count} "
                f"(success={row.successful_tool_call_count}, failed={row.failed_tool_call_count})\n"
                f"  computed_rows={','.join(row.computed_row_labels[:8]) or '(none)'}"
            )

            rendered_calls = 0
            for batch in row.batch_records:
                if rendered_calls >= max_calls_per_turn:
                    break
                lines.append(
                    f"  batch={batch.batch_index} reasoning={batch.batch_reasoning or '(none)'}"
                )
                for call in batch.calls:
                    if rendered_calls >= max_calls_per_turn:
                        break
                    status = "SUCCESS" if call.success else f"FAILURE ({call.error or ''})"
                    lines.append(
                        f"    call={call.tool_name} status={status} "
                        f"added_rows={call.added_row_count} summary={call.summary or '(none)'}"
                    )
                    rendered_calls += 1
            if row.tool_call_count > rendered_calls:
                lines.append(f"  ... and {row.tool_call_count - rendered_calls} more call(s)")
        return "\n".join(lines)

    @staticmethod
    def render_memory_summary(memory_summary: dict) -> str:
        if not memory_summary:
            return ""
        tools = memory_summary.get("tools_used") or []
        if not isinstance(tools, list):
            tools = []
        key_rows = (
            memory_summary.get("key_rows")
            or memory_summary.get("computed_rows")
            or []
        )
        if not isinstance(key_rows, list):
            key_rows = []
        completed = bool(memory_summary.get("task_completed", True))
        conclusion = trim_text(memory_summary.get("main_conclusion") or "", max_chars=220)
        return (
            f"tools={','.join(str(t) for t in tools[:5]) or 'none'}; "
            f"rows={','.join(str(r) for r in key_rows[:6]) or 'none'}; "
            f"task_completed={completed}; "
            f"conclusion={conclusion or 'N/A'}"
        )

    @classmethod
    def build_context_from_history_summaries(
        cls,
        turns: List[dict],
        window: int = 8,
    ) -> str:
        rows = cls.collect_agent_summaries_from_turns(
            turns=turns,
            agent_name=cls.AGENT_NAME,
        )
        if not rows:
            return ""
        lines: List[str] = []
        for ts, payload in rows[-window:]:
            rendered = cls.render_memory_summary(payload)
            if not rendered:
                rendered = cls.render_memory_summary_fallback(payload)
            lines.append(f"- [{ts}] {rendered}")
        return "\n".join(lines)
