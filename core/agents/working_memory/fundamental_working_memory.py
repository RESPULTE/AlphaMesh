from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

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
class FundamentalConversationWorkingMemory(
    ConversationWorkingMemoryBase[FundamentalTurnRelevantMemory]
):
    turn_records: List[FundamentalTurnRelevantMemory] = field(default_factory=list)


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
