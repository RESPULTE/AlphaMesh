import asyncio
import json
import re
from asyncio.log import logger
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage

from core.agents.financial_tools import ToolResult
from core.agents.models.fundamental_agent_models import (
    ChartSpec,
    CompletionReviewDecision,
    ExecutorBatchLog,
    VisualizationPlan,
)

MAX_TURN_TEXT_CHARS = 360


def _safe_create_task(coro) -> Optional[asyncio.Task]:
    """Create an asyncio task only when a running loop exists."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        logger.warning("_safe_create_task: no running event loop — task skipped.")
        return None


def _extract_last_human_message(messages: List[BaseMessage]) -> str:
    """Return the content of the most recent HumanMessage, or '' if none exists."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
    return ""


def _last_n_messages(messages: List[BaseMessage], n: int) -> List[BaseMessage]:
    """Return the last *n* messages; returns the full list when len ≤ n."""
    return messages[-n:] if len(messages) > n else list(messages)


def _safe_json(text: str) -> List[dict]:
    try:
        result = json.loads(text.strip())
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def trim_text(value: Any, *, max_chars: int = MAX_TURN_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalise_turn_timestamp(turn: dict) -> str:
    return str(turn.get("created_at") or turn.get("timestamp") or "").strip()


def extract_first_sentence(value: str, *, max_chars: int = 220) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text[:max_chars]


def build_turn_window_block(turns: List[dict], window: int) -> str:
    if not turns:
        return "(no prior turns)"
    lines: List[str] = []
    for idx, turn in enumerate(turns[-window:], start=1):
        ts = normalise_turn_timestamp(turn) or "unknown_time"
        user_message = trim_text(turn.get("user_message") or "")
        synthesis = trim_text(turn.get("assistant_synthesis") or "")
        lines.append(
            f"{idx}. [{ts}] User: {user_message or '(empty)'}\n"
            f"   Assistant: {synthesis or '(empty)'}"
        )
    return "\n".join(lines)


def render_memory_summary_fallback(summary: dict, *, max_chars: int = 350) -> str:
    return trim_text(json.dumps(summary, ensure_ascii=True), max_chars=max_chars)


def build_agent_memory_contexts(
    turns: List[dict],
    renderers: Dict[str, Callable[[dict], str]],
    *,
    window: int = 8,
) -> Dict[str, str]:
    by_agent: Dict[str, List[tuple[str, dict]]] = {}
    for turn in turns:
        summaries = turn.get("agent_memory_summaries") or {}
        if not isinstance(summaries, dict):
            continue
        ts = normalise_turn_timestamp(turn) or "unknown_time"
        for agent_name, payload in summaries.items():
            if not isinstance(payload, dict):
                continue
            by_agent.setdefault(str(agent_name), []).append((ts, payload))

    rendered: Dict[str, str] = {}
    for agent_name, rows in by_agent.items():
        renderer = renderers.get(agent_name)
        lines: List[str] = []
        for ts, payload in rows[-window:]:
            summary_text = (
                renderer(payload)
                if callable(renderer)
                else render_memory_summary_fallback(payload)
            )
            lines.append(f"- [{ts}] {summary_text}")
        rendered[agent_name] = "\n".join(lines)
    return rendered


def build_planner_memory_block(agent_memory_contexts: Dict[str, str]) -> str:
    if not agent_memory_contexts:
        return "(none)"
    lines: List[str] = []
    for agent_name in sorted(agent_memory_contexts.keys()):
        block = (agent_memory_contexts.get(agent_name) or "").strip()
        if not block:
            continue
        lines.append(f"[{agent_name}]\n{block}")
    return "\n\n".join(lines) if lines else "(none)"


def get_default_date_range(
    start_date: datetime | None,
    end_date: datetime | None,
    *,
    default_days_back: int = 30,
) -> tuple[datetime, datetime]:
    """Fill in missing start/end dates with defaults."""
    now_utc = datetime.now(timezone.utc)
    if end_date is None:
        end_date = now_utc
    if start_date is None:
        start_date = now_utc - timedelta(days=default_days_back)
    return start_date, end_date


def constrain_date_range(
    start_date: date,
    end_date: date,
    *,
    api_limit_days: int = 28,
) -> tuple[date, date]:
    """Constrain date range to bounded API window."""
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    now = datetime.now().date()
    api_limit_date = now - timedelta(days=api_limit_days)
    start_date_only = start_date
    end_date_only = end_date

    if end_date_only > now:
        end_date_only = now
    if start_date_only < api_limit_date:
        start_date_only = api_limit_date
    return start_date_only, end_date_only


def normalize_entity_tuple(name: Any, entity_type: Any) -> tuple[str, str] | None:
    normalized_name = str(name or "").strip()
    normalized_type = str(entity_type or "").strip()
    if not normalized_name or not normalized_type:
        return None
    return normalized_name, normalized_type


def coerce_entity_tuple(entity: Any) -> tuple[str, str] | None:
    if isinstance(entity, dict):
        return normalize_entity_tuple(
            entity.get("name") or entity.get("entity_name"),
            entity.get("entity_type"),
        )
    if isinstance(entity, (tuple, list)) and len(entity) >= 2:
        return normalize_entity_tuple(entity[0], entity[1])
    return normalize_entity_tuple(
        getattr(entity, "name", None),
        getattr(entity, "entity_type", None),
    )


def build_planner_relevance_context_block(
    chunks: List[Any],
    mapping: Dict[int, int],
    rationale_by_chunk_id: Dict[str, str] | None = None,
) -> str:
    rationale_by_chunk_id = rationale_by_chunk_id or {}
    lines: List[str] = []
    for idx, chunk in enumerate(chunks):
        source_id = mapping.get(idx, "?")
        chunk_id = str(getattr(chunk, "chunk_id", "") or "")
        rationale = rationale_by_chunk_id.get(chunk_id, "").strip()
        if not rationale:
            rationale = "Selected by planner as relevant to the query."
        chunk_text = str(getattr(chunk, "text", "") or "")
        lines.append(
            f"[{source_id}] Planner relevance rationale: {rationale}\n{chunk_text}"
        )
    return "\n\n".join(lines)


def build_analysis_context_prefix(
    *,
    company_context: str | None,
    agent_memory_context: str | None,
    cached_entities: List[tuple[str, str]],
) -> str:
    sections: List[str] = []
    if company_context:
        sections.append(f"Company Context:\n{company_context}")
    if agent_memory_context:
        sections.append(f"Agent Memory Context:\n{agent_memory_context}")
    if cached_entities:
        entity_lines = [
            f"  - {name} ({entity_type})" for name, entity_type in cached_entities
        ]
        sections.append("Known entities from prior turns:\n" + "\n".join(entity_lines))
    if not sections:
        return ""
    return "\n\n".join(sections) + "\n\n"


def remap_numeric_citations(
    analysis_text: str,
    sources: List[Any],
) -> tuple[str, List[Any]]:
    cited_ids = sorted(
        set(int(match) for match in re.findall(r"\[(\d+)\]", analysis_text))
    )
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(cited_ids, start=1)}

    def _remap(match: re.Match[str]) -> str:
        source_id = int(match.group(1))
        return (
            f"[{old_to_new[source_id]}]" if source_id in old_to_new else match.group(0)
        )

    remapped_text = re.sub(r"\[(\d+)\]", _remap, analysis_text)
    if not old_to_new:
        return remapped_text, []

    by_old_id: Dict[int, Any] = {}
    for source in sources:
        source_id = getattr(source, "source_id", None)
        if isinstance(source_id, int):
            by_old_id[source_id] = source

    remapped_sources: List[Any] = []
    for old_id, new_id in old_to_new.items():
        source = by_old_id.get(old_id)
        if source is None:
            continue
        if hasattr(source, "model_copy"):
            remapped_sources.append(source.model_copy(update={"source_id": new_id}))
            continue
        remapped_sources.append(source)
    return remapped_text, remapped_sources


def resolve_agent_memory_context(
    *,
    conversation_id: Optional[str],
    incoming_memory_context: Optional[str],
    memory_context_cache: Dict[str, str],
) -> tuple[str, str]:
    """
    Resolve the effective per-agent memory context for this turn.

    Rules:
    - If an incoming context is provided, it takes precedence and refreshes cache.
    - Otherwise, use cached context for the conversation if present.
    """
    conversation_key = (conversation_id or "").strip()
    incoming = (incoming_memory_context or "").strip()
    cached = memory_context_cache.get(conversation_key, "") if conversation_key else ""

    if conversation_key and incoming and incoming != cached:
        memory_context_cache[conversation_key] = incoming
    return conversation_key, (incoming or cached)


def persist_agent_memory_summary(
    *,
    conversation_id: str,
    rendered_summary: str,
    memory_context_cache: Dict[str, str],
) -> None:
    """Persist a rendered agent summary into the per-conversation memory cache."""
    if not conversation_id:
        return
    summary = (rendered_summary or "").strip()
    if not summary:
        return
    memory_context_cache[conversation_id] = summary


def _build_combined_company_context(
    tickers: List[str],
    context_blocks: Dict[str, str],
) -> Optional[str]:
    """
    Combine per-ticker company context blocks into a single string for
    injection into an agent's prompt. Returns None when no blocks exist.
    """
    blocks = [context_blocks[t] for t in tickers if t in context_blocks]
    return "\n\n---\n\n".join(blocks) if blocks else None


def effective_goal(state: Any) -> str:
    text = (getattr(state, "goal", None) or getattr(state, "query", "") or "").strip()
    if text:
        return text
    return "Perform fundamental analysis for the provided ticker."


def executor_logs_to_text(logs: List[ExecutorBatchLog]) -> str:
    if not logs:
        return "(none)"
    lines: List[str] = []
    for batch in logs[-20:]:
        lines.append(
            f"Batch {batch.batch_index} | reasoning: {batch.batch_reasoning or '(none)'}"
        )
        for call in batch.calls:
            status = "SUCCESS" if call.success else f"FAILURE ({call.error or ''})"
            lines.append(
                f"  - {call.tool_name} | {status} | params={call.parameters} | "
                f"rows={call.output_row_labels} | summary={call.summary}"
            )
    return "\n".join(lines)


def tool_results_to_text(results: List[ToolResult]) -> str:
    if not results:
        return "(none)"
    return "\n".join(
        f"[{r.tool_name}] {'SUCCESS' if r.success else 'FAILURE'} | "
        f"summary={r.summary or ''} | error={r.error or ''}"
        for r in results[-100:]
    )


def dedupe_preserve_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalise_data_mode(data_mode: str) -> str:
    normalised = (data_mode or "").strip().lower()
    if normalised in {"timeseries", "snapshot"}:
        return normalised
    return "timeseries"


def normalise_snapshot_period(snapshot_period: str) -> str:
    value = (snapshot_period or "").strip()
    return value or "latest"


def normalise_chart_spec(
    chart: ChartSpec,
    *,
    supported_chart_types: Set[str],
    snapshot_unsupported_types: Set[str],
) -> Tuple[str, str, str]:
    data_mode = normalise_data_mode(chart.data_mode)
    requested_type = (chart.chart_type or "").strip().lower()

    if requested_type not in supported_chart_types:
        requested_type = "bar" if data_mode == "snapshot" else "line"

    chart_type = requested_type
    if chart_type == "pie":
        data_mode = "snapshot"
    if data_mode == "snapshot" and chart_type in snapshot_unsupported_types:
        chart_type = "bar"

    snapshot_period = normalise_snapshot_period(chart.snapshot_period)
    return chart_type, data_mode, snapshot_period


def extract_relevant_rows(
    state: Any,
    financial_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deterministically select DataFrame rows relevant to fundamentals analysis.

    Sources:
    1. selected_row_labels from no-tools plans.
    2. Tool call parameter values matching existing row labels.
    3. added_rows keys from successful tool outputs.
    """
    labels: set[str] = set()
    available_index: list[str] = list(financial_data.index)
    available_set: set[str] = set(available_index)

    plan = getattr(state, "tool_plan", None)
    if plan is not None:
        for label in getattr(plan, "selected_row_labels", []) or []:
            if label in available_set:
                labels.add(label)

        for batch in getattr(plan, "batches", []) or []:
            for spec in getattr(batch, "calls", []) or []:
                parameters = getattr(spec, "parameters", {}) or {}
                for value in parameters.values():
                    if isinstance(value, str) and value in available_set:
                        labels.add(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item in available_set:
                                labels.add(item)

    for result in getattr(state, "tool_results", []) or []:
        if getattr(result, "success", False) and getattr(result, "added_rows", None):
            for row_label in result.added_rows:
                if row_label in available_set:
                    labels.add(row_label)

    if not labels:
        return financial_data

    ordered = [lbl for lbl in available_index if lbl in labels]
    return financial_data.loc[ordered]


def sanitize_visualization_plan(
    *,
    state: Any,
    financial_data: pd.DataFrame,
    decision: CompletionReviewDecision,
    max_rows_per_chart: int,
    max_raw_rows: int,
    supported_chart_types: Set[str],
    snapshot_unsupported_types: Set[str],
) -> Tuple[VisualizationPlan, pd.DataFrame]:
    available_index: List[str] = list(financial_data.index)
    available_set: Set[str] = set(available_index)
    used_chart_rows: Set[str] = set()
    sanitized_charts: List[ChartSpec] = []

    for chart in decision.charts:
        chart_type, data_mode, snapshot_period = normalise_chart_spec(
            chart,
            supported_chart_types=supported_chart_types,
            snapshot_unsupported_types=snapshot_unsupported_types,
        )
        candidate_rows = dedupe_preserve_order(
            [
                row
                for row in (chart.row_labels or [])
                if isinstance(row, str)
                and row in available_set
                and row not in used_chart_rows
            ]
        )
        if not candidate_rows:
            continue
        candidate_rows = candidate_rows[:max_rows_per_chart]

        if chart.group_rows:
            sanitized_charts.append(
                ChartSpec(
                    chart_type=chart_type,
                    data_mode=data_mode,
                    snapshot_period=snapshot_period,
                    title=(chart.title or "Financial Trend").strip(),
                    row_labels=candidate_rows,
                    group_rows=True,
                    rationale=chart.rationale,
                )
            )
            used_chart_rows.update(candidate_rows)
            continue

        for row_label in candidate_rows:
            sanitized_charts.append(
                ChartSpec(
                    chart_type=chart_type,
                    data_mode=data_mode,
                    snapshot_period=snapshot_period,
                    title=(chart.title or row_label).strip(),
                    row_labels=[row_label],
                    group_rows=False,
                    rationale=chart.rationale,
                )
            )
            used_chart_rows.add(row_label)

    if not sanitized_charts:
        fallback_df = extract_relevant_rows(state, financial_data)
        fallback_rows = [row for row in fallback_df.index if row in available_set]
        fallback_rows = fallback_rows[:max_rows_per_chart]
        if not fallback_rows and available_index:
            fallback_rows = available_index[:max_rows_per_chart]
        if fallback_rows:
            sanitized_charts = [
                ChartSpec(
                    chart_type="line",
                    data_mode="timeseries",
                    snapshot_period="latest",
                    title="Key Financial Trends",
                    row_labels=fallback_rows,
                    group_rows=True,
                    rationale="Fallback visualisation because no valid chart rows were selected.",
                )
            ]

    raw_row_labels = dedupe_preserve_order(
        [
            row
            for row in (decision.raw_row_labels or [])
            if isinstance(row, str) and row in available_set
        ]
    )
    if not raw_row_labels:
        fallback_raw_df = extract_relevant_rows(state, financial_data)
        raw_row_labels = [row for row in fallback_raw_df.index if row in available_set]
    raw_row_labels = raw_row_labels[:max_raw_rows]

    raw_display_df = (
        financial_data.loc[raw_row_labels]
        if raw_row_labels
        else pd.DataFrame(columns=financial_data.columns)
    )

    visualization_plan = VisualizationPlan(
        charts=sanitized_charts,
        raw_row_labels=raw_row_labels,
        reviewer_notes=decision.reviewer_notes,
    )
    return visualization_plan, raw_display_df
