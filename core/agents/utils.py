import asyncio
import json
import re
from asyncio.log import logger
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from core.agents.ticker_validation import TickerInfo


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


def _build_clarification_message(needs_confirmation: Dict[str, "TickerInfo"]) -> str:
    """Format a user-facing message asking for ticker confirmation."""
    lines = ["Before proceeding, I want to confirm the securities you're asking about:"]
    for ticker, info in needs_confirmation.items():
        if not info.is_valid and info.suggestions:
            suggestions_str = ", ".join(f"**{s}**" for s in info.suggestions[:3])
            lines.append(
                f"• **{ticker}** wasn't recognised as a valid ticker symbol. "
                f"Did you mean one of: {suggestions_str}?"
            )
        elif not info.is_valid:
            lines.append(
                f"• **{ticker}** wasn't recognised as a valid ticker symbol. "
                f"Please double-check the symbol and try again."
            )
        else:
            # Valid but non-equity (ETF, MUTUALFUND, etc.)
            qt = info.quote_type or "unknown type"
            lines.append(
                f"• **{ticker}** appears to be a `{qt}` rather than a common equity. "
                f"Is this correct, or did you mean a different symbol?"
            )
    lines.append("\nPlease reply with the correct ticker symbol(s) and I'll proceed.")
    return "\n".join(lines)


def _trim_text(value: Any, *, max_chars: int = MAX_TURN_TEXT_CHARS) -> str:
    return trim_text(value, max_chars=max_chars)


def _normalise_turn_timestamp(turn: dict) -> str:
    return normalise_turn_timestamp(turn)
