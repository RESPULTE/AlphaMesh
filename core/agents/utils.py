import asyncio
import json
from asyncio.log import logger
from typing import Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from core.agents.ticker_validation import TickerInfo


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
