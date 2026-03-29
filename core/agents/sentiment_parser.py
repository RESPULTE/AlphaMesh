"""
core/agents/sentiment_parser.py

Parses the structured <sentiment> JSON block emitted by agent LLM responses.

Both NewsAnalysisAgent and FundamentalAnalysisAgent use this — single
implementation, no duplication.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from core.agents.models.base_agent_models import AgentSentiment
from core.logger import get_logger

logger = get_logger(__name__)

_SENTIMENT_RE = re.compile(
    r"<sentiment>\s*(.*?)\s*</sentiment>", re.DOTALL | re.IGNORECASE
)

# Valid label set — used for safe fallback normalisation
_VALID_LABELS = frozenset({"STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"})


def parse_sentiment_block(raw: str) -> Optional[AgentSentiment]:
    """
    Extract and parse the <sentiment>...</sentiment> JSON block from an LLM
    response string.

    Returns None (rather than raising) on any parse failure so callers can
    proceed with a neutral default.
    """
    if not raw:
        return None

    match = _SENTIMENT_RE.search(raw)
    if not match:
        logger.debug("parse_sentiment_block: no <sentiment> block found")
        return None

    json_text = match.group(1).strip()
    if not json_text:
        return None

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("parse_sentiment_block: JSON parse failed: %s", exc)
        return None

    if not isinstance(data, dict):
        logger.warning(
            "parse_sentiment_block: expected JSON object, got %s", type(data)
        )
        return None

    # Validate and clamp score
    try:
        score = int(data.get("score", 50))
        score = max(0, min(100, score))
    except (TypeError, ValueError):
        score = 50

    # Validate label
    label = str(data.get("label", "NEUTRAL")).strip().upper()
    if label not in _VALID_LABELS:
        # Derive from score if label is unrecognised
        if score >= 75:
            label = "STRONG BUY"
        elif score >= 60:
            label = "BUY"
        elif score >= 40:
            label = "NEUTRAL"
        elif score >= 25:
            label = "SELL"
        else:
            label = "STRONG SELL"

    rationale = str(data.get("rationale", "")).strip()

    return AgentSentiment(score=score, label=label, rationale=rationale)


def strip_sentiment_block(text: str) -> str:
    """
    Remove the <sentiment>...</sentiment> block from analysis text before
    the text is stored or sent to the orchestrator synthesiser.

    The sentiment block is consumed by the API layer; it must not appear in
    the narrative shown to the user.
    """
    return _SENTIMENT_RE.sub("", text).strip()
