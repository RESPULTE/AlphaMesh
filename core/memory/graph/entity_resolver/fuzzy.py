"""Pure fuzzy-matching helpers for entity deduplication.

All functions here are stateless — no class wrapper needed.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

_CORP_SUFFIXES = frozenset(
    {
        "inc",
        "inc.",
        "incorporated",
        "corp",
        "corp.",
        "corporation",
        "co",
        "co.",
        "company",
        "ltd",
        "ltd.",
        "limited",
        "plc",
        "llc",
        "l.l.c.",
        "sa",
        "ag",
        "nv",
        "bv",
        "gmbh",
    }
)

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize_company_alias(name: str) -> str:
    """Strip corporate-suffix tokens and punctuation for alias comparison.

    >>> normalize_company_alias("Apple Inc.")
    'apple'
    >>> normalize_company_alias("Meta Platforms, Inc")
    'meta platforms'
    """
    cleaned = _PUNCT_RE.sub(" ", str(name or "").lower()).strip()
    if not cleaned:
        return ""
    tokens = [t for t in cleaned.split() if t and t not in _CORP_SUFFIXES]
    return " ".join(tokens)


def match_company_alias(
    *,
    name: str,
    entity_type: str,
    candidates: List[dict],
) -> Optional[str]:
    """Return the candidate ID whose alias matches *name*, or ``None``.

    Only applied when *entity_type* is ``"Company"``.
    """
    if entity_type != "Company" or not candidates:
        return None
    normalized = normalize_company_alias(name)
    if not normalized:
        return None
    for candidate in candidates:
        candidate_id = candidate.get("id")
        candidate_name = str(candidate.get("name") or "")
        if candidate_id and normalize_company_alias(candidate_name) == normalized:
            return str(candidate_id)
    return None


def pick_strongest_fuzzy_candidate(
    candidates: List[dict],
) -> Optional[Tuple[str, float]]:
    """Return ``(entity_id, similarity_score)`` for the highest-scoring candidate.

    *similarity_score* is the raw [0, 1] value from Neo4j.  Returns ``None``
    if no valid candidate exists.
    """
    best_id: Optional[str] = None
    best_score: float = -1.0
    for candidate in candidates:
        candidate_id = candidate.get("id")
        similarity = candidate.get("similarity")
        if not candidate_id:
            continue
        if not isinstance(similarity, (int, float)):
            continue
        score = float(similarity)
        if score > best_score:
            best_score = score
            best_id = str(candidate_id)
    if best_id is None:
        return None
    return best_id, best_score
