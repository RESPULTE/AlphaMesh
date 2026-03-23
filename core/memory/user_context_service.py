from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.memory.graph.nodeset_manager import NodeSetManager
from core.memory.stores.neo4j_adapter import Neo4jAdapter

CACHE_MAX_INTERESTS = 20  # raised from 10 to accommodate domain-grouped entries
_MAX_WEIGHT = 10.0  # normalization cap for score computation


@dataclass
class InterestCacheEntry:
    """One user interest signal, ready for prompt injection and cache ranking."""

    kind: Literal["investment", "learning"]
    category: str  # domain category (sector name or concept category)
    entity_name: str
    entity_type: str
    weight: float  # cumulative confidence weight from all reinforcing turns
    status: Literal["Active", "Invalidated", "Paused"]
    invalidated: bool
    cached_at: datetime
    reason: Optional[str] = None  # excerpt from user message that created this


class UserContext(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    investment_entries: List[Any] = Field(default_factory=list)
    learning_entries: List[Any] = Field(default_factory=list)


class UserContextService:
    def __init__(
        self,
        neo4j_adapter: Neo4jAdapter,
        nodeset_manager: NodeSetManager,
    ) -> None:
        self._neo4j_adapter = neo4j_adapter
        self._nodeset_manager = nodeset_manager
        self._logger = get_logger(__name__)
        self._cache: Dict[str, Dict[str, object]] = {}

    def _normalize_email(self, user_email: str) -> str:
        return str(user_email or "").strip().lower()

    def _build_user_context(self, entries: List[InterestCacheEntry]) -> UserContext:
        return UserContext(
            investment_entries=[e for e in entries if e.kind == "investment"],
            learning_entries=[e for e in entries if e.kind == "learning"],
        )

    def _rank_and_cap(
        self, entries: List[InterestCacheEntry]
    ) -> List[InterestCacheEntry]:
        """
        Composite score: 60% normalized weight + 40% recency (30-day linear decay).
        Invalidated entries are penalized (×0.3) but retained for context.
        """
        now_ts = datetime.now(timezone.utc).timestamp()

        def _score(e: InterestCacheEntry) -> float:
            age_days = max(0.0, (now_ts - e.cached_at.timestamp()) / 86400.0)
            recency_score = max(0.0, 1.0 - age_days / 30.0)
            normalized_weight = min(1.0, e.weight / _MAX_WEIGHT)
            base = 0.6 * normalized_weight + 0.4 * recency_score
            return base * 0.3 if e.invalidated else base

        entries.sort(key=_score, reverse=True)
        return entries[:CACHE_MAX_INTERESTS]

    async def load_for_user(self, user_email: str) -> UserContext:
        """
        Load interest data from Neo4j into the in-memory cache.
        Traverses NodeSet → UserInterestDomain → UserInterestEdge → Entity.
        Safe to call multiple times — returns cached result if already warm.
        """
        normalized = self._normalize_email(user_email)
        if not normalized:
            return UserContext()

        cached = self._cache.get(normalized)
        if cached and "entries" in cached:
            return self._build_user_context(cached["entries"])  # type: ignore[arg-type]

        _, nodeset_id = await self._nodeset_manager.get_or_create_user_nodeset(
            normalized
        )

        rows = await self._neo4j_adapter.get_user_interest_data(normalized, nodeset_id)

        now = datetime.now(timezone.utc)
        entries: List[InterestCacheEntry] = []

        for row in rows or []:
            d = dict(row.get("d") or {})
            e = dict(row.get("e") or {})
            entity = dict(row.get("entity") or {})

            if not d or not e or not entity:
                continue

            entries.append(
                InterestCacheEntry(
                    kind=d.get("domain_type", "investment"),
                    category=d.get("category", "general"),
                    entity_name=entity.get("name", ""),
                    entity_type=entity.get("entity_type", ""),
                    weight=float(e.get("weight", 0.0)),
                    status=e.get("status", "Active"),
                    invalidated=bool(e.get("invalidated", False)),
                    cached_at=now,
                    reason=None,
                )
            )

        ranked = self._rank_and_cap(entries)
        self._cache[normalized] = {"entries": ranked, "loaded_at": now}
        return self._build_user_context(ranked)

    def update_cache(
        self,
        new_entries: List[InterestCacheEntry],
        user_email: str,
    ) -> None:
        """
        Merge new interest entries into the in-memory cache without any graph write.

        Called immediately after schedule() dispatches the background graph task
        so that get_formatted_context() in subsequent turns of the same session
        reflects new signals without waiting for Neo4j.

        Merge semantics:
        - Reinforce: accumulate weight on existing entry.
        - Invalidate: replace entry with invalidated=True version.
        - New entity: insert fresh entry.
        """
        normalized = self._normalize_email(user_email)
        if not normalized:
            return

        cached = self._cache.get(normalized)
        existing: List[InterestCacheEntry] = (
            list(cached["entries"])  # type: ignore[arg-type]
            if cached and "entries" in cached
            else []
        )

        # Key: (kind, category, entity_name) — uniquely identifies one interest edge
        existing_map: Dict[tuple, InterestCacheEntry] = {
            (e.kind, e.category, e.entity_name): e for e in existing
        }

        for new_entry in new_entries:
            key = (new_entry.kind, new_entry.category, new_entry.entity_name)
            existing_entry = existing_map.get(key)
            if existing_entry:
                if new_entry.invalidated:
                    # Invalidation always wins — replace entirely
                    existing_map[key] = new_entry
                else:
                    # Reinforce: accumulate weight, refresh timestamp
                    existing_entry.weight += new_entry.weight
                    existing_entry.status = new_entry.status
                    existing_entry.cached_at = new_entry.cached_at
                    if new_entry.reason:
                        existing_entry.reason = new_entry.reason
            else:
                existing_map[key] = new_entry

        merged = self._rank_and_cap(list(existing_map.values()))
        self._cache[normalized] = {
            "entries": merged,
            "loaded_at": datetime.now(timezone.utc),
        }

    def get_formatted_context(
        self, user_email: Optional[str], limit: int = CACHE_MAX_INTERESTS
    ) -> str:
        """
        Format cached interest data into a structured block for LLM prompt injection.

        Groups active entries by domain category. Invalidated entries are listed
        separately with a directive to only surface them when highly relevant
        and to ask the user before doing so.
        """
        if not user_email:
            return "USER CONTEXT: None"
        normalized = self._normalize_email(user_email)
        cached = self._cache.get(normalized)
        if not cached or "entries" not in cached:
            return "USER CONTEXT: None"

        entries: List[InterestCacheEntry] = list(cached["entries"])[:limit]  # type: ignore[arg-type]
        if not entries:
            return "USER CONTEXT: None"

        def _format_section(
            kind_entries: List[InterestCacheEntry],
            section_title: str,
        ) -> str:
            if not kind_entries:
                return ""

            active = [e for e in kind_entries if not e.invalidated]
            invalidated = [e for e in kind_entries if e.invalidated]
            lines = [section_title]

            # Group active by category
            from collections import defaultdict

            by_category: Dict[str, List[InterestCacheEntry]] = defaultdict(list)
            for e in active:
                by_category[e.category].append(e)

            for category, cat_entries in sorted(by_category.items()):
                total_weight = sum(e.weight for e in cat_entries)
                lines.append(
                    f"\n{category.replace('_', ' ').title()} "
                    f"(total weight: {total_weight:.1f}):"
                )
                for e in sorted(cat_entries, key=lambda x: x.weight, reverse=True):
                    reason_tail = f" — {e.reason[:80]}" if e.reason else ""
                    lines.append(
                        f"  • {e.entity_name} [{e.status}, weight={e.weight:.1f}]"
                        f"{reason_tail}"
                    )

            if invalidated:
                lines.append(
                    "\n[Previously invalidated — surface ONLY if highly relevant. "
                    "If you reference these, first ask the user whether they would "
                    "like to continue with this topic:]"
                )
                for e in sorted(invalidated, key=lambda x: x.weight, reverse=True):
                    lines.append(
                        f"  • {e.entity_name} "
                        f"[weight={e.weight:.1f} before invalidation, {e.category}]"
                    )

            return "\n".join(lines)

        investment_entries = [e for e in entries if e.kind == "investment"]
        learning_entries = [e for e in entries if e.kind == "learning"]

        blocks = []
        inv = _format_section(investment_entries, "USER INVESTMENT PROFILE:")
        if inv:
            blocks.append(inv)
        learn = _format_section(learning_entries, "USER LEARNING PROFILE:")
        if learn:
            blocks.append(learn)

        return "\n\n".join(blocks) if blocks else "USER CONTEXT: None"

    def invalidate(self, user_email: str) -> None:
        """Evict a user's cache entry (e.g. on explicit session reset)."""
        normalized = self._normalize_email(user_email)
        if normalized in self._cache:
            del self._cache[normalized]
