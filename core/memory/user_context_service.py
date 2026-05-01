from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.memory.graph.nodeset_manager import NodeSetManager
from core.memory.stores.neo4j_adapter import Neo4jAdapter

CACHE_MAX_INTERESTS = 40
_MAX_WEIGHT = 12.0
_RECENCY_WINDOW_DAYS = 60.0
_CONTEXT_TOKEN_BUDGET = 320
_ACTIVE_SLOT_CAP = 6
_CONFLICT_SLOT_CAP = 4
_NUDGE_SLOT_CAP = 2


@dataclass
class InterestCacheEntry:
    """One user-interest aggregate for prompt injection and cache ranking."""

    kind: Literal["investment", "learning"]
    category: str
    entity_id: str
    entity_name: str
    entity_type: str
    cumulative_weight: float
    reinforcement_count: int
    invalidation_count: int
    current_stance: Literal["positive", "negative"]
    previous_stance: Optional[Literal["positive", "negative"]]
    last_changed_at: datetime
    cached_at: datetime
    reason: Optional[str] = None


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

    @staticmethod
    def _normalize_email(user_email: str) -> str:
        return str(user_email or "").strip().lower()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _parse_datetime(value: Any, default: datetime) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.strip())
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return default
        return default

    def _build_user_context(self, entries: List[InterestCacheEntry]) -> UserContext:
        return UserContext(
            investment_entries=[e for e in entries if e.kind == "investment"],
            learning_entries=[e for e in entries if e.kind == "learning"],
        )

    def _entry_score(self, entry: InterestCacheEntry) -> float:
        now_ts = datetime.now(timezone.utc).timestamp()
        age_days = max(0.0, (now_ts - entry.last_changed_at.timestamp()) / 86400.0)
        recency = max(0.0, 1.0 - (age_days / _RECENCY_WINDOW_DAYS))
        weight = min(1.0, max(0.0, entry.cumulative_weight) / _MAX_WEIGHT)
        reinforcement = min(1.0, max(0, entry.reinforcement_count) / 6.0)
        conflict = 0.15 if entry.previous_stance and entry.previous_stance != entry.current_stance else 0.0
        score = 0.5 * weight + 0.3 * recency + 0.2 * reinforcement + conflict
        if entry.current_stance == "negative":
            score *= 0.9
        return score

    def _rank_and_cap(self, entries: List[InterestCacheEntry]) -> List[InterestCacheEntry]:
        entries.sort(key=self._entry_score, reverse=True)
        return entries[:CACHE_MAX_INTERESTS]

    async def load_for_user(self, user_email: str) -> UserContext:
        """Load user-interest aggregates from Neo4j into the in-memory cache."""
        normalized = self._normalize_email(user_email)
        if not normalized:
            return UserContext()

        cached = self._cache.get(normalized)
        if cached and "entries" in cached:
            return self._build_user_context(cached["entries"])  # type: ignore[arg-type]

        _, nodeset_id = await self._nodeset_manager.get_or_create_user_nodeset(normalized)
        rows = await self._neo4j_adapter.get_user_interest_data(normalized, nodeset_id)

        now = datetime.now(timezone.utc)
        entries: List[InterestCacheEntry] = []
        for row in rows or []:
            d = dict(row.get("d") or {})
            e = dict(row.get("e") or {})
            entity = dict(row.get("entity") or {})
            latest_event = dict(row.get("latest_event") or {})
            previous_event = dict(row.get("previous_event") or {})
            if not d or not e or not entity:
                continue

            current_stance = str(e.get("current_stance") or "").strip().lower()
            if current_stance not in {"positive", "negative"}:
                continue

            previous_stance = str(previous_event.get("stance") or "").strip().lower()
            if previous_stance not in {"positive", "negative"}:
                previous_stance = None
            elif previous_stance == current_stance:
                previous_stance = None

            last_changed_at = self._parse_datetime(
                e.get("last_changed_at") or latest_event.get("observed_at") or e.get("last_updated_at"),
                now,
            )
            cumulative_weight = float(e.get("cumulative_weight") or 0.0)

            entries.append(
                InterestCacheEntry(
                    kind=d.get("domain_type", "investment"),
                    category=d.get("category", "general"),
                    entity_id=str(e.get("entity_id") or entity.get("id") or ""),
                    entity_name=str(entity.get("name") or ""),
                    entity_type=str(entity.get("entity_type") or ""),
                    cumulative_weight=cumulative_weight,
                    reinforcement_count=int(e.get("reinforcement_count") or 0),
                    invalidation_count=int(e.get("invalidation_count") or 0),
                    current_stance=current_stance,  # type: ignore[arg-type]
                    previous_stance=previous_stance,  # type: ignore[arg-type]
                    last_changed_at=last_changed_at,
                    cached_at=last_changed_at,
                    reason=(str(latest_event.get("source_excerpt") or "").strip() or None),
                )
            )

        ranked = self._rank_and_cap(entries)
        self._cache[normalized] = {"entries": ranked, "loaded_at": now}
        return self._build_user_context(ranked)

    def update_cache(self, new_entries: List[InterestCacheEntry], user_email: str) -> None:
        """Merge new user-interest aggregate deltas into in-memory cache."""
        normalized = self._normalize_email(user_email)
        if not normalized:
            return

        cached = self._cache.get(normalized)
        existing: List[InterestCacheEntry] = (
            list(cached["entries"])  # type: ignore[arg-type]
            if cached and "entries" in cached
            else []
        )
        existing_map: Dict[tuple, InterestCacheEntry] = {
            (e.kind, e.category, e.entity_id or e.entity_name.lower()): e for e in existing
        }

        for new_entry in new_entries:
            key = (new_entry.kind, new_entry.category, new_entry.entity_id or new_entry.entity_name.lower())
            old = existing_map.get(key)
            if not old:
                existing_map[key] = new_entry
                continue

            old.cumulative_weight += max(0.0, new_entry.cumulative_weight)
            old.reinforcement_count += max(0, new_entry.reinforcement_count)
            old.invalidation_count += max(0, new_entry.invalidation_count)
            if old.current_stance != new_entry.current_stance:
                old.previous_stance = old.current_stance  # keep immediate conflict context
                old.current_stance = new_entry.current_stance
            if new_entry.previous_stance and new_entry.previous_stance != old.current_stance:
                old.previous_stance = new_entry.previous_stance
            if new_entry.last_changed_at > old.last_changed_at:
                old.last_changed_at = new_entry.last_changed_at
            if new_entry.cached_at > old.cached_at:
                old.cached_at = new_entry.cached_at
            if new_entry.reason:
                old.reason = new_entry.reason

        merged = self._rank_and_cap(list(existing_map.values()))
        self._cache[normalized] = {"entries": merged, "loaded_at": datetime.now(timezone.utc)}

    def _build_conflict_nudges(self, entries: List[InterestCacheEntry]) -> List[dict]:
        conflicts = [
            e
            for e in entries
            if e.previous_stance and e.previous_stance != e.current_stance
        ]
        conflicts.sort(key=self._entry_score, reverse=True)
        nudges: List[dict] = []
        for entry in conflicts[:_NUDGE_SLOT_CAP]:
            old = entry.previous_stance or "unknown"
            new = entry.current_stance
            if old == "positive" and new == "negative":
                prompt = (
                    f"You previously liked {entry.entity_name} but moved away from it. "
                    "Would you like to revisit your current view?"
                )
            elif old == "negative" and new == "positive":
                prompt = (
                    f"You previously avoided {entry.entity_name} but recently showed renewed interest. "
                    "Should we update your strategy assumptions?"
                )
            else:
                prompt = f"Your preference for {entry.entity_name} changed recently. Should we reconfirm it?"
            nudges.append(
                {
                    "entity": entry.entity_name,
                    "domain_type": entry.kind,
                    "old_stance": old,
                    "new_stance": new,
                    "last_change_at": entry.last_changed_at.isoformat(),
                    "nudge_score": round(self._entry_score(entry), 4),
                    "suggested_prompt": prompt,
                }
            )
        return nudges

    def get_formatted_context(self, user_email: Optional[str], limit: int = CACHE_MAX_INTERESTS) -> str:
        """Format cached user-interest context + structured nudge candidates."""
        if not user_email:
            return "USER CONTEXT: None"
        normalized = self._normalize_email(user_email)
        cached = self._cache.get(normalized)
        if not cached or "entries" not in cached:
            return "USER CONTEXT: None"

        entries: List[InterestCacheEntry] = list(cached["entries"])[:limit]  # type: ignore[arg-type]
        if not entries:
            return "USER CONTEXT: None"

        for_section = sorted(entries, key=self._entry_score, reverse=True)
        active_lines: List[str] = []
        conflict_lines: List[str] = []

        active_count = 0
        conflict_count = 0
        for entry in for_section:
            if (
                entry.previous_stance
                and entry.previous_stance != entry.current_stance
                and conflict_count < _CONFLICT_SLOT_CAP
            ):
                conflict_lines.append(
                    f"- {entry.kind}:{entry.category}: user used to be {entry.previous_stance} on "
                    f"{entry.entity_name}, now {entry.current_stance} (changed {entry.last_changed_at.isoformat()})."
                )
                conflict_count += 1
                continue
            if active_count >= _ACTIVE_SLOT_CAP:
                continue
            active_lines.append(
                f"- {entry.kind}:{entry.category}: {entry.entity_name} "
                f"[stance={entry.current_stance}, weight={entry.cumulative_weight:.2f}, "
                f"reinforced={entry.reinforcement_count}, invalidated={entry.invalidation_count}]"
            )
            active_count += 1

        nudge_candidates = self._build_conflict_nudges(for_section)
        blocks: List[str] = []
        if active_lines:
            blocks.append("USER INTEREST PROFILE:\n" + "\n".join(active_lines))
        if conflict_lines:
            blocks.append("INTEREST CONFLICT TIMELINE:\n" + "\n".join(conflict_lines))
        if nudge_candidates:
            blocks.append(
                "NUDGE_CANDIDATES_JSON:\n"
                + json.dumps(nudge_candidates, ensure_ascii=True, separators=(",", ":"))
            )

        rendered = "\n\n".join(blocks) if blocks else "USER CONTEXT: None"
        if rendered == "USER CONTEXT: None":
            return rendered

        lines = rendered.splitlines()
        while lines and self._estimate_tokens("\n".join(lines)) > _CONTEXT_TOKEN_BUDGET:
            lines.pop()
        return "\n".join(lines) if lines else "USER CONTEXT: None"

    def invalidate(self, user_email: str) -> None:
        """Evict a user's cache entry (e.g. on explicit session reset)."""
        normalized = self._normalize_email(user_email)
        if normalized in self._cache:
            del self._cache[normalized]
