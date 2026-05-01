from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.memory.graph.nodeset_manager import NodeSetManager
from core.memory.stores.neo4j_adapter import Neo4jAdapter
from core.memory.user_interest_models import (
    UserInterestQueryResult,
    UserInterestQuerySpec,
)

CACHE_MAX_INTERESTS = 40
_MAX_WEIGHT = 12.0
_RECENCY_WINDOW_DAYS = 60.0
_CONTEXT_TOKEN_BUDGET = 320
_ACTIVE_SLOT_CAP = 6
_CONFLICT_SLOT_CAP = 4
_NUDGE_SLOT_CAP = 2
_TARGETED_INTEREST_CONTEXT_MAX_CHARS = 2200
_TARGETED_INTEREST_DOMAIN_FALLBACK_LIMIT = 3
_TARGETED_INTEREST_DOMAIN_LIMIT = 3
_TARGETED_INTEREST_EDGE_LIMIT = 8
_TARGETED_INTEREST_EXPANDED_ENTITY_LIMIT = 12


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
    def _truncate_targeted_context(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return "(none)"
        if len(value) <= _TARGETED_INTEREST_CONTEXT_MAX_CHARS:
            return value
        return value[: _TARGETED_INTEREST_CONTEXT_MAX_CHARS - 3].rstrip() + "..."

    @staticmethod
    def _is_low_confidence_query_spec(spec: UserInterestQuerySpec) -> bool:
        return not (
            spec.domain_type
            or (spec.category or "").strip()
            or list(spec.target_entities or [])
        )

    def _render_interest_domain_summary_block(self, rows: List[dict]) -> str:
        if not rows:
            return "(none)"
        lines: List[str] = ["Domain Summary:"]
        for idx, row in enumerate(
            rows[:_TARGETED_INTEREST_DOMAIN_FALLBACK_LIMIT], start=1
        ):
            domain = dict(row.get("domain") or {})
            category = str(domain.get("category") or "general")
            domain_type = str(domain.get("domain_type") or "investment")
            positive_weight = float(row.get("positive_weight") or 0.0)
            edge_count = int(row.get("edge_count") or 0)
            last_changed = str(row.get("last_changed_at") or "unknown_time")
            lines.append(
                f"{idx}. {domain_type}:{category} | positive_weight={positive_weight:.2f} "
                f"| edges={edge_count} | last_changed={last_changed}"
            )
        return self._truncate_targeted_context("\n".join(lines))

    def _render_targeted_interest_block(self, *, rows: List[dict], hops: int) -> str:
        if not rows:
            return "(none)"

        domain = dict(rows[0].get("domain") or {})
        domain_type = str(domain.get("domain_type") or "investment")
        category = str(domain.get("category") or "general")
        lines: List[str] = [f"Matched Domain:\n- {domain_type}:{category}"]
        lines.append("Top Interest Edges:")

        for idx, row in enumerate(rows[:_TARGETED_INTEREST_EDGE_LIMIT], start=1):
            edge = dict(row.get("edge") or {})
            entity = dict(row.get("entity") or {})
            entity_name = str(entity.get("name") or "unknown_entity")
            entity_type = str(entity.get("entity_type") or "UnknownType")
            stance = str(row.get("stance") or edge.get("current_stance") or "positive")
            weight = float(edge.get("cumulative_weight") or 0.0)
            changed = str(
                row.get("edge_last_changed")
                or edge.get("last_changed_at")
                or "unknown_time"
            )
            lines.append(
                f"{idx}. {entity_name} ({entity_type}) | stance={stance} "
                f"| weight={weight:.2f} | last_changed={changed}"
            )

        if hops > 0:
            expanded_set: set[Tuple[str, str]] = set()
            for row in rows:
                for neighbor in row.get("expanded_neighbors") or []:
                    if not isinstance(neighbor, dict):
                        continue
                    neighbor_name = str(neighbor.get("name") or "").strip()
                    neighbor_type = str(neighbor.get("entity_type") or "").strip()
                    if not neighbor_name or not neighbor_type:
                        continue
                    expanded_set.add((neighbor_name, neighbor_type))

            if expanded_set:
                lines.append(f"Expanded Context (hops={hops}):")
                for idx, (name, entity_type) in enumerate(
                    sorted(expanded_set)[:_TARGETED_INTEREST_EXPANDED_ENTITY_LIMIT],
                    start=1,
                ):
                    lines.append(f"{idx}. {name} ({entity_type})")

        return self._truncate_targeted_context("\n".join(lines))

    async def _build_targeted_query_spec(
        self,
        *,
        latest_user_message: str,
        baseline_user_context_block: str,
        portfolio_block: str,
        llm: Any,
    ) -> UserInterestQuerySpec:
        structured_llm = llm.with_structured_output(UserInterestQuerySpec)
        messages: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "You select a compact user-interest graph query for the orchestrator. "
                    "Infer domain_type ('investment'|'learning'), optional category, optional target_entities, "
                    "and hops (0..2). Set broad_fallback=true when user intent is broad/uncertain. "
                    "Set risk_or_avoidance_intent=true only when the user expresses avoidance, downside, or risk concern."
                )
            ),
            SystemMessage(
                content=f"USER CONTEXT:\n{baseline_user_context_block or 'USER CONTEXT: None'}"
            ),
            SystemMessage(content=f"PORTFOLIO:\n{portfolio_block or '[]'}"),
            HumanMessage(content=latest_user_message),
        ]
        spec: UserInterestQuerySpec = await structured_llm.ainvoke(messages)
        return spec.model_copy(update={"hops": max(0, min(2, int(spec.hops or 0)))})

    async def build_targeted_orchestrator_context(
        self,
        *,
        user_email: Optional[str],
        latest_user_message: str,
        baseline_user_context_block: str,
        portfolio_block: str,
        llm: Any,
    ) -> UserInterestQueryResult:
        if not user_email or not str(latest_user_message or "").strip():
            return UserInterestQueryResult(context_block="(none)")

        normalized = self._normalize_email(user_email)
        if not normalized:
            return UserInterestQueryResult(context_block="(none)")

        try:
            spec = await self._build_targeted_query_spec(
                latest_user_message=latest_user_message.strip(),
                baseline_user_context_block=baseline_user_context_block,
                portfolio_block=portfolio_block,
                llm=llm,
            )
            _, nodeset_id = await self._nodeset_manager.get_or_create_user_nodeset(
                normalized
            )
            fallback_mode = bool(spec.broad_fallback) or self._is_low_confidence_query_spec(
                spec
            )

            if fallback_mode:
                rows = await self._neo4j_adapter.get_user_interest_domain_summary(
                    user_email=normalized,
                    nodeset_id=nodeset_id,
                    limit=_TARGETED_INTEREST_DOMAIN_FALLBACK_LIMIT,
                )
                return UserInterestQueryResult(
                    query_spec=spec,
                    context_block=self._render_interest_domain_summary_block(rows),
                    debug_payload={
                        "mode": "fallback_domains_only",
                        "domain_rows": len(rows),
                        "hops": 0,
                    },
                )

            rows = await self._neo4j_adapter.query_user_interest_context(
                user_email=normalized,
                nodeset_id=nodeset_id,
                domain_type=spec.domain_type,
                category=spec.category,
                target_entities=[entity.model_dump() for entity in spec.target_entities],
                hops=spec.hops,
                risk_or_avoidance_intent=spec.risk_or_avoidance_intent,
                domain_limit=_TARGETED_INTEREST_DOMAIN_LIMIT,
                edge_limit=_TARGETED_INTEREST_EDGE_LIMIT,
                expanded_entity_limit=_TARGETED_INTEREST_EXPANDED_ENTITY_LIMIT,
            )
            return UserInterestQueryResult(
                query_spec=spec,
                context_block=self._render_targeted_interest_block(
                    rows=rows,
                    hops=spec.hops,
                ),
                debug_payload={
                    "mode": "targeted",
                    "row_count": len(rows),
                    "hops": spec.hops,
                    "domain_type": spec.domain_type,
                    "category": spec.category,
                },
            )
        except Exception:
            self._logger.exception(
                "build_targeted_orchestrator_context: failed for user '%s'",
                normalized,
            )
            return UserInterestQueryResult(
                context_block="(none)",
                debug_payload={"mode": "error"},
            )

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
