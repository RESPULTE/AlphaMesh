from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.memory.graph.models import (
    UserInvestmentInterestNode,
    UserLearningInterestNode,
)
from core.memory.graph.nodeset_manager import NodeSetManager
from core.memory.graph.utils import generate_uuid5
from core.memory.stores.neo4j_adapter import Neo4jAdapter

_STATUS_RANK: Dict[str, int] = {
    "Bought": 0,
    "Interested": 1,
    "Understood": 2,
    "Confused": 3,
    "Sold": 4,
    "Avoids": 5,
    "Not Interested": 6,
}

CACHE_MAX_INTERESTS = 10  # max entries kept per user in the in-process cache


@dataclass
class InterestCacheEntry:
    kind: Literal["investment", "learning"]
    node: Union[UserInvestmentInterestNode, UserLearningInterestNode]
    target_names: List[str]
    cached_at: datetime


class UserContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    investment_interests: List[UserInvestmentInterestNode] = Field(default_factory=list)
    learning_interests: List[UserLearningInterestNode] = Field(default_factory=list)


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

    def _parse_dt(self, value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except Exception:
                return datetime.fromtimestamp(0, tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.fromtimestamp(0, tz=timezone.utc)

    def _deterministic_id(
        self, user_email: str, status: str, target_ids: List[str], interest_type: str
    ) -> str:
        normalized = self._normalize_email(user_email)
        sorted_ids = sorted({tid for tid in target_ids if tid})
        key = f"{normalized}|{status}|{interest_type}|{','.join(sorted_ids)}"
        return generate_uuid5(key)

    def _rank_and_cap(
        self, entries: List[InterestCacheEntry]
    ) -> List[InterestCacheEntry]:
        """Sort by cached_at DESC, then updated_at DESC, then truncate."""
        entries.sort(
            key=lambda e: (
                -(e.cached_at.timestamp() if e.cached_at else 0),
                -(e.node.updated_at.timestamp() if e.node.updated_at else 0),
            )
        )
        return entries[:CACHE_MAX_INTERESTS]

    def _build_user_context(self, entries: List[InterestCacheEntry]) -> UserContext:
        """Reconstruct a UserContext from a (possibly truncated) entries list."""
        return UserContext(
            investment_interests=[e.node for e in entries if e.kind == "investment"],
            learning_interests=[e.node for e in entries if e.kind == "learning"],
        )

    async def load_for_user(self, user_email: str) -> UserContext:
        normalized = self._normalize_email(user_email)
        if not normalized:
            return UserContext()

        cached = self._cache.get(normalized)
        if cached and "entries" in cached:
            return self._build_user_context(cached["entries"])

        await self._nodeset_manager.get_or_create_user_nodeset(normalized)

        investment_rows, learning_rows = await asyncio.gather(
            self._neo4j_adapter.get_user_investment_interests(normalized),
            self._neo4j_adapter.get_user_learning_interests(normalized),
        )

        now = datetime.now(timezone.utc)
        entries: List[InterestCacheEntry] = []
        target_names_lookup: Dict[str, List[str]] = {}

        for row in investment_rows or []:
            node_raw = row.get("node") if isinstance(row, dict) else None
            props = dict(node_raw) if node_raw is not None else {}
            targets = row.get("targets") if isinstance(row, dict) else []
            target_ids = [t.get("id") for t in targets or [] if t and t.get("id")]
            names = [t.get("name") for t in targets or [] if t and t.get("name")]
            target_names_lookup[props.get("id", "")] = names
            node = UserInvestmentInterestNode(
                id=str(props.get("id") or ""),
                user_email=str(props.get("user_email") or normalized),
                status=str(props.get("status") or ""),
                reason=str(props.get("reason") or ""),
                confidence=str(props.get("confidence") or "low"),
                updated_at=self._parse_dt(props.get("updated_at")),
                target_entity_ids=target_ids,
            )
            entries.append(
                InterestCacheEntry("investment", node, names, cached_at=now)
            )

        for row in learning_rows or []:
            node_raw = row.get("node") if isinstance(row, dict) else None
            props = dict(node_raw) if node_raw is not None else {}
            targets = row.get("targets") if isinstance(row, dict) else []
            target_ids = [t.get("id") for t in targets or [] if t and t.get("id")]
            names = [t.get("name") for t in targets or [] if t and t.get("name")]
            target_names_lookup[props.get("id", "")] = names
            node = UserLearningInterestNode(
                id=str(props.get("id") or ""),
                user_email=str(props.get("user_email") or normalized),
                status=str(props.get("status") or ""),
                reason=str(props.get("reason") or ""),
                updated_at=self._parse_dt(props.get("updated_at")),
                target_entity_ids=target_ids,
            )
            entries.append(InterestCacheEntry("learning", node, names, cached_at=now))

        entries = self._rank_and_cap(entries)
        self._cache[normalized] = {
            "entries": entries,
            "target_names": target_names_lookup,
            "loaded_at": now,
        }
        return self._build_user_context(entries)

    def get_formatted_context(
        self, user_email: Optional[str], limit: int = CACHE_MAX_INTERESTS
    ) -> str:
        if not user_email:
            return "USER CONTEXT: None"
        normalized = self._normalize_email(user_email)
        cached = self._cache.get(normalized)
        if not cached or "entries" not in cached:
            return "USER CONTEXT: None"

        entries: List[InterestCacheEntry] = cached["entries"][:limit]
        if not entries:
            return "USER CONTEXT: None"

        investment_entries = [e for e in entries if e.kind == "investment"]
        learning_entries = [e for e in entries if e.kind == "learning"]

        blocks: List[str] = []

        if investment_entries:
            lines = []
            for idx, entry in enumerate(investment_entries, start=1):
                target_str = (
                    ", ".join(entry.target_names)
                    if entry.target_names
                    else ", ".join(entry.node.target_entity_ids) or "(unknown target)"
                )
                reason = entry.node.reason.strip() if entry.node.reason else ""
                tail = f" - {reason}" if reason else ""
                cached_ts = (
                    entry.cached_at.strftime("%Y-%m-%d %H:%M UTC")
                    if entry.cached_at
                    else "unknown"
                )
                lines.append(
                    f"{idx}. [{entry.node.status}] {target_str}{tail}  (cached {cached_ts})"
                )
            blocks.append("USER INVESTMENT PROFILE:\n" + "\n".join(lines))

        if learning_entries:
            lines = []
            for idx, entry in enumerate(learning_entries, start=1):
                target_str = (
                    ", ".join(entry.target_names)
                    if entry.target_names
                    else ", ".join(entry.node.target_entity_ids) or "(unknown target)"
                )
                reason = entry.node.reason.strip() if entry.node.reason else ""
                tail = f" - {reason}" if reason else ""
                cached_ts = (
                    entry.cached_at.strftime("%Y-%m-%d %H:%M UTC")
                    if entry.cached_at
                    else "unknown"
                )
                lines.append(
                    f"{idx}. [{entry.node.status}] {target_str}{tail}  (cached {cached_ts})"
                )
            blocks.append("USER LEARNING PROFILE:\n" + "\n".join(lines))

        return "\n\n".join(blocks)

    async def schedule_upsert(
        self,
        interest_node: Any,
        user_email: str,
    ) -> None:
        normalized = self._normalize_email(user_email)
        if not normalized:
            return

        try:
            _, nodeset_id = await self._nodeset_manager.get_or_create_user_nodeset(
                normalized
            )

            node_type = interest_node.__class__.__name__

            interest_node.id = self._deterministic_id(
                normalized,
                getattr(interest_node, "status", ""),
                getattr(interest_node, "target_entity_ids", []),
                node_type,
            )
            interest_node.user_email = normalized
            await self._neo4j_adapter.upsert_user_connected_nodes(
                interest_node, nodeset_id
            )

            now = datetime.now(timezone.utc)
            kind: Literal["investment", "learning"] = (
                "investment"
                if isinstance(interest_node, UserInvestmentInterestNode)
                else "learning"
            )
            new_entry = InterestCacheEntry(
                kind=kind,
                node=interest_node,
                target_names=[],
                cached_at=now,
            )
            cached = self._cache.get(normalized)
            if cached and "entries" in cached:
                existing: List[InterestCacheEntry] = cached["entries"]
                existing = [e for e in existing if e.node.id != interest_node.id]
                existing.append(new_entry)
                self._cache[normalized]["entries"] = self._rank_and_cap(existing)
        except Exception:
            self._logger.exception(
                "UserContextService.schedule_upsert failed for user %s", user_email
            )

    def invalidate(self, user_email: str) -> None:
        normalized = self._normalize_email(user_email)
        if normalized in self._cache:
            del self._cache[normalized]

    def schedule_upsert_fire_and_forget(
        self,
        interest_node: Any,
        user_email: str,
    ) -> None:
        async def _run():
            try:
                await self.schedule_upsert(interest_node, user_email)
            except Exception:
                self._logger.exception(
                    "UserContextService.schedule_upsert failed for user %s", user_email
                )

        asyncio.create_task(_run())
