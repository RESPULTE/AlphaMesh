from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.memory.graph.models import (
    ENTITY_NAMESPACE,
    UserInvestmentInterestNode,
    UserLearningInterestNode,
)
from core.memory.graph.nodeset_manager import NodeSetManager
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
        return str(uuid.uuid5(ENTITY_NAMESPACE, key))

    async def load_for_user(self, user_email: str) -> UserContext:
        normalized = self._normalize_email(user_email)
        if not normalized:
            return UserContext()

        cached = self._cache.get(normalized)
        if cached and "context" in cached:
            return cached["context"]  # type: ignore[return-value]

        await self._nodeset_manager.get_or_create_user_nodeset(normalized)

        investment_task = self._neo4j_adapter.get_user_investment_interests(normalized)
        learning_task = self._neo4j_adapter.get_user_learning_interests(normalized)
        investment_rows, learning_rows = await asyncio.gather(
            investment_task, learning_task
        )

        investment_nodes: List[UserInvestmentInterestNode] = []
        learning_nodes: List[UserLearningInterestNode] = []
        target_names: Dict[str, List[str]] = {}

        for row in investment_rows or []:
            node_raw = row.get("node") if isinstance(row, dict) else None
            props = dict(node_raw) if node_raw is not None else {}
            targets = row.get("targets") if isinstance(row, dict) else []
            target_ids = [t.get("id") for t in targets or [] if t and t.get("id")]
            target_names[props.get("id", "")] = [
                t.get("name") for t in targets or [] if t and t.get("name")
            ]
            investment_nodes.append(
                UserInvestmentInterestNode(
                    id=str(props.get("id") or ""),
                    user_email=str(props.get("user_email") or normalized),
                    status=str(props.get("status") or ""),
                    reason=str(props.get("reason") or ""),
                    confidence=str(props.get("confidence") or "low"),
                    updated_at=self._parse_dt(props.get("updated_at")),
                    target_entity_ids=target_ids,
                )
            )

        for row in learning_rows or []:
            node_raw = row.get("node") if isinstance(row, dict) else None
            props = dict(node_raw) if node_raw is not None else {}
            targets = row.get("targets") if isinstance(row, dict) else []
            target_ids = [t.get("id") for t in targets or [] if t and t.get("id")]
            target_names[props.get("id", "")] = [
                t.get("name") for t in targets or [] if t and t.get("name")
            ]
            learning_nodes.append(
                UserLearningInterestNode(
                    id=str(props.get("id") or ""),
                    user_email=str(props.get("user_email") or normalized),
                    status=str(props.get("status") or ""),
                    reason=str(props.get("reason") or ""),
                    updated_at=self._parse_dt(props.get("updated_at")),
                    target_entity_ids=target_ids,
                )
            )

        context = UserContext(
            investment_interests=investment_nodes,
            learning_interests=learning_nodes,
        )
        self._cache[normalized] = {"context": context, "target_names": target_names}
        return context

    def get_formatted_context(self, user_email: Optional[str], limit: int = 15) -> str:
        if not user_email:
            return "USER CONTEXT: None"
        normalized = self._normalize_email(user_email)
        cached = self._cache.get(normalized)
        if not cached:
            return "USER CONTEXT: None"

        context: UserContext = cached.get("context")  # type: ignore[assignment]
        target_names: Dict[str, List[str]] = cached.get("target_names", {})  # type: ignore[assignment]

        if not context.investment_interests and not context.learning_interests:
            return "USER CONTEXT: None"

        def sort_key(node):
            status_rank = _STATUS_RANK.get(node.status, 99)
            ts = node.updated_at.timestamp() if node.updated_at else 0
            return (status_rank, -ts)

        combined: List[Tuple[str, object]] = []
        for node in context.investment_interests:
            combined.append(("investment", node))
        for node in context.learning_interests:
            combined.append(("learning", node))

        combined.sort(key=lambda item: sort_key(item[1]))
        top = combined[:limit]

        investment_sorted = [node for kind, node in top if kind == "investment"]
        learning_sorted = [node for kind, node in top if kind == "learning"]

        blocks: List[str] = []

        if investment_sorted:
            lines = []
            for idx, node in enumerate(investment_sorted, start=1):
                names = target_names.get(node.id) or node.target_entity_ids
                target_str = ", ".join([n for n in names if n]) or "(unknown target)"
                reason = node.reason.strip() if node.reason else ""
                tail = f" - {reason}" if reason else ""
                lines.append(f"{idx}. [{node.status}] {target_str}{tail}")
            blocks.append("USER INVESTMENT PROFILE:\n" + "\n".join(lines))

        if learning_sorted:
            lines = []
            for idx, node in enumerate(learning_sorted, start=1):
                names = target_names.get(node.id) or node.target_entity_ids
                target_str = ", ".join([n for n in names if n]) or "(unknown target)"
                reason = node.reason.strip() if node.reason else ""
                tail = f" - {reason}" if reason else ""
                lines.append(f"{idx}. [{node.status}] {target_str}{tail}")
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
