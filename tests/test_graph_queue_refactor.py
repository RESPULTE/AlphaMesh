from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from core.config import settings
from core.memory.graph.entity_resolver import EntityResolution, ResolvedEdgeBatch
from core.memory.graph.graph_queue import GraphQueueManager, make_graph_task
from core.memory.graph.queue.pipeline import GraphWritePipeline
from core.memory.graph.queue.prompt_registry import PromptRegistry
from core.memory.graph.sql_store import GraphTaskSqlStore
from core.memory.graph.utils import entity_key, normalize_entity_name, normalize_entity_type


class FakeResolver:
    def __init__(self) -> None:
        self.batch_calls: List[Tuple[List[Tuple[str, str, Optional[dict]]], bool]] = []
        self.edge_calls: List[Tuple[List[dict], bool]] = []

    async def resolve_entities(
        self,
        entities: List[Tuple[str, str, Optional[dict]]],
        allow_create: bool = True,
    ) -> Dict[Tuple[str, str], EntityResolution]:
        self.batch_calls.append((entities, allow_create))
        resolved: Dict[Tuple[str, str], EntityResolution] = {}
        for name, entity_type, _props in entities:
            resolved[(name, entity_type)] = EntityResolution(
                entity_id=f"{entity_type}:{name.lower()}",
                match_stage="fake",
            )
        return resolved

    async def resolve_relationship_edges(
        self,
        relationships: List[dict],
        *,
        allow_create: bool,
    ) -> ResolvedEdgeBatch:
        self.edge_calls.append((relationships, allow_create))
        endpoint_inputs: List[Tuple[str, str, Optional[dict]]] = []
        for rel in relationships:
            endpoint_inputs.append(
                (
                    str(rel.get("from_name") or ""),
                    str(rel.get("from_type") or ""),
                    rel.get("from_node_props"),
                )
            )
            endpoint_inputs.append(
                (
                    str(rel.get("to_name") or ""),
                    str(rel.get("to_type") or ""),
                    rel.get("to_node_props"),
                )
            )
        endpoint_results = await self.resolve_entities(
            endpoint_inputs,
            allow_create=allow_create,
        )
        entity_cache: Dict[Tuple[str, str], str] = {}
        for key, resolution in endpoint_results.items():
            if resolution.entity_id:
                entity_cache[key] = resolution.entity_id
        return ResolvedEdgeBatch(
            relationships=list(relationships),
            entity_cache=entity_cache,
            skipped_relationships=0,
        )


class FakeWriter:
    def __init__(self) -> None:
        self.write_calls: List[Tuple[List[dict], Dict[Tuple[str, str], str]]] = []
        self.user_domains: List[str] = []
        self.user_edges: List[str] = []
        self.turn_nodes: List[str] = []

    async def write_relationships(
        self,
        relationships: List[dict],
        _conversation_id: str,
        _source_agent: str,
        entity_cache: Dict[Tuple[str, str], str],
    ) -> int:
        self.write_calls.append((relationships, dict(entity_cache)))
        written = 0
        for rel in relationships:
            from_name = normalize_entity_name(str(rel.get("from_name") or ""))
            from_type = str(rel.get("from_type") or "").strip()
            to_name = normalize_entity_name(str(rel.get("to_name") or ""))
            to_type = str(rel.get("to_type") or "").strip()

            if from_type not in {"UserInterestDomain", "UserInterestEdge", "TurnNode"}:
                from_type = normalize_entity_type(from_type) or ""
            if to_type not in {"UserInterestDomain", "UserInterestEdge", "TurnNode"}:
                to_type = normalize_entity_type(to_type) or ""

            if (
                from_name
                and to_name
                and from_type
                and to_type
                and entity_key(from_name, from_type) in entity_cache
                and entity_key(to_name, to_type) in entity_cache
            ):
                written += 1
        return written

    async def merge_user_interest_domain(self, domain_id: str, _props: dict) -> None:
        self.user_domains.append(domain_id)

    async def merge_user_interest_edge(
        self,
        edge_id: str,
        props: dict,
        operation: str,
        weight_delta: float,
    ) -> None:
        _ = (props, operation, weight_delta)
        self.user_edges.append(edge_id)

    async def merge_turn_node(self, turn_id: str, _props: dict) -> None:
        self.turn_nodes.append(turn_id)


class FakeRelationshipExtractor:
    async def extract(
        self,
        text: str,
        llm,
        system_prompt: str,
        max_attempts: int = 1,
    ) -> List[dict]:
        _ = (text, llm, system_prompt, max_attempts)
        return []


async def fake_entity_extractor(chunk_ids: List[str]) -> None:
    _ = chunk_ids


def fake_llm_provider(_config: Optional[dict]):
    return object()


def test_enqueue_immediate_uses_source_based_allow_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings, "GRAPH_ALLOW_CREATE_SOURCES", "taxonomy_bootstrap")
    resolver = FakeResolver()
    writer = FakeWriter()
    manager = GraphQueueManager(
        entity_resolver=resolver,
        graph_writer=writer,
        relationship_extractor=FakeRelationshipExtractor(),
        entity_extractor=fake_entity_extractor,
        llm_provider=fake_llm_provider,
        db_path=str(tmp_path / "graph_tasks.db"),
    )
    async def _run() -> None:
        await manager.start()
        try:
            task = make_graph_task(
                turn_id="turn-1",
                conversation_id="taxonomy_bootstrap",
                source_agent="taxonomy_bootstrap",
                relationships=[
                    {
                        "from_name": "Apple",
                        "from_type": "Company",
                        "relation": "BELONGS_TO",
                        "to_name": "Technology",
                        "to_type": "Sector",
                    }
                ],
                immediate=True,
            )
            task_id = await manager.enqueue(task)
            assert task_id == task.task_id
            assert resolver.batch_calls
            assert resolver.batch_calls[0][1] is True

            rows = await manager._store.fetchall(
                "SELECT allow_create FROM graph_tasks WHERE task_id = ?",
                (task.task_id,),
            )
            assert rows
            assert rows[0]["allow_create"] == 1
        finally:
            await manager.shutdown()

    asyncio.run(_run())


def test_pipeline_upserts_user_scoped_nodes_before_edge_write(
    tmp_path: Path,
):
    resolver = FakeResolver()
    writer = FakeWriter()
    store = GraphTaskSqlStore(str(tmp_path / "graph_tasks.db"))
    async def _run() -> None:
        await store.initialize()
        prompt_registry = PromptRegistry(store)
        pipeline = GraphWritePipeline(
            entity_resolver=resolver,
            graph_writer=writer,
            relationship_extractor=FakeRelationshipExtractor(),
            entity_extractor=fake_entity_extractor,
            llm_provider=fake_llm_provider,
            prompt_registry=prompt_registry,
        )

        relationships = [
            {
                "from_name": "edge-1",
                "from_type": "UserInterestEdge",
                "relation": "HAS_INTEREST_IN",
                "to_name": "domain-1",
                "to_type": "UserInterestDomain",
                "from_node_props": {"id": "edge-1", "operation": "reinforce"},
                "to_node_props": {"id": "domain-1"},
            },
            {
                "from_name": "edge-1",
                "from_type": "UserInterestEdge",
                "relation": "TARGETS",
                "to_name": "Apple",
                "to_type": "Company",
                "from_node_props": {"id": "edge-1", "operation": "reinforce"},
                "to_node_props": {},
            },
            {
                "from_name": "edge-1",
                "from_type": "UserInterestEdge",
                "relation": "SOURCED_FROM",
                "to_name": "turn-42",
                "to_type": "TurnNode",
                "from_node_props": {"id": "edge-1", "operation": "reinforce"},
                "to_node_props": {"id": "turn-42"},
            },
        ]

        domain_written, user_written = await pipeline.process_relationships(
            relationships=relationships,
            conversation_id="conversation-1",
            source_agent="orchestrator",
            allow_create=False,
        )
        assert domain_written == 0
        assert user_written == 3
        assert writer.user_domains == ["domain-1"]
        assert writer.user_edges == ["edge-1"]
        assert writer.turn_nodes == ["turn-42"]
        assert resolver.batch_calls
        assert resolver.batch_calls[0][1] is False

    asyncio.run(_run())


def test_ttl_eviction_closes_idle_live_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings, "GRAPH_ALLOW_CREATE_SOURCES", "")
    manager = GraphQueueManager(
        entity_resolver=FakeResolver(),
        graph_writer=FakeWriter(),
        relationship_extractor=FakeRelationshipExtractor(),
        entity_extractor=fake_entity_extractor,
        llm_provider=fake_llm_provider,
        db_path=str(tmp_path / "graph_tasks.db"),
    )
    async def _run() -> None:
        await manager.start()
        try:
            await manager.open_session("conv-idle")
            worker = manager._queues["conv-idle"]
            worker._last_activity = time.monotonic() - 3600
            await manager._evict_idle_queues()
            assert "conv-idle" not in manager._queues
        finally:
            await manager.shutdown()

    asyncio.run(_run())
