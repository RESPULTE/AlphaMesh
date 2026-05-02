from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from core.memory.graph.entity_resolver import EntityResolution, ResolvedEdgeBatch
from core.memory.graph.graph_queue import (
    TASK_KIND_EXTRACTION,
    TASK_KIND_SCOPED_EXTRACTION,
    GraphQueueManager,
    make_extraction_task,
    make_graph_task,
    prompt_id_from_text,
)
from core.memory.graph.models import ALLOWED_ENTITY_TYPES, ALLOWED_RELATIONSHIP_TYPES
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
        self.user_domain_props: List[dict] = []
        self.user_edges: List[str] = []
        self.user_events: List[str] = []
        self.session_nodes: List[str] = []
        self.chunk_entities_rows: List[dict] = []

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

            if from_type not in {
                "UserInterestDomain",
                "UserInterestEdge",
                "UserInterestEvent",
                "SessionNode",
            }:
                from_type = normalize_entity_type(from_type) or ""
            if to_type not in {
                "UserInterestDomain",
                "UserInterestEdge",
                "UserInterestEvent",
                "SessionNode",
            }:
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

    async def merge_user_interest_domain(self, domain_id: str, props: dict) -> None:
        self.user_domains.append(domain_id)
        self.user_domain_props.append(dict(props))

    async def merge_user_interest_edge(
        self,
        edge_id: str,
        props: dict,
        operation: str,
        weight_delta: float,
    ) -> None:
        _ = (props, operation, weight_delta)
        self.user_edges.append(edge_id)

    async def merge_user_interest_event(self, node_id: str, props: dict) -> None:
        _ = props
        self.user_events.append(node_id)

    async def merge_session_node(self, node_id: str, props: dict) -> None:
        _ = props
        self.session_nodes.append(node_id)

    async def get_entities_for_chunks(self, chunk_ids: List[str]) -> List[dict]:
        _ = chunk_ids
        return list(self.chunk_entities_rows)


class FakeRelationshipExtractor:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    async def extract(
        self,
        *,
        mode: str = "relationships",
        text: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        llm=None,
        system_prompt: Optional[str] = None,
        force: bool = False,
    ) -> List[dict]:
        self.calls.append(
            {
                "mode": mode,
                "text": text,
                "chunk_ids": list(chunk_ids or []),
                "llm": llm,
                "system_prompt": system_prompt,
                "force": force,
            }
        )
        _ = (mode, text, chunk_ids, llm, system_prompt, force)
        return []


def fake_llm_provider(_config: Optional[dict]):
    return object()


def test_enqueue_immediate_relationship_task_disables_creation(tmp_path: Path):
    resolver = FakeResolver()
    writer = FakeWriter()
    manager = GraphQueueManager(
        entity_resolver=resolver,
        graph_writer=writer,
        relationship_extractor=FakeRelationshipExtractor(),
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
            await manager.close_session(task.conversation_id)
            assert resolver.batch_calls
            assert resolver.batch_calls[0][1] is False
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
            llm_provider=fake_llm_provider,
            prompt_registry=prompt_registry,
        )

        relationships = [
            {
                "from_name": "domain-1",
                "from_type": "UserInterestDomain",
                "relation": "HAS_INTEREST_IN",
                "to_name": "edge-1",
                "to_type": "UserInterestEdge",
                "from_node_props": {"id": "domain-1", "nodeset_id": "nodeset-1"},
                "to_node_props": {"id": "edge-1", "operation": "reinforce"},
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
                "relation": "HAS_EVENT",
                "to_name": "event-1",
                "to_type": "UserInterestEvent",
                "from_node_props": {"id": "edge-1", "operation": "reinforce"},
                "to_node_props": {"id": "event-1"},
            },
            {
                "from_name": "event-1",
                "from_type": "UserInterestEvent",
                "relation": "OBSERVED_IN",
                "to_name": "session-1",
                "to_type": "SessionNode",
                "from_node_props": {"id": "event-1"},
                "to_node_props": {"id": "session-1"},
            },
        ]

        domain_written, user_written = await pipeline.process_relationships(
            relationships=relationships,
            conversation_id="conversation-1",
            source_agent="orchestrator",
            allow_create=False,
        )
        assert domain_written == 0
        assert user_written == 4
        assert writer.user_domains == ["domain-1"]
        assert writer.user_domain_props[0].get("nodeset_id") == "nodeset-1"
        assert writer.user_edges == ["edge-1"]
        assert writer.user_events == ["event-1"]
        assert writer.session_nodes == ["session-1"]
        assert resolver.batch_calls
        assert resolver.batch_calls[0][1] is False

    asyncio.run(_run())


def test_ttl_eviction_closes_idle_live_workers(tmp_path: Path):
    manager = GraphQueueManager(
        entity_resolver=FakeResolver(),
        graph_writer=FakeWriter(),
        relationship_extractor=FakeRelationshipExtractor(),
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


def test_pipeline_upserts_session_and_event_nodes_before_edge_write(
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
            llm_provider=fake_llm_provider,
            prompt_registry=prompt_registry,
        )

        relationships = [
            {
                "from_name": "domain-1",
                "from_type": "UserInterestDomain",
                "relation": "HAS_INTEREST_IN",
                "to_name": "edge-1",
                "to_type": "UserInterestEdge",
                "from_node_props": {"id": "domain-1", "nodeset_id": "nodeset-1"},
                "to_node_props": {"id": "edge-1", "operation": "reinforce"},
            },
            {
                "from_name": "edge-1",
                "from_type": "UserInterestEdge",
                "relation": "HAS_EVENT",
                "to_name": "event-1",
                "to_type": "UserInterestEvent",
                "from_node_props": {"id": "edge-1", "operation": "reinforce"},
                "to_node_props": {"id": "event-1"},
            },
            {
                "from_name": "event-1",
                "from_type": "UserInterestEvent",
                "relation": "OBSERVED_IN",
                "to_name": "session-1",
                "to_type": "SessionNode",
                "from_node_props": {"id": "event-1"},
                "to_node_props": {"id": "session-1"},
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
        ]

        domain_written, user_written = await pipeline.process_relationships(
            relationships=relationships,
            conversation_id="conversation-1",
            source_agent="orchestrator",
            allow_create=False,
        )
        assert domain_written == 0
        assert user_written == 4
        assert writer.user_domains == ["domain-1"]
        assert writer.user_edges == ["edge-1"]
        assert writer.user_events == ["event-1"]
        assert writer.session_nodes == ["session-1"]

    asyncio.run(_run())


def test_enqueue_skips_invalid_chunk_and_empty_tasks(tmp_path: Path):
    manager = GraphQueueManager(
        entity_resolver=FakeResolver(),
        graph_writer=FakeWriter(),
        relationship_extractor=FakeRelationshipExtractor(),
        llm_provider=fake_llm_provider,
        db_path=str(tmp_path / "graph_tasks.db"),
    )

    async def _run() -> None:
        await manager.start()
        try:
            invalid_chunk_task = make_extraction_task(
                turn_id="turn-1",
                conversation_id="conv-1",
                source_agent="agent-a",
                task_kind=TASK_KIND_SCOPED_EXTRACTION,
                chunk_ids=[],
                extraction_text="extract me",
                system_prompt="extract relationships",
            )
            empty_relationship_task = make_graph_task(
                turn_id="turn-1",
                conversation_id="conv-1",
                source_agent="agent-a",
                relationships=[],
            )
            missing_prompt_task = make_extraction_task(
                turn_id="turn-1",
                conversation_id="conv-1",
                source_agent="agent-a",
                extraction_text="extract me",
                system_prompt=None,
            )

            await manager.enqueue(invalid_chunk_task)
            await manager.enqueue(empty_relationship_task)
            await manager.enqueue(missing_prompt_task)

            rows = await manager._store.fetchall("SELECT task_id FROM graph_tasks")
            assert not rows
        finally:
            await manager.shutdown()

    asyncio.run(_run())


def test_recover_pending_tasks_without_allow_create_policy(tmp_path: Path):
    resolver = FakeResolver()
    writer = FakeWriter()
    manager = GraphQueueManager(
        entity_resolver=resolver,
        graph_writer=writer,
        relationship_extractor=FakeRelationshipExtractor(),
        llm_provider=fake_llm_provider,
        db_path=str(tmp_path / "graph_tasks.db"),
    )

    async def _run() -> None:
        await manager._store.initialize()

        task = make_graph_task(
            turn_id="turn-1",
            conversation_id="conv-1",
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
        )
        await manager._store.persist_task(task.to_payload())

        async def _capture_process(tasks: List[object]) -> Dict[str, int]:
            assert len(tasks) == 1
            assert not hasattr(tasks[0], "allow_create")
            return {"domain_edges": 0, "user_edges": 0}

        manager._pipeline.process_tasks = _capture_process  # type: ignore[method-assign]
        await manager._recover_pending_tasks()

    asyncio.run(_run())


def test_make_extraction_task_normalizes_scope_and_rejects_unknown_values() -> None:
    task = make_extraction_task(
        turn_id="turn-1",
        conversation_id="conv-1",
        source_agent="agent-a",
        extraction_text="analysis text",
        system_prompt="extract relationships",
        allowed_entity_types=["Sector", "Company", "Company"],
        allowed_relationship_types=["BELONGS_TO", "BELONGS_TO", "RELATED_TO"],
    )

    assert task.allowed_entity_types == ["Company", "Sector"]
    assert task.allowed_relationship_types == ["BELONGS_TO", "RELATED_TO"]
    assert task.system_prompt is not None
    assert "Task-scoped extraction constraints:" in task.system_prompt
    assert task.system_prompt_id == prompt_id_from_text(task.system_prompt)

    with pytest.raises(ValueError, match="Unknown entity_type\\(s\\): UnknownType"):
        make_extraction_task(
            turn_id="turn-1",
            conversation_id="conv-1",
            source_agent="agent-a",
            extraction_text="analysis text",
            system_prompt="extract relationships",
            allowed_entity_types=["UnknownType"],
        )

    with pytest.raises(ValueError, match="Unknown relationship_type\\(s\\): UNKNOWN_EDGE"):
        make_extraction_task(
            turn_id="turn-1",
            conversation_id="conv-1",
            source_agent="agent-a",
            extraction_text="analysis text",
            system_prompt="extract relationships",
            allowed_relationship_types=["UNKNOWN_EDGE"],
        )


def test_graph_task_store_roundtrip_persists_extraction_scope(tmp_path: Path) -> None:
    store = GraphTaskSqlStore(str(tmp_path / "graph_tasks.db"))

    async def _run() -> None:
        await store.initialize()
        task = make_extraction_task(
            turn_id="turn-1",
            conversation_id="conv-1",
            source_agent="agent-a",
            task_kind=TASK_KIND_SCOPED_EXTRACTION,
            extraction_text="analysis text",
            system_prompt="extract relationships",
            chunk_system_prompt="extract chunk entities",
            chunk_ids=["chunk-1"],
            allowed_entity_types=["Company", "Sector"],
            allowed_relationship_types=["BELONGS_TO"],
        )
        await store.persist_task(task.to_payload())

        rows = await store.load_pending_tasks()
        assert len(rows) == 1
        loaded = rows[0]
        assert loaded["allowed_entity_types"] == ["Company", "Sector"]
        assert loaded["allowed_relationship_types"] == ["BELONGS_TO"]
        assert loaded["chunk_system_prompt_id"] == prompt_id_from_text(
            "extract chunk entities"
        )

    asyncio.run(_run())


def test_pipeline_drops_out_of_scope_extracted_relationships(tmp_path: Path) -> None:
    class ScopedExtractor:
        async def extract(
            self,
            *,
            mode: str = "relationships",
            text: Optional[str] = None,
            chunk_ids: Optional[List[str]] = None,
            llm=None,
            system_prompt: Optional[str] = None,
            force: bool = False,
        ) -> List[dict]:
            _ = (mode, text, chunk_ids, llm, system_prompt, force)
            if mode != "relationships":
                return []
            return [
                {
                    "from_name": "Apple",
                    "from_type": "Company",
                    "relation": "BELONGS_TO",
                    "to_name": "Technology",
                    "to_type": "Sector",
                },
                {
                    "from_name": "Apple",
                    "from_type": "Company",
                    "relation": "RELATED_TO",
                    "to_name": "US",
                    "to_type": "Market",
                },
                {
                    "from_name": "Apple",
                    "from_type": "Company",
                    "relation": "BELONGS_TO",
                    "to_name": "Earnings Beat",
                    "to_type": "FinancialEvent",
                },
            ]

    resolver = FakeResolver()
    writer = FakeWriter()
    store = GraphTaskSqlStore(str(tmp_path / "graph_tasks.db"))

    async def _run() -> None:
        await store.initialize()
        prompt_registry = PromptRegistry(store)
        pipeline = GraphWritePipeline(
            entity_resolver=resolver,
            graph_writer=writer,
            relationship_extractor=ScopedExtractor(),
            llm_provider=fake_llm_provider,
            prompt_registry=prompt_registry,
        )

        task = make_extraction_task(
            turn_id="turn-1",
            conversation_id="conv-1",
            source_agent="agent-a",
            extraction_text="analysis text",
            system_prompt="extract relationships",
            allowed_entity_types=["Company", "Sector"],
            allowed_relationship_types=["BELONGS_TO"],
        )
        assert task.system_prompt is not None
        await prompt_registry.register(task.system_prompt)

        results = await pipeline.process_tasks([task])
        assert results["domain_edges"] == 0
        assert results["user_edges"] == 0
        assert resolver.edge_calls
        scoped_relationships, _allow_create = resolver.edge_calls[0]
        assert len(scoped_relationships) == 1
        assert scoped_relationships[0]["relation"] == "BELONGS_TO"
        assert len(writer.write_calls) == 1
        written_relationships, _entity_cache = writer.write_calls[0]
        assert len(written_relationships) == 1
        assert written_relationships[0]["relation"] == "BELONGS_TO"
        assert written_relationships[0]["from_type"] in ALLOWED_ENTITY_TYPES
        assert written_relationships[0]["to_type"] in ALLOWED_ENTITY_TYPES
        assert written_relationships[0]["relation"] in ALLOWED_RELATIONSHIP_TYPES

    asyncio.run(_run())


def test_pipeline_groups_chunk_extraction_by_prompt_id(tmp_path: Path) -> None:
    class RecordingExtractor:
        def __init__(self) -> None:
            self.chunk_calls: List[dict] = []

        async def extract(
            self,
            *,
            mode: str = "relationships",
            text: Optional[str] = None,
            chunk_ids: Optional[List[str]] = None,
            llm=None,
            system_prompt: Optional[str] = None,
            force: bool = False,
        ) -> List[dict]:
            _ = (text, force)
            if mode == "chunk_entities":
                self.chunk_calls.append(
                    {
                        "chunk_ids": list(chunk_ids or []),
                        "system_prompt": system_prompt,
                        "llm": llm,
                    }
                )
            return []

    resolver = FakeResolver()
    writer = FakeWriter()
    store = GraphTaskSqlStore(str(tmp_path / "graph_tasks.db"))
    extractor = RecordingExtractor()

    async def _run() -> None:
        await store.initialize()
        prompt_registry = PromptRegistry(store)
        pipeline = GraphWritePipeline(
            entity_resolver=resolver,
            graph_writer=writer,
            relationship_extractor=extractor,
            llm_provider=fake_llm_provider,
            prompt_registry=prompt_registry,
        )

        prompt_a = "prompt A"
        prompt_b = "prompt B"
        prompt_a_id = await prompt_registry.register(prompt_a)
        prompt_b_id = await prompt_registry.register(prompt_b)

        tasks = [
            make_extraction_task(
                turn_id="turn-1",
                conversation_id="conv-1",
                source_agent="agent-a",
                task_kind=TASK_KIND_SCOPED_EXTRACTION,
                chunk_ids=["c1", "c2", "c1"],
                chunk_system_prompt=prompt_a,
            ),
            make_extraction_task(
                turn_id="turn-1",
                conversation_id="conv-1",
                source_agent="agent-a",
                task_kind=TASK_KIND_SCOPED_EXTRACTION,
                chunk_ids=["c3"],
                chunk_system_prompt=prompt_b,
            ),
            make_extraction_task(
                turn_id="turn-1",
                conversation_id="conv-1",
                source_agent="agent-a",
                task_kind=TASK_KIND_SCOPED_EXTRACTION,
                chunk_ids=["c4"],
                chunk_system_prompt=None,
            ),
        ]
        tasks[0].chunk_system_prompt_id = prompt_a_id
        tasks[1].chunk_system_prompt_id = prompt_b_id

        await pipeline.process_tasks(tasks)

        assert len(extractor.chunk_calls) == 3
        prompt_to_chunks = {
            call["system_prompt"]: sorted(call["chunk_ids"])
            for call in extractor.chunk_calls
        }
        assert prompt_to_chunks[prompt_a] == ["c1", "c2"]
        assert prompt_to_chunks[prompt_b] == ["c3"]
        assert prompt_to_chunks[None] == ["c4"]

    asyncio.run(_run())


def test_pipeline_injects_chunk_entity_context_and_filters_relationships(
    tmp_path: Path,
) -> None:
    class RecordingExtractor:
        def __init__(self) -> None:
            self.calls: List[dict] = []

        async def extract(
            self,
            *,
            mode: str = "relationships",
            text: Optional[str] = None,
            chunk_ids: Optional[List[str]] = None,
            llm=None,
            system_prompt: Optional[str] = None,
            force: bool = False,
        ) -> List[dict]:
            self.calls.append(
                {
                    "mode": mode,
                    "text": text,
                    "chunk_ids": list(chunk_ids or []),
                    "llm": llm,
                    "system_prompt": system_prompt,
                    "force": force,
                }
            )
            return [
                {
                    "from_name": "Apple",
                    "from_type": "Company",
                    "relation": "RELATED_TO",
                    "to_name": "Revenue Growth",
                    "to_type": "FinancialConcept",
                },
                {
                    "from_name": "Apple",
                    "from_type": "Company",
                    "relation": "RELATED_TO",
                    "to_name": "Unknown Entity",
                    "to_type": "FinancialConcept",
                },
            ]

    resolver = FakeResolver()
    writer = FakeWriter()
    writer.chunk_entities_rows = [
        {
            "entity_name": "Apple",
            "entity_type": "Company",
            "source_chunk_id": "chunk-1",
        },
        {
            "entity_name": "Revenue Growth",
            "entity_type": "FinancialConcept",
            "source_chunk_id": "chunk-1",
        },
    ]
    store = GraphTaskSqlStore(str(tmp_path / "graph_tasks.db"))
    extractor = RecordingExtractor()

    async def _run() -> None:
        await store.initialize()
        prompt_registry = PromptRegistry(store)
        pipeline = GraphWritePipeline(
            entity_resolver=resolver,
            graph_writer=writer,
            relationship_extractor=extractor,
            llm_provider=fake_llm_provider,
            prompt_registry=prompt_registry,
        )

        task = make_extraction_task(
            turn_id="turn-1",
            conversation_id="conv-1",
            source_agent="agent-a",
            task_kind=TASK_KIND_SCOPED_EXTRACTION,
            extraction_text="analysis text",
            chunk_ids=["chunk-1"],
            system_prompt="extract relationships",
            allowed_entity_types=["Company", "FinancialConcept"],
            allowed_relationship_types=["RELATED_TO"],
        )
        assert task.system_prompt is not None
        await prompt_registry.register(task.system_prompt)

        result = await pipeline.process_tasks([task])
        assert result["retry_tasks"] == []
        assert result["processed_task_ids"] == [task.task_id]
        assert extractor.calls
        relationship_calls = [
            call for call in extractor.calls if call.get("mode") == "relationships"
        ]
        assert relationship_calls
        injected_text = str(relationship_calls[0]["text"] or "")
        assert "Known entities extracted for the referenced chunks" in injected_text
        assert "Apple (Company)" in injected_text
        assert "Revenue Growth (FinancialConcept)" in injected_text
        assert resolver.edge_calls
        scoped_relationships, _allow_create = resolver.edge_calls[0]
        assert len(scoped_relationships) == 1
        assert scoped_relationships[0]["to_name"] == "Revenue Growth"

    asyncio.run(_run())


def test_pipeline_no_entities_schedules_retry_then_exhausts(tmp_path: Path) -> None:
    class EmptyExtractor:
        async def extract(
            self,
            *,
            mode: str = "relationships",
            text: Optional[str] = None,
            chunk_ids: Optional[List[str]] = None,
            llm=None,
            system_prompt: Optional[str] = None,
            force: bool = False,
        ) -> List[dict]:
            _ = (mode, text, chunk_ids, llm, system_prompt, force)
            return []

    resolver = FakeResolver()
    writer = FakeWriter()
    writer.chunk_entities_rows = []
    store = GraphTaskSqlStore(str(tmp_path / "graph_tasks.db"))

    async def _run() -> None:
        await store.initialize()
        prompt_registry = PromptRegistry(store)
        pipeline = GraphWritePipeline(
            entity_resolver=resolver,
            graph_writer=writer,
            relationship_extractor=EmptyExtractor(),
            llm_provider=fake_llm_provider,
            prompt_registry=prompt_registry,
        )

        task = make_extraction_task(
            turn_id="turn-1",
            conversation_id="conv-1",
            source_agent="agent-a",
            task_kind=TASK_KIND_SCOPED_EXTRACTION,
            extraction_text="analysis text",
            chunk_ids=["chunk-1"],
            system_prompt="extract relationships",
            retry_count=0,
            max_retries=3,
            retry_delay_seconds=300,
        )
        assert task.system_prompt is not None
        await prompt_registry.register(task.system_prompt)

        result = await pipeline.process_tasks([task])
        assert result["processed_task_ids"] == [task.task_id]
        assert len(result["retry_tasks"]) == 1
        retry_task = result["retry_tasks"][0]
        assert retry_task.retry_count == 1
        assert retry_task.max_retries == 3
        assert retry_task.not_before is not None

        exhausted_task = make_extraction_task(
            turn_id="turn-1",
            conversation_id="conv-1",
            source_agent="agent-a",
            task_kind=TASK_KIND_SCOPED_EXTRACTION,
            extraction_text="analysis text",
            chunk_ids=["chunk-1"],
            system_prompt="extract relationships",
            retry_count=3,
            max_retries=3,
            retry_delay_seconds=300,
        )
        assert exhausted_task.system_prompt is not None
        await prompt_registry.register(exhausted_task.system_prompt)
        exhausted_result = await pipeline.process_tasks([exhausted_task])
        assert exhausted_result["processed_task_ids"] == [exhausted_task.task_id]
        assert exhausted_result["retry_tasks"] == []

    asyncio.run(_run())


def test_graph_task_store_roundtrip_persists_retry_metadata(tmp_path: Path) -> None:
    store = GraphTaskSqlStore(str(tmp_path / "graph_tasks.db"))

    async def _run() -> None:
        await store.initialize()
        task = make_extraction_task(
            turn_id="turn-1",
            conversation_id="conv-1",
            source_agent="agent-a",
            extraction_text="analysis text",
            system_prompt="extract relationships",
            retry_count=2,
            max_retries=4,
            retry_delay_seconds=120,
            not_before=12345.0,
        )
        await store.persist_task(task.to_payload())

        rows = await store.load_pending_tasks()
        assert len(rows) == 1
        loaded = rows[0]
        assert loaded["retry_count"] == 2
        assert loaded["max_retries"] == 4
        assert loaded["retry_delay_seconds"] == 120
        assert loaded["not_before"] == 12345.0

    asyncio.run(_run())


def test_manager_rejects_legacy_task_kinds(tmp_path: Path) -> None:
    manager = GraphQueueManager(
        entity_resolver=FakeResolver(),
        graph_writer=FakeWriter(),
        relationship_extractor=FakeRelationshipExtractor(),
        llm_provider=fake_llm_provider,
        db_path=str(tmp_path / "graph_tasks.db"),
    )

    async def _run() -> None:
        await manager.start()
        try:
            legacy_task = make_extraction_task(
                turn_id="turn-1",
                conversation_id="conv-1",
                source_agent="agent-a",
                task_kind=TASK_KIND_EXTRACTION,
                extraction_text="analysis text",
                system_prompt="extract relationships",
            )
            legacy_task.task_kind = "relationships"
            with pytest.raises(ValueError, match="unsupported task_kind"):
                await manager.enqueue(legacy_task)
        finally:
            await manager.shutdown()

    asyncio.run(_run())

