from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest

from core.config import settings
from core.memory.graph.entity_resolver import EntityResolver
from core.memory.graph.utils import canonical_entity_id


@dataclass
class FakeVectorDoc:
    id: Optional[str]
    metadata: Optional[dict] = None


class FakeNeo4jAdapter:
    def __init__(self) -> None:
        self.existing_ids: set[str] = set()
        self.fuzzy_candidates: Dict[str, List[dict]] = {}
        self.events: List[str] = []
        self.entity_exists_calls: List[str] = []
        self.find_fuzzy_calls: List[dict] = []
        self.merge_calls: List[object] = []

    async def entity_exists(self, entity_id: str) -> bool:
        self.events.append("neo4j.entity_exists")
        self.entity_exists_calls.append(entity_id)
        return entity_id in self.existing_ids

    async def find_fuzzy_entity_candidates(
        self,
        entity_type: str,
        name: str,
        exclude_id: str = "",
        threshold: float = 0.5,
        limit: int = 10,
    ) -> List[dict]:
        self.events.append("neo4j.find_fuzzy")
        self.find_fuzzy_calls.append(
            {
                "entity_type": entity_type,
                "name": name,
                "exclude_id": exclude_id,
                "threshold": threshold,
                "limit": limit,
            }
        )
        return list(self.fuzzy_candidates.get(name, []))

    async def merge_entity_node(self, node) -> None:
        self.events.append("neo4j.merge_entity")
        self.merge_calls.append(node)
        self.existing_ids.add(node.id)


class FakeChromaAdapter:
    def __init__(self) -> None:
        self.query_results: List[Tuple[FakeVectorDoc, float]] = []
        self.events: List[str] = []
        self.query_calls: List[dict] = []
        self.upsert_calls: List[dict] = []

    async def query_entity_similar(
        self,
        text: str,
        entity_type: str,
        n_results: int = 10,
    ) -> List[Tuple[FakeVectorDoc, float]]:
        self.events.append("chroma.query")
        self.query_calls.append(
            {
                "text": text,
                "entity_type": entity_type,
                "n_results": n_results,
            }
        )
        return list(self.query_results)

    async def upsert_entity_embedding(
        self,
        entity_id: str,
        name: str,
        description: str,
        entity_type: str,
    ) -> None:
        self.events.append("chroma.upsert")
        self.upsert_calls.append(
            {
                "entity_id": entity_id,
                "name": name,
                "description": description,
                "entity_type": entity_type,
            }
        )


def test_resolution_order_exact_then_fuzzy_then_vector_then_create() -> None:
    async def _run() -> None:
        neo4j = FakeNeo4jAdapter()
        chroma = FakeChromaAdapter()
        resolver = EntityResolver(neo4j, chroma)

        exact_name = "Apple"
        exact_id = canonical_entity_id(exact_name, "Company")
        neo4j.existing_ids.add(exact_id)

        resolved_exact = await resolver.resolve_entity(
            name=exact_name,
            entity_type="Company",
        )
        assert resolved_exact.entity_id == exact_id
        assert neo4j.events == ["neo4j.entity_exists"]
        assert chroma.events == []

        neo4j.events.clear()
        chroma.events.clear()

        create_name = "NewCo"
        create_id = canonical_entity_id(create_name, "Company")
        resolved_created = await resolver.resolve_entity(
            name=create_name,
            entity_type="Company",
            allow_create=True,
        )
        assert resolved_created.entity_id == create_id
        assert neo4j.events[0] == "neo4j.entity_exists"
        assert neo4j.events[1] == "neo4j.find_fuzzy"
        assert chroma.events[0] == "chroma.query"
        assert neo4j.events[2] == "neo4j.merge_entity"
        assert chroma.events[1] == "chroma.upsert"

    asyncio.run(_run())


def test_fuzzy_threshold_normalized_from_0_100_to_0_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        monkeypatch.setattr(settings, "EXTRACTION_FUZZY_THRESHOLD", 69.0)
        neo4j = FakeNeo4jAdapter()
        resolver = EntityResolver(
            neo4j_adapter=neo4j,
            entity_chroma_adapter=None,
        )

        resolved = await resolver.resolve_entity(
            name="Acme",
            entity_type="Company",
            allow_create=False,
        )
        assert resolved.entity_id is None
        assert neo4j.find_fuzzy_calls
        assert neo4j.find_fuzzy_calls[0]["threshold"] == pytest.approx(
            0.69,
            rel=0,
            abs=1e-9,
        )

    asyncio.run(_run())


def test_vector_candidate_below_threshold_falls_back_to_create_when_allowed() -> None:
    async def _run() -> None:
        neo4j = FakeNeo4jAdapter()
        chroma = FakeChromaAdapter()
        candidate_id = "existing-company-id"
        neo4j.existing_ids.add(candidate_id)
        neo4j.fuzzy_candidates["Acme Labs"] = [
            {"id": candidate_id, "name": "Acme", "similarity": 0.75}
        ]
        chroma.query_results = [(FakeVectorDoc(id=candidate_id), 0.35)]

        resolver = EntityResolver(
            neo4j_adapter=neo4j,
            entity_chroma_adapter=chroma,
            vector_distance_threshold=0.2,
        )

        resolution = await resolver.resolve_entity(
            name="Acme Labs",
            entity_type="Company",
            allow_create=True,
        )
        resolved_id = resolution.entity_id
        expected_id = canonical_entity_id("Acme Labs", "Company")

        assert resolved_id == expected_id
        assert all(call_id != candidate_id for call_id in neo4j.entity_exists_calls)
        assert len(neo4j.merge_calls) == 1
        assert neo4j.merge_calls[0].id == expected_id

    asyncio.run(_run())


def test_resolve_batch_deduplicates_inputs_and_reuses_cache() -> None:
    async def _run() -> None:
        neo4j = FakeNeo4jAdapter()
        resolver = EntityResolver(
            neo4j_adapter=neo4j,
            entity_chroma_adapter=None,
        )

        first_batch = [
            ("Apple", "Company", None),
            ("apple", "Company", None),
            ("NVIDIA", "Company", None),
            ("nvidia", "Company", None),
        ]

        resolved_first = await resolver.resolve_entities(first_batch, allow_create=True)
        assert len(resolved_first) == 2
        assert ("apple", "Company") in resolved_first
        assert ("nvidia", "Company") in resolved_first
        assert resolved_first[("apple", "Company")].entity_id
        assert resolved_first[("nvidia", "Company")].entity_id
        assert len(neo4j.entity_exists_calls) == 2
        assert len(neo4j.merge_calls) == 2

        entity_exists_before = len(neo4j.entity_exists_calls)
        merge_before = len(neo4j.merge_calls)

        second_batch = [
            ("apple", "Company", None),
            ("nvidia", "Company", None),
        ]
        resolved_second = await resolver.resolve_entities(
            second_batch,
            allow_create=True,
        )

        assert len(resolved_second) == 2
        assert ("apple", "Company") in resolved_second
        assert ("nvidia", "Company") in resolved_second
        assert resolved_second[("apple", "Company")].entity_id
        assert resolved_second[("nvidia", "Company")].entity_id
        assert len(neo4j.entity_exists_calls) == entity_exists_before
        assert len(neo4j.merge_calls) == merge_before

    asyncio.run(_run())
