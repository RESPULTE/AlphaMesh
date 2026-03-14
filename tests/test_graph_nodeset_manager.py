"""Unit tests for NodeSetManager."""

import uuid

import pytest

from core.memory.graph.models import ENTITY_NAMESPACE
from core.memory.graph.nodeset_manager import NodeSetManager


class DummyResult:
    def __init__(self, records):
        self._records = records

    async def data(self):
        return self._records


class FakeNeo4jAdapter:
    def __init__(self) -> None:
        self.nodes = {}
        self.writes = []

    async def _execute_read(self, cypher, params):
        records = [{"id": vid, "name": name} for name, vid in self.nodes.items()]
        return DummyResult(records)

    async def merge_nodeset_node(self, nodeset_id, name, description):
        self.nodes[name] = nodeset_id

    async def _execute_write(self, cypher, params):
        self.writes.append((cypher, params))


@pytest.mark.asyncio
async def test_get_or_create_registers_nodeset():
    adapter = FakeNeo4jAdapter()
    manager = NodeSetManager(adapter)
    nodeset_id = await manager.get_or_create("TestSet", "desc")
    assert nodeset_id == str(uuid.uuid5(ENTITY_NAMESPACE, "TestSet"))
    assert await manager.get_id("TestSet") == nodeset_id


@pytest.mark.asyncio
async def test_assign_to_chunk_metadata_appends_nodeset():
    adapter = FakeNeo4jAdapter()
    manager = NodeSetManager(adapter)
    metadata = {"nodeset_ids": ["existing"]}
    result = manager.assign_to_chunk_metadata(metadata, "new-id")
    assert "new-id" in result["nodeset_ids"]
