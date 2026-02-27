import asyncio
import logging
from unittest.mock import AsyncMock, patch
import pytest

# Mock generate_node_id before importing the module that uses it
with patch(
    "cognee.modules.engine.utils.generate_node_id.generate_node_id"
) as mock_generate_node_id:
    mock_generate_node_id.side_effect = lambda x: f"id-{x.strip('.')}"
    from core.memory.entity_merger import run_entity_merging_neo4j

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.asyncio
async def test_entity_merging_neo4j():
    # Mock graph client with a mock execute method
    mock_client = AsyncMock()
    mock_graph = AsyncMock()
    mock_execute = AsyncMock()
    mock_client.graph = mock_graph
    mock_graph.execute = mock_execute

    # Also set direct execute in case it uses that
    mock_client.execute = mock_execute

    # 1. First call to fetch_query should return some nodes
    # We will simulate "Apple Inc." and "Apple Inc" mapping to the same ID.
    mock_execute.side_effect = [
        # Call 1: Fetch
        [
            {"neo4j_id": 101, "name": "Apple Inc."},
            {"neo4j_id": 102, "name": "Apple Inc"},
            {"neo4j_id": 201, "name": "Microsoft Corp"},
        ],
        # Call 2: Merge for the Apple group
        [{"merged_id": "merged-apple-id", "name": "Apple Inc."}],
    ]

    await run_entity_merging_neo4j(mock_client)

    # Verifications
    assert mock_execute.call_count == 2

    # Check fetch query
    fetch_call = mock_execute.call_args_list[0]
    assert "MATCH (n:`__Node__`)" in fetch_call.args[0]
    assert (
        "WHERE (n:Company OR n:Sector OR n:GlobalEvent OR n:MacroTrend OR n:FinancialConcept)"
        in fetch_call.args[0]
    )

    # Check merge query
    merge_call = mock_execute.call_args_list[1]
    assert "MATCH (n:`__Node__`)" in merge_call.args[0]
    assert "CALL apoc.refactor.mergeNodes" in merge_call.args[0]

    # 101 and 102 should be the node IDs passed to the merge query
    params = merge_call.args[1]
    assert "node_ids" in params
    assert len(params["node_ids"]) == 2
    assert 101 in params["node_ids"]
    assert 102 in params["node_ids"]
    assert 201 not in params["node_ids"]  # Microsoft should not be merged
