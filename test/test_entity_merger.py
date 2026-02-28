import asyncio
import logging
from unittest.mock import AsyncMock, patch
import pytest

# Mock generate_node_id before importing the module that uses it
from core.memory.entity_merger import run_entity_merging_neo4j

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.asyncio
async def test_entity_merging_neo4j():
    # Mock graph client with a mock execute method
    mock_client = AsyncMock()
    mock_query = AsyncMock()
    # Handle the fact that we use graph_client.query now
    mock_client.query = mock_query

    # 1. First call to fetch_query should return some nodes
    # We will simulate "Apple Inc." and "Apple Inc" which are fuzzy matches (ratio > 0.85)
    # And "Microsoft Corp" and "Microsoft" which are subsets (ratio > 0.50 or subset)
    # but we'll mock the CHUNKS search to NOT find a semantic match for Microsoft here.
    mock_query.side_effect = [
        # Call 1: Fetch
        [
            {
                "neo4j_id": 101,
                "cognee_id": "c-101",
                "name": "Apple Inc.",
                "labels": ["Company"],
            },
            {
                "neo4j_id": 102,
                "cognee_id": "c-102",
                "name": "Apple Inc",
                "labels": ["Company"],
            },
            {
                "neo4j_id": 201,
                "cognee_id": "c-201",
                "name": "Microsoft Corp",
                "labels": ["Company"],
            },
            {
                "neo4j_id": 202,
                "cognee_id": "c-202",
                "name": "Microsoft",
                "labels": ["Company"],
            },
        ],
        # Call 2: Merge for the Apple group ONLY (unless we mock the search to succeed)
        [{"merged_neo4j_id": "102", "cognee_id": "c-102", "name": "Apple Inc"}],
    ]

    # Patch the `search` function so it returns no semantic match for Microsoft
    with patch(
        "core.memory.entity_merger.search", new_callable=AsyncMock
    ) as mock_search:
        mock_search.return_value = [{"text": "Something completely different"}]

        # Patch the relational engine so we don't try to actually connect to sqlite/pg
        with patch("core.memory.entity_merger.get_relational_engine") as mock_gre:
            # We don't need it to actually do anything
            await run_entity_merging_neo4j(mock_client)

            # Verifications
            assert mock_query.call_count == 2

            # Check fetch query
            fetch_call = mock_query.call_args_list[0]
            assert "MATCH (n:`__Node__`)" in fetch_call.args[0]
            assert (
                "WHERE (n:Company OR n:Sector OR n:GlobalEvent OR n:MacroTrend OR n:FinancialConcept)"
                in fetch_call.args[0]
            )

            # Check merge query for Apple
            merge_call = mock_query.call_args_list[1]
            assert "MATCH (n:`__Node__`)" in merge_call.args[0]
            assert "CALL apoc.refactor.mergeNodes" in merge_call.args[0]

            # 101 and 102 should be the node IDs passed to the merge query
            params = merge_call.kwargs.get(
                "args", merge_call.args[1] if len(merge_call.args) > 1 else {}
            )
            assert "node_ids" in params
            assert len(params["node_ids"]) == 2
            assert 101 in params["node_ids"]
            assert 102 in params["node_ids"]
            assert (
                201 not in params["node_ids"]
            )  # Microsoft should not be merged because search check failed
