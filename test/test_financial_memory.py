"""
Test suite for FinancialKnowledgeMemory module.

Tests namespace isolation, cross-namespace edge linking, and privacy enforcement.
Updated to test the new dual-namespace pipeline.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("google.genai").setLevel(logging.WARNING)

async def main():
    """Run all tests for FinancialKnowledgeMemory."""
    from core.memory.graphiti_memory import FinancialKnowledgeMemory

    print("=" * 60)
    print("FINANCIAL KNOWLEDGE MEMORY TEST SUITE")
    print("=" * 60)

    # Initialize memory with env vars
    memory = FinancialKnowledgeMemory(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        api_key=os.getenv("LLM_BINDING_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
    )

    # Build indices
    await memory.build_indices()

    # Test user IDs
    user_a = "test_user_alice"
    user_b = "test_user_bob"

    try:
        # # =====================================================================
        # # TEST 1: Global episode extraction with AddEpisodeResults
        # # =====================================================================
        # print("\n[TEST 1] Global episode with AddEpisodeResults return type")
        # global_result = await memory.add_global_episode(
        #     name="test_apple_info",
        #     episode_body="Apple Inc (AAPL) is a technology company with a market cap of $3 trillion.",
        #     source_description="Global financial data",
        # )
        # print(f"  ✓ Global episode added - {len(global_result.nodes)} nodes extracted")
        # for node in global_result.nodes:
        #     print(f"    - Node: {node.name} (labels: {node.labels})")

        # # =====================================================================
        # # TEST 2: User episode with global_nodes context
        # # =====================================================================
        # print("\n[TEST 2] User episode with global_nodes context")
        # user_result = await memory.add_user_episode(
        #     user_id=user_a,
        #     name="test_alice_portfolio",
        #     episode_body="Alice is a 30 year old investor who holds 100 shares of Apple at cost basis $150.",
        #     source_description="User portfolio",
        #     global_nodes=global_result.nodes,  # Pass global nodes for context
        # )
        # print(f"  ✓ User episode added - {len(user_result.nodes)} nodes extracted")
        # for node in user_result.nodes:
        #     print(f"    - Node: {node.name} (labels: {node.labels})")

        # =====================================================================
        # TEST 3: Unified add_episode with dual-namespace pipeline
        # =====================================================================
        print("\n[TEST 3] Unified add_episode with dual-namespace pipeline")
        result = await memory.add_episode(
            user_id=user_a,
            name="test_mixed_content",
            episode_body="Bob wants to buy Tesla stock. Tesla (TSLA) is an EV company worth $800B. Bob is 35 and works as an engineer.",
        )
        print(f"  ✓ Dual-namespace episode added")
        print(f"    - Global nodes: {len(result['global_result'].nodes)}")
        print(f"    - User nodes: {len(result['user_result'].nodes)}")
        print(f"    - Shared episode UUID: {result['shared_episode_uuid'][:8]}...")
        
        # List global nodes
        print("    Global entities:")
        for node in result['global_result'].nodes:
            print(f"      - {node.name}")
        
        # List user nodes
        print("    User entities:")
        for node in result['user_result'].nodes:
            print(f"      - {node.name}")

        # # =====================================================================
        # # TEST 4: Namespace isolation - global-only search shouldn't expose user data
        # # =====================================================================
        # print("\n[TEST 4] Namespace isolation - global search shouldn't expose user details")
        # # Search with include_user=False to only get global results
        # global_results = await memory.search(
        #     query="Apple stock",
        #     user_id=user_a,
        #     include_global=True,
        #     include_user=False,  # Only search global namespace
        # )
        
        # # Check that user-specific details are not in global results
        # user_data_exposed = False
        # for res in global_results:
        #     fact = getattr(res, "fact", "") or ""
        #     if "Alice" in fact or "100 shares" in fact:
        #         user_data_exposed = True
        #         break
        
        # if user_data_exposed:
        #     print("  ✗ FAILED: User data exposed in global-only search!")
        # else:
        #     print("  ✓ Namespace isolation verified - no user data in global-only search")

        # # =====================================================================
        # # TEST 5: User search includes both user and global content
        # # =====================================================================
        # print("\n[TEST 5] User search includes global content")
        # user_results = await memory.search(
        #     query="Apple",
        #     user_id=user_a,
        #     include_global=True,
        #     include_user=True,
        # )
        # print(f"  Found {len(user_results)} results for user search")
        # print("  ✓ User search completed")

        # # =====================================================================
        # # TEST 6: Cross-namespace edge linking verification
        # # =====================================================================
        # print("\n[TEST 6] Cross-namespace edge linking")
        
        # # Query for cross-namespace edges
        # query = """
        # MATCH (u:Entity)-[e]->(g:Entity)
        # WHERE u.group_id STARTS WITH 'user_' 
        #   AND g.group_id = 'GLOBAL'
        #   AND e.group_id STARTS WITH 'user_'
        # RETURN u.name AS user_entity, e.name AS edge_type, g.name AS global_entity
        # LIMIT 10
        # """
        # records, _, _ = await memory._graphiti.driver.execute_query(query)
        
        # if records:
        #     print(f"  ✓ Found {len(records)} cross-namespace edges:")
        #     for record in records:
        #         print(f"    - {record['user_entity']} --[{record['edge_type']}]--> {record['global_entity']}")
        # else:
        #     print("  (No cross-namespace edges found - may need more data)")

        # # =====================================================================
        # # TEST 7: Privacy - User B cannot see User A's data
        # # =====================================================================
        # print("\n[TEST 7] Privacy - User B cannot see User A's data")
        # user_b_search = await memory.search(
        #     query="Alice portfolio",
        #     user_id=user_b,
        #     include_global=False,
        #     include_user=True,  # Only search User B's namespace
        # )
        
        # alice_data_found = False
        # for res in user_b_search:
        #     fact = getattr(res, "fact", "") or ""
        #     if "Alice" in fact or "100 shares" in fact:
        #         alice_data_found = True
        #         break
        
        # if alice_data_found:
        #     print("  ✗ FAILED: User A's data visible to User B!")
        # else:
        #     print("  ✓ Privacy verified - User B cannot see User A's data")

        # # =====================================================================
        # # TEST 8: No edges between different users
        # # =====================================================================
        # print("\n[TEST 8] Verify no cross-user edges exist")
        
        # # Query for any edges between user namespaces
        # query = """
        # MATCH (a:Entity)-[e]->(b:Entity)
        # WHERE a.group_id STARTS WITH 'user_' 
        #   AND b.group_id STARTS WITH 'user_'
        #   AND a.group_id <> b.group_id
        # RETURN count(e) as cross_user_edges
        # """
        # records, _, _ = await memory._graphiti.driver.execute_query(query)
        # cross_user_count = records[0]["cross_user_edges"] if records else 0
        
        # if cross_user_count > 0:
        #     print(f"  ✗ FAILED: Found {cross_user_count} cross-user edges!")
        # else:
        #     print("  ✓ Privacy verified - no edges between different user namespaces")

        # # =====================================================================
        # # TEST 9: User entity with profile fields
        # # =====================================================================
        # print("\n[TEST 9] User entity with profile fields")
        # user_nodes = await memory.search_nodes(
        #     query="User Alice",
        #     user_id=user_a,
        #     include_global=False,
        #     include_user=True,
        # )
        
        # user_found = False
        # for node in user_nodes.nodes:
        #     if hasattr(node, 'name'):
        #         print(f"  Found node: {node.name} (labels: {getattr(node, 'labels', [])})")
        #         if 'User' in getattr(node, 'labels', []) or 'alice' in node.name.lower():
        #             user_found = True
        
        # if user_found:
        #     print("  ✓ User entity found in user namespace")
        # else:
        #     print("  (User entity not extracted - may depend on LLM extraction)")

        # print("\n" + "=" * 60)
        # print("ALL TESTS COMPLETED")
        # print("=" * 60)

    finally:
        # Cleanup test data
        print("\nCleaning up test data...")
        # await memory.delete_user_data(user_a)
        # await memory.delete_user_data(user_b)
        
        # Close connection
        await memory.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
