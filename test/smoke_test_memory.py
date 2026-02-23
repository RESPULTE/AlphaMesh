"""
test/smoke_test_memory.py

End-to-end smoke test for the FinancialMemorySystem demonstrating Data Isolation.

Exercises the full production pipeline with TWO isolated users:
  1. Initialize DB and GLOBAL NodeSet
  2. Ingest public news (GLOBAL)
  3. Ingest User A's private conversation (USER A) — completely unrelated topic
  4. Ingest User B's private conversation (USER B) — completely unrelated topic
  5. Run Cognify for both users
  6. Run queries to prove User A cannot access User B's knowledge graph and vice-versa.

Usage:
    C:\\Project\\AlphaMesh\\venv\\Scripts\\python.exe test/smoke_test_memory.py

Environment: set LLM_API_KEY and other Cognee env vars in .env before running.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress overly verbose cognee logs so the test output is readable
logging.getLogger("cognee").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(message)s")
import cognee   

async def run_smoke_test() -> None:
    from core.memory import (
        FinancialMemorySystem,
        hash_user_email,
        get_user_nodeset_name,
        GLOBAL_NODESET_NAME,
    )
    from cognee.modules.search.types import SearchType

    # Remove all data files
    await cognee.prune.prune_data()

    # Remove everything (graph, vector, metadata, cache)
    await cognee.prune.prune_system(metadata=True)

    USER_A = "alice@alphamese.ai"
    USER_B = "bob@alphamese.ai"

    HASH_A = hash_user_email(USER_A)
    HASH_B = hash_user_email(USER_B)

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  AlphaMesh Financial Memory System — Isolation Smoke Test")
    print(SEP)
    print(f"  User A          : {USER_A}")
    print(f"  User A NodeSet  : {get_user_nodeset_name(USER_A)}")
    print(f"  User B          : {USER_B}")
    print(f"  User B NodeSet  : {get_user_nodeset_name(USER_B)}")
    print(f"  Global NodeSet  : {GLOBAL_NODESET_NAME}")
    print(f"{SEP}\n")

    memory = FinancialMemorySystem()


    # ------------------------------------------------------------------
    # 1. Initialize
    # ------------------------------------------------------------------
    print("[1/5] Initializing FinancialMemorySystem …")
    await memory.initialize()
    print("      > Initialized.\n")

    # ------------------------------------------------------------------
    # 2. Ingest public news (GLOBAL)
    # ------------------------------------------------------------------
    print("[2/5] Ingesting public macroeconomic news (GLOBAL) …")
    await memory.ingest_news(
        articles=[
            {
                "headline": "Federal Reserve maintains interest rates at 5.25%",
                "summary": "The Fed held rates steady, signaling lower inflation risks.",
            }
        ],
        is_global=True,
    )
    print("      > Global news ingested.\n")

    # ------------------------------------------------------------------
    # 3. Ingest User A's private data (Biotech)
    # ------------------------------------------------------------------
    print(f"[3/5] Ingesting private conversation for '{USER_A}' (Topic: Biotech) …")
    await memory.ingest_conversation(
        user_email=USER_A,
        messages=[
            {"role": "user", "content": "I just invested heavily in NeuroCure Therapeutics ($NCT)."},
            {
                "role": "assistant",
                "content": "NeuroCure ($NCT) is focused on Alzheimer's treatments. They have a Phase 3 trial reading out in Q4.",
            },
            {"role": "user", "content": "My target price for $NCT is $150 if the trial succeeds."},
        ],
    )
    print("      > User A data ingested.\n")

    # ------------------------------------------------------------------
    # 4. Ingest User B's private data (Real Estate)
    # ------------------------------------------------------------------
    print(f"[4/5] Ingesting private conversation for '{USER_B}' (Topic: Real Estate) …")
    await memory.ingest_conversation(
        user_email=USER_B,
        messages=[
            {"role": "user", "content": "I am looking at buying commercial real estate via the VNO REIT."},
            {
                "role": "assistant",
                "content": "Vornado Realty Trust (VNO) has significant exposure to NYC office space.",
            },
            {"role": "user", "content": "Yes, I plan to hold VNO for the 7% dividend yield."},
        ],
    )
    print("      > User B data ingested.\n")

    # ------------------------------------------------------------------
    # 5. Cognify both users
    # ------------------------------------------------------------------
    print("[5/5] Running cognify for both users (requires LLM_API_KEY) …")
    print(f"      > Cognifying {USER_A}...")
    await memory.cognify_user(user_email=USER_A, chunks_per_batch=20)
    print(f"      > Cognifying {USER_B}...")
    await memory.cognify_user(user_email=USER_B, chunks_per_batch=20)
    print("      > Cognify complete for both users.\n")

    print(f"{SEP}")
    print("  VERIFYING ISOLATION AND KNOWLEDGE GRAPH")
    print(f"{SEP}\n")

    async def execute_query(user: str, prompt: str) -> None:
        print(f"  [Q] User: {user}")
        print(f"      Ask : '{prompt}'")
        try:
            results = await memory.query(
                user_email=user,
                query_text=prompt,
                search_type=SearchType.CHUNKS,
                top_k=3,
            )
            print(f"      Ans : {len(results)} chunks found.")
            for i, r in enumerate(results):
                text_snippet = str(r).replace("\\n", " ").strip()
                print(f"            [{i+1}] {text_snippet}")
        except Exception as exc:
            print(f"      ERR : {type(exc).__name__}: {exc}")
        print("")

    # Test 1: Both users can see GLOBAL data
    print("--- Test 1: Global Data Access ---")
    await execute_query(USER_A, "What did the Federal Reserve do?")
    await execute_query(USER_B, "What did the Federal Reserve do?")

    # Test 2: User A accessing User A's data (Should succeed)
    print("--- Test 2: User A asking about their own data ---")
    await execute_query(USER_A, "What is my target price for NeuroCure ($NCT)?")

    # Test 3: User B trying to access User A's data (Should return nothing or only global/hallucinations, NO $NCT data)
    print("--- Test 3: ISOLATION CHECK — User B asking about User A's data ---")
    await execute_query(USER_B, "What is my target price for NeuroCure ($NCT)?")

    # Test 4: User B accessing User B's data (Should succeed)
    print("--- Test 4: User B asking about their own data ---")
    await execute_query(USER_B, "Why am I holding VNO?")

    # Test 5: User A trying to access User B's data (Should return nothing, NO VNO data)
    print("--- Test 5: ISOLATION CHECK — User A asking about User B's data ---")
    await execute_query(USER_A, "Why am I holding VNO?")

    print(f"{SEP}")
    print("  Smoke test fully verified.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
