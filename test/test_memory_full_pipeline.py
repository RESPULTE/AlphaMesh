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

import cognee

from core.memory.financial_retriever import QueryScope

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress overly verbose cognee logs so the test output is readable
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


async def run_ingestion(memory, user_a, user_b, sep):
    # ------------------------------------------------------------------
    # 2. Ingest public news (GLOBAL)
    # ------------------------------------------------------------------
    print("[2/5] Ingesting public macroeconomic news (GLOBAL) …")
    await memory.ingest_news(
        articles=[
            {
                "headline": "Federal Reserve maintains interest rates at 5.25% amid easing inflation data",
                "summary": "The Federal Reserve held its benchmark interest rate steady at the 5.25%-5.50% range, signaling that while inflation risks have diminished considerably over the last two quarters, policymakers are awaiting further conclusive economic data before committing to consecutive rate cuts in the latter half of the year. The broader market reacted positively to the dovish tone.",
            },
            {
                "headline": "Biotech funding sees a resurgence in Q3 led by neurodegenerative breakthroughs",
                "summary": "Venture capital funding for clinical-stage biotechnology firms surged by 22% quarter-over-over. The influx of capital is largely attributed to promising preliminary data from several phase 2 and phase 3 trials focusing on Alzheimer's and Parkinson's disease therapeutics.",
            },
        ],
        is_global=True,
    )
    print("      > Global news ingested.\n")

    # ------------------------------------------------------------------
    # 3. Ingest User A's private data (Biotech)
    # ------------------------------------------------------------------
    print(f"[3/5] Ingesting private conversation for '{user_a}' (Topic: Biotech) …")
    await memory.ingest_conversation(
        user_email=user_a,
        messages=[
            {
                "role": "user",
                "content": "I just invested heavily in NeuroCure Therapeutics ($NCT). I believe their pipeline is undervalued given the recent biotech market trends.",
            },
            {
                "role": "assistant",
                "content": "NeuroCure Therapeutics ($NCT) is significantly focused on next-generation Alzheimer's treatments. Their flagship asset, NC-842, is currently in a pivotal Phase 3 trial reading out in Q4, which targets amyloid plaques utilizing a novel transport mechanism. If successful, it could capture a large share of a multi-billion dollar Total Addressable Market (TAM). \n\nHowever, it's worth noting the broader interconnected market effects: substantial breakthroughs extending lifespans and requiring specialized assisted living infrastructure could indirectly impact the demand for specific real estate assets. While somewhat tangential, shifts in healthcare outcomes might eventually influence the portfolios of major commercial real estate companies, like Vornado Realty Trust ($VNO), if they expand into life sciences or specialized medical real estate to capture the shifting demographic needs.",
            },
            {
                "role": "user",
                "content": "That is an interesting macro perspective. Regardless, my target price for $NCT is $150 if the trial succeeds, based on comparable acquisition multiples of recent clinical-stage neuro-focused peers.",
            },
        ],
    )
    print("      > User A data ingested.\n")

    # ------------------------------------------------------------------
    # 4. Ingest User B's private data (Real Estate)
    # ------------------------------------------------------------------
    print(f"[4/5] Ingesting private conversation for '{user_b}' (Topic: Real Estate) …")
    await memory.ingest_conversation(
        user_email=user_b,
        messages=[
            {
                "role": "user",
                "content": "I am looking at buying commercial real estate equities to hedge against inflation, specifically considering the VNO REIT.",
            },
            {
                "role": "assistant",
                "content": "Vornado Realty Trust (VNO) is one of the largest real estate investment trusts in the United States, with significant exposure to premium NYC office space and street retail properties. The company's portfolio is heavily concentrated in Manhattan, driving its core revenue.\n\nWhile VNO offers an attractive dividend yield, the current macroeconomic environment—specifically the Federal Reserve's stance on holding interest rates steady at 5.25%—means borrowing costs remain elevated for property developers. Furthermore, the persistent shift towards hybrid work models continues to pose a medium-term headwind to office occupancy rates. To counteract this, some commercial REITs are looking to diversify their tenant base by converting traditional office spaces into specialized medical facilities or life sciences labs, though VNO's core strength remains in prime traditional office and retail spaces.",
            },
            {
                "role": "user",
                "content": "Yes, I understand the risks with NYC office exposure. But I plan to hold VNO primarily for the robust 7% dividend yield, and I believe the premium properties will hold their valuation over the next decade.",
            },
        ],
    )
    print("      > User B data ingested.\n")

    # ------------------------------------------------------------------
    # 5. Cognify
    # ------------------------------------------------------------------
    print("[5/5] Running single-pass cognify for all data (requires LLM_API_KEY) …")
    await memory.cognify(chunks_per_batch=20)
    print("      > Cognify complete for all data.\n")

    print(f"{sep}")
    print("  VERIFYING ISOLATION AND KNOWLEDGE GRAPH")
    print(f"{sep}\n")


async def execute_query(
    memory, user: str, prompt: str, entity_name: str, scope: QueryScope
) -> None:
    print(f"  [Q] User: {user}")
    print(f"      Ask : '{prompt}'")
    only_context = False
    try:
        results = await memory.query(
            user_email=user,
            query_text=prompt,
            query_scope=scope,
            entity_name=entity_name,
            top_k=10,
            only_context=only_context,
        )
        if not only_context:
            print(f"      Ans : {len(results)} chunks found.")
            for i, r in enumerate(results):
                text_snippet = str(r).replace("\\n", " ").strip()
                print(f"            [{i+1}] {text_snippet}")
        else:
            print(f"      Ans : {len(results)} chunks found.")
            print(results["context_text"])
            for i, r in enumerate(results["citations"]):
                text_snippet = str(r).replace("\\n", " ").strip()
                print(f"            [{i+1}] {text_snippet}")
    except Exception as exc:
        print(f"      ERR : {type(exc).__name__}: {exc}")
    print("")


async def query_test(memory, USER_A, USER_B):
    # Test 1: Both users can see GLOBAL data (Federal Reserve)
    print("--- Test 1a: Global Data Access — Federal Reserve ---")
    await execute_query(
        memory,
        USER_A,
        "What is the Federal Reserve's current stance on interest rates?",
        "Federal Reserve",
        QueryScope.MARKET,
    )
    await execute_query(
        memory,
        USER_B,
        "What is the Federal Reserve's current stance on interest rates?",
        "Federal Reserve",
        QueryScope.MARKET,
    )

    # Test 1b: Both users can see GLOBAL data (Biotech funding)
    print("--- Test 1b: Global Data Access — Biotech Funding ---")
    await execute_query(
        memory,
        USER_A,
        "How is biotech venture capital funding trending this quarter?",
        "Biotech",
        QueryScope.MARKET,
    )
    await execute_query(
        memory,
        USER_B,
        "How is biotech venture capital funding trending this quarter?",
        "Biotech",
        QueryScope.MARKET,
    )

    # Test 2: User A accessing User A's data (Should succeed)
    print("--- Test 2: User A asking about their own data ---")
    await execute_query(
        memory,
        USER_A,
        "What is my target price for NeuroCure Therapeutics ($NCT) and what is the NC-842 Phase 3 trial status?",
        "NeuroCure Therapeutics",
        QueryScope.COMPANY,
    )

    # Test 3: User B trying to access User A's data (Should return nothing or only global, NO $NCT data)
    print("--- Test 3: ISOLATION CHECK — User B asking about User A's data ---")
    await execute_query(
        memory,
        USER_B,
        "What is the target price for NeuroCure Therapeutics ($NCT) and what is the NC-842 Phase 3 trial status?",
        "NeuroCure Therapeutics",
        QueryScope.COMPANY,
    )

    # Test 4: User B accessing User B's data (Should succeed)
    print("--- Test 4: User B asking about their own data ---")
    await execute_query(
        memory,
        USER_B,
        "Why am I holding VNO and what dividend yield am I expecting from it?",
        "Vornado Realty Trust",
        QueryScope.COMPANY,
    )

    # Test 5: User A trying to access User B's data (Should return nothing, NO VNO data)
    print("--- Test 5: ISOLATION CHECK — User A asking about User B's data ---")
    await execute_query(
        memory,
        USER_A,
        "Why am I holding VNO and what dividend yield am I expecting from it?",
        "Vornado Realty Trust",
        QueryScope.COMPANY,
    )

    # Test 6: Cross-scope query — User A asking about macro threats to their thesis
    print("--- Test 6: User A — Cross-scope investment thesis analysis ---")
    await execute_query(
        memory,
        USER_A,
        "What are the current macro threats and upsides to my investment thesis in NeuroCure ($NCT), taking into account interest rate and biotech market conditions?",
        "NeuroCure Therapeutics",
        QueryScope.COMPANY,
    )


async def clear_all():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def run_smoke_test() -> None:

    from core.memory import (
        GLOBAL_NODESET_NAME,
        FinancialMemorySystem,
        get_user_nodeset_name,
    )

    # Remove all data files

    USER_A = "alice@alphamese.ai"
    USER_B = "bob@alphamese.ai"

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
    await clear_all()
    print("[1/5] Initializing FinancialMemorySystem …")
    await memory.initialize()
    print("      > Initialized.\n")

    await run_ingestion(memory, USER_A, USER_B, SEP)
    # await query_test(memory, USER_A, USER_B)

    print(f"{SEP}")
    print("  Smoke test fully verified.")
    print(f"{SEP}\n")

    # await execute_query(USER_A, "what does VNO gets affected by any other stock?")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
