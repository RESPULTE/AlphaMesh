"""
test/test_user_knowledge_context.py

Integration test for FinancialMemorySystem.get_user_knowledge_context()
Demonstrates the retrieval of user interest nodes (Investment and Learning)
derived from a multi-turn conversation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import cognee

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.memory import FinancialMemorySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UserContextTest")

# Suppress overly verbose cognee logs
logging.getLogger("cognee").setLevel(logging.WARNING)


async def clear_all():
    """Wipe all cognee data and system metadata."""
    logger.info("Cleaning Cognee environment...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def run_test():
    await clear_all()

    memory = FinancialMemorySystem()
    await memory.initialize()

    user_email = "alice@alphamesh.ai"

    # ------------------------------------------------------------------
    # 1. Ingest a lengthy conversation covering multiple interests
    # ------------------------------------------------------------------
    logger.info(f"Ingesting conversation for {user_email}...")

    messages = [
        {
            "role": "user",
            "content": "Hi, I've been following the semiconductor sector closely. I'm thinking about increasing my position in NVIDIA ($NVDA). The AI demand seems persistent, but I'm worried about the Blackwell architecture delays and how it might affect their Q4 guidance. What do you think about the risk/reward here?",
        },
        {
            "role": "assistant",
            "content": "NVIDIA ($NVDA) is indeed at a pivotal point. The Blackwell architecture is expected to be a massive performance leap, but manufacturing yield issues at TSMC have pushed the timeline back slightly. For Q4 guidance, the primary risk is whether Hopper (H100/H200) demand can fill the gap left by any Blackwell volume shortfall. Most analysts believe the backlog for H200 is strong enough to sustain growth, but the valuation already reflects a lot of optimism. It's a high-conviction play with near-term execution risk.",
        },
        {
            "role": "user",
            "content": "That makes sense. On a technical note, I'm trying to deepen my understanding of the underlying tech. Can you explain how Transformer architectures actually utilize GPU memory bandwidth? I want to understand if the H100 bottleneck is compute-bound (TFLOPS) or memory-bound (HBM3 bandwidth) for large language model inference.",
        },
        {
            "role": "assistant",
            "content": "Great question. For LLM inference, the bottleneck shifts depending on batch size. At low batch sizes (typical for single-user latency-sensitive tasks), the model is almost always memory-bandwidth bound. This is because every single weight of the model must be loaded from HBM into the GPU's memory for every single token generated. The compute engines effectively 'starve' waiting for data. As batch sizes increase, you become more compute-bound as the cost of loading the weights is amortized across many tokens. This is why the H100's HBM3 bandwidth (3TB/s+) is often more critical than its raw TFLOPS for effective inference performance.",
        },
        {
            "role": "user",
            "content": "Thanks! This helps me evaluate the competitive moat of NVDA versus custom silicon like TPU or Groq. I'm definitely interested in learning more about KV coaching and other memory optimization techniques for Transformers.",
        },
    ]

    await memory.ingest_conversation(user_email=user_email, messages=messages)

    # ------------------------------------------------------------------
    # 2. Run Cognify to extract entities and states
    # ------------------------------------------------------------------
    logger.info("Running Cognify (this may take a minute as it calls the LLM)...")
    await memory.cognify()
    logger.info("Cognify complete.")

    # ------------------------------------------------------------------
    # 3. Retrieve and Verify User Knowledge Context
    # ------------------------------------------------------------------
    logger.info(f"Retrieving knowledge context for {user_email}...")

    # We call the method we just implemented
    context_records = await memory.get_user_knowledge_context(user_email, top_k=10)

    print("\n" + "=" * 80)
    print(f"USER KNOWLEDGE CONTEXT FOR {user_email}")
    print("=" * 80)

    if not context_records:
        print(
            "No context records found. Check if the LLM correctly extracted UserInvestmentInterest/UserLearningInterest nodes."
        )
    else:
        for i, rec in enumerate(context_records):
            print(f"[{i+1}] TYPE: {rec.node_type}")
            print(f"    STATUS: {rec.status}")
            print(f"    REASON: {rec.reason}")
            print(f"    UPDATED: {rec.updated_at}")
            print("-" * 40)

    # Simple validations
    has_investment = any(
        r.node_type == "UserInvestmentInterest" for r in context_records
    )
    has_learning = any(r.node_type == "UserLearningInterest" for r in context_records)

    if has_investment:
        logger.info("SUCCESS: Found investment interest.")
    else:
        logger.warning("FAILED: Could not find investment interest in the graph.")

    if has_learning:
        logger.info("SUCCESS: Found learning interest.")
    else:
        logger.warning("FAILED: Could not find learning interest in the graph.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except Exception:
        logger.exception("Test failed with an error:")
        sys.exit(1)
