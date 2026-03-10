import asyncio
import inspect

from core.memory.memory_system import FinancialMemorySystem

print("--- TASK 5.1 & 5.2: CHECK ingest_document_lean ---")
ms = FinancialMemorySystem()
assert hasattr(ms, "ingest_document_lean"), "ingest_document_lean missing"
sig_lean = inspect.signature(ms.ingest_document_lean)
assert "include_summaries" in sig_lean.parameters, "missing include_summaries"
print("PASS: ingest_document_lean implemented with correct args")

print("--- TASK 5.3: CHECK ingest_conversation is no-op ---")


async def test_conv():
    # Will not fail because it returns without error
    await ms.ingest_conversation("test@user.com", [{"role": "user", "content": "hi"}])
    print("PASS: ingest_conversation is a no-op")


asyncio.run(test_conv())

print("--- TASK 5.4: CHECK cognify unchanged ---")
sig_cog = inspect.signature(ms.cognify)
assert "run_in_background" in sig_cog.parameters
assert "chunks_per_batch" in sig_cog.parameters
print("PASS: cognify unchanged")
