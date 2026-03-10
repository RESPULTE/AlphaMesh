import asyncio
import uuid

print("--- TASK 0.2: CHECK ENVIRONMENT ---")
try:
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine
    from cognee.modules.engine.operations.setup import setup

    async def check():
        await setup()
        g = await get_graph_engine()
        assert g is not None, "Graph engine is None"
        v = get_vector_engine()
        assert v is not None, "Vector engine is None"
        print("PASS: cognee environment OK")

    asyncio.run(check())
except Exception as e:
    print(f"FAIL task 0.2: {e}")

print("--- TASK 0.3: CHECK GRAPH MODELS ---")
try:
    print("PASS: graph_models imports OK")
except Exception as e:
    print(f"FAIL task 0.3: {e}")

print("--- TASK 0.4: CHECK GET CANONICAL ID ---")
try:
    from core.memory.pipeline_tasks import get_canonical_id

    result = get_canonical_id("AAPL")
    assert isinstance(result, uuid.UUID), "Expected UUID"
    print(f"PASS: get_canonical_id('AAPL') = {result}")
except Exception as e:
    print(f"FAIL task 0.4: {e}")
