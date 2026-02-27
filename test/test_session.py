import asyncio

import cognee
from cognee import visualize_graph
from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
    persist_sessions_in_knowledge_graph_pipeline,
)
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user, create_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.permissions.methods import give_permission_on_dataset
from cognee.modules.data.methods import create_authorized_dataset
logger = get_logger("conversation_session_persistence_example")
import os
from cognee.modules.engine.operations.setup import setup
async def main():
    # NOTE: CACHING has to be enabled for this example to work
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"
    await setup()
    a = await create_user("different_user@gmail.com", "123456789",  is_superuser=True)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text_1 = "Cognee is a solution that can build knowledge graph from text, creating an AI memory system"
    text_2 = "Germany is a country located next to the Netherlands"

    await cognee.add([text_1, text_2], dataset_id="main_dataset")
    await cognee.cognify()

    question = "What can I use to create a knowledge graph?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
    )
    print("\nSession ID: default_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "You sure about that?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=question
    )
    print("\nSession ID: default_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "This is awesome!"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=question
    )
    print("\nSession ID: default_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "Where is Germany?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
        session_id="different_session",
    )
    print("\nSession ID: different_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "Next to which country again?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
        session_id="different_session",
    )
    print("\nSession ID: different_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "So you remember everything I asked from you?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
        session_id="different_session",
    )
    print("\nSession ID: different_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    await persist_sessions_in_knowledge_graph_pipeline(
        user=await get_default_user(),
        session_ids=["default_session"],
        dataset="main_dataset"
    )
    

    
    await persist_sessions_in_knowledge_graph_pipeline(
        user=a,
        session_ids=["different_session"],
        dataset="main_dataset"
    )

    await visualize_graph()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())