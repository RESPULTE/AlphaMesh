
import asyncio
import json
import logging
import os
from datetime import datetime
from logging import INFO

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

#################################################
# CONFIGURATION
#################################################

logging.basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig

# Google API key configuration
api_key = "AIzaSyAnl5kNhS0Xe9ZR-tJYEZ-bb-QA7iOZXuY"

# Graphiti initialization moved to main()
from datetime import datetime, timezone

from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphiti_tracer import GraphitiTracer

async def main():
    tracer = GraphitiTracer()
    
    # Initialize Graphiti with Gemini clients and tracer
    llm_client = GeminiClient(
        config=LLMConfig(api_key=api_key, model="gemini-2.5-flash-lite")
    )
    llm_client.set_tracer(tracer)

    cross_encoder = GeminiRerankerClient(
        config=LLMConfig(api_key=api_key, model="gemini-2.5-flash-lite")
    )
    # Reranker might not support set_tracer directly based on inspection, 
    # but if it does or if we wrapper it:
    if hasattr(cross_encoder, 'set_tracer'):
        cross_encoder.set_tracer(tracer)

    graphiti = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=llm_client,
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=api_key, embedding_model="gemini-embedding-001", embedding_dim=3072
            )
        ),
        cross_encoder=cross_encoder,
        max_coroutines=1,
    )
    
    tracer.instrument_graphiti(graphiti)

    try:
        episodes = [
            {
                "content": "Kamala Harris is the Attorney General of California. She was previously "
                "the district attorney for San Francisco.",
                "type": EpisodeType.text,
                "description": "podcast transcript",
            },
            {
                "content": "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
                "type": EpisodeType.text,
                "description": "podcast transcript",
            },
            {
                "content": {
                    "name": "Gavin Newsom",
                    "position": "Governor",
                    "state": "California",
                    "previous_role": "Lieutenant Governor",
                    "previous_location": "San Francisco",
                },
                "type": EpisodeType.json,
                "description": "podcast metadata",
            },
            {
                "content": {
                    "name": "Gavin Newsom",
                    "position": "Governor",
                    "term_start": "January 7, 2019",
                    "term_end": "Present",
                },
                "type": EpisodeType.json,
                "description": "podcast metadata",
            },
        ]

        # # Add episodes to the graph
        # for i, episode in enumerate(episodes):
        #     await graphiti.add_episode(
        #         name=f"Freakonomics Radio {i}",
        #         episode_body=(
        #             episode["content"]
        #             if isinstance(episode["content"], str)
        #             else json.dumps(episode["content"])
        #         ),
        #         source=episode["type"],
        #         source_description=episode["description"],
        #         reference_time=datetime.now(timezone.utc),
        #     )
        #     print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################
        print("_______________________________________________________________________")

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search("Who was the California Attorney General?")

        # Print search results
        print("\nSearch Results:")
        for result in results:
            print(f"UUID: {result.uuid}")
            print(f"Fact: {result.fact}")
            if hasattr(result, "valid_at") and result.valid_at:
                print(f"Valid from: {result.valid_at}")
            if hasattr(result, "invalid_at") and result.invalid_at:
                print(f"Valid until: {result.invalid_at}")
            print("---")

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################
        print("_______________________________________________________________________")
        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print("\nReranking search results based on graph distance:")
            print(f"Using center node UUID: {center_node_uuid}")

            reranked_results = await graphiti.search(
                "Who was the California Attorney General?",
                center_node_uuid=center_node_uuid,
            )

            # Print reranked search results
            print("\nReranked Search Results:")
            for result in reranked_results:
                print(f"UUID: {result.uuid}")
                print(f"Fact: {result.fact}")
                if hasattr(result, "valid_at") and result.valid_at:
                    print(f"Valid from: {result.valid_at}")
                if hasattr(result, "invalid_at") and result.invalid_at:
                    print(f"Valid until: {result.invalid_at}")
                print("---")
        else:
            print("No results found in the initial search to use as center node.")
        print("_______________________________________________________________________")

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            "\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:"
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query="California Governor",
            config=node_search_config,
        )

        # Print node search results
        print("\nNode Search Results:")
        for node in node_search_results.nodes:
            print(f"Node UUID: {node.uuid}")
            print(f"Node Name: {node.name}")
            node_summary = (
                node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
            )
            print(f"Content Summary: {node_summary}")
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f"Created At: {node.created_at}")
            if hasattr(node, "attributes") and node.attributes:
                print("Attributes:")
                for key, value in node.attributes.items():
                    print(f"  {key}: {value}")
            print("---")

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print("\nConnection closed")
        
        tracer.report()


if __name__ == "__main__":
    asyncio.run(main())
