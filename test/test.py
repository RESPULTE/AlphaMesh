"""
Example usage of the RAG System

This file demonstrates all major features of the RAG system including:
- Document insertion (foreground and background)
- Entity management (create, edit, delete, merge)
- File processing
- Querying with different modes
- Citation tracking
- Data export
"""

import asyncio
import logging
from pathlib import Path

from core.memory.light_rag import ProcessingMode

# Assuming this structure
from core.updated_servcies import service_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basic usage example: initialize and query"""
    logger.info("=== Basic Usage Example ===")

    # Initialize RAG system
    rag = await service_manager.get_rag_system(
        working_dir="./rag_storage_example",
        workspace="example_workspace",
        enable_citations=True,
        enable_reranking=True,
    )

    # Insert some documents
    result = await rag.insert_documents(
        texts=[
            "LightRAG is a simple and fast retrieval-augmented generation system.",
            "It uses knowledge graphs to organize information efficiently.",
            "The system supports multiple file types including PDF, DOCX, and TXT.",
        ],
        mode=ProcessingMode.FOREGROUND,
        file_paths=["doc1.txt", "doc2.txt", "doc3.txt"],
    )
    logger.info(f"Insertion result: {result}")

    # Query the system
    query_result = await rag.query(
        query_text="What is LightRAG?", mode="hybrid", enable_rerank=True
    )
    logger.info(f"Query result: {query_result['result']}")

    return rag


async def example_background_processing():
    """Example of background document processing"""
    logger.info("=== Background Processing Example ===")

    rag = await service_manager.get_rag_system(working_dir="./rag_storage_background")

    # Large batch of documents for background processing
    large_batch = [
        f"Document {i}: This is a sample document about topic {i % 5}."
        for i in range(100)
    ]

    # Insert in background (non-blocking)
    result = await rag.insert_documents(
        texts=large_batch, mode=ProcessingMode.BACKGROUND
    )
    logger.info(f"Background insertion queued: {result}")

    # You can continue with other work while processing happens in background
    logger.info("Continuing with other work...")

    # Check status
    status = await rag.get_status()
    logger.info(f"System status: {status}")

    # Wait a bit for processing to complete
    await asyncio.sleep(5)

    # Check status again
    status = await rag.get_status()
    logger.info(f"System status after processing: {status}")

    return rag


async def example_entity_management():
    """Example of entity management operations"""
    logger.info("=== Entity Management Example ===")

    rag = await service_manager.get_rag_system(working_dir="./rag_storage_entities")

    # Create entities
    entity1 = await rag.create_entity(
        entity_name="LightRAG",
        attributes={
            "description": "A fast and simple RAG system",
            "entity_type": "SOFTWARE",
        },
    )
    logger.info(f"Created entity: {entity1}")

    entity2 = await rag.create_entity(
        entity_name="Knowledge Graph",
        attributes={
            "description": "A graph-based knowledge representation",
            "entity_type": "CONCEPT",
        },
    )
    logger.info(f"Created entity: {entity2}")

    # Create relationship
    relation = await rag.create_relation(
        source_entity="LightRAG",
        target_entity="Knowledge Graph",
        attributes={
            "description": "LightRAG uses Knowledge Graphs",
            "keywords": "uses implements",
            "weight": 1.5,
        },
    )
    logger.info(f"Created relation: {relation}")

    # Edit entity
    updated_entity = await rag.edit_entity(
        entity_name="LightRAG",
        attributes={
            "description": "A fast, simple, and efficient RAG system with knowledge graph support",
            "entity_type": "SOFTWARE",
            "version": "2.0",
        },
    )
    logger.info(f"Updated entity: {updated_entity}")

    # Merge entities (for deduplication)
    # First create duplicate entities
    await rag.create_entity(
        entity_name="Light RAG",
        attributes={"description": "A RAG system", "entity_type": "SOFTWARE"},
    )

    await rag.create_entity(
        entity_name="LightRAG System",
        attributes={"description": "RAG implementation", "entity_type": "SOFTWARE"},
    )

    # Merge duplicates
    merge_result = await rag.merge_entities(
        source_entities=["Light RAG", "LightRAG System"],
        target_entity="LightRAG",
        merge_strategy={"description": "concatenate", "entity_type": "keep_first"},
    )
    logger.info(f"Merge result: {merge_result}")

    return rag


async def example_file_processing():
    """Example of processing different file types"""
    logger.info("=== File Processing Example ===")

    rag = await service_manager.get_rag_system(working_dir="./rag_storage_files")

    # Create sample files for demonstration
    sample_dir = Path("./sample_documents")
    sample_dir.mkdir(exist_ok=True)

    # Create a sample text file
    sample_txt = sample_dir / "sample.txt"
    sample_txt.write_text(
        "This is a sample document about artificial intelligence.\n"
        "AI systems can process large amounts of data efficiently."
    )

    # Process the file
    result = await rag.process_file(
        file_path=str(sample_txt),
        mode=ProcessingMode.FOREGROUND,
        doc_id="sample_doc_001",
    )
    logger.info(f"File processing result: {result}")

    # Query about the processed file
    query_result = await rag.query(query_text="What can AI systems do?", mode="hybrid")
    logger.info(f"Query result: {query_result['result']}")

    return rag


async def example_query_modes():
    """Example of different query modes"""
    logger.info("=== Query Modes Example ===")

    rag = await service_manager.get_rag_system(working_dir="./rag_storage_query")

    # Insert sample data
    await rag.insert_documents(
        texts=[
            "The solar system consists of the Sun and celestial bodies.",
            "Earth is the third planet from the Sun.",
            "Mars is known as the Red Planet due to its iron oxide surface.",
            "Jupiter is the largest planet in our solar system.",
        ]
    )

    query = "Tell me about planets in the solar system"

    # Try different query modes
    modes = ["local", "global", "hybrid", "naive", "mix"]

    for mode in modes:
        logger.info(f"\n--- Query Mode: {mode} ---")
        result = await rag.query(
            query_text=query, mode=mode, top_k=60, enable_rerank=True
        )
        logger.info(f"Result: {result['result'][:200]}...")  # First 200 chars

    return rag


async def example_data_export():
    """Example of exporting knowledge graph data"""
    logger.info("=== Data Export Example ===")

    rag = await service_manager.get_rag_system(working_dir="./rag_storage_export")

    # Insert some data
    await rag.insert_documents(
        texts=[
            "Python is a high-level programming language.",
            "JavaScript is widely used for web development.",
            "Both Python and JavaScript are popular among developers.",
        ]
    )

    # Export in different formats
    export_dir = Path("./exports")
    export_dir.mkdir(exist_ok=True)

    # CSV export
    result_csv = await rag.export_data(
        output_path=str(export_dir / "knowledge_graph.csv"),
        file_format="csv",
        include_vector_data=False,
    )
    logger.info(f"CSV export: {result_csv}")

    # Excel export
    result_xlsx = await rag.export_data(
        output_path=str(export_dir / "knowledge_graph.xlsx"),
        file_format="excel",
        include_vector_data=False,
    )
    logger.info(f"Excel export: {result_xlsx}")

    # Markdown export
    result_md = await rag.export_data(
        output_path=str(export_dir / "knowledge_graph.md"),
        file_format="md",
        include_vector_data=False,
    )
    logger.info(f"Markdown export: {result_md}")

    return rag


async def example_cache_management():
    """Example of cache management"""
    logger.info("=== Cache Management Example ===")

    rag = await service_manager.get_rag_system(working_dir="./rag_storage_cache")

    # Make some queries to populate cache
    await rag.insert_documents(texts=["Cache management is important for performance."])

    await rag.query("What is important for performance?", mode="local")
    await rag.query("Tell me about cache", mode="global")

    # Clear specific cache
    result = await rag.clear_cache(modes=["local"])
    logger.info(f"Cleared local cache: {result}")

    # Clear all cache
    result = await rag.clear_cache()
    logger.info(f"Cleared all cache: {result}")

    return rag


async def example_complete_workflow():
    """Complete workflow example combining multiple features"""
    logger.info("=== Complete Workflow Example ===")

    # Initialize system
    rag = await service_manager.get_rag_system(
        working_dir="./rag_storage_complete",
        workspace="production",
        enable_citations=True,
        enable_entity_merging=True,
        enable_reranking=True,
    )

    try:
        # Step 1: Insert documents
        logger.info("Step 1: Inserting documents...")
        await rag.insert_documents(
            texts=[
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Natural language processing enables computers to understand text.",
            ],
            file_paths=["ml_intro.txt", "dl_basics.txt", "nlp_guide.txt"],
        )

        # Step 2: Create custom entities
        logger.info("Step 2: Creating entities...")
        await rag.create_entity(
            entity_name="Machine Learning",
            attributes={
                "description": "A field of AI focusing on learning from data",
                "entity_type": "CONCEPT",
            },
        )

        await rag.create_entity(
            entity_name="Deep Learning",
            attributes={
                "description": "Advanced ML using neural networks",
                "entity_type": "CONCEPT",
            },
        )

        # Step 3: Create relationships
        logger.info("Step 3: Creating relationships...")
        await rag.create_relation(
            source_entity="Deep Learning",
            target_entity="Machine Learning",
            attributes={
                "description": "Deep Learning is a subset of Machine Learning",
                "keywords": "subset specialization",
                "weight": 2.0,
            },
        )

        # Step 4: Query with different modes
        logger.info("Step 4: Querying...")
        query_result = await rag.query(
            query_text="How does deep learning relate to machine learning?",
            mode="hybrid",
            enable_rerank=True,
        )
        logger.info(f"Query result: {query_result['result']}")

        # Step 5: Export data
        logger.info("Step 5: Exporting data...")
        await rag.export_data(
            output_path="./complete_workflow_export.csv", file_format="csv"
        )

        # Step 6: Get system status
        status = await rag.get_status()
        logger.info(f"Final system status: {status}")

    finally:
        # Always shutdown gracefully
        logger.info("Shutting down...")
        await service_manager.shutdown_rag_system()


async def main():
    """Run all examples"""
    try:
        # Run examples one by one
        # await example_basic_usage()
        # await example_background_processing()
        # await example_entity_management()
        # await example_file_processing()
        # await example_query_modes()
        # await example_data_export()
        # await example_cache_management()

        # Run complete workflow
        await example_complete_workflow()

    except Exception as e:
        logger.error(f"Error in examples: {e}", exc_info=True)
    finally:
        # Ensure proper cleanup
        await service_manager.shutdown_rag_system()


if __name__ == "__main__":
    asyncio.run(main())
