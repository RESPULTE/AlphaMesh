"""Integration tests for the AlphaMesh dual-store ingestion pipeline.

These tests require real Neo4j and ChromaDB instances to be running and reachable
via environment variables used by `core.config.settings`.

Run with:
    pytest -q

The tests ingest a small article, then assert data parity between the vector
store (ChromaDB) and the graph store (Neo4j), including metadata and linkage.
"""

from __future__ import annotations

import pytest

from core.services import service_manager


@pytest.mark.asyncio
async def test_dual_store_ingestion_writes_to_vector_and_graph_store() -> None:
    """Ingest a sample article and verify it exists in both stores."""

    try:
        # --- Arrange ---
        articles = [
            {
                "title": "AlphaMesh Test Article (integration)",
                "url": "https://example.com/integration-article",
                "publishedAt": "2026-03-10T10:00:00Z",
                "description": "Integration test description.",
                "content": "Integration test content. This is a deterministic payload.",
            }
        ]

        ingestor = service_manager.get_ingestor()
        chroma_adapter = service_manager.get_chroma_adapter()
        neo4j_adapter = service_manager.get_neo4j_adapter()
        nodeset_manager = service_manager.get_nodeset_manager()

        # --- Act ---
        chunk_ids, _chunks = await ingestor.ingest_articles(articles)

        # --- Assert ---
        assert chunk_ids, "Expected at least one chunk ID to be returned."

        # Validate vector store (Chroma) contents
        chroma_result = await chroma_adapter.get_by_ids(chunk_ids)
        chroma_ids = set(chroma_result.get("ids") or [])
        assert set(chunk_ids) == chroma_ids

        global_anchor_id = await nodeset_manager.get_global_financial_events_id()

        metadatas = chroma_result.get("metadatas") or []
        # All returned metadatas should include the chunk id, article title, and source URL.
        for metadata in metadatas:
            assert metadata["chunk_id"] in chunk_ids
            assert metadata["document_id"]
            assert metadata["article_title"] == "AlphaMesh Test Article (integration)"
            assert metadata["source_url"] == "https://example.com/integration-article"

            # Verify linkage to nodeset IDs (global anchor) exists.
            assert "nodeset_ids" in metadata
            assert global_anchor_id in metadata["nodeset_ids"]

        # Validate graph store (Neo4j) contents
        cypher_chunks = (
            "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS id, c.text AS text"
        )
        neo4j_data = await neo4j_adapter._execute_read(
            cypher_chunks, {"ids": chunk_ids}
        )
        neo4j_ids = {record["id"] for record in neo4j_data}
        assert set(chunk_ids) == neo4j_ids

        # Ensure the stored text includes the original content
        for record in neo4j_data:
            assert "Integration test content" in record["text"]

        # Verify document linkage (Chunk -> Document)
        cypher_docs = (
            "MATCH (c:Chunk) WHERE c.id IN $ids "
            "MATCH (c)-[:BELONGS_TO_DOCUMENT]->(d:Document) "
            "RETURN DISTINCT d.id AS id, d.title AS title, d.source_url AS source_url"
        )
        doc_data = await neo4j_adapter._execute_read(cypher_docs, {"ids": chunk_ids})
        assert doc_data, "Expected a document node linked from the chunk."
        for d in doc_data:
            assert d["title"] == "AlphaMesh Test Article (integration)"
            assert d["source_url"] == "https://example.com/integration-article"

        # Verify global anchor linkage exists
        cypher_anchor = (
            "MATCH (d:Document)-[:BELONGS_TO_NODESET]->(g:NodeSet {id: $anchor_id}) "
            "WHERE d.id IN $doc_ids RETURN COUNT(d) AS count"
        )
        doc_ids = [d["id"] for d in doc_data]
        anchor_data = await neo4j_adapter._execute_read(
            cypher_anchor, {"anchor_id": global_anchor_id, "doc_ids": doc_ids}
        )
        anchor_count = anchor_data[0]["count"] if anchor_data else 0
        assert anchor_count == len(doc_ids)
    except Exception as exc:
        pytest.fail(f"Dual-store ingestion failed: {exc}")

    finally:
        # --- Cleanup ---
        # Remove the inserted chunks and documents so tests can be re-run safely.
        try:
            await chroma_adapter.delete_by_ids(chunk_ids)
        except Exception:
            # Best effort cleanup - do not fail tests on cleanup.
            pass

        try:
            cleanup_cypher = (
                "MATCH (c:Chunk) WHERE c.id IN $ids "
                "OPTIONAL MATCH (c)-[:BELONGS_TO_DOCUMENT]->(d:Document) "
                "DETACH DELETE c, d"
            )
            await neo4j_adapter._execute_write(cleanup_cypher, {"ids": chunk_ids})
        except Exception:
            pass


@pytest.mark.asyncio
async def test_chroma_adapter_client_is_cached() -> None:
    """Ensure Chroma adapter caches the client between calls."""

    chroma_adapter = service_manager.get_chroma_adapter()

    first_client = await chroma_adapter._get_vectorstore()
    second_client = await chroma_adapter._get_vectorstore()

    assert first_client is second_client


@pytest.mark.asyncio
async def test_dual_store_ingestion_creates_multiple_chunks_when_overflowing() -> None:
    """Ingest a large article and ensure it gets split into multiple chunks."""
    try:
        long_content = "Lorem ipsum " * 200  # ~2200 characters, exceeding chunk_size
        articles = [
            {
                "title": "AlphaMesh Overflow Article (integration)",
                "url": "https://example.com/overflow-article",
                "publishedAt": "2026-03-10T10:00:00Z",
                "description": "Overflow test description.",
                "content": long_content,
            }
        ]

        ingestor = service_manager.get_ingestor()
        chroma_adapter = service_manager.get_chroma_adapter()
        neo4j_adapter = service_manager.get_neo4j_adapter()
        chunk_ids, _chunks = await ingestor.ingest_articles(articles)

        assert (
            len(chunk_ids) > 1
        ), "Expected the long article to be split into multiple chunks."

        # Ensure both stores contain the same chunk IDs
        chroma_ids = set((await chroma_adapter.get_by_ids(chunk_ids)).get("ids") or [])
        assert set(chunk_ids) == chroma_ids

        cypher_chunks = "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS id"
        neo4j_chunks = await neo4j_adapter._execute_read(
            cypher_chunks, {"ids": chunk_ids}
        )
        neo4j_ids = {record["id"] for record in neo4j_chunks}
        assert set(chunk_ids) == neo4j_ids
    except Exception as exc:
        pytest.fail(f"Dual-store ingestion with overflow failed: {exc}")

    finally:
        # Cleanup
        try:
            await chroma_adapter.delete_by_ids(chunk_ids)
        except Exception:
            pass

        try:
            cleanup_cypher = (
                "MATCH (c:Chunk) WHERE c.id IN $ids "
                "OPTIONAL MATCH (c)-[:BELONGS_TO_DOCUMENT]->(d:Document) "
                "DETACH DELETE c, d"
            )
            await neo4j_adapter._execute_write(cleanup_cypher, {"ids": chunk_ids})
        except Exception:
            pass


