from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from langchain_core.documents import Document

from core.config import settings
from core.memory.retrieval.dual_store_retriever import DualStoreRetriever
from core.memory.retrieval.models import RetrievedChunk, RewrittenQueries
from core.memory.retrieval.reranker import CompositePrefilter
from core.memory.retrieval.tracing import (
    NetworkXRetrievalTraceSink,
    RetrievalTraceEvent,
)

pytest.importorskip("networkx")


class FakeChromaAdapter:
    def __init__(self) -> None:
        self._query_map: Dict[str, List[Tuple[Document, float]]] = {
            "company query": [
                (
                    Document(
                        page_content="company seed one",
                        metadata={"chunk_id": "c-seed-1"},
                        id="c-seed-1",
                    ),
                    0.92,
                ),
                (
                    Document(
                        page_content="company seed two",
                        metadata={"chunk_id": "c-seed-2"},
                        id="c-seed-2",
                    ),
                    0.88,
                ),
            ],
            "market query": [
                (
                    Document(
                        page_content="market seed one",
                        metadata={"chunk_id": "m-seed-1"},
                        id="m-seed-1",
                    ),
                    0.9,
                )
            ],
        }

    async def query(self, *, query_text: str, **kwargs):
        _ = kwargs
        return list(self._query_map.get(query_text, []))


class FakeNeo4jAdapter:
    async def get_entities_for_chunks(self, chunk_ids: List[str]) -> List[dict]:
        row_map = {
            "c-seed-1": {
                "entity_id": "e-comp-1",
                "entity_name": "Apple",
                "entity_type": "Company",
            },
            "c-seed-2": {
                "entity_id": "e-comp-2",
                "entity_name": "iPhone",
                "entity_type": "FinancialConcept",
            },
            "m-seed-1": {
                "entity_id": "e-mkt-1",
                "entity_name": "NASDAQ",
                "entity_type": "Market",
            },
        }
        rows: List[dict] = []
        for chunk_id in chunk_ids:
            mapped = row_map.get(chunk_id)
            if mapped is None:
                continue
            rows.append({**mapped, "source_chunk_id": chunk_id})
        return rows

    async def get_entity_neighbors(
        self, entity_ids: List[str], exclude_ids: List[str]
    ) -> List[dict]:
        _ = exclude_ids
        rows: List[dict] = []
        for entity_id in entity_ids:
            if entity_id == "e-comp-1":
                rows.append(
                    {
                        "source_entity_id": "e-comp-1",
                        "source_entity_name": "Apple",
                        "neighbor_entity_id": "e-common-1",
                        "neighbor_name": "Supply Chain",
                        "neighbor_type": "FinancialConcept",
                        "relationship_type": "RELATED_TO",
                        "reason": "Apple depends on supply chain continuity.",
                    }
                )
                rows.append(
                    {
                        "source_entity_id": "e-comp-1",
                        "source_entity_name": "Apple",
                        "neighbor_entity_id": "e-unused-1",
                        "neighbor_name": "Unused Entity",
                        "neighbor_type": "FinancialConcept",
                        "relationship_type": "RELATED_TO",
                        "reason": "Synthetic non-contributing neighbor for trace pruning.",
                    }
                )
            elif entity_id == "e-comp-2":
                rows.append(
                    {
                        "source_entity_id": "e-comp-2",
                        "source_entity_name": "iPhone",
                        "neighbor_entity_id": "e-common-2",
                        "neighbor_name": "Consumer Demand",
                        "neighbor_type": "FinancialConcept",
                        "relationship_type": "RELATED_TO",
                        "reason": "iPhone demand drives revenue outlook.",
                    }
                )
                rows.append(
                    {
                        "source_entity_id": "e-comp-2",
                        "source_entity_name": "iPhone",
                        "neighbor_entity_id": "e-common-1",
                        "neighbor_name": "Supply Chain",
                        "neighbor_type": "FinancialConcept",
                        "relationship_type": "RELATED_TO",
                        "reason": "Device availability is tied to components supply.",
                    }
                )
            elif entity_id == "e-mkt-1":
                rows.append(
                    {
                        "source_entity_id": "e-mkt-1",
                        "source_entity_name": "NASDAQ",
                        "neighbor_entity_id": "e-common-1",
                        "neighbor_name": "Supply Chain",
                        "neighbor_type": "FinancialConcept",
                        "relationship_type": "RELATED_TO",
                        "reason": "Broad supply chain risk impacts index constituents.",
                    }
                )
        return rows

    async def get_chunks_for_entities(
        self, entity_ids: List[str], exclude_chunk_ids: List[str]
    ) -> List[dict]:
        rows: List[dict] = []
        for entity_id in entity_ids:
            if entity_id == "e-common-1":
                rows.extend(
                    [
                        {
                            "chunk_id": "graph-shared",
                            "chunk_text": "shared graph chunk",
                            "chunk_index": 7,
                            "document_id": "doc-g-1",
                            "article_title": "Graph Shared",
                            "source_url": "https://example.com/shared",
                            "published_at": None,
                            "extraction_status": "EXTRACTED",
                            "supporting_entity_id": "e-common-1",
                        }
                    ]
                )
            elif entity_id == "e-common-2":
                rows.extend(
                    [
                        {
                            "chunk_id": "graph-shared",
                            "chunk_text": "shared graph chunk",
                            "chunk_index": 7,
                            "document_id": "doc-g-1",
                            "article_title": "Graph Shared",
                            "source_url": "https://example.com/shared",
                            "published_at": None,
                            "extraction_status": "EXTRACTED",
                            "supporting_entity_id": "e-common-2",
                        },
                        {
                            "chunk_id": "graph-company",
                            "chunk_text": "company-only graph chunk",
                            "chunk_index": 8,
                            "document_id": "doc-g-2",
                            "article_title": "Graph Company",
                            "source_url": "https://example.com/company",
                            "published_at": None,
                            "extraction_status": "PENDING",
                            "supporting_entity_id": "e-common-2",
                        },
                    ]
                )

        excluded = set(exclude_chunk_ids or [])
        return [row for row in rows if row["chunk_id"] not in excluded]


class FakeReranker:
    def __init__(
        self,
        *,
        delay_seconds: float = 0.0,
        failing_domains: set[str] | None = None,
    ) -> None:
        self._delay_seconds = delay_seconds
        self._failing_domains = failing_domains or set()

    async def rank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        _ = query
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)

        domain = (chunks[0].domain or "").strip() if chunks else ""
        if domain in self._failing_domains:
            raise RuntimeError(f"simulated reranker failure for domain={domain}")

        ranked: List[RetrievedChunk] = []
        for idx, chunk in enumerate(chunks):
            base_score = chunk.relevance_score if chunk.relevance_score is not None else 0.0
            ranked.append(
                chunk.model_copy(
                    update={
                        "relevance_score": float(base_score + (len(chunks) - idx)),
                        "relevance_source": "jina",
                    }
                )
            )
        return ranked


def _build_retriever(
    trace_sink: NetworkXRetrievalTraceSink | None = None,
    *,
    reranker: FakeReranker | None = None,
) -> DualStoreRetriever:
    prefilter = CompositePrefilter(alpha=0.8, beta=0.2, prefilter_k=10)
    return DualStoreRetriever(
        neo4j_adapter=FakeNeo4jAdapter(),
        chroma_adapter=FakeChromaAdapter(),
        prefilter=prefilter,
        reranker=reranker or FakeReranker(),
        trace_sink=trace_sink,
    )


def _chunk_ids(chunks: List[RetrievedChunk]) -> List[str]:
    return [chunk.chunk_id for chunk in chunks]


@pytest.mark.asyncio
async def test_retrieve_output_unchanged_when_tracing_disabled() -> None:
    without_tracing = _build_retriever(trace_sink=None)
    with_tracing = _build_retriever(trace_sink=NetworkXRetrievalTraceSink(max_runs=20))

    chunks_without = await without_tracing.retrieve("company query")
    chunks_with = await with_tracing.retrieve("company query", run_id="run-with")

    assert sorted(_chunk_ids(chunks_without)) == sorted(_chunk_ids(chunks_with))

    shared_chunk = next(chunk for chunk in chunks_with if chunk.chunk_id == "graph-shared")
    assert sorted(shared_chunk.metadata.get("supporting_entity_ids", [])) == [
        "e-common-1",
        "e-common-2",
    ]

    relationships = shared_chunk.metadata.get("selected_neighbor_relationships", [])
    assert relationships
    assert {
        "frontier_name": "Apple",
        "selected_neighbor_name": "Supply Chain",
        "relationship_type": "RELATED_TO",
        "reason": "Apple depends on supply chain continuity.",
    } in relationships
    assert {
        "frontier_name": "iPhone",
        "selected_neighbor_name": "Supply Chain",
        "relationship_type": "RELATED_TO",
        "reason": "Device availability is tied to components supply.",
    } in relationships


@pytest.mark.asyncio
async def test_trace_graph_contains_layered_vector_and_graph_edges() -> None:
    sink = NetworkXRetrievalTraceSink(max_runs=20)
    retriever = _build_retriever(trace_sink=sink)

    await retriever.retrieve(
        "company query",
        run_id="run-company",
        domain="company",
        parent_run_id="run-parent",
    )
    graph = sink.get_run_graph("run-company")
    assert graph is not None
    assert graph.graph["parent_run_id"] == "run-parent"
    assert graph.graph["domain"] == "company"

    edge_tuples = [(u, v, d) for u, v, d in graph.edges(data=True)]
    assert any(
        graph.nodes[u].get("node_type") == "query"
        and graph.nodes[v].get("node_type") == "chunk"
        and d.get("stage") == "vector_seed"
        and d.get("layer") == 0
        for u, v, d in edge_tuples
    )
    assert any(
        graph.nodes[u].get("node_type") == "chunk"
        and graph.nodes[v].get("node_type") == "entity"
        and d.get("stage") == "seed_entities"
        and d.get("layer") == 0
        for u, v, d in edge_tuples
    )
    assert any(
        graph.nodes[u].get("node_type") == "entity"
        and graph.nodes[v].get("node_type") == "chunk"
        and d.get("stage") == "frontier_chunks"
        and d.get("layer") == 1
        for u, v, d in edge_tuples
    )

    graph_chunk_nodes = [
        attrs
        for _node_id, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "chunk" and attrs.get("source") == "graph"
    ]
    assert graph_chunk_nodes
    assert all(attrs.get("domain") == "company" for attrs in graph_chunk_nodes)
    assert all(int(attrs.get("layer") or 0) >= 1 for attrs in graph_chunk_nodes)


@pytest.mark.asyncio
async def test_comprehensive_retrieve_creates_parent_and_child_runs() -> None:
    sink = NetworkXRetrievalTraceSink(max_runs=20)
    retriever = _build_retriever(trace_sink=sink)

    rewritten = RewrittenQueries(
        company_query="company query",
        market_query="market query",
        active_domains=["company", "market"],
    )

    context = await retriever.comprehensive_retrieve(rewritten)
    assert context.chunks

    run_ids = sink.list_runs()
    assert len(run_ids) == 3

    parent_run_id = None
    for run_id in run_ids:
        graph = sink.get_run_graph(run_id)
        if graph is not None and graph.graph.get("domain") == "comprehensive":
            parent_run_id = run_id
            break
    assert parent_run_id is not None

    child_graphs = []
    for run_id in run_ids:
        graph = sink.get_run_graph(run_id)
        if graph is not None and graph.graph.get("parent_run_id") == parent_run_id:
            child_graphs.append(graph)
    assert len(child_graphs) == 2

    parent_graph = sink.get_run_graph(parent_run_id)
    assert parent_graph is not None
    assert any(
        attrs.get("prefilter_rank") is not None
        for _, attrs in parent_graph.nodes(data=True)
    )


@pytest.mark.asyncio
async def test_comprehensive_retrieve_parallel_rerank_and_fallback() -> None:
    retriever = _build_retriever(
        reranker=FakeReranker(delay_seconds=0.3, failing_domains={"market"})
    )
    rewritten = RewrittenQueries(
        company_query="company query",
        market_query="market query",
        active_domains=["company", "market"],
    )

    started = time.monotonic()
    context = await retriever.comprehensive_retrieve(rewritten)
    elapsed = time.monotonic() - started

    assert context.chunks
    # Two domain rerank calls are each delayed by 0.3s. Serial execution would
    # exceed ~0.6s in this fixture, so this bound validates parallel reranking.
    assert elapsed < 0.55

    company_chunks = [chunk for chunk in context.chunks if chunk.domain == "company"]
    market_chunks = [chunk for chunk in context.chunks if chunk.domain == "market"]
    assert company_chunks
    assert market_chunks
    assert all(chunk.relevance_source == "jina" for chunk in company_chunks)
    # Market rerank is forced to fail, so fallback should avoid Jina source labels.
    assert any(chunk.relevance_source != "jina" for chunk in market_chunks)


def test_trace_sink_lru_eviction() -> None:
    sink = NetworkXRetrievalTraceSink(max_runs=2)
    sink.record(
        RetrievalTraceEvent(
            run_id="run-1",
            parent_run_id=None,
            domain="company",
            stage="vector_seed",
            hop=0,
            layer=0,
            payload={"query": "q1", "chunks": []},
        )
    )
    sink.record(
        RetrievalTraceEvent(
            run_id="run-2",
            parent_run_id=None,
            domain="company",
            stage="vector_seed",
            hop=0,
            layer=0,
            payload={"query": "q2", "chunks": []},
        )
    )
    sink.record(
        RetrievalTraceEvent(
            run_id="run-3",
            parent_run_id=None,
            domain="company",
            stage="vector_seed",
            hop=0,
            layer=0,
            payload={"query": "q3", "chunks": []},
        )
    )

    assert sink.list_runs() == ["run-2", "run-3"]


@pytest.mark.asyncio
async def test_trace_visualization_artifacts_are_exported() -> None:
    sink = NetworkXRetrievalTraceSink(max_runs=20)
    retriever = _build_retriever(trace_sink=sink)

    await retriever.retrieve("company query", run_id="viz-run", domain="company")

    artifact_dir = Path("data/retrieval_trace_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    html_path = artifact_dir / "retrieval_trace_demo.html"
    graphml_path = artifact_dir / "retrieval_trace_demo.graphml"
    sink.export_html("viz-run", str(html_path))
    sink.export_graphml("viz-run", str(graphml_path))

    assert html_path.exists()
    assert graphml_path.exists()
    assert html_path.stat().st_size > 0
    assert graphml_path.stat().st_size > 0


@pytest.mark.asyncio
async def test_export_prunes_to_paths_that_lead_to_chunk_hits(tmp_path: Path) -> None:
    sink = NetworkXRetrievalTraceSink(max_runs=20)
    retriever = _build_retriever(trace_sink=sink)

    run_id = "prune-run"
    await retriever.retrieve("company query", run_id=run_id, domain="company")

    raw_graph = sink.get_run_graph(run_id)
    assert raw_graph is not None
    assert raw_graph.has_node("entity:e-unused-1")

    node_link_path = tmp_path / "pruned.json"
    sink.export_node_link_json(run_id, str(node_link_path))
    payload = json.loads(node_link_path.read_text(encoding="utf-8"))

    exported_nodes = {str(node["id"]) for node in payload.get("nodes", [])}
    assert "entity:e-unused-1" not in exported_nodes

    exported_links = payload.get("links", [])
    assert any(str(link.get("stage") or "") == "vector_seed" for link in exported_links)
    assert any(
        str(link.get("stage") or "") == "frontier_chunks" for link in exported_links
    )


@pytest.mark.asyncio
async def test_export_html_uses_entity_names_and_uniform_node_size(tmp_path: Path) -> None:
    sink = NetworkXRetrievalTraceSink(max_runs=20)
    retriever = _build_retriever(trace_sink=sink)

    run_id = "viz-label-run"
    await retriever.retrieve("company query", run_id=run_id, domain="company")

    html_path = tmp_path / "trace.html"
    sink.export_html(run_id, str(html_path))
    html = html_path.read_text(encoding="utf-8")

    assert '"label": "Apple"' in html
    assert "Apple\\nI0" not in html
    assert "id=entity:e-comp-1" in html
    assert '"size": 23' in html


def test_export_is_empty_when_no_chunk_hit_paths_exist(tmp_path: Path) -> None:
    sink = NetworkXRetrievalTraceSink(max_runs=20)
    run_id = "no-hit-run"
    sink.record(
        RetrievalTraceEvent(
            run_id=run_id,
            parent_run_id=None,
            domain="comprehensive",
            stage="prefilter_output",
            hop=0,
            layer=0,
            payload={
                "ranked_chunks": [
                    {
                        "chunk_id": "chunk-only",
                        "domain": "comprehensive",
                        "chunk_text": "prefilter only",
                        "rank": 1,
                        "selected": True,
                    }
                ]
            },
        )
    )

    node_link_path = tmp_path / "no_hit.json"
    sink.export_node_link_json(run_id, str(node_link_path))
    payload = json.loads(node_link_path.read_text(encoding="utf-8"))
    assert payload.get("nodes", []) == []
    assert payload.get("links", []) == []


@pytest.mark.asyncio
async def test_auto_export_is_toggled_by_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = NetworkXRetrievalTraceSink(max_runs=20)
    retriever = _build_retriever(trace_sink=sink)

    monkeypatch.setattr(settings, "RETRIEVAL_TRACE_AUTO_EXPORT", True)
    monkeypatch.setattr(settings, "RETRIEVAL_TRACE_AUTO_EXPORT_DIR", str(tmp_path))

    run_id = "auto-export-run"
    await retriever.retrieve("company query", run_id=run_id, domain="company")

    html_path = tmp_path / f"company_{run_id}.html"
    graphml_path = tmp_path / f"company_{run_id}.graphml"
    assert html_path.exists()
    assert graphml_path.exists()
    assert html_path.stat().st_size > 0
    assert graphml_path.stat().st_size > 0
