from __future__ import annotations

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
                        "neighbor_entity_id": "e-common-1",
                        "neighbor_name": "Supply Chain",
                        "neighbor_type": "FinancialConcept",
                        "relationship_type": "RELATED_TO",
                    }
                )
            elif entity_id == "e-comp-2":
                rows.append(
                    {
                        "source_entity_id": "e-comp-2",
                        "neighbor_entity_id": "e-common-2",
                        "neighbor_name": "Consumer Demand",
                        "neighbor_type": "FinancialConcept",
                        "relationship_type": "RELATED_TO",
                    }
                )
            elif entity_id == "e-mkt-1":
                rows.append(
                    {
                        "source_entity_id": "e-mkt-1",
                        "neighbor_entity_id": "e-common-1",
                        "neighbor_name": "Supply Chain",
                        "neighbor_type": "FinancialConcept",
                        "relationship_type": "RELATED_TO",
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


def _build_retriever(
    trace_sink: NetworkXRetrievalTraceSink | None = None,
) -> DualStoreRetriever:
    return DualStoreRetriever(
        neo4j_adapter=FakeNeo4jAdapter(),
        chroma_adapter=FakeChromaAdapter(),
        prefilter=CompositePrefilter(alpha=0.8, beta=0.2, prefilter_k=10),
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
        u == "query:run-company"
        and v == "chunk:c-seed-1"
        and d.get("stage") == "vector_seed"
        and d.get("layer") == 0
        for u, v, d in edge_tuples
    )
    assert any(
        u == "chunk:c-seed-1"
        and v == "entity:e-comp-1"
        and d.get("stage") == "seed_entities"
        and d.get("layer") == 0
        for u, v, d in edge_tuples
    )
    assert any(
        u == "entity:e-common-1"
        and v == "chunk:graph-shared"
        and d.get("stage") == "frontier_chunks"
        and d.get("layer") == 1
        for u, v, d in edge_tuples
    )

    node_attrs = graph.nodes["chunk:graph-shared"]
    assert node_attrs.get("domain") == "company"
    assert node_attrs.get("layer") == 1


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
