from core.memory.retrieval.models import MemoryChunk, RetrievedChunk
from core.memory.retrieval.reranker import CompositeReranker


def test_composite_reranker_ranks_correctly():
    reranker = CompositeReranker(alpha=0.5, beta=0.5, top_k=2)

    # We create some memory chunks manually.
    c1 = MemoryChunk(
        chunk_id="c1",
        text="text 1",
        source="vector",
        domain="company",
        embedding_score=0.8,  # 0.5 * 0.8 = 0.4
        graph_depth=0,  # 0.5 * (1/1) = 0.5 -> total 0.9
        composite_score=0.0,
        metadata={},
    )

    c2 = MemoryChunk(
        chunk_id="c2",
        text="text 2",
        source="graph",
        domain="company",
        embedding_score=0.6,  # 0.5 * 0.6 = 0.3
        graph_depth=1,  # 0.5 * (1/2) = 0.25 -> total 0.55
        composite_score=0.0,
        metadata={},
    )

    c3 = MemoryChunk(
        chunk_id="c3",
        text="text 3",
        source="vector",
        domain="sector",
        embedding_score=0.9,  # 0.5 * 0.9 = 0.45
        graph_depth=0,  # 0.5 * (1/1) = 0.5 -> total 0.95
        composite_score=0.0,
        metadata={},
    )

    ranked = reranker.rank([c1, c2, c3])

    assert len(ranked) == 2
    assert ranked[0].chunk_id == "c3"
    assert ranked[0].composite_score == 0.95
    assert ranked[1].chunk_id == "c1"
    assert ranked[1].composite_score == 0.9


def test_composite_reranker_deduplicates_by_chunk_id():
    reranker = CompositeReranker(alpha=1.0, beta=0.0, top_k=5)

    c1 = MemoryChunk(
        chunk_id="c1",
        text="text 1",
        source="vector",
        domain="company",
        embedding_score=0.5,
        graph_depth=0,
        composite_score=0.0,
        metadata={},
    )
    c1_better = MemoryChunk(
        chunk_id="c1",
        text="text 1 better",
        source="graph",
        domain="company",
        embedding_score=0.9,
        graph_depth=1,
        composite_score=0.0,
        metadata={},
    )

    ranked = reranker.rank([c1, c1_better])

    assert len(ranked) == 1
    assert ranked[0].embedding_score == 0.9
    assert ranked[0].text == "text 1 better"


def test_composite_reranker_from_retrieved_chunk():
    rc = RetrievedChunk(
        chunk_id="r1",
        text="text r",
        source="vector",
        score=0.8,
        metadata={"meta": "data"},
    )
    mc = CompositeReranker.from_retrieved_chunk(rc, domain="market")

    assert mc.chunk_id == "r1"
    assert mc.domain == "market"
    assert mc.embedding_score == 0.8
    assert mc.graph_depth == 0
    assert mc.metadata == {"meta": "data"}

    # Graph depth when source is graph
    rc2 = RetrievedChunk(
        chunk_id="r2", text="text r2", source="graph", score=None, metadata={}
    )
    mc2 = CompositeReranker.from_retrieved_chunk(rc2, domain="sector")
    assert mc2.embedding_score == 0.0
    assert mc2.graph_depth == 1
