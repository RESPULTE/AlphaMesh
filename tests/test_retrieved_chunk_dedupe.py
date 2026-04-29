from core.memory.retrieval.models import RetrievedChunk


def _chunk(
    chunk_id: str,
    *,
    url: str,
    title: str,
    text: str,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="vector",
        relevance_score=score,
        source_url=url,
        article_title=title,
        metadata={"source_url": url, "article_title": title},
    )


def test_dedupe_chunks_by_article_text_keeps_highest_score_within_same_article() -> None:
    chunks = [
        _chunk(
            "c1",
            url="https://example.com/a?utm_source=x",
            title="Article A",
            text="Same sentence.   With extra spacing.",
            score=0.3,
        ),
        _chunk(
            "c2",
            url="https://example.com/a",
            title="Article A",
            text="same sentence. with extra spacing.",
            score=0.9,
        ),
    ]

    deduped = RetrievedChunk._dedupe_chunks_by_article_text(chunks)
    assert len(deduped) == 1
    assert deduped[0].chunk_id == "c2"


def test_dedupe_chunks_by_article_text_preserves_same_text_across_different_articles() -> None:
    chunks = [
        _chunk(
            "c1",
            url="https://example.com/a",
            title="Article A",
            text="Shared wire copy line.",
            score=0.5,
        ),
        _chunk(
            "c2",
            url="https://example.com/b",
            title="Article B",
            text="Shared wire copy line.",
            score=0.6,
        ),
    ]

    deduped = RetrievedChunk._dedupe_chunks_by_article_text(chunks)
    assert len(deduped) == 2
    assert {chunk.chunk_id for chunk in deduped} == {"c1", "c2"}

