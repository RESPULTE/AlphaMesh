from datetime import datetime, timezone


from core.ingestion.chunker import ArticleChunker


def test_article_chunker_parses_published_at_correctly():
    chunker = ArticleChunker(chunk_size=512, chunk_overlap=64)
    dt = chunker._parse_published_at("2026-03-14T12:00:00Z")
    assert dt.year == 2026
    assert dt.month == 3
    assert dt.day == 14
    assert dt.hour == 12
    assert dt.tzinfo == timezone.utc


def test_article_chunker_falls_back_to_now_on_bad_date():
    chunker = ArticleChunker(chunk_size=512, chunk_overlap=64)
    dt = chunker._parse_published_at("invalid-date-string")
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc


def test_article_chunker_splits_article_correctly():
    chunker = ArticleChunker(chunk_size=50, chunk_overlap=10)

    article = {
        "title": "Test Article",
        "description": "Short desc.",
        "content": "This is a much longer content that should ideally be split into multiple chunks because the chunk size is only fifty characters.",
        "url": "https://example.com/test",
        "publishedAt": "2026-03-14T12:00:00Z",
    }
    companies = ["TEST"]

    doc_meta, chunks = chunker.chunk_article(article, companies)

    assert doc_meta.title == "Test Article"
    assert doc_meta.source_url == "https://example.com/test"
    assert doc_meta.companies_involved == ["TEST"]

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.document_id == doc_meta.document_id
        assert chunk.chunk_index == i
        assert chunk.article_title == "Test Article"
        assert chunk.source_url == "https://example.com/test"
        assert chunk.companies_involved == ["TEST"]
        assert isinstance(chunk.chunk_id, str)
