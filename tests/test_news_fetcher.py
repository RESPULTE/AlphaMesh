from core.agents.news_fetcher import _normalise_tavily_result_to_article


def test_tavily_normalization_avoids_description_content_duplication() -> None:
    article = _normalise_tavily_result_to_article(
        {
            "url": "https://example.com/a",
            "title": "Example",
            "content": "Full body text from tavily.",
            "score": 0.87,
        }
    )

    assert article["content"] == "Full body text from tavily."
    assert article["description"] == ""
    assert article["content_source"] == "tavily"

