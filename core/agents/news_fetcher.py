"""
news_fetcher.py
───────────────
Combines NewsAPI (article discovery) with Trafilatura (full-content extraction).

Two public entry-points:
  • build_news_query(...)   – constructs an advanced NewsAPI `q` string
  • fetch_articles(...)     – fetches + scrapes articles, returns enriched dicts

Usage (standalone / testing):
    import asyncio
    from news_fetcher import fetch_articles, build_news_query

    q = build_news_query(
        ticker="AAPL",
        must_include=["earnings", "revenue"],
        must_exclude=["rumour"],
    )
    articles = asyncio.run(fetch_articles(q, from_date="2025-01-01", to_date="2025-03-01"))
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import trafilatura
from newsapi import NewsApiClient

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trusted financial-news domains (passed as `domains=` to NewsAPI).
# NewsAPI accepts a comma-separated list; max 20 sources on paid plans.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. Query builder
# ---------------------------------------------------------------------------


def build_news_query(
    ticker: str,
    *,
    company_name: Optional[str] = None,
    must_include: Optional[List[str]] = None,
    must_exclude: Optional[List[str]] = None,
    any_of: Optional[List[str]] = None,
    exact_phrase: Optional[str] = None,
) -> str:
    """
    Build an advanced NewsAPI `q` string using boolean operators.

    NewsAPI supports:
      • "phrase"      → exact match
      • +word         → must appear
      • -word         → must NOT appear
      • AND / OR / NOT and parentheses for grouping

    Examples
    --------
    >>> build_news_query("AAPL", must_include=["earnings"], must_exclude=["rumour"])
    '(AAPL OR Apple) +earnings -rumour'

    >>> build_news_query(
    ...     "MSFT",
    ...     company_name="Microsoft",
    ...     any_of=["Azure", "cloud", "AI"],
    ...     must_exclude=["lawsuit"],
    ...     exact_phrase="quarterly results",
    ... )
    '(MSFT OR Microsoft) AND (Azure OR cloud OR AI) +"quarterly results" -lawsuit'

    Parameters
    ----------
    ticker:         Stock ticker (always included).
    company_name:   Human-readable company name (ORed with ticker).
    must_include:   Words/phrases that MUST appear (+prefix).
    must_exclude:   Words/phrases that must NOT appear (-prefix).
    any_of:         Words that should match at least one (OR group).
    exact_phrase:   An exact phrase wrapped in quotes.

    Returns
    -------
    A URL-safe query string (NewsAPI handles URL encoding internally).
    Max 500 chars — truncated with a warning if exceeded.
    """
    parts: List[str] = []

    # Core subject: ticker OR company name
    if company_name:
        parts.append(f"({ticker} OR {company_name})")
    else:
        parts.append(ticker)

    # Optional OR group
    if any_of:
        group = " OR ".join(any_of)
        parts.append(f"AND ({group})")

    # Exact phrase
    if exact_phrase:
        parts.append(f'+"{exact_phrase}"')

    # Must-include terms (+prefix)
    for word in must_include or []:
        parts.append(f"+{word}")

    # Must-exclude terms (-prefix)
    for word in must_exclude or []:
        parts.append(f"-{word}")

    q = " ".join(parts)

    if len(q) > 500:
        logger.warning("NewsAPI query exceeds 500 chars (%d). Truncating.", len(q))
        q = q[:500]

    return q


# ---------------------------------------------------------------------------
# 2. Full-content scraper (trafilatura, async)
# ---------------------------------------------------------------------------


async def _scrape_url(
    url: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """
    Download and extract the main article body from `url` using trafilatura.

    Runs the blocking trafilatura calls in a thread pool so the event loop
    stays free.  Returns the extracted text or None on failure.
    """
    async with semaphore:

        def _fetch_and_extract() -> Optional[str]:
            try:
                # trafilatura.fetch_url handles redirects, encoding, etc.
                downloaded = trafilatura.fetch_url(
                    url,
                    no_ssl=False,
                )
                if not downloaded:
                    return None

                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,  # use readability as fallback
                    favor_recall=True,  # prefer more text over precision
                    deduplicate=True,
                )
                return text
            except Exception as exc:
                logger.debug("trafilatura failed for %s: %s", url, exc)
                return None

        return await asyncio.to_thread(_fetch_and_extract)


# ---------------------------------------------------------------------------
# 3. Main public function
# ---------------------------------------------------------------------------


async def fetch_articles(
    q: str,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page_size: int = 10,
    page: int = 1,
    language: str = "en",
    sort_by: str = "relevancy",
    scrape_full_content: bool = True,
) -> List[dict]:
    """
    Discover articles via NewsAPI and optionally scrape their full content
    with trafilatura.

    Parameters
    ----------
    q:
        Advanced query string. Use `build_news_query()` to construct it, or
        pass your own using NewsAPI's boolean syntax.
    from_date / to_date:
        ISO date strings "YYYY-MM-DD".  NewsAPI free tier: max 28 days ago.
    page_size:
        Results per page (max 100).
    page:
        Page number.
    language:
        ISO 639-1 language code.
    sort_by:
        "relevancy" | "popularity" | "publishedAt"

    Returns
    -------
    List of article dicts, each containing at minimum:
        title, description, content, url, urlToImage,
        publishedAt, source (dict with id/name).
    Articles where both content and description are empty are dropped.
    """
    loop = asyncio.get_running_loop()
    client = NewsApiClient(api_key=settings.NEWSAPI_KEY)

    # NewsAPI is a synchronous library — run in thread pool
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.get_everything(
                q=q,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by=sort_by,
                page=page,
                page_size=page_size,
                domains=settings.FINANCIAL_DOMAINS,
            ),
        )
    except Exception as exc:
        logger.error("NewsAPI request failed: %s", exc)
        raise

    if response.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {response.get('message', 'unknown error')}")

    raw_articles: List[dict] = response.get("articles", [])
    logger.info("NewsAPI returned %d articles.", len(raw_articles))

    if not raw_articles:
        return []

    if not scrape_full_content:
        # Return as-is, filtering out articles with no usable text
        return [a for a in raw_articles if a.get("content") or a.get("description")]

    # -----------------------------------------------------------------------
    # Scrape full content concurrently with trafilatura
    # -----------------------------------------------------------------------
    semaphore = asyncio.Semaphore(settings._SCRAPE_CONCURRENCY)
    urls = [a.get("url", "") for a in raw_articles]

    scraped_texts: List[Optional[str]] = await asyncio.gather(
        *[_scrape_url(url, semaphore) for url in urls]
    )

    enriched: List[dict] = []
    for article, scraped in zip(raw_articles, scraped_texts):
        url = article.get("url", "")

        # Prefer scraped full text; fall back to NewsAPI snippet
        if scraped and len(scraped) >= settings._MIN_BODY_LENGTH:
            article["content"] = scraped
            article["content_source"] = "trafilatura"
            logger.debug("Scraped %d chars from %s", len(scraped), url)
        else:
            # Keep NewsAPI truncated content if scraping failed
            fallback = article.get("content") or article.get("description") or ""
            if not fallback:
                logger.debug("Dropping article with no content: %s", url)
                continue
            article["content_source"] = "newsapi_snippet"

        enriched.append(article)

    logger.info(
        "Returning %d/%d articles after scraping "
        "(%d with full text, %d with snippets).",
        len(enriched),
        len(raw_articles),
        sum(1 for a in enriched if a.get("content_source") == "trafilatura"),
        sum(1 for a in enriched if a.get("content_source") == "newsapi_snippet"),
    )
    return enriched
