"""
news_fetcher.py
───────────────
Combines NewsAPI (article discovery) with Trafilatura (full-content extraction)
and Tavily (web search).

Public entry-points:
  • build_news_query(...)        – constructs an advanced NewsAPI `q` string
  • fetch_articles(...)          – NewsAPI fetch + trafilatura scrape
  • fetch_articles_from_tavily() – Tavily web search, normalized to article dicts
  • fetch_news(action, query)    – unified dispatcher for the agent's fetch nodes

Usage (standalone / testing):
    import asyncio
    from news_fetcher import fetch_news, build_news_query

    articles = asyncio.run(fetch_news("newsapi", "AAPL earnings", from_date="2025-01-01"))
    articles = asyncio.run(fetch_news("web_search", "Apple AI strategy"))
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
# 3. Tavily web search
# ---------------------------------------------------------------------------


def _normalise_published_at(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            return raw
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _call_tavily_search(
    payload: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    req = Request(
        settings.TAVILY_SEARCH_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _normalise_tavily_result_to_article(result: Dict[str, Any]) -> dict:
    url = str(result.get("url") or "").strip()
    title = str(result.get("title") or "").strip() or "Untitled"
    snippet = str(result.get("content") or "").strip()
    raw_content = result.get("raw_content")
    if isinstance(raw_content, dict):
        raw_content = json.dumps(raw_content, ensure_ascii=True)
    content = str(raw_content or snippet).strip()
    published_at = _normalise_published_at(
        result.get("published_date") or result.get("published_at")
    )
    source_name = _extract_domain(url) or "web"

    return {
        "title": title,
        "description": snippet[:500],
        "content": content or snippet,
        "url": url,
        "urlToImage": None,
        "publishedAt": published_at,
        "source": {"id": None, "name": source_name},
        "content_source": "tavily",
    }


async def fetch_articles_from_tavily(
    query: str,
    *,
    max_results: int = settings.TAVILY_SEARCH_MAX_RESULTS,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    topic: str = settings.TAVILY_TOPIC,
    search_depth: str = settings.TAVILY_SEARCH_DEPTH,
    include_raw_content: bool = True,
) -> List[dict]:
    """
    Search the web using Tavily and normalize results to NewsAPI-like articles.

    Returns article dicts compatible with the existing chunker/ingestor:
    title, description, content, url, publishedAt, source.
    """
    if not settings.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not configured. Skipping Tavily web search.")
        return []

    max_results = max(1, min(int(max_results or 1), 20))
    payload: Dict[str, Any] = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "topic": topic,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_raw_content": include_raw_content,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        response: Dict[str, Any] = await asyncio.to_thread(
            _call_tavily_search,
            payload,
            settings._SCRAPE_TIMEOUT,
        )
    except URLError as exc:
        logger.error("Tavily request failed for query '%s': %s", query, exc)
        return []
    except Exception as exc:
        logger.error("Unexpected Tavily failure for query '%s': %s", query, exc)
        return []

    results = response.get("results") or []
    articles = [
        _normalise_tavily_result_to_article(r)
        for r in results
        if isinstance(r, dict) and r.get("url")
    ]
    logger.info(
        "Tavily returned %d normalized articles for query '%s'.",
        len(articles),
        query,
    )
    return [a for a in articles if a.get("content") or a.get("description")]


# ---------------------------------------------------------------------------
# 4. Main public function
# ---------------------------------------------------------------------------


async def fetch_articles(
    q: str,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page_size: int = settings.NEWS_FETCH_MAX_ARTICLES,
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
    if page_size > settings.NEWS_FETCH_MAX_ARTICLES:
        page_size = settings.NEWS_FETCH_MAX_ARTICLES

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


# ---------------------------------------------------------------------------
# 5. Unified dispatcher
# ---------------------------------------------------------------------------


async def fetch_news(
    action: Literal["newsapi", "web_search"],
    query: str,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[dict]:
    """
    Unified entry-point for the agent's online-fetch branch.

    Routes to NewsAPI (+ trafilatura scrape) or Tavily based on *action*.
    Returns a normalized list of article dicts compatible with the ingestor.
    """
    if action == "newsapi":
        return await fetch_articles(
            q=query,
            from_date=from_date,
            to_date=to_date,
            page_size=max_results,
        )
    if action == "web_search":
        return await fetch_articles_from_tavily(
            query=query,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
    logger.warning("fetch_news: unknown action '%s'; returning []", action)
    return []
