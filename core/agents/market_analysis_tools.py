from datetime import datetime, timedelta, timezone

import yfinance as yf
from core.services import service_manager  # Assuming this provides singletons
from langchain_core.tools import tool
from newspaper import Article

# --- Helper Logic ---


def is_article_stale(publish_str: str, hours_threshold: int = 3) -> bool:
    """Parses date string and checks if it's older than X hours."""
    try:
        # Handle various date formats or ISO strings
        if not publish_str:
            return True

        # Assuming ISO format from ingestion
        pub_date = datetime.fromisoformat(str(publish_str))

        # If naive, assume local/UTC matching system time (simplified for example)
        if pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=None)

        diff = datetime.now(timezone.utc) - pub_date
        return diff > timedelta(hours=hours_threshold)
    except Exception:
        # If we can't parse the date, assume it's stale to be safe
        return True


# --- Tool Definitions ---


class MarketAnalysisTools:

    @staticmethod
    @tool
    def retrieve_ticker_data(query: str, ticker: str):
        """
        Search the internal memory for news about a stock ticker.
        Returns the content and the 'publish_time' of the articles.
        """
        print(f"--- [Tool] Retrieving Memory for {ticker} ---")
        manager = service_manager.get_vector_store_manager()

        docs = manager.retrieve(query=query, filter_dict={"ticker": ticker})

        if not docs:
            return "No documents found in memory. Status: EMPTY."

        results = []
        stale_count = 0

        for doc in docs:
            pub_time = doc.metadata.get("publish_time", "Unknown")
            is_stale = is_article_stale(pub_time, hours_threshold=3)
            if is_stale:
                stale_count += 1

            results.append(
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Date: {pub_time}\n"
                f"Content: {doc.page_content}"
            )

        context = "\n\n".join(results)

        # We append a 'System Note' to the tool output to guide the LLM
        status_msg = ""
        if stale_count == len(docs):
            status_msg = "\n\n[SYSTEM NOTE]: All retrieved documents are older than 3 hours. Consideration: Fetch new data."

        return f"{context}{status_msg}"

    @staticmethod
    @tool
    def fetch_fresh_news(ticker: str, max_articles: int = 5):
        """
        Fetches the absolute latest news from Yahoo Finance for a ticker and ingests it.
        Use this if memory is empty or data is stale (> 3 hours old).
        """
        print(f"--- [Tool] Fetching Fresh News for {ticker} ---")
        manager = service_manager.get_vector_store_manager()

        try:
            stock = yf.Ticker(ticker)
            news = stock.get_news(max_articles)
        except Exception as e:
            return f"Error fetching news: {e}"

        if not news:
            return "No new articles found on external source."

        ingested_count = 0
        for item in news:
            # Skip videos
            if item.get("content", {}).get("contentType") == "VIDEO":
                continue

            # URL Extraction Logic
            content = item.get("content", {})
            url = content.get("clickThroughUrl", {}).get("url") or content.get(
                "canonicalUrl", {}
            ).get("url")

            if not url:
                continue

            try:
                article = Article(url)
                article.download()
                article.parse()

                # Metadata prep
                meta = {
                    "url": url,
                    "title": content.get("title"),
                    "source": "Yahoo Finance",
                    "ticker": ticker,
                    # Ensure we save time in a parseable format
                    "publish_time": content.get("pubDate", datetime.now().isoformat()),
                }

                if manager.ingest_article(article.text, meta):
                    ingested_count += 1
            except Exception:
                continue

        return f"Successfully fetched and ingested {ingested_count} new articles. You may now retrieve them."


# List for the Agent
tools_list = [
    MarketAnalysisTools.retrieve_ticker_data,
    MarketAnalysisTools.fetch_fresh_news,
]
