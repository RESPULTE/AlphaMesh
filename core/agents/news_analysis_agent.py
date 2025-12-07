import asyncio
import json
import logging
import operator
from datetime import datetime, timedelta
from typing import Annotated, List, Optional, Type

from core.agents.base_agent import AbstractAgent
from core.agents.models import BaseAgentInput
from core.services import service_manager
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from newspaper import Article, ArticleException
from pydantic import BaseModel, Field

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NewsAnalysisAgent")

# --- Configuration & Toggles ---
LOG_INGESTED_TITLES = True
MAX_SEARCH_ATTEMPTS = 2
MAX_LOOKBACK_DAYS = 29
BATCH_SIZE = 10


# --- Structured Outputs ---
class SufficiencyCheck(BaseModel):
    is_sufficient: bool = Field(description="True if context answers the question.")
    reasoning: str = Field(description="Explanation.")


class CitedSource(BaseModel):
    source_id: int = Field(description="The numeric ID used in the text, e.g., 1.")
    title: str = Field(description="The title of the article.")
    url: str = Field(description="The URL of the article.")
    page_content: str = Field(description="The content of the article.")


class NewsAnalysisOutput(BaseModel):
    """
    REFACTORED: This model now serves as a data container for the Orchestrator.
    The 'detailed_analysis' field is removed as analysis is now centralized.
    """

    sources: List[CitedSource] = Field(
        description="The list of raw source articles gathered by the agent."
    )


# --- Internal State ---
class _AgentState(BaseAgentInput):
    current_page: int = 1
    last_total_results: int = 0
    attempt_count: int = 0

    latest_retrieved: Optional[datetime] = None
    news_context: Annotated[List[CitedSource], operator.add] = Field(
        default_factory=list
    )

    needs_more_data: bool = False
    is_fully_resolved: bool = False

    sufficiency_reasoning: str = ""

    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)


# --- Tools ---


def _download_article_sync(url: str) -> Optional[str]:
    """Blocking helper to download and parse article text."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text) < 200:
            return None
        return article.text
    except (Exception, ArticleException):
        return None


@tool
async def ingest_stock_news_tool(
    ticker: str, from_date: str, to_date: str, page: int
) -> str:
    """
    Fetches news metadata from an API, downloads content, and ingests it directly
    into the vector store. The vector store handles summarization and embedding.
    Returns a JSON string with results count.
    """
    logger.info(
        f"🛠️  Tool Call: Fetching '{ticker}' | Dates: {from_date} to {to_date} | Page: {page}"
    )

    try:
        loop = asyncio.get_running_loop()

        # 1. Fetch Metadata from NewsAPI
        response = await loop.run_in_executor(
            None,
            lambda: service_manager.get_news_api().get_everything(
                q=ticker,
                from_param=from_date,
                to=to_date,
                language="en",
                sort_by="relevancy",
                page=page,
                page_size=BATCH_SIZE,
            ),
        )

        if response.get("status") != "ok":
            logger.error(f"   ❌ API Error: {response.get('message')}")
            return json.dumps(
                {"success": False, "error": response.get("message"), "total_results": 0}
            )

        articles_meta = response.get("articles", [])
        total_results = response.get("totalResults", 0)

        if not articles_meta:
            return json.dumps(
                {
                    "success": True,
                    "count": 0,
                    "total_results": 0,
                    "message": "No articles found.",
                }
            )

        # 2. Download Raw Content
        logger.info(
            f"   -> Downloading raw content for {len(articles_meta)} articles..."
        )

        sem = asyncio.Semaphore(10)

        async def _fetch_and_ingest(meta):
            async with sem:
                url = meta.get("url")
                if not url:
                    return False

                text = await loop.run_in_executor(None, _download_article_sync, url)
                if not text:
                    return False

                title = meta.get("title", "Unknown")

                source_meta = {
                    "url": url,
                    "title": title,
                    "source": "NewsAPI",
                    "ticker": ticker,
                    "publish_time": meta.get("publishedAt", ""),
                }

                success = (
                    await service_manager.get_vector_store_manager().ingest_article(
                        raw_text=text,
                        source_metadata=source_meta,
                        should_summarize=False,
                    )
                )

                if success and LOG_INGESTED_TITLES:
                    logger.info(f"      + Ingested: {title}")

                return success

        results = await asyncio.gather(*[_fetch_and_ingest(a) for a in articles_meta])
        ingest_count = sum(1 for r in results if r)

        logger.info(
            f"✅ Batch Complete. Ingested {ingest_count}/{len(articles_meta)} articles."
        )

        return json.dumps(
            {
                "success": True,
                "count": ingest_count,
                "total_results": total_results,
                "message": f"Ingested {ingest_count} articles.",
            }
        )

    except Exception as e:
        logger.error(f"❌ Critical Tool Error: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "total_results": 0})


# --- Main Agent ---


class NewsAnalysisAgent(AbstractAgent):
    def __init__(self):
        super().__init__()
        self.tools = [ingest_stock_news_tool]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = service_manager.get_agent().bind_tools(self.tools)
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "news_agent"

    @property
    def description(self) -> str:
        return "Gathers raw news articles and sources based on a query. Does not perform analysis."

    @classmethod
    def get_output_schema_class(cls) -> Type[BaseModel]:
        return NewsAnalysisOutput

    async def run(self, input_data: BaseAgentInput) -> NewsAnalysisOutput:
        logger.info(f"🚀 [Agent Start] Ticker: {input_data.ticker}")

        final_state = await self._graph.ainvoke(input_data.model_dump())

        return NewsAnalysisOutput(
            sources=final_state.get("news_context", []),
        )

    def _build_graph(self):
        workflow = StateGraph(_AgentState)

        workflow.add_node("retrieve", self._retrieve_news)
        workflow.add_node("evaluate_sufficiency", self._evaluate_sufficiency)
        workflow.add_node("strategize_search", self._strategize_search)
        workflow.add_node("execute_search_action", self._execute_search_action)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate_sufficiency")

        # REWRITTEN: Use direct attribute access in conditional logic
        workflow.add_conditional_edges(
            "evaluate_sufficiency",
            lambda state: END if state.is_fully_resolved else "strategize_search",
        )

        workflow.add_conditional_edges(
            "strategize_search",
            lambda state: "execute_search_action" if state.needs_more_data else END,
        )

        workflow.add_edge("execute_search_action", "retrieve")

        return workflow.compile()

    # --- Node Implementations ---

    async def _retrieve_news(self, state: _AgentState) -> dict:
        # REWRITTEN: Use state.attribute access
        logger.info(
            f"🔍 [Step: Retrieve] Checking vector store (Attempt {state.attempt_count})..."
        )

        docs = await asyncio.to_thread(
            service_manager.get_vector_store_manager().retrieve,
            query=state.query,
            filter_dict={"ticker": state.ticker},
            k=15,
        )

        context_pieces = []
        current_article_count = len(state.news_context)
        start_id = current_article_count + 1
        latest_retrieved = None

        if docs:
            logger.info(f"   -> Found {len(docs)} existing documents.")
            for i, doc in enumerate(docs, start=start_id):
                meta = doc.metadata
                title = meta.get("title", "Unknown Title")
                url = meta.get("url", "#")
                pub_time_str = meta.get("publish_time", "")
                content = meta.get("summary", doc.page_content)

                context_pieces.append(
                    CitedSource(source_id=i, title=title, url=url, page_content=content)
                )
                if pub_time_str:
                    try:
                        latest_retrieved = datetime.fromisoformat(
                            pub_time_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass  # Ignore if date format is unexpected
        else:
            logger.info("   -> No documents found in store.")

        return {
            "news_context": context_pieces,
            "latest_retrieved": latest_retrieved,
        }

    async def _evaluate_sufficiency(self, state: _AgentState) -> dict:
        """
        Evaluates if enough data has been gathered based on heuristics.
        """
        # REWRITTEN: Use state.attribute access
        logger.info(
            f"🤔 [Step: Evaluate Sufficiency] Attempt {state.attempt_count}/{MAX_SEARCH_ATTEMPTS}"
        )

        article_count = len(state.news_context)

        if state.attempt_count >= MAX_SEARCH_ATTEMPTS:
            logger.info("   -> Max attempts reached. Concluding search.")
            return {"is_fully_resolved": True}

        if article_count >= 15:
            logger.info("   -> Content saturation (15+ articles). Concluding search.")
            return {"is_fully_resolved": True}

        if (
            state.start_date
            and (datetime.now() - state.start_date).days > MAX_LOOKBACK_DAYS
        ):
            logger.info("   -> Max lookback window reached. Concluding search.")
            return {"is_fully_resolved": True}

        logger.info("   -> Insufficient data based on heuristics. Continuing search.")
        return {"is_fully_resolved": False}

    def _strategize_search(self, state: _AgentState) -> dict:
        # REWRITTEN: Use state.attribute access
        logger.info("🧠 [Step: Strategize] Calculating next search parameters...")

        today = datetime.now()
        limit_date = today - timedelta(days=MAX_LOOKBACK_DAYS)

        articles_fetched = state.current_page * BATCH_SIZE
        can_paginate = state.last_total_results > articles_fetched

        if can_paginate:
            logger.info(
                f"   -> Strategy: Pagination. Moving to page {state.current_page + 1}."
            )
            return {
                "needs_more_data": True,
                "current_page": state.current_page + 1,
                "attempt_count": state.attempt_count + 1,
            }

        potential_from = state.start_date - timedelta(days=7)
        if potential_from < limit_date:
            logger.info("   -> Strategy: Date expansion limit reached. Stopping.")
            return {"needs_more_data": False}

        logger.info(
            f"   -> Strategy: Expanding range to {potential_from.strftime('%Y-%m-%d')}."
        )
        return {
            "needs_more_data": True,
            "current_page": 1,
            "start_date": potential_from,
            "attempt_count": state.attempt_count + 1,
        }

    async def _execute_search_action(self, state: _AgentState) -> dict:
        """
        Executes the news search tool based on the current strategy.
        """
        # REWRITTEN: Use state.attribute access
        logger.info("🤖 [Step: Execute Search Action] Running tool...")

        tool_args = {
            "ticker": state.ticker,
            "from_date": state.start_date.strftime("%Y-%m-%d"),
            "to_date": state.end_date.strftime("%Y-%m-%d"),
            "page": state.current_page,
        }

        tool_output_str = await ingest_stock_news_tool.ainvoke(tool_args)

        total_results = 0
        try:
            tool_output_json = json.loads(tool_output_str)
            total_results = tool_output_json.get("total_results", 0)
            logger.info(f"   -> Tool reported {total_results} total results available.")
        except json.JSONDecodeError:
            logger.error("   -> Failed to parse JSON from tool output.")

        return {"last_total_results": total_results}


if __name__ == "__main__":

    async def main():
        agent = NewsAnalysisAgent()
        input_data = BaseAgentInput(
            ticker="AAPL",
            query="reason for price drop Apple",
            start_date="2025-12-01",
            end_date="2025-12-05",
        )
        res = await agent.run(input_data)
        print("\n--- RAW AGENT OUTPUT ---\n")
        print(f"Found {len(res.sources)} sources:")
        for source in res.sources:
            print(f"- {source.title}")

    asyncio.run(main())
