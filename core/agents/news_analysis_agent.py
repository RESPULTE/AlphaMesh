import asyncio
import logging
import operator
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Type

from core.agents.base_agent import AbstractAgent, AgentOutput
from core.services import service_manager
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from newspaper import Article
from pydantic import BaseModel, Field

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NewsAnalysisAgent")

# --- Configuration & Toggles ---
# Set this to True to see the specific titles of articles being ingested or retrieved.
# Set to False to keep the logs cleaner (counts only).
LOG_INGESTED_TITLES = True

# --- Constants ---
MAX_SEARCH_ATTEMPTS = 2
MAX_LOOKBACK_DAYS = 30
BATCH_SIZE = 10


# --- Input Schema ---
class NewsAnalysisInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL).")
    question: str = Field(description="The search-optimized question.")
    from_date: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)."
    )
    to_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD).")


# --- Structured Outputs ---
class SufficiencyCheck(BaseModel):
    is_sufficient: bool = Field(description="True if context answers the question.")
    reasoning: str = Field(description="Explanation.")


class ProcessedArticle(BaseModel):
    """Schema for a single article processed by the LLM."""

    url: str = Field(description="Original URL of the article.")
    title: str = Field(description="Title of the article.")
    publish_date: str = Field(description="Publication date.")
    summary: str = Field(
        description="Concise summary of facts relevant to the ticker/query."
    )
    relevance_score: int = Field(
        description="Score 0-10. 0 is irrelevant/spam, 10 is critical info."
    )


class BatchArticleProcessing(BaseModel):
    """Schema for the batch response from the LLM."""

    articles: List[ProcessedArticle] = Field(description="List of processed articles.")


# --- Internal State ---
class _AgentState(BaseModel):
    ticker: str
    query: str
    search_from_date: str
    search_to_date: str
    current_page: int = 1
    last_total_results: int = 0
    attempt_count: int = 0
    news_context: Annotated[str, operator.add] = ""
    messages: Annotated[List[BaseMessage], operator.add] = []

    needs_more_data: bool = False
    is_fully_resolved: bool = False


class _OutputState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]


# --- Tools ---


def _download_article_sync(url: str) -> Optional[str]:
    """
    Blocking helper to download and parse article text.
    Returns the raw text or None if failed.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text) < 200:
            return None
        return article.text
    except Exception:
        return None


@tool
async def ingest_stock_news_tool(
    ticker: str, query: str, from_date: str, to_date: str, page: int
) -> Dict[str, Any]:
    """
    Fetches a batch of news, processes them via LLM for relevance/summary,
    and ingests the high-quality summaries into the vector store.
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
                q=query,
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
            return {
                "success": False,
                "error": response.get("message"),
                "total_results": 0,
            }

        articles_meta = response.get("articles", [])
        total_results = response.get("totalResults", 0)

        logger.info(
            f"   -> API found {total_results} total results. Processing batch of {len(articles_meta)}."
        )

        if not articles_meta:
            return {
                "success": True,
                "count": 0,
                "total_results": 0,
                "message": "No articles found.",
            }

        # 2. Download Raw Content (Parallel)
        logger.info(
            f"   -> Downloading raw content for {len(articles_meta)} articles..."
        )

        raw_contents = []
        sem = asyncio.Semaphore(10)

        async def _fetch_content(meta):
            async with sem:
                url = meta.get("url")
                if not url:
                    return None
                text = await loop.run_in_executor(None, _download_article_sync, url)
                if text:
                    return {
                        "url": url,
                        "title": meta.get("title", "Unknown"),
                        "date": meta.get("publishedAt", ""),
                        "text": text[
                            :4000
                        ],  # Truncate to avoid context overflow if articles are huge
                    }
                return None

        results = await asyncio.gather(*[_fetch_content(a) for a in articles_meta])
        valid_articles = [r for r in results if r is not None]

        logger.info(
            f"   -> Successfully downloaded {len(valid_articles)}/{len(articles_meta)} articles."
        )

        if not valid_articles:
            return {
                "success": True,
                "count": 0,
                "total_results": total_results,
                "message": "All articles failed to download or were empty.",
            }

        # 3. Batch Process with LLM
        logger.info(f"   -> Analyzing relevance and summarizing via LLM...")

        llm = service_manager.get_agent().with_structured_output(BatchArticleProcessing)

        # Construct Context for LLM
        articles_context = ""
        for i, art in enumerate(valid_articles):
            articles_context += (
                f"--- ARTICLE {i} ---\n"
                f"URL: {art['url']}\n"
                f"Title: {art['title']}\n"
                f"Date: {art['date']}\n"
                f"Content: {art['text']}\n\n"
            )

        prompt = f"""You are a financial news filter.
        Ticker: {ticker}
        Query: {query}
        
        Task:
        1. Analyze the following {len(valid_articles)} news articles.
        2. Filter out spam, irrelevant articles, or generic market noise (Score < 5).
        3. For relevant articles (Score >= 5), write a concise summary focusing on facts.
        4. Return the structured list.

        ARTICLES:
        {articles_context}
        """

        try:
            processed_batch: BatchArticleProcessing = await llm.ainvoke(prompt)
        except Exception as e:
            logger.error(f"   ❌ LLM Batch Processing Failed: {e}")
            return {"success": False, "error": "LLM Processing Failed"}

        # 4. Ingest Processed Results
        ingest_count = 0
        skipped_count = 0

        for p_art in processed_batch.articles:
            if p_art.relevance_score < 5:
                skipped_count += 1
                continue  # Skip low relevance

            source_meta = {
                "url": p_art.url,
                "title": p_art.title,
                "source": "NewsAPI (LLM Processed)",
                "ticker": ticker,
                "publish_time": p_art.publish_date,
                "relevance": p_art.relevance_score,
            }

            # We ingest the LLM-generated summary, which is cleaner than raw text
            success = service_manager.get_vector_store_manager().ingest_article(
                raw_text=f"Summary: {p_art.summary}\nFull Title: {p_art.title}",
                source_metadata=source_meta,
            )
            if success:
                ingest_count += 1
                if LOG_INGESTED_TITLES:
                    logger.info(
                        f"      + Ingested [Score {p_art.relevance_score}]: {p_art.title}"
                    )

        logger.info(
            f"✅ Ingestion Summary: {ingest_count} kept, {skipped_count} skipped (low relevance)."
        )

        return {
            "success": True,
            "count": ingest_count,
            "total_results": total_results,
            "message": f"Processed {len(valid_articles)} articles. Ingested {ingest_count} relevant summaries.",
        }

    except Exception as e:
        logger.error(f"❌ Critical Tool Error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "count": 0, "total_results": 0}


# --- Main Agent ---


class NewsAnalysisAgent(AbstractAgent):
    def __init__(self):
        super().__init__()
        self.tools = [ingest_stock_news_tool]
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "news_agent"

    @property
    def description(self) -> str:
        return "Qualitative analysis with batch LLM processing and source citations."

    @classmethod
    def get_input_schema_class(cls) -> Type[BaseModel]:
        return NewsAnalysisInput

    async def run(self, input_data: NewsAnalysisInput) -> AgentOutput:
        logger.info(
            f"🚀 [Agent Start] Ticker: {input_data.ticker} | Query: {input_data.question}"
        )

        today = datetime.now()
        fmt = "%Y-%m-%d"
        t_date = input_data.to_date if input_data.to_date else today.strftime(fmt)
        f_date = (
            input_data.from_date
            if input_data.from_date
            else (today - timedelta(days=7)).strftime(fmt)
        )

        initial_state = {
            "ticker": input_data.ticker,
            "query": input_data.question,
            "search_from_date": f_date,
            "search_to_date": t_date,
            "current_page": 1,
            "last_total_results": 0,
            "attempt_count": 0,
            "messages": [],
            "news_context": "",
        }

        final_state = await self._graph.ainvoke(initial_state)
        output_content = final_state["messages"][-1].content
        logger.info("🏁 [Agent Finish] Analysis Complete.")
        return AgentOutput(agent_name=self.name, output=output_content)

    def _build_graph(self):
        workflow = StateGraph(_AgentState, output_schema=_OutputState)

        workflow.add_node("retrieve", self._retrieve_news)
        workflow.add_node("evaluate_sufficiency", self._evaluate_sufficiency)
        workflow.add_node("strategize_search", self._strategize_search)
        workflow.add_node("execute_fetch", self._execute_fetch)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate_sufficiency")

        workflow.add_conditional_edges(
            "evaluate_sufficiency",
            lambda state: (
                "generate_answer" if state.is_fully_resolved else "strategize_search"
            ),
        )

        workflow.add_conditional_edges(
            "strategize_search",
            lambda state: (
                "execute_fetch" if state.needs_more_data else "generate_answer"
            ),
        )

        workflow.add_edge("execute_fetch", "retrieve")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    # --- Node Implementations ---

    async def _retrieve_news(self, state: _AgentState) -> dict:
        logger.info(
            f"🔍 [Step: Retrieve] Checking vector store (Attempt {state.attempt_count})..."
        )

        docs = await asyncio.to_thread(
            service_manager.get_vector_store_manager().retrieve,
            query=state.query,
            filter_dict={"ticker": state.ticker},
        )

        context_pieces = []
        if docs:
            logger.info(f"   -> Found {len(docs)} existing documents/summaries.")
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                title = meta.get("title", "Unknown Title")
                url = meta.get("url", "#")
                pub_time = meta.get("publish_time", "Unknown Date")

                if LOG_INGESTED_TITLES:
                    logger.info(f"      > Retrieved: {title}")

                piece = (
                    f"--- ARTICLE {i} ---\n"
                    f"Title: {title}\n"
                    f"Date: {pub_time}\n"
                    f"URL: {url}\n"
                    f"Content: {doc.page_content}\n"
                )
                context_pieces.append(piece)
        else:
            logger.info("   -> No documents found in store.")

        return {"news_context": "\n".join(context_pieces)}

    async def _evaluate_sufficiency(self, state: _AgentState) -> dict:
        logger.info(
            "🤔 [Step: Evaluate Sufficiency] Checking if context answers the question..."
        )

        if state.attempt_count >= MAX_SEARCH_ATTEMPTS:
            logger.info(
                "   -> Max attempts reached. Proceeding to answer with available data."
            )
            return {"is_fully_resolved": True}

        if not state.news_context:
            logger.info("   -> Context empty. Need more data.")
            return {"is_fully_resolved": False}

        llm = service_manager.get_agent().with_structured_output(SufficiencyCheck)

        prompt = f"""You are a strict evaluator.
        User Question: "{state.query}"
        
        Retrieved News Context:
        {state.news_context[:4000]}...
        
        Does this context contain enough information to meaningfully answer the user's question?
        """

        try:
            result: SufficiencyCheck = await llm.ainvoke(prompt)
            logger.info(
                f"   -> Result: {'Sufficient' if result.is_sufficient else 'Insufficient'}"
            )
            logger.info(f"   -> Reasoning: {result.reasoning}")
            return {"is_fully_resolved": result.is_sufficient}
        except Exception as e:
            logger.error(f"   -> Sufficiency check failed: {e}")
            return {"is_fully_resolved": False}

    def _strategize_search(self, state: _AgentState) -> dict:
        logger.info("🧠 [Step: Strategize] Calculating next search parameters...")

        fmt = "%Y-%m-%d"
        current_from = datetime.strptime(state.search_from_date, fmt)
        current_to = datetime.strptime(state.search_to_date, fmt)
        today = datetime.now()
        limit_date = today - timedelta(days=MAX_LOOKBACK_DAYS)

        articles_fetched = state.current_page * BATCH_SIZE
        can_paginate = state.last_total_results > articles_fetched

        new_page = state.current_page
        new_from = current_from
        new_to = current_to
        action_taken = False
        strategy_msg = ""

        if state.attempt_count == 0:
            action_taken = True
            strategy_msg = "Initial search."
        elif can_paginate:
            new_page += 1
            action_taken = True
            strategy_msg = f"Pagination available. Moving to page {new_page}."
        else:
            new_page = 1
            potential_from = current_from - timedelta(days=7)

            if current_to.date() < today.date():
                potential_to = current_to + timedelta(days=7)
            else:
                potential_to = current_to

            if potential_from < limit_date:
                potential_from = limit_date
            if potential_to > today:
                potential_to = today

            if potential_from == current_from and potential_to == current_to:
                logger.info("   -> Cannot expand search window further (hit limits).")
                return {"needs_more_data": False}

            new_from = potential_from
            new_to = potential_to
            action_taken = True
            strategy_msg = f"Expanding date range to {new_from.strftime(fmt)} - {new_to.strftime(fmt)}."

        if action_taken:
            logger.info(f"   -> Strategy: {strategy_msg}")
            return {
                "needs_more_data": True,
                "current_page": new_page,
                "search_from_date": new_from.strftime(fmt),
                "search_to_date": new_to.strftime(fmt),
                "attempt_count": state.attempt_count + 1,
            }

        return {"needs_more_data": False}

    async def _execute_fetch(self, state: _AgentState) -> dict:
        # Logging handled inside the tool
        result = await ingest_stock_news_tool.ainvoke(
            {
                "ticker": state.ticker,
                "query": state.query,
                "from_date": state.search_from_date,
                "to_date": state.search_to_date,
                "page": state.current_page,
            }
        )
        return {
            "last_total_results": result.get("total_results", 0),
            "messages": [
                ToolMessage(
                    content=result.get("message", ""),
                    tool_call_id="system",
                    name="ingest_tool",
                )
            ],
        }

    async def _generate_answer(self, state: _AgentState) -> dict:
        logger.info("✍️  [Step: Generate Answer] Synthesizing final response...")

        has_news = len(state.news_context) > 50

        template = (
            "You are a financial analyst. Answer the question based ONLY on the provided context.\n"
            "Question: '{question}'\n\n"
            "Context:\n{context}\n\n"
            "### CITATION RULES (CRITICAL):\n"
            "1. Every factual statement must have a citation.\n"
            "2. Use the URL provided in the context blocks.\n"
            "3. Format citations as Markdown links at the end of the relevant sentence.\n"
            "   Example: 'Tesla shares rose 5% [Source Title](https://news.com/tesla).'\n"
            "4. If the context is empty, state that no data was found."
        )

        prompt = template.format(
            question=state.query,
            context=state.news_context if has_news else "No relevant news found.",
        )

        response = await service_manager.get_agent().ainvoke(prompt)
        return {"messages": [response]}


if __name__ == "__main__":

    async def main():
        agent = NewsAnalysisAgent()

        input_data = NewsAnalysisInput(
            ticker="MSFT",
            question="What is the reason for the recent price drop?",
            from_date="2025-12-01",
            to_date="2025-12-05",
        )

        res = await agent.run(input_data)
        print("\nFINAL OUTPUT:\n", res.output)

    asyncio.run(main())
