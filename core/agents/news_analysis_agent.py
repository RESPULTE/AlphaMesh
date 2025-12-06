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

# --- Constants ---
MAX_SEARCH_ATTEMPTS = 3
MAX_LOOKBACK_DAYS = 30


# --- Input Schema ---
class NewsAnalysisInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL).")
    question: str = Field(description="The search-optimized question.")
    from_date: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)."
    )
    to_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD).")


# --- Structured Output for Sufficiency Check ---
class SufficiencyCheck(BaseModel):
    is_sufficient: bool = Field(description="True if context answers the question.")
    reasoning: str = Field(description="Explanation.")


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


def _ingest_article_sync(url: str, title: str, pubtime: str, ticker: str) -> bool:
    try:
        article_raw = Article(url)
        article_raw.download()
        article_raw.parse()

        if len(article_raw.text) < 200:
            return False

        short_title = (title[:40] + "...") if len(title) > 40 else title

        source_meta = {
            "url": url,
            "title": title,
            "source": "NewsAPI",
            "ticker": ticker,
            "publish_time": pubtime,
        }
        success = service_manager.get_vector_store_manager().ingest_article(
            raw_text=article_raw.text, source_metadata=source_meta
        )

        if success:
            logger.info(f"📄 Ingested: {short_title}")

        return success
    except Exception:
        return False


@tool
async def ingest_stock_news_tool(
    ticker: str, query: str, from_date: str, to_date: str, page: int
) -> Dict[str, Any]:
    """Fetches news via NewsAPI and ingests it."""
    logger.info(
        f"🛠️ Tool Call: Fetching '{ticker}' | Date: {from_date} to {to_date} | Page: {page}"
    )

    try:
        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(
            None,
            lambda: service_manager.get_news_api().get_everything(
                q=ticker,
                from_param=from_date,
                to=to_date,
                language="en",
                sort_by="relevancy",
                page=page,
                page_size=5,
            ),
        )

        if response.get("status") != "ok":
            return {
                "success": False,
                "error": response.get("message"),
                "count": 0,
                "total_results": 0,
            }

        articles = response.get("articles", [])
        total_results = response.get("totalResults", 0)

        if not articles:
            return {
                "success": True,
                "count": 0,
                "total_results": 0,
                "message": "No articles found.",
            }

        sem = asyncio.Semaphore(10)

        async def _process(art):
            async with sem:
                url = art.get("url")
                title = art.get("title")
                pub = art.get("publishedAt")
                if url and title:
                    return await loop.run_in_executor(
                        None, _ingest_article_sync, url, title, pub, ticker
                    )
                return False

        results = await asyncio.gather(*[_process(a) for a in articles])
        ingested = sum(results)

        logger.info(f"✅ Ingestion Complete. Added {ingested} articles.")

        return {
            "success": True,
            "count": ingested,
            "total_results": total_results,
            "message": f"Ingested {ingested} articles.",
        }

    except Exception as e:
        logger.error(f"❌ Critical Tool Error: {e}")
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
        return "Qualitative analysis with source citations."

    @classmethod
    def get_input_schema_class(cls) -> Type[BaseModel]:
        return NewsAnalysisInput

    async def run(self, input_data: NewsAnalysisInput) -> AgentOutput:
        logger.info(f"🚀 [Agent Start] Ticker: {input_data.ticker}")

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
        logger.info(f"🔍 [Step: Retrieve] Attempt {state.attempt_count}")

        docs = await asyncio.to_thread(
            service_manager.get_vector_store_manager().retrieve,
            query=state.query,
            filter_dict={"ticker": state.ticker},
        )

        context_pieces = []
        if docs:
            # --- UPDATED: Format includes URL explicitly ---
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                title = meta.get("title", "Unknown Title")
                url = meta.get("url", "#")  # Default to # if missing
                pub_time = meta.get("publish_time", "Unknown Date")

                piece = (
                    f"--- ARTICLE {i} ---\n"
                    f"Title: {title}\n"
                    f"Date: {pub_time}\n"
                    f"URL: {url}\n"
                    f"Content: {doc.page_content}\n"
                )
                context_pieces.append(piece)

            logger.info(f"   -> Found {len(docs)} documents.")
        else:
            logger.info("   -> No documents found.")

        return {"news_context": "\n".join(context_pieces)}

    async def _evaluate_sufficiency(self, state: _AgentState) -> dict:
        logger.info("🤔 [Step: Evaluate Sufficiency]")

        if state.attempt_count >= MAX_SEARCH_ATTEMPTS:
            return {"is_fully_resolved": True}

        if not state.news_context:
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
            logger.info(f"   -> Sufficient: {result.is_sufficient}")
            return {"is_fully_resolved": result.is_sufficient}
        except Exception:
            return {"is_fully_resolved": False}

    def _strategize_search(self, state: _AgentState) -> dict:
        logger.info("🧠 [Step: Strategize]")

        fmt = "%Y-%m-%d"
        current_from = datetime.strptime(state.search_from_date, fmt)
        current_to = datetime.strptime(state.search_to_date, fmt)
        today = datetime.now()
        limit_date = today - timedelta(days=MAX_LOOKBACK_DAYS)

        articles_fetched = state.current_page * 20
        can_paginate = state.last_total_results > articles_fetched

        new_page = state.current_page
        new_from = current_from
        new_to = current_to
        action_taken = False

        if state.attempt_count == 0:
            action_taken = True
        elif can_paginate:
            new_page += 1
            action_taken = True
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
                return {"needs_more_data": False}

            new_from = potential_from
            new_to = potential_to
            action_taken = True

        if action_taken:
            return {
                "needs_more_data": True,
                "current_page": new_page,
                "search_from_date": new_from.strftime(fmt),
                "search_to_date": new_to.strftime(fmt),
                "attempt_count": state.attempt_count + 1,
            }

        return {"needs_more_data": False}

    async def _execute_fetch(self, state: _AgentState) -> dict:
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
        logger.info("✍️ [Step: Generate Answer]")

        has_news = len(state.news_context) > 50

        # --- UPDATED: Prompt enforces citations ---
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
            ticker="NVDA",
            question="What are the reason for the recent price drop?",
            from_date="2025-12-01",
            to_date="2025-12-05",
        )

        res = await agent.run(input_data)
        print("\nFINAL OUTPUT:\n", res.output)

    asyncio.run(main())
