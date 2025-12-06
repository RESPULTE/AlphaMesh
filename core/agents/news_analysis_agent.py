import asyncio
import json
import logging
import operator
from datetime import datetime, timedelta
from typing import Annotated, List, Optional, Type

from core.agents.base_agent import AbstractAgent, AgentOutput
from core.services import service_manager
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
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
LOG_INGESTED_TITLES = True
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
    """Blocking helper to download and parse article text."""
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

                # Download
                text = await loop.run_in_executor(None, _download_article_sync, url)
                if not text:
                    return False

                title = meta.get("title", "Unknown")

                # Prepare Metadata for Vector Store
                source_meta = {
                    "url": url,
                    "title": title,
                    "source": "NewsAPI",
                    "ticker": ticker,
                    "publish_time": meta.get("publishedAt", ""),
                }

                # 3. Direct Ingestion (Delegating summarization to Vector Store)
                # We set should_summarize=True so the VS pipeline creates the summary
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
        # Bind tools to the LLM to allow for future expansion and dynamic selection
        self.llm_with_tools = service_manager.get_agent().bind_tools(self.tools)
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "news_agent"

    @property
    def description(self) -> str:
        return "Qualitative analysis using bound tools and vector store ingestion."

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

        # Extract final answer
        output_content = "No response generated."
        if final_state["messages"]:
            output_content = final_state["messages"][-1].content

        logger.info("🏁 [Agent Finish] Analysis Complete.")
        return AgentOutput(agent_name=self.name, output=output_content)

    def _build_graph(self):
        workflow = StateGraph(_AgentState, output_schema=_OutputState)

        # Define Nodes
        workflow.add_node("retrieve", self._retrieve_news)
        workflow.add_node("evaluate_sufficiency", self._evaluate_sufficiency)
        workflow.add_node("strategize_search", self._strategize_search)

        # New: LLM Node to decide tool calls
        workflow.add_node("call_tool_agent", self._call_tool_agent)
        # New: Standard ToolNode to execute bound tools
        workflow.add_node("execute_tools", ToolNode(self.tools))
        # New: Node to update state based on tool output
        workflow.add_node("process_tool_output", self._process_tool_output)

        workflow.add_node("generate_answer", self._generate_answer)

        # Define Edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate_sufficiency")

        # Sufficiency Logic
        workflow.add_conditional_edges(
            "evaluate_sufficiency",
            lambda state: (
                "generate_answer" if state.is_fully_resolved else "strategize_search"
            ),
        )

        # Strategy Logic
        workflow.add_conditional_edges(
            "strategize_search",
            lambda state: (
                "call_tool_agent" if state.needs_more_data else "generate_answer"
            ),
        )

        # Tool Execution Flow
        workflow.add_edge("call_tool_agent", "execute_tools")
        workflow.add_edge("execute_tools", "process_tool_output")
        workflow.add_edge(
            "process_tool_output", "retrieve"
        )  # Loop back to check what we got

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
            logger.info(f"   -> Found {len(docs)} existing documents.")
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                title = meta.get("title", "Unknown Title")
                url = meta.get("url", "#")
                pub_time = meta.get("publish_time", "Unknown Date")

                # Check if this document has a summary from the ingestion pipeline
                content = meta.get("summary", doc.page_content)

                if LOG_INGESTED_TITLES:
                    logger.info(f"      > Retrieved: {title}")

                piece = (
                    f"--- ARTICLE {i} ---\n"
                    f"Title: {title}\n"
                    f"Date: {pub_time}\n"
                    f"URL: {url}\n"
                    f"Content: {content}\n"
                )
                context_pieces.append(piece)
        else:
            logger.info("   -> No documents found in store.")

        return {"news_context": "\n".join(context_pieces)}

    async def _evaluate_sufficiency(self, state: _AgentState) -> dict:
        """
        Separate node to check if the retrieved context is enough.
        This replaces the internal filter logic previously in the tool.
        """
        logger.info("🤔 [Step: Evaluate Sufficiency]")

        if state.attempt_count >= MAX_SEARCH_ATTEMPTS:
            logger.info("   -> Max attempts reached. Proceeding.")
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
        except Exception:
            return {"is_fully_resolved": False}

    def _strategize_search(self, state: _AgentState) -> dict:
        logger.info("🧠 [Step: Strategize] Calculating next search parameters...")

        fmt = "%Y-%m-%d"
        current_from = datetime.strptime(state.search_from_date, fmt)
        current_to = datetime.strptime(state.search_to_date, fmt)
        today = datetime.now()
        limit_date = today - timedelta(days=MAX_LOOKBACK_DAYS)

        articles_fetched = state.current_page * BATCH_SIZE
        # We rely on 'last_total_results' being updated by _process_tool_output
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
                return {"needs_more_data": False}

            new_from = potential_from
            new_to = potential_to
            action_taken = True
            strategy_msg = (
                f"Expanding range: {new_from.strftime(fmt)} to {new_to.strftime(fmt)}."
            )

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

    async def _call_tool_agent(self, state: _AgentState) -> dict:
        """
        Uses the LLM (with bound tools) to generate the specific tool call.
        """
        logger.info("🤖 [Step: Call Tool Agent] Invoking LLM to call tools...")

        system_msg = SystemMessage(
            content="You are a research assistant. Use the available tools to fetch data based on the user's strategy."
        )

        # Explicitly instruct the LLM to use the parameters derived from the strategy step
        user_msg = (
            f"Please fetch news for ticker '{state.ticker}' regarding '{state.query}'. "
            f"Use the date range {state.search_from_date} to {state.search_to_date}. "
            f"Fetch page {state.current_page}."
        )

        response = await self.llm_with_tools.ainvoke([system_msg, user_msg])
        return {"messages": [response]}

    def _process_tool_output(self, state: _AgentState) -> dict:
        """
        Parses the output of the tool execution to update internal state (specifically total_results).
        """
        last_message = state.messages[-1]

        total_results = 0

        if isinstance(last_message, ToolMessage):
            try:
                # The tool returns a JSON string, so we parse it
                content_json = json.loads(last_message.content)
                total_results = content_json.get("total_results", 0)
                logger.info(
                    f"   -> Tool output processed. Total API Results available: {total_results}"
                )
            except Exception:
                logger.warning("   -> Could not parse tool output JSON.")

        return {"last_total_results": total_results}

    async def _generate_answer(self, state: _AgentState) -> dict:
        logger.info("✍️  [Step: Generate Answer]")

        has_news = len(state.news_context) > 50

        template = (
            "You are a financial analyst. Answer the question based ONLY on the provided context.\n"
            "Question: '{question}'\n\n"
            "Context:\n{context}\n\n"
            "### CITATION RULES:\n"
            "1. Every factual statement must have a citation.\n"
            "2. Use the URL provided in the context blocks.\n"
            "3. Format citations as Markdown links: [Source Title](https://...).\n"
            "4. If context is empty, state that no data was found."
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
