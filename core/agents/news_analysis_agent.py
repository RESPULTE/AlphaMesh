import asyncio
import json
import logging
import operator
from datetime import datetime, timedelta
from typing import Annotated, List, Optional, Type

from core.agents.base_agent import AbstractAgent
from core.services import service_manager
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
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


class CitedSource(BaseModel):
    source_id: int = Field(description="The numeric ID used in the text, e.g., 1.")
    title: str = Field(description="The title of the article.")
    url: str = Field(description="The URL of the article.")


class StructuredNewsAnalysis(BaseModel):
    analysis: str = Field(
        description="The answer text containing inline citations in the format [1], [2], etc."
    )
    sources: List[CitedSource] = Field(
        description="The list of sources that correspond to the inline citations."
    )


# --- Internal State ---
class _AgentState(BaseModel):
    # ... (existing fields)
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
    sufficiency_reasoning: str = ""

    # NEW FIELD
    final_structured_output: Optional[StructuredNewsAnalysis] = None


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

    @classmethod
    def get_output_schema_class(cls) -> Type[BaseModel]:
        return StructuredNewsAnalysis

    async def run(self, input_data: NewsAnalysisInput) -> dict:
        logger.info(f"🚀 [Agent Start] Ticker: {input_data.ticker}")

        # ... (Date setup logic remains the same) ...
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
            "sufficiency_reasoning": "",
            "final_structured_output": None,
        }

        return await self._graph.ainvoke(initial_state)

    def _build_graph(self):
        workflow = StateGraph(_AgentState, output_schema=StructuredNewsAnalysis)

        # Define Nodes
        workflow.add_node("retrieve", self._retrieve_news)
        workflow.add_node("evaluate_sufficiency", self._evaluate_sufficiency)
        workflow.add_node("strategize_search", self._strategize_search)

        # NEW: The combined node
        workflow.add_node("execute_search_action", self._execute_search_action)

        workflow.add_node("generate_answer", self._generate_answer)

        # Define Edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate_sufficiency")

        # Sufficiency Logic (With the dictionary fix from the previous step)
        workflow.add_conditional_edges(
            "evaluate_sufficiency",
            lambda state: (
                "generate_answer" if state.is_fully_resolved else "strategize_search"
            ),
            {
                "generate_answer": "generate_answer",
                "strategize_search": "strategize_search",
            },
        )

        # Strategy Logic
        # Note: We now point to 'execute_search_action' instead of 'call_tool_agent'
        workflow.add_conditional_edges(
            "strategize_search",
            lambda state: (
                "execute_search_action" if state.needs_more_data else "generate_answer"
            ),
            {
                "execute_search_action": "execute_search_action",
                "generate_answer": "generate_answer",
            },
        )

        # Simplified Flow
        # execute_search_action now loops directly back to retrieve
        workflow.add_edge("execute_search_action", "retrieve")

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
        # Calculate the starting ID based on how many articles are already in the context
        # This prevents ID collision if we loop multiple times (e.g., [1]-[10], then [11]-[20])
        current_article_count = state.news_context.count("--- ARTICLE")
        start_id = current_article_count + 1

        if docs:
            logger.info(f"   -> Found {len(docs)} existing documents.")
            for i, doc in enumerate(docs, start=start_id):
                meta = doc.metadata
                title = meta.get("title", "Unknown Title")
                url = meta.get("url", "#")
                pub_time = meta.get("publish_time", "Unknown Date")
                content = meta.get("summary", doc.page_content)

                # Format clearly with the ID so the LLM can reference it
                piece = (
                    f"--- ARTICLE [{i}] ---\n"
                    f"Source ID: {i}\n"
                    f"Title: {title}\n"
                    f"Date: {pub_time}\n"
                    f"URL: {url}\n"
                    f"Content: {content}\n"
                )
                context_pieces.append(piece)
        else:
            logger.info("   -> No documents found in store.")

        return {"news_context": "\n".join(context_pieces)}

    async def _evaluate_sufficiency(self, state: _AgentState) -> _AgentState:
        """
        Refined sufficiency check that accounts for:
        1. Search exhaustion (Max attempts / Max lookback).
        2. Content saturation (Too many articles).
        3. 'Negative Result' sufficiency (No news found is a valid answer).
        """
        logger.info(
            f"🤔 [Step: Evaluate Sufficiency] Attempt {state.attempt_count}/{MAX_SEARCH_ATTEMPTS}"
        )

        # --- 1. Calculate Metrics ---
        # Count articles based on the separator used in _retrieve_news
        article_count = state.news_context.count("--- ARTICLE")

        # Parse dates to see how far back we have gone
        fmt = "%Y-%m-%d"
        try:
            current_from = datetime.strptime(state.search_from_date, fmt)
            limit_date = datetime.now() - timedelta(days=MAX_LOOKBACK_DAYS)
            days_searched_depth = (datetime.now() - current_from).days
            hit_max_lookback = current_from <= limit_date
        except Exception:
            # Fallback if parsing fails
            days_searched_depth = 0
            hit_max_lookback = False

        logger.info(
            f"   -> Metrics: Articles={article_count} | Days Depth={days_searched_depth} | Hit Limit={hit_max_lookback}"
        )

        # --- 2. Heuristic Hard Stops (No LLM needed) ---

        # A. Max Attempts Reached
        if state.attempt_count >= MAX_SEARCH_ATTEMPTS:
            logger.info("   -> Max attempts reached. Forcing resolution.")
            return {"is_fully_resolved": True}

        # B. Content Saturation
        # If we have > 15 articles, we likely have the major stories.
        # Don't waste time searching for minor details.
        if article_count >= 15:
            logger.info("   -> Content saturation (15+ articles). Forcing resolution.")
            return {"is_fully_resolved": True}

        # C. Date Exhaustion
        # If we have searched back to the MAX_LOOKBACK_DAYS and found nothing,
        # we can't search further.
        if hit_max_lookback and state.attempt_count > 0:
            logger.info("   -> Max lookback window reached. Stopping search.")
            return {"is_fully_resolved": True}

        # --- 3. Pragmatic LLM Evaluation ---

        # If we have NO articles and haven't hit limits, we definitely need to try at least once more
        # (unless we just did the first attempt and got 0 results, then the strategy node handles expansion)
        if article_count == 0 and state.attempt_count == 0:
            logger.info(
                "   -> Zero articles on first attempt. Need strategy expansion."
            )
            return {"is_fully_resolved": False}

        llm = service_manager.get_agent().with_structured_output(SufficiencyCheck)

        # We construct a prompt that encourages the LLM to accept "No News" as an answer
        # if the search depth seems reasonable.
        prompt = f"""You are a pragmatic financial analyst. determine if we have enough context to answer the user's question, OR if further searching is futile.

        User Question: "{state.query}"
        
        --- SEARCH STATUS ---
        - Search Attempts: {state.attempt_count} / {MAX_SEARCH_ATTEMPTS}
        - Articles Found: {article_count}
        - Days Looked Back: {days_searched_depth} days
        
        --- RETRIEVED CONTEXT START ---
        {state.news_context[:3000]}...
        --- RETRIEVED CONTEXT END ---

        ### GUIDELINES:
        1. If the context contains a direct answer -> SUFFICIENT.
        2. If we found relevant articles but they don't explicitly mention the specific event, and we have searched multiple times -> SUFFICIENT (We can assume the event wasn't major news).
        3. If we have found 0 articles but have searched back {days_searched_depth} days -> SUFFICIENT (The answer is "No news was found regarding this").
        4. ONLY mark as 'False' (Insufficient) if you believe a *specific* different search query or date range would yield better results immediately.

        Is this sufficient?
        """

        try:
            result: SufficiencyCheck = await llm.ainvoke(prompt)
            logger.info(
                f"   -> LLM Decision: {'Sufficient' if result.is_sufficient else 'Insufficient'}"
            )
            logger.info(f"   -> Reasoning: {result.reasoning}")
            return {
                "is_fully_resolved": result.is_sufficient,
                "sufficiency_reasoning": result.reasoning,
            }
        except Exception as e:
            logger.error(f"   -> LLM Check failed ({e}). Defaulting to continue.")
            return {"is_fully_resolved": False}

    def _strategize_search(self, state: _AgentState) -> _AgentState:
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

    # async def _call_tool_agent(self, state: _AgentState) -> _AgentState:
    #     """
    #     Uses the LLM (with bound tools) to generate the specific tool call.
    #     """
    #     logger.info("🤖 [Step: Call Tool Agent] Invoking LLM to call tools...")

    #     system_msg = SystemMessage(
    #         content="You are a research assistant. Use the available tools to fetch data based on the user's strategy."
    #     )

    #     # Explicitly instruct the LLM to use the parameters derived from the strategy step
    #     user_msg = (
    #         f"Please fetch news for ticker '{state.ticker}' regarding '{state.query}'. "
    #         f"Use the date range {state.search_from_date} to {state.search_to_date}. "
    #         f"Fetch page {state.current_page}."
    #     )

    #     response = await self.llm_with_tools.ainvoke([system_msg, user_msg])
    #     logger.info(f"   -> LLM generated tool call: {response.tool_calls}")
    #     return {"messages": [response]}

    async def _execute_search_action(self, state: _AgentState) -> _AgentState:
        """
        Merged Node:
        1. Calls LLM to decide on tool parameters.
        2. Executes the tool immediately.
        3. Processes the output (parsing JSON for total_results).
        """
        logger.info("🤖 [Step: Execute Search Action] deciding and running tools...")

        # --- 1. Prepare LLM Call ---
        system_msg = SystemMessage(
            content="You are a research assistant. Use the available tools to fetch data based on the user's strategy."
        )
        user_msg = (
            f"Please fetch news for ticker '{state.ticker}' regarding '{state.query}'. "
            f"Use the date range {state.search_from_date} to {state.search_to_date}. "
            f"Fetch page {state.current_page}."
        )

        # Call LLM
        response = await self.llm_with_tools.ainvoke([system_msg, user_msg])
        output_messages = [response]
        total_results = 0

        # --- 2. Check and Execute Tools Manually ---
        if response.tool_calls:
            for tool_call in response.tool_calls:
                logger.info(f"   -> Executing Tool: {tool_call['name']}")

                # In a larger app, you might look this up in a dict,
                # but here we know we only have one tool.
                if tool_call["name"] == "ingest_stock_news_tool":
                    # Execute the tool
                    tool_output = await ingest_stock_news_tool.ainvoke(
                        tool_call["args"]
                    )

                    # Create the ToolMessage (Required for chat history consistency)
                    tool_msg = ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                    output_messages.append(tool_msg)

                    # --- 3. Process Logic (formerly process_tool_output) ---
                    try:
                        content_json = json.loads(tool_output)
                        total_results = content_json.get("total_results", 0)
                        logger.info(
                            f"   -> Search executed. API Total Results: {total_results}"
                        )
                    except Exception:
                        logger.warning("   -> Could not parse tool output JSON.")

        # Return combined updates
        return {"messages": output_messages, "last_total_results": total_results}

    # def _process_tool_output(self, state: _AgentState) -> _AgentState:
    #     """
    #     Parses the output of the tool execution to update internal state (specifically total_results).
    #     """
    #     last_message = state.messages[-1]

    #     total_results = 0

    #     if isinstance(last_message, ToolMessage):
    #         try:
    #             # The tool returns a JSON string, so we parse it
    #             content_json = json.loads(last_message.content)
    #             total_results = content_json.get("total_results", 0)
    #             logger.info(
    #                 f"   -> Tool output processed. Total API Results available: {total_results}"
    #             )
    #         except Exception:
    #             logger.warning("   -> Could not parse tool output JSON.")

    #     return {"last_total_results": total_results}

    async def _generate_answer(self, state: _AgentState) -> StructuredNewsAnalysis:
        logger.info("✍️  [Step: Generate Answer]")

        has_news = len(state.news_context) > 50

        # Bind the structured output schema
        llm = service_manager.get_agent().with_structured_output(StructuredNewsAnalysis)

        template = (
            "You are a financial analyst. Answer the user question based ONLY on the provided context.\n"
            "Question: '{question}'\n\n"
            "### SEARCH METADATA:\n"
            "The search agent provided the following note regarding data completeness:\n"
            '"{reasoning}"\n\n'
            "### CONTEXT:\n{context}\n\n"
            "### CITATION RULES (STRICT):\n"
            "1. You MUST use inline citations in the format [ID], e.g., [1] or [1][2].\n"
            "2. The ID corresponds to the 'Source ID' provided in the context blocks.\n"
            "3. Do NOT include the URL or Title in the text body, only the bracketed number.\n"
            "4. If the metadata indicates no news was found, state that clearly in the analysis.\n"
        )

        prompt = template.format(
            question=state.query,
            reasoning=state.sufficiency_reasoning,
            context=(
                state.news_context if has_news else "No relevant news articles found."
            ),
        )

        # The LLM returns a Pydantic object now, not a Message
        return await llm.ainvoke(prompt)


if __name__ == "__main__":

    async def main():
        agent = NewsAnalysisAgent()
        input_data = NewsAnalysisInput(
            ticker="MSFT",
            question="What is the reason for the data center removal?",
            from_date="2025-12-01",
            to_date="2025-12-05",
        )
        res = await agent.run(input_data)
        print("\nFINAL OUTPUT:\n", res.analysis)

    asyncio.run(main())
    # agent = NewsAnalysisAgent()
    # png_bytes = agent._graph.get_graph().draw_mermaid_png()

    # with open("graph.png", "wb") as f:
    #     f.write(png_bytes)

    # print("Saved graph as graph.png")
