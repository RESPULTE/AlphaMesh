import asyncio
import json
import logging
import operator
from datetime import datetime, timedelta
from typing import Annotated, List, Optional, Type

from core.agents.base_agent import AbstractAgent
from core.services import service_manager
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
    page_content: str = Field(description="The content of the article.")


class StructuredNewsAnalysis(BaseModel):
    detailed_analysis: str = Field(
        description="A comprehensive, multi-paragraph markdown report answering the user's question with inline citations. [1], [1][2] and etc."
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
        self.tool_map = {t.name: t for t in self.tools}
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

    async def run(self, input_data: NewsAnalysisInput) -> StructuredNewsAnalysis:
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
            "latest_retrieved": None,
            "messages": [],
            "news_context": [],
            "sufficiency_reasoning": "",
        }

        retval = await self._graph.ainvoke(initial_state)

        return StructuredNewsAnalysis(
            agent_name=self.name,
            detailed_analysis=retval["detailed_analysis"],
            sources=retval["sources"],
        )

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
            k=15,
        )

        context_pieces = []
        # Calculate the starting ID based on how many articles are already in the context
        # This prevents ID collision if we loop multiple times (e.g., [1]-[10], then [11]-[20])
        current_article_count = len(state.news_context)
        start_id = current_article_count + 1
        latest_retrieved = None

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
                context_pieces.append(
                    CitedSource(source_id=i, title=title, url=url, page_content=content)
                )
                latest_retrieved = pub_time

        else:
            logger.info("   -> No documents found in store.")

        return {
            "news_context": context_pieces,
            "latest_retrieved": latest_retrieved,
        }

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
        article_count = len(state.news_context)

        # Parse dates to see how far back we have gone
        fmt = "%Y-%m-%d"
        now = datetime.now()
        latest = state.latest_retrieved
        try:
            current_from = datetime.strptime(state.search_from_date, fmt)
            limit_date = now - timedelta(days=MAX_LOOKBACK_DAYS)
            days_searched_depth = (now - current_from).days
            hit_max_lookback = current_from <= limit_date
        except Exception:
            # Fallback if parsing fails
            days_searched_depth = 0
            hit_max_lookback = False

        logger.info(
            f"""   
            -> Metrics: Articles={article_count} | Days Depth={days_searched_depth} | 
            Hit Limit={hit_max_lookback} | Attempt={state.attempt_count}/{MAX_SEARCH_ATTEMPTS} | 
            latest_retrieved={latest or 'N/A'}
            """
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

        if latest is not None and (now - latest) > timedelta(hours=6):
            logger.info(
                "   -> Latest retrieved article is older than 6 hours. Need to refresh."
            )
            return {"is_fully_resolved": False}

        llm = service_manager.get_agent().with_structured_output(SufficiencyCheck)
        context = "\n".join([n.model_dump_json() for n in state.news_context])

        # We construct a prompt that encourages the LLM to accept "No News" as an answer
        # if the search depth seems reasonable.
        prompt = f"""You are a thorough and critical financial analyst. Determine if the provided news context is sufficient to answer the user's question, or if we must dig deeper.

        User Question: "{state.query}"
        
        --- RETRIEVED CONTEXT START ---
        {context}...

        --- RETRIEVED CONTEXT END ---

        ### EVALUATION GUIDELINES:
        1. **Direct Evidence:** If the context explicitly answers the question -> SUFFICIENT.
        2. **Tangential & Correlated Events:** If the exact event is not mentioned, but you see **related events** (e.g., sector-wide trends, macro news, or broader company announcements like restructuring/earnings) that logically explain or correlate with the user's query -> SUFFICIENT. 
           *Example:* If the user asks "Why did the stock crash?" and you find "Tech sector sell-off", that is a sufficient answer.

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

    async def _execute_search_action(self, state: _AgentState) -> _AgentState:
        """
        FIXED: Generic Tool Execution Node.
        1. Calls LLM with bound tools.
        2. Iterates over ANY tool calls returned.
        3. Looks up the tool in self.tool_map and executes it.
        """
        logger.info("🤖 [Step: Execute Search Action] deciding and running tools...")

        # 1. Prepare LLM Call
        system_msg = SystemMessage(
            content="You are a research assistant. Use the available tools to fetch data based on the user's strategy."
        )
        # We explicitly guide the LLM using the state calculated in the previous 'strategize' node
        user_msg = (
            f"Please execute the search strategy for ticker '{state.ticker}'. "
            f"Use Date Range: {state.search_from_date} to {state.search_to_date}. "
            f"Page: {state.current_page}."
        )

        response = await self.llm_with_tools.ainvoke([system_msg, user_msg])
        output_messages = [response]
        total_results = 0

        # 2. Dynamic Tool Execution
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                logger.info(f"   -> LLM selected Tool: {tool_name}")

                # Retrieve the actual tool function from the map
                selected_tool = self.tool_map.get(tool_name)

                if selected_tool:
                    try:
                        # Execute the tool dynamically
                        tool_output = await selected_tool.ainvoke(tool_call["args"])

                        # Create the ToolMessage (Required for chat history consistency)
                        tool_msg = ToolMessage(
                            content=tool_output,
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                        output_messages.append(tool_msg)

                        # 3. Post-Processing (Attempt to parse JSON for state updates)
                        # This logic allows specific tools to update the state variable 'last_total_results'
                        # while ignoring others that might return plain text.
                        try:
                            if isinstance(tool_output, str):
                                data = json.loads(tool_output)
                                # Only update if the specific key exists
                                if "total_results" in data:
                                    total_results = data["total_results"]
                                    logger.info(
                                        f"   -> Extracted total_results: {total_results}"
                                    )
                        except json.JSONDecodeError:
                            # Not a JSON output, or not relevant to total_results
                            pass
                    except Exception as e:
                        logger.error(f"   ❌ Error executing tool {tool_name}: {e}")
                else:
                    logger.warning(
                        f"   ⚠️ Tool '{tool_name}' selected by LLM but not found in tool_map."
                    )

        return {"messages": output_messages, "last_total_results": total_results}

    async def _generate_answer(self, state: _AgentState) -> StructuredNewsAnalysis:
        logger.info("✍️  [Step: Generate Answer]")

        # 1. FIX: Check if we actually have context
        # Check if the string is not empty and contains the article separator
        has_news = len(state.news_context) > 0

        # 2. FIX: specific instructions for the "System"
        system_instructions = (
            "You are a professional financial analyst. Your task is to produce a rigorous, "
            "multi-facet analysis that is grounded in the provided context.\n\n"
            "### CITATION RULES (STRICT):\n"
            "1. Use inline citations with bracketed IDs like [1](url_1) or [1][2](url_1, url_2).\n"
            "2. Each ID must correspond to a 'Source ID' from the provided context.\n"
            "4. Every factual claim must have at least one citation.\n"
            "5. Do NOT cite nonexistent or fabricated IDs.\n\n"
            "### OUTPUT FORMAT:\n"
            "You must output a JSON object. The 'analysis' field must contain the full "
            "narrative report (multiple paragraphs), not just a summary or entity name."
        )

        context = "\n".join([n.model_dump_json() for n in state.news_context])
        user_content = (
            f"### USER QUESTION:\n'{state.query}'\n\n"
            f"### SEARCH METADATA:\n"
            f'Reasoning: "{state.sufficiency_reasoning}"\n\n'
            f"### CONTEXT:\n"
            f"{context if has_news else 'No relevant news articles found.'}\n\n"
            f"Please generate the detailed analysis now."
        )

        # Bind the structured output schema

        # 4. FIX: Pass messages instead of a single formatted string
        messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_content),
        ]
        try:
            retval = await service_manager.get_agent().ainvoke(messages)

            return StructuredNewsAnalysis(
                detailed_analysis=retval.content, sources=state.news_context
            )
        except Exception as e:
            logger.error(f"❌ Error in generation: {e}")
            # Fallback if generation fails
            return StructuredNewsAnalysis(
                detailed_analysis="Error generating analysis due to model failure.",
                sources=[],
            )


if __name__ == "__main__":

    async def main():
        agent = NewsAnalysisAgent()
        input_data = NewsAnalysisInput(
            ticker="AAPL",
            question="reason for price drop Apple",
            from_date="2025-12-01",
            to_date="2025-12-05",
        )
        res = await agent.run(input_data)
        print("\nFINAL OUTPUT:\n", res.detailed_analysis)

    asyncio.run(main())
    # agent = NewsAnalysisAgent()
    # png_bytes = agent._graph.get_graph().draw_mermaid_png()

    # with open("new_analysis.png", "wb") as f:
    #     f.write(png_bytes)

    # print("Saved graph as graph.png")
