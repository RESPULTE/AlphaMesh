import asyncio
import json
import logging
import operator
from datetime import datetime, timedelta
from typing import Annotated, List, Optional, Type

from core.agents.base_agent import AbstractAgent
from core.agents.models import BaseAgentInput, BaseAgentOutput
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


class NewsAnalysisOutput(BaseAgentOutput):
    """Data container for the News Analysis Agent."""

    agent_name: str = "news_agent"
    sources: List[CitedSource] = Field(
        description="The list of raw source articles gathered by the agent."
    )

    def get_llm_context_str(self) -> str:
        """Formats the list of sources into a numbered, citable block for the analyst LLM."""
        if not self.sources:
            return "### REPORT FROM news_agent\nNo relevant news articles were found."

        header = "### REPORT FROM news_agent (Qualitative News Analysis)\n"
        # Format each source with its citation ID prominently displayed
        formatted_sources = "\n".join(
            [
                f"[{s.source_id}] Title: {s.title}\n"
                f"    URL: {s.url}\n"
                f'    Content Snippet: "{s.page_content[:300]}..."'
                for s in self.sources
            ]
        )
        return header + formatted_sources


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

        return NewsAnalysisOutput(**final_state)

    def _build_graph(self):
        workflow = StateGraph(_AgentState, output_schema=NewsAnalysisOutput)

        # Define Nodes
        workflow.add_node("parse_input", self._parse_input)
        workflow.add_node("retrieve", self._retrieve_news)
        workflow.add_node("evaluate_sufficiency", self._evaluate_sufficiency)
        workflow.add_node("strategize_search", self._strategize_search)

        # NEW: The combined node
        workflow.add_node("execute_search_action", self._execute_search_action)

        workflow.add_node("generate_analysis", self._generate_analysis)

        # Define Edges
        workflow.add_edge(START, "parse_input")
        workflow.add_edge("parse_input", "retrieve")
        workflow.add_edge("retrieve", "evaluate_sufficiency")

        # Sufficiency Logic (With the dictionary fix from the previous step)
        workflow.add_conditional_edges(
            "evaluate_sufficiency",
            lambda state: (
                "generate_analysis" if state.is_fully_resolved else "strategize_search"
            ),
            {
                "generate_analysis": "generate_analysis",
                "strategize_search": "strategize_search",
            },
        )

        # Strategy Logic
        # Note: We now point to 'execute_search_action' instead of 'call_tool_agent'
        workflow.add_conditional_edges(
            "strategize_search",
            lambda state: (
                "execute_search_action"
                if state.needs_more_data
                else "generate_analysis"
            ),
            {
                "execute_search_action": "execute_search_action",
                "generate_analysis": "generate_analysis",
            },
        )

        # Simplified Flow
        # execute_search_action now loops directly back to retrieve
        workflow.add_edge("execute_search_action", "retrieve")

        workflow.add_edge("generate_analysis", END)

        return workflow.compile()

    # --- Node Implementations ---

    def _parse_input(self, state: BaseAgentInput) -> _AgentState:
        if (datetime.now() - state.start_date) > timedelta(days=MAX_LOOKBACK_DAYS):
            state.start_date = datetime.now() - timedelta(days=MAX_LOOKBACK_DAYS)

        return state.model_dump().copy()

    async def _retrieve_news(self, state: _AgentState) -> dict:
        # REWRITTEN: Use state.attribute access
        logger.info(
            f"🔍 [Step: Retrieve] Checking vector store (Attempt {state.attempt_count})..."
        )

        docs = await asyncio.to_thread(
            service_manager.get_vector_store_manager().retrieve,
            query=state.vector_query,
            filter_dict={"ticker": state.ticker},
            k=20,
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

        if state.start_date and (
            (datetime.now() - state.start_date).days > MAX_LOOKBACK_DAYS
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
            potential_from = limit_date

        logger.info(
            f"   -> Strategy: Expanding range to {potential_from.strftime('%Y-%m-%d')}."
        )
        return {
            "needs_more_data": True,
            "current_page": 1,
            "start_date": potential_from,
            "attempt_count": state.attempt_count + 1,
        }

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
            f"Use Date Range: {state.start_date} to {state.end_date}. "
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

    async def _generate_analysis(self, state: _AgentState) -> NewsAnalysisOutput:
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

            return NewsAnalysisOutput(
                analysis=retval.content, sources=state.news_context
            )
        except Exception as e:
            logger.error(f"❌ Error in generation: {e}")
            # Fallback if generation fails
            return NewsAnalysisOutput(
                analysis="Error generating analysis due to model failure.",
                sources=[],
            )


if __name__ == "__main__":

    async def main():
        agent = NewsAnalysisAgent()
        input_data = BaseAgentInput(
            ticker="AAPL",
            vector_query="reason price rise Apple",
            query="why did apple's share price rise so much recently?",
            start_date="2025-12-01",
            end_date="2025-12-05",
        )
        res = await agent.run(input_data)
        print("\n--- RAW AGENT OUTPUT ---\n")
        print(f"Found {len(res.sources)} sources:")
        for source in res.sources:
            print(f"- {source.title}")

        print(res.analysis)

        for s in res.sources:
            print(s.model_dump_json(indent=2))

    asyncio.run(main())

    # agent = NewsAnalysisAgent()
    # png_bytes = agent._graph.get_graph().draw_mermaid_png()

    # with open("new_analysis.png", "wb") as f:
    #     f.write(png_bytes)

    # print("Saved graph as graph.png")
