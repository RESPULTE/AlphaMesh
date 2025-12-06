import asyncio
import datetime
import operator
from datetime import datetime, timedelta, timezone
from typing import Annotated, Callable, List, Optional, Type

from core.agents.base_agent import AbstractAgent, AgentOutput
from core.services import service_manager
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from newspaper import Article
from pydantic import BaseModel, Field

# --- Tool Definitions ---


def _is_article_stale(publish_str: str, days_threshold: int = 2) -> bool:
    """Helper to check if news is too old."""
    try:
        if not publish_str:
            return True
        pub_date = datetime.fromisoformat(str(publish_str))
        if pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - pub_date) > timedelta(days=days_threshold)
    except Exception:
        return True


def _ingest_article_sync(url: str, title: str, pubtime: str, ticker: str) -> bool:
    """
    Synchronous helper to download and ingest a single article.
    This performs blocking I/O (download) and CPU work (parse).
    """
    try:
        article_raw = Article(url)
        article_raw.download()
        article_raw.parse()

        if len(article_raw.text) < 200:
            return False

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
        return success
    except Exception as e:
        return False


@tool
async def ingest_stock_news_tool(
    ticker: str,
    query: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page: int = 1,
) -> str:
    """
    Fetches news for a stock ticker using NewsAPI and ingests it into the vector DB ASYNCHRONOUSLY.
    """
    print(f"--- [Tool] NewsAPI: Fetching {ticker} (Page {page}) ---")

    # 1. Setup Defaults
    if not query:
        query = ticker
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")
    if not from_date:
        dt_to = datetime.strptime(to_date, "%Y-%m-%d")
        from_date = (dt_to - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        # 2. Call NewsAPI (This part is fast/lightweight, usually fine to keep sync or wrap)
        # We wrap it in a thread just to be safe if the network call hangs
        loop = asyncio.get_running_loop()

        print(f"    Query: '{query}' | Date: {from_date} to {to_date} | Page: {page}")

        response = await loop.run_in_executor(
            None,
            lambda: service_manager.get_news_api().get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language="en",
                sort_by="relevancy",
                page=page,
                page_size=20,
            ),
        )

        status = response.get("status")
        total_results = response.get("totalResults", 0)
        articles = response.get("articles", [])

        if status != "ok":
            return f"Error from NewsAPI: {response.get('message', 'Unknown error')}"

        if not articles:
            return f"No articles found for {ticker} between {from_date} and {to_date}."

        # 3. Asynchronous Ingestion
        # We use a Semaphore to limit concurrent downloads to 10 to avoid rate limiting or timeouts
        sem = asyncio.Semaphore(10)

        async def _process_article(art):
            async with sem:
                url = art.get("url")
                title = art.get("title")
                pub_date = art.get("publishedAt")

                if not url or not title:
                    return False

                # Offload the blocking _ingest_article_sync to a thread
                success = await loop.run_in_executor(
                    None, _ingest_article_sync, url, title, pub_date, ticker
                )
                if success:
                    print(f"   -> Ingested: {title[:40]}...")
                return success

        # Launch all tasks
        tasks = [_process_article(art) for art in articles]
        results = await asyncio.gather(*tasks)
        ingested_count = sum(results)

        # 4. Return Summary
        return (
            f"Success. Ingested {ingested_count} articles from Page {page}. "
            f"Total available matches: {total_results}. "
            f"If needed, call again with 'page={page+1}'."
        )

    except Exception as e:
        return f"Critical Error fetching news for {ticker}: {str(e)}"


# --- Input Schemas ---


class NewsAnalysisInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol to research.")
    question: str = Field(
        description="The specific question to answer based on the news."
    )


# --- Internal State ---


class _AgentState(BaseModel):
    query: str
    ticker: str
    news_context: Annotated[str, operator.add] = ""
    messages: Annotated[List[BaseMessage], operator.add] = []
    need_query_news: bool = False
    no_news_data: bool = False


class _OutputState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]


# --- Main Agent Class ---


class NewsAnalysisAgent(AbstractAgent):
    """Agent for qualitative analysis using NewsAPI (Async)."""

    def __init__(self):
        super().__init__()
        self.tools = [ingest_stock_news_tool]
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "news_agent"

    @property
    def description(self) -> str:
        return "Analyzes news, sentiment, and macro events using NewsAPI."

    @classmethod
    def get_input_schema_class(cls) -> Type[BaseModel]:
        return NewsAnalysisInput

    def register_tool(self, new_tool: Callable):
        self.tools.append(new_tool)
        self._graph = self._build_graph()

    async def run(self, input_data: NewsAnalysisInput) -> AgentOutput:
        """Executes the news analysis workflow asynchronously."""
        print(f"--- [Agent: {self.name}] Executing with input: {input_data.dict()} ---")

        initial_state = {
            "ticker": input_data.ticker,
            "query": input_data.question,
            "messages": [HumanMessage(content=input_data.question)],
            "news_context": "",
        }

        # Use ainvoke for async graph execution
        final_state = await self._graph.ainvoke(initial_state)
        output_content = final_state["messages"][-1].content

        return AgentOutput(agent_name=self.name, output=output_content)

    def _build_graph(self):
        workflow = StateGraph(_AgentState, output_schema=_OutputState)

        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("retrieve", self._retrieve_news_from_vector_store)
        workflow.add_node("decide_action", self._decide_action)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            self._route_retrieval,
            {"decide_action": "decide_action", "generate_answer": "generate_answer"},
        )

        workflow.add_conditional_edges(
            "decide_action",
            self._route_tool_execution,
            {"execute_tools": "execute_tools", "generate_answer": "generate_answer"},
        )

        workflow.add_edge("execute_tools", "retrieve")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    # --- Node Implementations ---

    async def _rewrite_query(self, state: _AgentState) -> dict:
        print("--- Rewriting Query ---")
        template = REWRITE_PROMPT_NO_NEWS if state.no_news_data else REWRITE_PROMPT
        llm = service_manager.get_agent()
        prompt = ChatPromptTemplate.from_template(template)
        # ainvoke for async LLM call
        rewritten_query = await (prompt | llm | StrOutputParser()).ainvoke(
            {"question": state.query}
        )
        print(f"--- Rewritten Query: {rewritten_query} ---")
        return {"query": rewritten_query}

    async def _retrieve_news_from_vector_store(self, state: _AgentState) -> dict:
        print(f"--- Retrieving News for {state.ticker} ---")
        filter_dict = {"ticker": state.ticker}

        # Run vector store retrieval in thread pool if it's blocking,
        # otherwise assume service_manager might have an async method.
        # Here we assume it's blocking, so we offload it.
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(
            None,
            lambda: service_manager.get_vector_store_manager().retrieve(
                query=state.query, filter_dict=filter_dict
            ),
        )

        stale_count = 0
        if docs:
            stale_count = sum(
                1 for d in docs if _is_article_stale(d.metadata.get("publish_time"))
            )

        is_stale_or_empty = (not docs) or (stale_count == len(docs))
        if state.no_news_data:
            is_stale_or_empty = False

        context = (
            "\n\n".join(
                [
                    f"Source: {d.metadata.get('title')} ({d.metadata.get('publish_time')})\nContent: {d.page_content}"
                    for d in docs
                ]
            )
            if docs
            else ""
        )

        return {"news_context": context, "need_query_news": is_stale_or_empty}

    async def _decide_action(self, state: _AgentState) -> dict:
        print("--- Deciding Action (LLM) ---")
        llm = service_manager.get_agent()
        llm_with_tools = llm.bind_tools(self.tools)

        today_str = datetime.now().strftime("%Y-%m-%d")
        sys_msg = (
            f"You are a smart financial researcher. Today is {today_str}.\n"
            f"The user is asking about {state.ticker}. "
            f"Current retrieved context is either empty or stale.\n\n"
            f"You have a tool 'ingest_stock_news_tool' that uses NewsAPI.\n"
            f"- If you need recent news, call it with the ticker.\n"
            f"- If previous results were insufficient, call it again with 'page=2' or a wider date range."
        )

        messages = [SystemMessage(content=sys_msg)] + state.messages
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def _execute_tools(self, state: _AgentState) -> dict:
        """
        Executes tools asynchronously using ainvoke.
        """
        print("--- Executing Tools ---")
        last_message = state.messages[-1]
        tool_map = {t.name: t for t in self.tools}
        outputs = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in tool_map:
                print(f"--- Invoking {tool_name} with {tool_args} ---")
                tool_instance = tool_map[tool_name]

                # IMPORTANT: Use ainvoke to support the async tool definition
                tool_output = await tool_instance.ainvoke(tool_args)

                outputs.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )
            else:
                outputs.append(
                    ToolMessage(
                        content="Error: Tool not found.",
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )

        found_no_data = any("No articles found" in str(o.content) for o in outputs)
        return {
            "messages": outputs,
            "no_news_data": found_no_data,
            "need_query_news": False,
        }

    async def _generate_answer(self, state: _AgentState) -> dict:
        print("--- Generating Answer ---")
        template = GENERATE_PROMPT_NO_NEWS if state.no_news_data else GENERATE_PROMPT
        prompt = template.format(
            ticker=state.ticker, question=state.query, context=state.news_context
        )
        response = await service_manager.get_agent().ainvoke(prompt)
        return {"messages": [response]}

    # --- Routing ---

    def _route_retrieval(self, state: _AgentState) -> str:
        if state.need_query_news:
            return "decide_action"
        return "generate_answer"

    def _route_tool_execution(self, state: _AgentState) -> str:
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "execute_tools"
        return "generate_answer"


# --- Prompts ---
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)
GENERATE_PROMPT_NO_NEWS = (
    "You are a financial analyst. No specific news was found for {ticker} in the requested range. "
    "Based on the following general context, provide a possible explanation for the user's question. "
    "Original Question: {question}\n"
    "General Context: {context}"
)
REWRITE_PROMPT = "Rewrite the user's question to be concise and focused on keywords. Return only the rewritten question.\nOriginal: {question}"
REWRITE_PROMPT_NO_NEWS = "Rewrite the user's question to search for general market sentiment. Return only the rewritten question.\nOriginal: {question}"


# --- Async Execution Example ---
if __name__ == "__main__":

    async def main():
        agent = NewsAnalysisAgent()

        # Example Request
        req = NewsAnalysisInput(
            ticker="TSLA", question="What is the latest news on Tesla's deliveries?"
        )

        print("Starting Async Agent...")
        res = await agent.run(req)

        print("\n" + "=" * 40)
        print("Final Output:")
        print(res.output)

    asyncio.run(main())
