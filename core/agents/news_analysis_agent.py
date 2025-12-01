import datetime
import operator
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Type

import yfinance as yf
from core.agents.base_agent import AbstractAgent, AgentOutput
from core.services import service_manager
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from newspaper import Article, ArticleException
from pydantic import BaseModel, Field


class NewsAnalysisInput(BaseModel):
    """Input schema for the News Analysis Agent."""

    ticker: str = Field(description="The stock ticker symbol to research.")
    question: str = Field(
        description="The specific question to answer based on the news."
    )


# --- Internal State and Models for the Graph ---
class _AgentState(BaseModel):
    query: str
    ticker: str
    news_context: Annotated[str, operator.add] = ""
    need_query_news: bool = False
    no_news_data: bool = False


class _OutputState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]


class NewsAnalysisAgent(AbstractAgent):
    """Agent for qualitative analysis of news, sentiment, and market events."""

    def __init__(self):
        super().__init__()
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "news_agent"

    @property
    def description(self) -> str:
        return (
            "Focuses on qualitative data: news, market sentiment, "
            "reasons for price volatility, and macro events. Use this for 'why' questions "
            "related to stock price movements or recent developments."
        )

    @classmethod
    def get_input_schema_class(cls) -> Type[BaseModel]:
        return NewsAnalysisInput

    def run(self, input_data: NewsAnalysisInput) -> AgentOutput:
        """Executes the news analysis workflow."""
        print(f"--- [Agent: {self.name}] Executing with input: {input_data.dict()} ---")

        initial_state = {
            "ticker": input_data.ticker,
            "query": input_data.question,
        }

        final_state = self._graph.invoke(initial_state)
        output_content = final_state["messages"][-1].content

        return AgentOutput(agent_name=self.name, output=output_content)

    def _build_graph(self):
        """Builds and compiles the LangGraph workflow."""
        workflow = StateGraph(_AgentState, output_schema=_OutputState)

        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("retrieve", self._retrieve_news)
        workflow.add_node("fetch_data", self._query_and_ingest_stock_news)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._route_query_data,
            {"fetch_data": "fetch_data", "generate_answer": "generate_answer"},
        )
        workflow.add_conditional_edges(
            "fetch_data",
            self._route_news_data,
            {"rewrite_query": "rewrite_query", "retrieve": "retrieve"},
        )
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _is_article_stale(self, publish_str: str, days_threshold: int = 2) -> bool:
        try:
            if not publish_str:
                return True
            pub_date = datetime.fromisoformat(str(publish_str))
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - pub_date) > timedelta(
                days=days_threshold
            )
        except Exception:
            return True

    def _ingest_article(self, url: str, title: str, pubtime: str, ticker: str) -> bool:
        try:
            article_raw = Article(url)
            article_raw.download()
            article_raw.parse()
            source_meta = {
                "url": url,
                "title": title,
                "source": "Yahoo Finance",
                "ticker": ticker,
                "publish_time": pubtime,
            }
            success = service_manager.get_vector_store_manager().ingest_article(
                raw_text=article_raw.text, source_metadata=source_meta
            )
            if success:
                print(f"Successfully ingested: {title}")
            return success
        except ArticleException as e:
            print(f"Skipped article at {url} due to download/parse error: {e}")
            return False

    def _query_and_ingest_stock_news(self, state: _AgentState) -> dict:
        if not state.need_query_news:
            return {}
        print(f"--- Fetching and Ingesting News for {state.ticker} ---")
        stock = yf.Ticker(state.ticker)
        news = stock.get_news(count=5)

        count = 0
        for item in news:
            url = (
                item["content"]["canonicalUrl"]["url"]
                if item["content"]["canonicalUrl"]
                else item["content"]["clickThroughUrl"]["url"]
            )
            title = item["content"]["title"]
            pubdate = item["content"]["pubDate"]

            if self._ingest_article(url, title, pubdate, state.ticker):
                count += 1

        print(f"--- Ingestion Complete. Added {count} new articles. ---")
        return {"need_query_news": False, "no_news_data": count == 0}

    def _retrieve_news(self, state: _AgentState) -> dict:
        print(f"--- [Tool] Retrieving News for {state.ticker} ---")
        filter_dict = {"ticker": state.ticker} if not state.no_news_data else {}
        docs = service_manager.get_vector_store_manager().retrieve(
            query=state.query, filter_dict=filter_dict
        )
        if not docs:
            return {"need_query_news": not state.no_news_data}
        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}"
                for doc in docs
            ]
        )
        stale_count = sum(
            1 for d in docs if self._is_article_stale(d.metadata.get("publish_time"))
        )
        return {
            "news_context": context,
            "need_query_news": stale_count == len(docs) and not state.no_news_data,
        }

    def _generate_answer(self, state: _AgentState) -> dict:
        print("--- Generating Answer ---")
        question = state.query
        context = state.news_context
        template = GENERATE_PROMPT_NO_NEWS if state.no_news_data else GENERATE_PROMPT
        prompt = template.format(
            ticker=state.ticker, question=question, context=context
        )
        response = service_manager.get_agent().invoke(prompt)
        return {"messages": [response]}

    def _route_query_data(self, state: _AgentState) -> str:
        return "fetch_data" if state.need_query_news else "generate_answer"

    def _route_news_data(self, state: _AgentState) -> str:
        # If we just fetched data and still found nothing, we should not loop again.
        # This simple check avoids infinite loops if a ticker truly has no news.
        # A more robust solution could involve a counter in the state.
        if state.no_news_data:
            # We will try one broad retrieval, but won't fetch again.
            return "rewrite_query"
        return "retrieve"

    def _rewrite_query(self, state: _AgentState) -> dict:
        print("--- Rewriting Query ---")
        template = REWRITE_PROMPT_NO_NEWS if state.no_news_data else REWRITE_PROMPT
        llm = service_manager.get_agent()
        prompt = ChatPromptTemplate.from_template(template)
        rewritten_query = (prompt | llm | StrOutputParser()).invoke(
            {"question": state.query}
        )
        print(f"--- Rewritten Query: {rewritten_query} ---")
        # Important: After rewriting for "no_news", prevent another fetch loop.
        return {"query": rewritten_query, "need_query_news": False}


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
    "You are a financial analyst. No specific news was found for {ticker}. "
    "Based on the following general context, provide a possible explanation for the user's question. "
    "Clearly state this is a broader analysis. Keep it to 3 sentences.\n"
    "Original Question: {question}\n"
    "General Context: {context}"
)
REWRITE_PROMPT = (
    "You are a question re-writer for a vector database. "
    "Rewrite the user's question to be concise and focused on keywords and entities. "
    "Return only the rewritten question.\n"
    "Original question: {question}"
)
REWRITE_PROMPT_NO_NEWS = (
    "You are a financial question re-writer. A search for specific news for a stock returned no results. "
    "Rewrite the user's question to search for general market sentiment related to the original topic. "
    "Return only the rewritten question.\n"
    "Original question: {question}"
)


# Example of how to run the new agent
if __name__ == "__main__":
    # 1. Create an instance of the agent
    news_agent = NewsAnalysisAgent()

    # 2. Define the structured input
    user_request = NewsAnalysisInput(
        ticker="NVDA", question="Why did the stock price drop recently?"
    )

    # 3. Execute the agent
    print("Starting News Analysis Agent...")
    final_output = news_agent.run(user_request)

    # 4. Print the result
    print("\n" + "=" * 40)
    print(f"Agent '{final_output.agent_name}' completed its analysis.")
    print("Final Answer:")
    print(final_output.output)
