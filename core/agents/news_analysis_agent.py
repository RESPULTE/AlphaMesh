import datetime
import operator
from datetime import datetime, timedelta, timezone
from typing import Annotated, List

import yfinance as yf

# --- Import Services ---
from core.services import service_manager
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from newspaper import Article, ArticleException
from pydantic import BaseModel

# --- Constants for Prompts ---
# --- Constants for Prompts ---


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

GRADER_PROMPT = (
    "You are a grader assessing whether a retrieved financial news article contains "
    "sufficient information to answer a user question.\n\n"
    "Retrieved Context:\n{context}\n\n"
    "User Question:\n{question}\n\n"
    "Assess if the context contains specific facts, numbers, or explanations required to answer the question. "
    "If the context is vague, unrelated, or empty, mark as insufficient."
)
# --- Ingestion Logic ---


class AgentState(BaseModel):
    query: str
    ticker: str
    news_context: Annotated[str, operator.add] = ""

    need_query_news: bool = False
    no_news_data: bool = False


class OutputState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]


# --- Helper Functions ---


def is_article_stale(publish_str: str, days_threshold: int = 2) -> bool:
    """Parses date string and checks if it's older than X hours."""
    try:
        # Handle various date formats or ISO strings
        if not publish_str:
            return True

        # Assuming ISO format from ingestion
        pub_date = datetime.fromisoformat(str(publish_str))

        # If naive, assume local/UTC matching system time (simplified for example)
        if pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=timezone.utc)

        diff = datetime.now(timezone.utc) - pub_date
        return diff > timedelta(days=days_threshold)
    except Exception:
        # If we can't parse the date, assume it's stale to be safe
        return True


def query_and_ingest_stock_news(state: AgentState):
    """
    Fetches raw articles using yfinance/newspaper and ingests them
    into the centralized Vector Store via the Service Manager.
    """

    if not state.need_query_news:
        return {}

    print(f"--- Fetching and Ingesting News for {state.ticker} ---")

    # Get the manager instance
    rag_manager = service_manager.get_vector_store_manager()

    stock = yf.Ticker(state.ticker)
    news = stock.get_news(1)  #!change back to 5

    count = 0
    for new in news:
        if new["content"]["contentType"] == "VIDEO":
            continue

        url = (
            new["content"]["clickThroughUrl"].get("url")
            if new["content"]["clickThroughUrl"] is not None
            else new["content"]["canonicalUrl"].get("url")
        )

        try:
            # Download Raw Article
            article_raw = Article(url)
            article_raw.download()
            article_raw.parse()

            # Prepare Metadata
            source_meta = {
                "url": url,
                "title": new["content"]["title"],
                "source": "Yahoo Finance",  # or extract from article_raw
                "ticker": state.ticker,
                "publish_time": new["content"]["pubDate"],
            }

            # Ingest into RAG System (Handles summarization & chunking internally)
            success = rag_manager.ingest_article(
                raw_text=article_raw.text, source_metadata=source_meta
            )

            if success:
                print(f"Successfully ingested: {new['content']['title']}")
                count += 1
            else:
                print(f"Skipped (Duplicate or Empty): {new['content']['title']}")

        except (Exception, ArticleException) as e:
            print(f"Error processing {url}: {e}")

    print(f"--- Ingestion Complete. Added {count} articles. ---")

    return {"need_query_news": False, "no_news_data": True}


# --- Tool Definition ---


def retrieve_news(state: AgentState) -> str:
    """
    Search for news and financial data related to the stock.
    Returns relevant context strings.
    """

    print(f"--- [Tool] Retrieving News for {state.ticker} ---")
    # Use the manager's retrieve method which handles embedding and grading
    filter_dict = {"ticker": state.ticker} if not state.no_news_data else {}
    docs = service_manager.get_vector_store_manager().retrieve(
        query=state.query, filter_dict=filter_dict
    )

    if not docs:
        return {"need_query_news": not state.no_news_data}

    # Format documents into a context string
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}"
            for doc in docs
        ]
    )
    stale_count = 0
    for d in docs:
        if is_article_stale(d.metadata.get("publish_time")):
            stale_count += 1

    return {
        "news_context": context,
        "need_query_news": stale_count == len(docs) and not state.no_news_data,
    }


# --- Graph Nodes ---


# def generate_query_or_respond(state: MessagesState, llm: BaseChatModel, retriever_tool):
#     """Decide whether to use the retriever tool or respond directly."""
#     # Bind the specific tool for this ticker
#     response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
#     return {"messages": [response]}


# --- Data Models ---


# class RetrievalGrade(BaseModel):
#     """Binary score for retrieval sufficiency."""

#     is_sufficient: bool = Field(
#         description="True if the context provides enough information to answer the question, False otherwise."
#     )
#     reason: str = Field(
#         description="Brief explanation of why the context is sufficient or insufficient."
#     )


# --- Hybrid Grading Function ---


# def hybrid_grade_documents(
#     state: MessagesState, llm: BaseChatModel
# ) -> Literal["generate_answer", "rewrite_question"]:
#     """
#     Option B: Hybrid Heuristic + LLM Grading.

#     1. Heuristic: Checks for empty content or missing keywords (fast fail).
#     2. LLM: Self-assessment on semantic sufficiency.
#     """
#     question = state["messages"][0].content
#     last_message = state["messages"][-1]

#     # --- 1. Heuristics (Fast Fail) ---

#     # A. Check if tool was actually called
#     if not isinstance(last_message, ToolMessage):
#         return "rewrite_question"

#     context = last_message.content.strip()

#     # B. Check for specific failure strings from the tool
#     if NO_NEWS_ERROR_MESSAGE in context or not context:
#         print("--- Heuristic Fail: Empty or No News Found ---")
#         return "rewrite_question"

#     grader_llm = llm.with_structured_output(RetrievalGrade)

#     prompt = GRADER_PROMPT.format(question=question, context=context)
#     grade_result = grader_llm.invoke(prompt)

#     if grade_result.is_sufficient:
#         print(f"--- Grading Passed: {grade_result.reason} ---")
#         return "generate_answer"
#     else:
#         print(f"--- Grading Failed: {grade_result.reason} ---")
#         return "rewrite_question"


def generate_answer(state: AgentState, llm: BaseChatModel):
    """Generate a final answer using the retrieved context."""
    print("--- Generating Answer ---")
    question = state.query
    context = state.news_context

    if state.no_news_data:
        # No specific news was found, use a prompt for reasoning with general context
        prompt_template = (
            "You are a financial analyst providing insights. No specific news was found for the company {ticker}. "
            "However, the following context describes the general market or sector sentiment.\n\n"
            "Based on this general context, provide a possible explanation for the user's question. "
            "Clearly state that this is a broader analysis due to the lack of company-specific information. "
            "Keep the answer concise (3 sentences max).\n\n"
            "Original Question: {question}\n"
            "General Context: {context}\n\n"
            "Your reasoned analysis:"
        )
        prompt = prompt_template.format(
            ticker=state.ticker, question=question, context=context
        )
    else:
        # Specific news was found, use the standard generation prompt
        prompt = GENERATE_PROMPT.format(question=question, context=context)

    response = llm.invoke(prompt)
    return {"messages": [response]}


# --- Workflow Construction ---


def route_query_data(state: AgentState):
    return "fetch_data" if state.need_query_news else "generate_answer"


def route_news_data(state: AgentState):
    return "rewrite_query" if state.no_news_data else "retrieve"


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrites the user's query to be more effective for retrieval."""
    print("--- Rewriting Query ---")

    if state.no_news_data:
        # This is the second pass, no specific news was found.
        # Broaden the query to find general market sentiment.
        template = (
            "You are a financial question re-writer. A search for specific news for a stock ticker returned no results. "
            "Your goal is to re-write the user's question to search for general market sentiment or news related to the original query's topic, but not specific to the ticker. "
            "The rewritten question should be concise and focus on keywords and entities. "
            "Return only the rewritten question, with no other text or explanation. "
            "Original question: {question}\n"
            "Rewritten question for general sentiment:"
        )
    else:
        # This is the first pass. Optimize for specific document retrieval.
        template = (
            "You are a question re-writer. Your goal is to re-write a user's question "
            "to be more effective for retrieving relevant documents from a vector database. "
            "The rewritten question should be concise and focus on keywords and entities. "
            "Return only the rewritten question, with no other text or explanation. "
            "Original question: {question}\n"
            "Rewritten question:"
        )

    llm = service_manager.get_agent()
    prompt = ChatPromptTemplate.from_template(template)
    rewritten_query = (prompt | llm | StrOutputParser()).invoke(
        {"question": state.query}
    )
    print(f"--- Rewritten Query: {rewritten_query} ---")
    return {"query": rewritten_query}


def create_graph_workflow(llm):
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState, output_schema=OutputState)

    # Add nodes
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve_news)
    workflow.add_node("fetch_data", query_and_ingest_stock_news)
    # workflow.add_node("grade", hybrid_grade_documents)
    workflow.add_node("generate_answer", lambda state: generate_answer(state, llm))

    # Edges
    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        route_query_data,
        {"fetch_data": "fetch_data", "generate_answer": "generate_answer"},
    )

    workflow.add_conditional_edges(
        "fetch_data",
        route_news_data,
        {"rewrite_query": "rewrite_query", "retrieve": "retrieve"},
    )
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def run_analysis(ticker: str, question: str) -> str:
    """
    Main function to run the stock analysis agent.
    """
    print("--- Initializing News Analysis Services ---")
    llm = service_manager.get_agent()

    # 2. Create Tool wrapping the ServiceManager Retrieval
    print(f"\n--- Creating Retriever Tool for {ticker} ---")

    # 3. Build Graph
    print("\n--- Building News Graph Workflow ---")
    graph = create_graph_workflow(llm)

    # 4. Execute
    print("\n--- Executing News Graph ---")
    initial_state = {"ticker": ticker, "query": question}
    final_state = graph.invoke(initial_state)

    return final_state["messages"][-1].content


if __name__ == "__main__":
    # --- User Input ---
    stock_ticker = "MSFT"
    user_question = "MSFT company revenue rise"
    result = run_analysis(stock_ticker, user_question)
    print("\n--- Final News Analysis Result ---")
    print(result)
