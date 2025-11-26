from typing import Literal

from pydantic import BaseModel, Field
import yfinance as yf
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from newspaper import Article

# --- Import Services ---
from core.services import service_manager

# --- Constants for Prompts ---
# --- Constants for Prompts ---
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question to search a financial database:"
)

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
NO_NEWS_ERROR_MESSAGE = "No relevant news found."


def fetch_and_ingest_stock_news(ticker: str, max_articles: int = 5):
    """
    Fetches raw articles using yfinance/newspaper and ingests them
    into the centralized Vector Store via the Service Manager.
    """
    print(f"--- Fetching and Ingesting News for {ticker} ---")

    # Get the manager instance
    rag_manager = service_manager.get_vector_store_manager()

    stock = yf.Ticker(ticker)
    news = stock.get_news(max_articles)

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
                "ticker": ticker,
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

        except Exception as e:
            print(f"Error processing {url}: {e}")

    print(f"--- Ingestion Complete. Added {count} articles. ---")


# --- Tool Definition ---


def create_retriever_tool(ticker: str):
    """
    Creates a tool that uses the ServiceManager's RAG capabilities
    specifically filtered for the requested ticker.
    """
    rag_manager = service_manager.get_vector_store_manager()

    @tool
    def retrieve_news(query: str) -> str:
        """
        Search for news and financial data related to the stock.
        Returns relevant context strings.
        """
        # Use the manager's retrieve method which handles embedding and grading
        docs = rag_manager.retrieve(query=query, filter_dict={"ticker": ticker})

        if not docs:
            return NO_NEWS_ERROR_MESSAGE

        # Format documents into a context string
        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}"
                for doc in docs
            ]
        )
        return context

    return retrieve_news


# --- Graph Nodes ---


def generate_query_or_respond(state: MessagesState, llm: BaseChatModel, retriever_tool):
    """Decide whether to use the retriever tool or respond directly."""
    # Bind the specific tool for this ticker
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# --- Data Models ---


class RetrievalGrade(BaseModel):
    """Binary score for retrieval sufficiency."""

    is_sufficient: bool = Field(
        description="True if the context provides enough information to answer the question, False otherwise."
    )
    reason: str = Field(
        description="Brief explanation of why the context is sufficient or insufficient."
    )


# --- Hybrid Grading Function ---


def hybrid_grade_documents(
    state: MessagesState, llm: BaseChatModel
) -> Literal["generate_answer", "rewrite_question"]:
    """
    Option B: Hybrid Heuristic + LLM Grading.

    1. Heuristic: Checks for empty content or missing keywords (fast fail).
    2. LLM: Self-assessment on semantic sufficiency.
    """
    question = state["messages"][0].content
    last_message = state["messages"][-1]

    # --- 1. Heuristics (Fast Fail) ---

    # A. Check if tool was actually called
    if not isinstance(last_message, ToolMessage):
        return "rewrite_question"

    context = last_message.content.strip()

    # B. Check for specific failure strings from the tool
    if NO_NEWS_ERROR_MESSAGE in context or not context:
        print("--- Heuristic Fail: Empty or No News Found ---")
        return "rewrite_question"

    # C. Check for length (Too short implies lack of substance)
    if len(context) < 100:
        print("--- Heuristic Fail: Context too short ---")
        return "rewrite_question"

    # --- 2. LLM Self-Assessment (Slow Check) ---
    print("--- Heuristics Passed. Running LLM Self-Assessment... ---")

    grader_llm = llm.with_structured_output(RetrievalGrade)

    prompt = GRADER_PROMPT.format(question=question, context=context)
    grade_result = grader_llm.invoke(prompt)

    if grade_result.is_sufficient:
        print(f"--- Grading Passed: {grade_result.reason} ---")
        return "generate_answer"
    else:
        print(f"--- Grading Failed: {grade_result.reason} ---")
        return "rewrite_question"


def rewrite_question(state: MessagesState, llm: BaseChatModel):
    """Rewrite the original user question for better retrieval."""
    print("--- Rewriting Question ---")
    # Find the original user question (usually the first message)
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    return {"messages": [{"role": "user", "content": response.content}]}


def generate_answer(state: MessagesState, llm: BaseChatModel):
    """Generate a final answer using the retrieved context."""
    print("--- Generating Answer ---")

    # Get the original question
    question = state["messages"][0].content

    # Get the context from the last ToolMessage
    # We iterate backwards to find the tool output
    context = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            context = msg.content
            break

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke(prompt)
    return {"messages": [response]}


# --- Workflow Construction ---


def create_graph_workflow(llm, retriever_tool):
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node(
        "generate_query_or_respond",
        lambda state: generate_query_or_respond(state, llm, retriever_tool),
    )
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", lambda state: rewrite_question(state, llm))
    workflow.add_node("generate_answer", lambda state: generate_answer(state, llm))

    # Edges
    workflow.add_edge(START, "generate_query_or_respond")

    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )

    # Updated Conditional Edge: Uses hybrid_grade_documents
    workflow.add_conditional_edges(
        "retrieve",
        lambda state: hybrid_grade_documents(state, llm),
        {"generate_answer": "generate_answer", "rewrite_question": "rewrite_question"},
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


if __name__ == "__main__":

    def run_analysis(ticker: str, question: str):
        """
        Main function to run the stock analysis agent.
        """
        print("--- Initializing Services ---")
        llm = service_manager.get_agent()

        # 1. Fetch & Store Data (Automatic Ingestion)
        fetch_and_ingest_stock_news(ticker)

        # 2. Create Tool wrapping the ServiceManager Retrieval
        print(f"\n--- Creating Retriever Tool for {ticker} ---")
        retriever_tool = create_retriever_tool(ticker)

        # 3. Build Graph
        print("\n--- Building Graph Workflow ---")
        graph = create_graph_workflow(llm, retriever_tool)

        # 4. Execute
        print("\n--- Executing Graph ---")
        initial_state = {"messages": [{"role": "user", "content": question}]}

        for chunk in graph.stream(initial_state):
            for node, update in chunk.items():
                print(f"\n--- Update from node: {node} ---")
                if hasattr(update["messages"][-1], "pretty_print"):
                    update["messages"][-1].pretty_print()
                else:
                    print(update)

    # --- User Input ---
    stock_ticker = "NVDA"
    user_question = "Why did NVDA stock bad?"

    run_analysis(stock_ticker, user_question)
