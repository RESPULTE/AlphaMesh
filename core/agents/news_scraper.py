from typing import Literal

import numpy as np
import yfinance as yf
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from newspaper import article

# --- Constants for Prompts and Settings ---
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

SIMILARITY_THRESHOLD = 0.4  # Tune this based on experiments


def fetch_stock_news(
    ticker: str, llm: GoogleGenerativeAI, max_articles: int = 3
) -> list[Document]:
    """Fetch raw articles and convert them into LangChain Documents (unsummarized)."""

    def _summarize_article(text: str) -> str:
        """Summarizes a single article using the provided language model."""
        prompt = (
            f"Summarize the following article about {ticker} stock. "
            f"Include only the relevant parts related to the company's performance, "
            f"financials, market reactions, or major events.\n\n{text}"
        )
        summary = llm.invoke(prompt).content
        return summary

    stock = yf.Ticker(ticker)
    news = stock.get_news(max_articles)
    docs = []
    for new in news:
        try:
            url = new["content"]["clickThroughUrl"]["url"]
            text = _summarize_article(article(url).text)
            docs.append(
                Document(
                    page_content=text,
                    metadata={"title": new["content"]["title"], "url": url},
                )
            )
        except Exception as e:
            print(f"Error parsing {url}: {e}")
    return docs


def create_retriever_for_stock(
    ticker: str,
    docs: Document,
    embedding_function: GoogleGenerativeAIEmbeddings,
):
    """
    Creates a retriever tool for a given stock ticker by fetching,
    loading, and processing its latest news.
    """

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=embedding_function
    )
    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        f"retrieve_stock_news_{ticker}",
        f"Search and return information about {ticker} stock news.",
    )
    return retriever_tool


# --- Utility Functions ---
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --- Graph Nodes ---
def generate_query_or_respond(state: MessagesState, llm, retriever_tool):
    """Decide whether to use the retriever tool or respond directly."""
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


def grade_documents(
    state: MessagesState, embedding_func: GoogleGenerativeAIEmbeddings
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine if retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    question_emb = embedding_func.embed_query(question)
    context_emb = embedding_func.embed_query(context)

    similarity = _cosine_similarity(np.array(question_emb), np.array(context_emb))

    if similarity > SIMILARITY_THRESHOLD:
        return "generate_answer"
    else:
        return "rewrite_question"


def rewrite_question(state: MessagesState, llm: ChatGoogleGenerativeAI):
    """Rewrite the original user question for better retrieval."""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    return {"messages": [{"role": "user", "content": response.content}]}


def generate_answer(state: MessagesState, llm: ChatGoogleGenerativeAI):
    """Generate a final answer using the retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke(prompt)
    return {"messages": [response]}


def create_graph_workflow(
    llm: ChatGoogleGenerativeAI,
    embedding_func: GoogleGenerativeAIEmbeddings,
    retriever_tool,
):
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(MessagesState)

    # Add nodes to the graph
    workflow.add_node(
        "generate_query_or_respond",
        lambda state: generate_query_or_respond(state, llm, retriever_tool),
    )
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", lambda state: rewrite_question(state, llm))
    workflow.add_node("generate_answer", lambda state: generate_answer(state, llm))

    # Define the graph edges
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: "generate_answer"},
    )
    workflow.add_conditional_edges(
        "retrieve", lambda state: grade_documents(state, embedding_func)
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


from core.services import ServiceManager


def run_analysis(ticker: str, question: str):
    """
    Main function to run the stock analysis agent.
    """
    print("--- Initializing Services ---")
    service_manager = ServiceManager()
    llm = service_manager.get_llm()
    embedding_func = service_manager.get_embedding_func()

    print(f"\n--- retrieving News Data for {ticker} ---")
    docs = fetch_stock_news(ticker, llm)

    print(f"\n--- Creating Retriever for {ticker} ---")
    retriever_tool = create_retriever_for_stock(
        ticker=ticker, docs=docs, embedding_function=embedding_func
    )

    if not retriever_tool:
        print("Failed to create retriever tool. Exiting.")
        return

    print("\n--- Building Graph Workflow ---")
    graph = create_graph_workflow(llm, embedding_func, retriever_tool)

    print("\n--- Executing Graph ---")
    initial_state = {"messages": [{"role": "user", "content": question}]}

    for chunk in graph.stream(initial_state):
        for node, update in chunk.items():
            print(f"\n--- Update from node: {node} ---")
            # The final output is a message object, others might be dicts
            if hasattr(update["messages"][-1], "pretty_print"):
                update["messages"][-1].pretty_print()
            else:
                print(update)


if __name__ == "__main__":
    # --- User Input ---
    stock_ticker = "NVDA"
    user_question = "Why did NVDA stock go up recently?"

    run_analysis(stock_ticker, user_question)
