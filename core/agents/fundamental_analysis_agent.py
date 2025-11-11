import sqlite3
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from core.services import ServiceManager
from edgar import Company, set_identity
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
from realtime import SystemMessage

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
        prompt = SystemMessage(
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

    print(f"split '{len(docs)}' documents into '{len(doc_splits)}' splits")

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


class FundamentalAnalysisAgent:
    def __init__(self, db_name="financial_data.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self._create_table()
        set_identity("yeapzing@gmail.com")

    def _create_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS income_statements (
                ticker TEXT,
                concept TEXT,
                label TEXT,
                date TEXT,
                value REAL,
                UNIQUE(ticker, concept, date)
            )
        """
        )
        self.conn.commit()

    def get_financial_data(self, ticker: str, years: int = 5) -> pd.DataFrame:
        # First, try to fetch from the database
        df = pd.read_sql(
            f"SELECT * FROM income_statements WHERE ticker='{ticker}'", self.conn
        )

        if not df.empty:
            print(f"Data for {ticker} found in the database.")
            return df

        # If not in DB, fetch from EDGAR
        print(f"Fetching data for {ticker} from EDGAR.")
        company = Company(ticker)
        filings = company.get_filings(form="10-K").latest(years)
        all_income_dfs = []

        for filing in filings:
            try:
                xbrl = filing.xbrl()
                if not xbrl:
                    continue
                income_statement = xbrl.statements.income_statement()
                df = income_statement.to_dataframe()
                all_income_dfs.append(df)
            except Exception as e:
                print(f"Could not process filing from {filing.filing_date}. Error: {e}")

        if not all_income_dfs:
            return pd.DataFrame()

        # Consolidate and store in DB
        consolidated_df = self._consolidate_data(all_income_dfs)
        self._store_data(ticker, consolidated_df)

        return consolidated_df

    def _consolidate_data(self, all_income_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        # (This logic is adapted from your provided sample)
        merge_keys = ["concept", "label"]
        if not all_income_dfs:
            return pd.DataFrame()

        consolidated_df = all_income_dfs[0].set_index(merge_keys)
        date_cols = [
            col for col in consolidated_df.columns if self.can_convert_to_datetime(col)
        ]
        consolidated_df = consolidated_df[date_cols]

        for next_df in all_income_dfs[1:]:
            next_df_indexed = next_df.set_index(merge_keys)
            next_date_cols = [
                col
                for col in next_df_indexed.columns
                if self.can_convert_to_datetime(col)
            ]
            new_cols_to_add = [
                col for col in next_date_cols if col not in consolidated_df.columns
            ]

            if new_cols_to_add:
                consolidated_df = consolidated_df.join(
                    next_df_indexed[new_cols_to_add], how="outer"
                )

        return consolidated_df.reset_index()

    def _store_data(self, ticker: str, df: pd.DataFrame):
        # Melt the DataFrame to a long format suitable for SQL
        melted_df = df.melt(
            id_vars=["concept", "label"], var_name="date", value_name="value"
        )
        melted_df["ticker"] = ticker

        # Write to SQL
        melted_df.to_sql(
            "income_statements", self.conn, if_exists="append", index=False
        )
        print(f"Data for {ticker} stored in the database.")

    def can_convert_to_datetime(self, s: str) -> bool:
        try:
            pd.to_datetime(s)
            return True
        except (ValueError, TypeError):
            return False

    def calculate_cumulative_growth(
        self, ticker: str, metric: str, years: int = 5
    ) -> Dict[str, Any]:
        df = self.get_financial_data(ticker, years)
        if df.empty:
            return {"error": f"No data for {ticker}"}

        # Find the metric
        metric_row = df[df["concept"] == metric]
        if metric_row.empty:
            return {"error": f"Metric {metric} not found for {ticker}"}

        # Extract values and calculate growth
        date_cols = [col for col in df.columns if self.can_convert_to_datetime(col)]
        values = metric_row[date_cols].iloc[0].dropna()

        if len(values) < 2:
            return {"error": "Not enough data to calculate growth"}

        initial_value = values.iloc[0]
        final_value = values.iloc[-1]

        growth = (final_value - initial_value) / initial_value
        return {
            "ticker": ticker,
            "metric": metric,
            "cumulative_growth": f"{growth:.2%}",
            "initial_value": initial_value,
            "final_value": final_value,
        }

    def compare_metrics(
        self, tickers: List[str], metric: str, years: int = 5
    ) -> go.Figure:
        fig = go.Figure()

        for ticker in tickers:
            df = self.get_financial_data(ticker, years)
            if df.empty:
                continue

            metric_row = df[df["concept"] == metric]
            if metric_row.empty:
                continue

            date_cols = [col for col in df.columns if self.can_convert_to_datetime(col)]
            values = metric_row[date_cols].iloc[0].dropna()

            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(values.index),
                    y=values.values,
                    mode="lines+markers",
                    name=ticker,
                )
            )

        fig.update_layout(
            title=f"Comparison of {metric}",
            xaxis_title="Year",
            yaxis_title="Value",
            legend_title="Tickers",
        )
        return fig


if __name__ == "__main__":
    # --- User Input ---
    stock_ticker = "NVDA"
    user_question = "Why did NVDA stock go down recently?"

    # run_analysis(stock_ticker, user_question)

    # Example usage of the new agent
    agent = FundamentalAnalysisAgent()

    # 1. Fetch data for a single company (will be stored in DB)
    print("\n--- Fetching single company data ---")
    aapl_data = agent.get_financial_data("AAPL")
    print(aapl_data.head())

    # 2. Calculate cumulative growth
    print("\n--- Calculating Cumulative Growth ---")
    growth = agent.calculate_cumulative_growth("AAPL", "us-gaap_Revenues")
    print(growth)

    # 3. Compare metrics for multiple companies
    print("\n--- Comparing Metrics ---")
    fig = agent.compare_metrics(["AAPL", "MSFT", "GOOGL"], "us-gaap_NetIncomeLoss")
    fig.show()
