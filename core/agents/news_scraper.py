from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.agents import AgentType, initialize_agent
from langchain_community.vectorstores import Chroma
from langchain_core import Document

from core.services import get_embedding_func, get_llm


tools = [YahooFinanceNewsTool()]

vector_store = Chroma(embedding_function=get_embedding_func())

# --- 5. Create the LangChain Agent ---
# Initialize the agent with the Gemini model and the yfinance tool.
# We are using the ZERO_SHOT_REACT_DESCRIPTION agent type, which is a versatile choice.
agent = initialize_agent(
    tools, get_llm(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# --- 6. Define the Main Logic to Fetch and Store News ---
def get_and_store_news(query: str):
    """
    Uses the agent to fetch news about a query and stores it in the Chroma vector store.
    """
    print(f"Fetching news for: {query}")
    # Run the agent to get the news from the yfinance tool
    news_result = agent.run(query)

    print("\n--- News Fetched ---")
    print(news_result)

    # Create a LangChain Document object from the news to store it in Chroma
    news_document = Document(page_content=news_result, metadata={"source": "yfinance"})

    # Add the news document to the Chroma vector store
    vector_store.add_documents([news_document])
    print("\n--- News stored in Chroma vector store ---")

    # You can now query the vector store for the stored information
    # For example, let's do a similarity search for the original query
    similar_docs = vector_store.similarity_search(query)
    print("\n--- Verifying storage with a similarity search ---")
    print(similar_docs[0].page_content)


# --- 7. Run the agent ---
if __name__ == "__main__":
    get_and_store_news("Get the latest news about Microsoft")
