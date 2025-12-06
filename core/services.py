# services.py
from core.config import settings


class ServiceManager:
    """
    A centralized manager for initializing and providing access to external services.
    This pattern avoids global variables and makes dependencies explicit.
    """

    def __init__(self):
        self._llm = None
        self._embedding_func = None
        self._graph = None
        self._vector_store = None

        self._financial_db = None
        self._vector_store_manager = None

    def get_agent(self, temperature=0.0):
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

        """Initializes and returns the language model instance."""
        if self._llm is None:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=settings.LLM_MODEL,
                    google_api_key=settings.GOOGLE_API_KEY,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                raise
        if temperature != self._llm.temperature:
            self._llm.temperature = temperature

        return self._llm

    def get_embedding_func(self):
        from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

        """Initializes and returns the embedding model instance."""
        if self._embedding_func is None:
            try:
                self._embedding_func = GoogleGenerativeAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    google_api_key=settings.GOOGLE_API_KEY,
                )
            except Exception as e:
                print(f"Error initializing embedding function: {e}")
                raise
        return self._embedding_func

    def get_graph(self):
        from langchain_neo4j import Neo4jGraph

        """Initializes and returns the Neo4j graph instance."""
        if self._graph is None:
            try:
                self._graph = Neo4jGraph(
                    url=settings.NEO4J_URL,
                    username=settings.NEO4J_USERNAME,
                    password=settings.NEO4J_PASSWORD,
                )
            except Exception as e:
                print(f"Error initializing Neo4j graph: {e}")
                raise
        return self._graph

    def get_vector_store(self):
        from langchain_chroma import Chroma

        """Initializes and returns the Chroma vector store instance."""
        if self._vector_store is None:
            try:
                self._vector_store = Chroma(
                    collection_name=settings.CHROMA_NAME,
                    embedding_function=self.get_embedding_func(),
                    persist_directory=settings.CHROMA_PATH,
                )
            except Exception as e:
                print(f"Error initializing Chroma vector store: {e}")
                raise
        return self._vector_store

    def get_vector_store_retriever(self):
        """Returns a retriever from the Chroma vector store."""
        vector_store = self.get_vector_store()
        return vector_store.as_retriever()

    def get_financial_database(self):
        from core.agents.get_financial_data import FinancialDatabase

        if self._financial_db is None:
            try:
                self._financial_db = FinancialDatabase()
            except Exception as e:
                print(f"Error initializing Financial Database: {e}")
                raise
        return self._financial_db

    def get_vector_store_manager(self):
        from core.agents.rag_agent import VectorStoreManager

        if self._vector_store_manager is None:
            try:
                vector_store = self.get_vector_store()
                self._vector_store_manager = VectorStoreManager(
                    vector_store.as_retriever(),
                    self.get_agent(temperature=0.0),
                    self.get_embedding_func(),
                    vector_store,
                )
            except Exception as e:
                print(f"Error initializing Vector Store Manager: {e}")
                raise
        return self._vector_store_manager

    def get_news_api(self):
        from newsapi import NewsApiClient

        return NewsApiClient(api_key=settings.NEWSAPI_KEY)


service_manager = ServiceManager()
