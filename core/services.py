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
        self._memory_system = None
        self._financial_db = None
        self._neo4j_adapter = None
        self._chroma_adapter = None
        self._nodeset_manager = None
        self._dual_store_ingestor = None

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

    def get_graph_search_type(self) -> str:
        """Returns the Cognee search type for graph retrieval."""
        return "graph_completion"

    def get_chunk_search_type(self) -> str:
        """Returns the Cognee search type for vector/chunk retrieval."""
        return "chunks"

    def get_financial_database(self):
        from core.agents.get_financial_data import FinancialDatabase

        if self._financial_db is None:
            try:
                self._financial_db = FinancialDatabase()
            except Exception as e:
                print(f"Error initializing Financial Database: {e}")
                raise
        return self._financial_db

    def get_memory_system(self):
        from core.memory.memory_system import FinancialMemorySystem

        if self._memory_system is None:
            try:
                self._memory_system = FinancialMemorySystem()
            except Exception as e:
                print(f"Error initializing Financial Memory System: {e}")
                raise
        return self._memory_system

    def get_news_api(self):
        from newsapi import NewsApiClient

        return NewsApiClient(api_key=settings.NEWSAPI_KEY)

    def get_neo4j_adapter(self):
        from core.stores.neo4j_adapter import Neo4jAdapter

        if self._neo4j_adapter is None:
            try:
                self._neo4j_adapter = Neo4jAdapter(
                    uri=settings.NEO4J_URI,
                    username=settings.NEO4J_USERNAME,
                    password=settings.NEO4J_PASSWORD,
                    database=settings.NEO4J_DATABASE,
                )
            except Exception as e:
                print(f"Error initializing Neo4j adapter: {e}")
                raise
        return self._neo4j_adapter

    def get_chroma_adapter(self):
        from core.stores.chroma_adapter import ChromaDBAdapter

        if self._chroma_adapter is None:
            try:
                self._chroma_adapter = ChromaDBAdapter(
                    collection_name=settings.CHROMA_COLLECTION_NEWS,
                    persist_directory=settings.CHROMA_PATH,
                )
            except Exception as e:
                print(f"Error initializing ChromaDB adapter: {e}")
                raise
        return self._chroma_adapter

    def get_nodeset_manager(self):
        from core.graph.nodeset_manager import NodeSetManager

        if self._nodeset_manager is None:
            try:
                self._nodeset_manager = NodeSetManager(self.get_neo4j_adapter())
            except Exception as e:
                print(f"Error initializing NodeSetManager: {e}")
                raise
        return self._nodeset_manager

    def get_ingestor(self):
        from core.ingestion.chunker import ArticleChunker
        from core.ingestion.ingestor import DualStoreIngestor

        if self._dual_store_ingestor is None:
            try:
                chunker = ArticleChunker(
                    chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
                )
                self._dual_store_ingestor = DualStoreIngestor(
                    neo4j_adapter=self.get_neo4j_adapter(),
                    chroma_adapter=self.get_chroma_adapter(),
                    nodeset_manager=self.get_nodeset_manager(),
                    embedding_func=self.get_embedding_func(),
                    chunker=chunker,
                )
            except Exception as e:
                print(f"Error initializing DualStoreIngestor: {e}")
                raise
        return self._dual_store_ingestor


service_manager = ServiceManager()
