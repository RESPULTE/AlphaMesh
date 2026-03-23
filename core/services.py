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
        self._financial_db = None
        self._neo4j_adapter = None
        self._chroma_adapter = None
        self._entity_chroma_adapter = None
        self._nodeset_manager = None
        self._dual_store_ingestor = None
        self._retriever = None
        self._reranker = None
        self._memory_retrieval_service = None
        self._user_context_service = None
        self._subgraph_store = None
        self._subgraph_service = None
        self._ticker_validator = None

    def get_agent(self, temperature=0.0):
        """Initializes and returns the language model instance."""
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

        if self._llm is None:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=settings.LLM_MODEL,
                    temperature=temperature,
                    location=settings.GOOGLE_CLOUD_LOCATION,
                    project=settings.GOOGLE_CLOUD_PROJECT,
                )
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                raise
        if temperature != self._llm.temperature:
            self._llm.temperature = temperature

        return self._llm

    def get_embedding_func(self):
        """Initializes and returns the embedding model instance."""
        from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

        if self._embedding_func is None:
            try:
                self._embedding_func = GoogleGenerativeAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    google_api_key=settings.GOOGLE_API_KEY,
                    location=settings.GOOGLE_CLOUD_LOCATION,
                    project=settings.GOOGLE_CLOUD_PROJECT,
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
        # FIX: was importing from non-existent 'core.agents.get_financial_data'
        from core.agents.financial_db import FinancialDatabase

        if self._financial_db is None:
            try:
                self._financial_db = FinancialDatabase()
            except Exception as e:
                print(f"Error initializing Financial Database: {e}")
                raise
        return self._financial_db

    def get_news_api(self):
        from newsapi import NewsApiClient

        return NewsApiClient(api_key=settings.NEWSAPI_KEY)

    def get_neo4j_adapter(self):
        from core.memory.stores.neo4j_adapter import Neo4jAdapter

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
        from core.memory.stores.chroma_adapter import ChromaDBAdapter

        if self._chroma_adapter is None:
            try:
                self._chroma_adapter = ChromaDBAdapter(
                    collection_name=settings.CHROMA_COLLECTION_NEWS,
                    persist_directory=settings.CHROMA_PATH,
                    embedding_function=self.get_embedding_func(),
                )
            except Exception as e:
                print(f"Error initializing ChromaDB adapter: {e}")
                raise
        return self._chroma_adapter

    def get_entity_chroma_adapter(self):
        from core.memory.stores.chroma_adapter import ChromaDBAdapter

        if self._entity_chroma_adapter is None:
            try:
                self._entity_chroma_adapter = ChromaDBAdapter(
                    collection_name=settings.CHROMA_COLLECTION_ENTITIES,
                    persist_directory=settings.CHROMA_PATH,
                    embedding_function=self.get_embedding_func(),
                )
            except Exception as e:
                print(f"Error initializing Entity ChromaDB adapter: {e}")
                raise
        return self._entity_chroma_adapter

    def get_nodeset_manager(self):
        from core.memory.graph.nodeset_manager import NodeSetManager

        if self._nodeset_manager is None:
            try:
                self._nodeset_manager = NodeSetManager(
                    neo4j_adapter=self.get_neo4j_adapter(),
                    entity_chroma_adapter=self.get_entity_chroma_adapter(),
                )
            except Exception as e:
                print(f"Error initializing NodeSetManager: {e}")
                raise
        return self._nodeset_manager

    def get_ingestor(self):
        from core.memory.ingestion.chunker import ArticleChunker
        from core.memory.ingestion.ingestor import DualStoreIngestor

        if self._dual_store_ingestor is None:
            try:
                chunker = ArticleChunker(
                    chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
                )
                self._dual_store_ingestor = DualStoreIngestor(
                    neo4j_adapter=self.get_neo4j_adapter(),
                    chroma_adapter=self.get_chroma_adapter(),
                    entity_chroma_adapter=self.get_entity_chroma_adapter(),
                    nodeset_manager=self.get_nodeset_manager(),
                    embedding_func=self.get_embedding_func(),
                    chunker=chunker,
                    llm=self.get_agent(),
                    subgraph_store=self.get_subgraph_store(),
                )
            except Exception as e:
                print(f"Error initializing DualStoreIngestor: {e}")
                raise
        return self._dual_store_ingestor

    def get_retriever(self):
        from core.memory.retrieval.dual_store_retriever import DualStoreRetriever

        if self._retriever is None:
            try:
                self._retriever = DualStoreRetriever(
                    neo4j_adapter=self.get_neo4j_adapter(),
                    chroma_adapter=self.get_chroma_adapter(),
                    llm=self.get_agent(),
                    reranker=self.get_reranker(),
                )
            except Exception as e:
                print(f"Error initializing DualStoreRetriever: {e}")
                raise
        return self._retriever

    def get_reranker(self):
        from core.memory.retrieval.reranker import CompositeReranker

        if self._reranker is None:
            try:
                self._reranker = CompositeReranker(
                    alpha=settings.RERANK_ALPHA,
                    beta=settings.RERANK_BETA,
                    top_k=settings.RERANK_FINAL_TOP_K,
                )
            except Exception as e:
                print(f"Error initializing CompositeReranker: {e}")
                raise
        return self._reranker

    def get_user_context_service(self):
        from core.memory.user_context_service import UserContextService

        if self._user_context_service is None:
            try:
                self._user_context_service = UserContextService(
                    neo4j_adapter=self.get_neo4j_adapter(),
                    nodeset_manager=self.get_nodeset_manager(),
                )
            except Exception as e:
                print(f"Error initializing UserContextService: {e}")
                raise
        return self._user_context_service

    def get_subgraph_store(self):
        from core.memory.stores.subgraph_store import SubgraphStore

        if self._subgraph_store is None:
            try:
                self._subgraph_store = SubgraphStore(
                    redis_url=settings.REDIS_URL,
                    ttl=settings.SUBGRAPH_TTL_SECONDS,
                )
            except Exception as e:
                print(f"Error initializing SubgraphStore: {e}")
                raise
        return self._subgraph_store

    def get_subgraph_service(self):
        from core.memory.graph.subgraph_service import SubgraphExtractionService

        if self._subgraph_service is None:
            self._subgraph_service = SubgraphExtractionService(
                ingestor=self.get_ingestor(),
                embedding_func=self.get_embedding_func(),
                fuzzy_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
                semantic_threshold=settings.EXTRACTION_SEMANTIC_THRESHOLD,
            )
        return self._subgraph_service

    def get_ticker_validator(self):
        from core.agents.ticker_validation import TickerValidator

        if self._ticker_validator is None:
            try:
                self._ticker_validator = TickerValidator(
                    neo4j_adapter=self.get_neo4j_adapter(),
                    entity_chroma_adapter=self.get_entity_chroma_adapter(),
                )
            except Exception as e:
                print(f"Error initializing TickerValidator: {e}")
                raise
        return self._ticker_validator

    async def startup(self) -> None:
        """
        Run once at application startup before serving any requests.
        Bootstraps the Market + Sector entity taxonomy in both stores
        and initializes all default NodeSets.
        """
        await self.get_nodeset_manager().initialize_default_nodesets()


service_manager = ServiceManager()
