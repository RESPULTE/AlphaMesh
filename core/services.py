# core/services.py
from core.config import settings


class ServiceManager:
    """
    Centralised manager for all service singletons.

    Changes from previous version
    ------------------------------
    - get_retriever(): uses prefilter= instead of reranker= � DualStoreRetriever
      now takes a CompositePrefilter directly; the full TwoStageReranker (with
      Jina) is used only at the final selection point in _rendezvous_node.
    - get_reranker(): returns TwoStageReranker (CompositePrefilter + Jina).
    - get_prefilter(): new � returns the shared CompositePrefilter singleton.
    """

    def __init__(self):
        self._llm = None
        self._embedding_func = None
        self._financial_db = None
        self._neo4j_adapter = None
        self._chroma_adapter = None
        self._entity_chroma_adapter = None
        self._nodeset_manager = None
        self._entity_resolver = None
        self._relationship_extractor = None
        self._graph_queue_manager = None
        self._dual_store_ingestor = None
        self._retriever = None
        self._prefilter = None
        self._reranker = None
        self._memory_retrieval_service = None
        self._user_context_service = None
        self._subgraph_service = None
        self._ticker_validator = None
        self._market_data_service = None
        self._orchestrator_agent = None

    def get_agent(self, temperature=0.0):
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

        if self._llm is None:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=settings.LLM_MODEL,
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                )
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                raise
        if temperature != self._llm.temperature:
            self._llm.temperature = temperature
        return self._llm

    def get_embedding_func(self):
        from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

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
        return "graph_completion"

    def get_chunk_search_type(self) -> str:
        return "chunks"

    def get_financial_database(self):
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
                print(f"Error initializing Neo4j db: {e}")
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
                print(f"Error initializing ChromaDB db: {e}")
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
                print(f"Error initializing Entity ChromaDB db: {e}")
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

    # -- EntityResolver --------------------------------------------------------

    def get_entity_resolver(self):
        from core.memory.graph.entity_resolver import EntityResolver

        if self._entity_resolver is None:
            try:
                self._entity_resolver = EntityResolver(
                    neo4j_adapter=self.get_neo4j_adapter(),
                    entity_chroma_adapter=self.get_entity_chroma_adapter(),
                    neo4j_fuzzy_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
                    rapidfuzz_threshold=settings.EXTRACTION_FUZZY_THRESHOLD,
                )
            except Exception as e:
                print(f"Error initializing EntityResolver: {e}")
                raise
        return self._entity_resolver

    def get_relationship_extractor(self):
        from core.memory.graph.queue.relationship_extractor import RelationshipExtractor

        if self._relationship_extractor is None:
            self._relationship_extractor = RelationshipExtractor()
        return self._relationship_extractor

    # -- GraphQueueManager -----------------------------------------------------

    def get_graph_queue_manager(self):
        from core.memory.graph.graph_queue import GraphQueueManager

        if self._graph_queue_manager is None:
            try:

                def _llm_provider(config: dict | None):
                    temperature = 0.0
                    if config and isinstance(config, dict):
                        temperature = float(config.get("temperature", 0.0))
                    return self.get_agent(temperature=temperature)

                self._graph_queue_manager = GraphQueueManager(
                    entity_resolver=self.get_entity_resolver(),
                    graph_writer=self.get_neo4j_adapter(),
                    relationship_extractor=self.get_relationship_extractor(),
                    entity_extractor=self.get_ingestor().extract_entities_for_chunks,
                    llm_provider=_llm_provider,
                )
            except Exception as e:
                print(f"Error initializing GraphQueueManager: {e}")
                raise
        return self._graph_queue_manager

    # -- Ingestor --------------------------------------------------------------

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
                    entity_resolver=self.get_entity_resolver(),
                    chunker=chunker,
                    llm=self.get_agent(),
                )
            except Exception as e:
                print(f"Error initializing DualStoreIngestor: {e}")
                raise
        return self._dual_store_ingestor

    def get_retriever(self):
        from core.memory.retrieval.dual_store_retriever import DualStoreRetriever
        from core.memory.retrieval.tracing import NetworkXRetrievalTraceSink

        if self._retriever is None:
            try:
                self._retriever = DualStoreRetriever(
                    neo4j_adapter=self.get_neo4j_adapter(),
                    chroma_adapter=self.get_chroma_adapter(),
                    prefilter=self.get_prefilter(),
                )
            except Exception as e:
                print(f"Error initializing DualStoreRetriever: {e}")
                raise
        return self._retriever

    def get_prefilter(self):
        """
        Return the shared CompositePrefilter singleton.

        Used directly by DualStoreRetriever for fast intermediate ordering of
        memory chunks, and as stage 1 inside TwoStageReranker.
        """
        from core.memory.retrieval.reranker import CompositePrefilter

        if self._prefilter is None:
            try:
                self._prefilter = CompositePrefilter(
                    alpha=settings.RERANK_ALPHA,
                    beta=settings.RERANK_BETA,
                    prefilter_k=settings.RERANK_PREFILTER_K,
                )
            except Exception as e:
                print(f"Error initializing CompositePrefilter: {e}")
                raise
        return self._prefilter

    def get_reranker(self):
        """
        Return the TwoStageReranker singleton.

        Used at the final selection point (_rendezvous_node) where all chunk
        sources are combined and the Jina cross-encoder makes the definitive
        top-k selection.
        """
        from core.memory.retrieval.reranker import TwoStageReranker

        if self._reranker is None:
            try:
                self._reranker = TwoStageReranker(
                    prefilter=self.get_prefilter(),
                    top_k=settings.RERANK_FINAL_TOP_K,
                    jina_api_key=settings.JINA_API_KEY,
                    jina_model=settings.JINA_RERANKER_MODEL,
                )
            except Exception as e:
                print(f"Error initializing TwoStageReranker: {e}")
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

    def get_market_data_service(self):
        from core.market_data_service import MarketDataService

        if self._market_data_service is None:
            try:
                self._market_data_service = MarketDataService()
            except Exception as e:
                print(f"Error initializing MarketDataService: {e}")
                raise
        return self._market_data_service

    def get_orchestrator_agent(self):
        from core.agents.orchestrator_agent import OrchestratorAgent

        if self._orchestrator_agent is None:
            try:
                self._orchestrator_agent = OrchestratorAgent()
            except Exception as e:
                print(f"Error initializing OrchestratorAgent: {e}")
                raise
        return self._orchestrator_agent

    async def startup(self) -> None:
        """
        Run once at application startup.
        Order matters: NodeSetManager bootstrap must finish before GraphQueueManager
        starts (recovery may write to Neo4j).
        """
        await self.get_nodeset_manager().initialize_default_nodesets()
        await self.get_graph_queue_manager().start()

    async def shutdown(self) -> None:
        """Run at application teardown to drain graph queues gracefully."""
        if self._graph_queue_manager is not None:
            await self._graph_queue_manager.shutdown()


service_manager = ServiceManager()
