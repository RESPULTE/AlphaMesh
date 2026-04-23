from pathlib import Path
from typing import List, Optional

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

# Determine the path to .env file relative to this config.py file
_config_dir = Path(__file__).parent.parent
_env_file = _config_dir / ".env"


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=str(_env_file),
        extra="ignore",
        populate_by_name=True,
    )

    # ── LLM and Embedding ─────────────────────────────────────────────────────
    GOOGLE_API_KEY: Optional[str] = Field(default=None, validation_alias="LLM_API_KEY")
    EMBEDDING_API_KEY: Optional[str] = Field(
        default=None, validation_alias="EMBEDDING_API_KEY"
    )
    LLM_MODEL: str = "gemini-2.5-flash-lite"
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NEWS: str = "news_chunks"
    CHROMA_COLLECTION_ENTITIES: str = "entity_nodes"
    CHROMA_PATH: str = "./data/chroma_db"
    CHROMA_NAME: Optional[str] = None

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Extraction ────────────────────────────────────────────────────────────
    EXTRACTION_BATCH_SIZE: int = 8
    EXTRACTION_MAX_CONCURRENCY: int = 10
    EXTRACTION_ENABLED: bool = True
    EXTRACTION_IMMEDIATE: bool = False
    EXTRACTION_LLM_RETRY_ATTEMPTS: int = 3
    EXTRACTION_NEO4J_RETRY_ATTEMPTS: int = 3

    ENTITY_EMBEDDING_BATCH_SIZE: int = 32
    ENTITY_EMBEDDING_MAX_CONCURRENCY: int = 4

    # ── Subgraph / Redis ──────────────────────────────────────────────────────
    REDIS_URL: str = ""
    SUBGRAPH_TTL_SECONDS: int = 3600

    # ── In-memory dedup thresholds ────────────────────────────────────────────
    EXTRACTION_FUZZY_THRESHOLD: float = 69.0
    EXTRACTION_SEMANTIC_THRESHOLD: float = 0.8

    # ── Portfolio ─────────────────────────────────────────────────────────────
    PORTFOLIO_JSON_PATH: str = "data/portfolio.json"

    # ── Retriever ─────────────────────────────────────────────────────────────
    RETRIEVER_MAX_ITERATIONS: int = 2
    RETRIEVER_SEED_TOP_K: int = 10
    RETRIEVER_MAX_PARALLEL_NODES: int = 3
    RETRIEVER_MAX_NEIGHBOR_CANDIDATES: int = 15
    RETRIEVAL_TRACE_ENABLED: bool = False
    RETRIEVAL_TRACE_MAX_RUNS: int = 20
    RETRIEVAL_TRACE_AUTO_EXPORT: bool = False
    RETRIEVAL_TRACE_AUTO_EXPORT_DIR: str = "./data/retrieval_trace_artifacts"

    # ── Memory retrieval ──────────────────────────────────────────────────────
    MEMORY_VECTOR_TOP_K: int = 10
    MEMORY_SIMILARITY_THRESHOLD: float = 0.72

    # ── Re-ranking ────────────────────────────────────────────────────────────
    RERANK_ALPHA: float = 0.8
    RERANK_BETA: float = 0.2
    RERANK_FINAL_TOP_K: int = 15
    RERANK_PREFILTER_K: int = 40
    JINA_API_KEY: Optional[str] = None
    JINA_RERANKER_MODEL: str = "jina-reranker-v2-base-multilingual"

    # ── NewsAPI ───────────────────────────────────────────────────────────────
    NEWSAPI_KEY: Optional[str] = None
    NEWS_FETCH_MAX_ARTICLES: int = 8

    FINANCIAL_DOMAINS: str = ",".join(
        [
            "reuters.com",
            "bbc.co.uk",
            "bbc.com",
            "theguardian.com",
            "ft.com",
            "bloomberg.com",
            "wsj.com",
            "cnbc.com",
            "marketwatch.com",
            "forbes.com",
            "businessinsider.com",
            "economist.com",
            "apnews.com",
            "finance.yahoo.com",
            "seekingalpha.com",
            "investing.com",
            "morningstar.com",
            "barrons.com",
        ]
    )

    # ── Scraping ──────────────────────────────────────────────────────────────
    _SCRAPE_CONCURRENCY: int = 8
    _SCRAPE_TIMEOUT: int = 12
    _MIN_BODY_LENGTH: int = 150

    # ── Google Cloud ──────────────────────────────────────────────────────────
    GOOGLE_CLOUD_LOCATION: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None

    # ── Graph Queue ───────────────────────────────────────────────────────────
    GRAPH_QUEUE_DB_PATH: str = "./data/graph_tasks.db"

    # ── Orchestrator ──────────────────────────────────────────────────────────
    # Hard ceiling on a single analysis run; prevents hung SSE connections.
    ORCHESTRATOR_TIMEOUT_SECONDS: float = 120.0

    # =========================================================================
    # API Layer settings  (consumed by api/ only; still one .env file)
    # =========================================================================

    # ── Authentication ────────────────────────────────────────────────────────
    # Dummy secret used by the placeholder auth adapter; will be replaced by
    # Firebase service account credentials when auth is hardened.
    JWT_SECRET_KEY: str = "changeme-replace-with-firebase-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list is read as a string and split at startup.
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # ── Rate limiting (requests per minute per user) ───────────────────────────
    RATE_LIMIT_DEFAULT: int = 60  # all other authenticated routes

    # ── Session / History persistence ─────────────────────────────────────────
    # Conversations, messages, and login sessions share this SQLite DB file.
    CONVERSATIONS_DB_PATH: str = "./data/conversations.db"

    # ── Market data cache TTLs (seconds) ─────────────────────────────────────
    # Shared in-process cache keyed by ticker; public data, no per-user
    # isolation needed.  The FundamentalAnalysisAgent writes here; the
    # /api/market/{ticker} endpoint reads from the same cache.
    MARKET_CACHE_DB_PATH: str = "./data/market_cache.db"
    MARKET_QUOTE_TTL: int = 60
    MARKET_INTRADAY_TTL: int = 300


settings = Settings()
