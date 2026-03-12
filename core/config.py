from pathlib import Path
from typing import Optional

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

    # LLM and Embedding Configuration
    GOOGLE_API_KEY: Optional[str] = Field(default=None, validation_alias="LLM_API_KEY")
    LLM_MODEL: str = Field(default="gemini-2.5-flash-lite")
    EMBEDDING_API_KEY: Optional[str] = Field(
        default=None, validation_alias="EMBEDDING_API_KEY"
    )
    EMBEDDING_MODEL: str = Field(default="gemini-embedding-001")

    # Neo4j Configuration
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"

    # ChromaDB Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NEWS: str = "news_chunks"
    CHROMA_PATH: str = "./chroma_db"
    CHROMA_NAME: Optional[str] = None

    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # NewsAPI Configuration
    NEWSAPI_KEY: Optional[str] = None


settings = Settings()
