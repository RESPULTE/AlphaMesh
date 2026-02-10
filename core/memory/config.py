import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LightRAG Settings
    working_dir: str = Field(default="./rag_storage", validation_alias="WORKING_DIR")
    max_async: int = Field(default=4, validation_alias="MAX_ASYNC")
    max_parallel_insert: int = Field(default=2, validation_alias="MAX_PARALLEL_INSERT")
    chunk_size: int = Field(default=1200, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, validation_alias="CHUNK_OVERLAP_SIZE")

    # Neo4j Settings
    neo4j_uri: str = Field(..., validation_alias="NEO4J_URI")
    neo4j_username: str = Field(..., validation_alias="NEO4J_USERNAME")
    neo4j_password: str = Field(..., validation_alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", validation_alias="NEO4J_DATABASE")

    # Qdrant Settings
    qdrant_url: str = Field(..., validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., validation_alias="QDRANT_API_KEY")

    # Gemini Settings
    gemini_api_key: str = Field(..., validation_alias="GEMINI_API_KEY")
    llm_model: str = Field(default="gemini-2.5-flash-lite", validation_alias="LLM_MODEL")
    embedding_model: str = Field(default="models/text-embedding-004", validation_alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=768, validation_alias="EMBEDDING_DIM")


memory_config = MemoryConfig()
