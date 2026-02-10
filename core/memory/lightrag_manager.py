import os
import asyncio
from typing import Dict, Optional
from lightrag.lightrag import LightRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs
import numpy as np

from core.memory.config import memory_config
from core.memory.models import GLOBAL_ENTITY_TYPES, USER_ENTITY_TYPES
from core.memory.extraction_prompts import build_global_extraction_prompt, build_user_extraction_prompt


class LightRAGManager:
    """
    Manages LightRAG instances with workspace isolation.
    All instances share the same Neo4j and Qdrant backends.
    """

    def __init__(self):
        self._global_rag: Optional[LightRAG] = None
        self._user_rags: Dict[str, LightRAG] = {}
        self._lock = asyncio.Lock()

    async def get_global(self) -> LightRAG:
        """Returns the global LightRAG instance."""
        async with self._lock:
            if self._global_rag is None:
                self._global_rag = await self._create_instance(
                    workspace="global", 
                    entity_types=GLOBAL_ENTITY_TYPES,
                    prompt=build_global_extraction_prompt()
                )
            return self._global_rag

    async def get_user(self, user_id: str) -> LightRAG:
        """Returns a user-specific LightRAG instance."""
        workspace = f"user_{user_id}"
        async with self._lock:
            if workspace not in self._user_rags:
                self._user_rags[workspace] = await self._create_instance(
                    workspace=workspace,
                    entity_types=USER_ENTITY_TYPES,
                    prompt=build_user_extraction_prompt()
                )
            return self._user_rags[workspace]

    async def _create_instance(self, workspace: str, entity_types: list[str], prompt: str) -> LightRAG:
        """Initializes a LightRAG instance with the specified workspace."""
        
        # Inject storage-specific environmental variables BEFORE creating the LightRAG instance
        # This is critical because LightRAG checks these variables in __post_init__
        os.environ["NEO4J_URI"] = memory_config.neo4j_uri
        os.environ["NEO4J_USERNAME"] = memory_config.neo4j_username
        os.environ["NEO4J_PASSWORD"] = memory_config.neo4j_password
        os.environ["NEO4J_DATABASE"] = memory_config.neo4j_database
        
        os.environ["QDRANT_URL"] = memory_config.qdrant_url
        os.environ["QDRANT_API_KEY"] = memory_config.qdrant_api_key

        # Setup Gemini LLM function
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
            return await gemini_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=memory_config.gemini_api_key,
                model_name=memory_config.llm_model,
                **kwargs
            )

        # Setup Gemini Embedding function
        @wrap_embedding_func_with_attrs(
            embedding_dim=memory_config.embedding_dim,
            max_token_size=2048,
            model_name=memory_config.embedding_model
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await gemini_embed.func(
                texts, 
                api_key=memory_config.gemini_api_key, 
                model=memory_config.embedding_model
            )

        rag = LightRAG(
            workspace=workspace,
            working_dir=os.path.join(memory_config.working_dir, workspace),
            llm_model_func=llm_model_func,
            llm_model_name=memory_config.llm_model,
            embedding_func=embedding_func,
            # Storage Config
            kv_storage="JsonKVStorage", # Small scale metadata locally, Neo4j/Qdrant for core
            doc_status_storage="JsonDocStatusStorage",
            vector_storage="QdrantVectorDBStorage",
            graph_storage="Neo4JStorage",
            # Concurrency
            llm_model_max_async=memory_config.max_async,
            max_parallel_insert=memory_config.max_parallel_insert,
            # Addon params for extraction
            addon_params={
                "entity_types": entity_types,
                "extract_prompt": prompt
            }
        )
        
        await rag.initialize_storages()
        return rag

    async def close_all(self):
        """Finalizes all RAG instances."""
        if self._global_rag:
            await self._global_rag.finalize_storages()
        for rag in self._user_rags.values():
            await rag.finalize_storages()
        self._user_rags.clear()
        self._global_rag = None
