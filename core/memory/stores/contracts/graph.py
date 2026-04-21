"""Graph-store contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from core.memory.graph.models import DocumentNode, EntityNode
from core.memory.retrieval.models import RetrievedChunk


class GraphStoreAdapter(ABC):
    """Abstract adapter for graph database operations."""

    @abstractmethod
    async def entity_exists(self, entity_id: str) -> bool:
        pass

    @abstractmethod
    async def find_fuzzy_entity_candidates(
        self,
        entity_type: str,
        name: str,
        exclude_id: str = "",
        threshold: float = 0.50,
        limit: int = 10,
    ) -> List[dict]:
        pass

    @abstractmethod
    async def merge_document_node(self, node: DocumentNode) -> None:
        pass

    @abstractmethod
    async def merge_chunk_node(self, node: RetrievedChunk) -> None:
        pass

    @abstractmethod
    async def merge_entity_node(self, node: EntityNode) -> None:
        pass

    @abstractmethod
    async def merge_relationship(
        self, source_id: str, target_id: str, rel_type: str, props: Dict[str, object]
    ) -> None:
        pass

    @abstractmethod
    async def write_relationships(
        self,
        relationships: List[dict],
        conversation_id: str,
        source_agent: str,
        entity_cache: Dict[Tuple[str, str], str],
    ) -> int:
        pass

    @abstractmethod
    async def get_chunk_extraction_status(self, chunk_ids: List[str]) -> Dict[str, str]:
        pass

    @abstractmethod
    async def update_chunk_extraction_status(self, chunk_id: str, status: str) -> None:
        pass

    @abstractmethod
    async def get_entities_for_chunks(self, chunk_ids: List[str]) -> List[dict]:
        pass

    @abstractmethod
    async def get_entity_neighbors(
        self, entity_ids: List[str], exclude_ids: List[str]
    ) -> List[dict]:
        pass

    @abstractmethod
    async def get_chunks_for_entities(
        self, entity_ids: List[str], exclude_chunk_ids: List[str]
    ) -> List[dict]:
        pass

    @abstractmethod
    async def get_entity_category(self, entity_id: str) -> Optional[str]:
        pass

