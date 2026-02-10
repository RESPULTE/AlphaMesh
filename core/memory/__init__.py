"""
Financial Memory Module — public interface.

Provides a dual-namespace LightRAG-based memory system with:
  - Global workspace: shared financial domain knowledge (no PII)
  - User workspaces: personal financial context per user
  - Cross-namespace linking via direct Neo4j edges
"""

from core.memory.lightrag_memory import FinancialMemory
from core.memory.models import (
    GLOBAL_ENTITY_TYPES,
    USER_ENTITY_TYPES,
    IngestionResult,
    IngestionStatus,
    QueryResult,
)

__all__ = [
    "FinancialMemory",
    "IngestionResult",
    "IngestionStatus",
    "QueryResult",
    "GLOBAL_ENTITY_TYPES",
    "USER_ENTITY_TYPES",
]
