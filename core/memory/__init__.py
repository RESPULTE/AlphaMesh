# core/memory/__init__.py
"""
Financial Knowledge Memory Module.

Provides Graphiti-based memory with dual-namespace architecture:
- GLOBAL namespace for shared financial knowledge
- User-specific namespaces for personal data and preferences
"""

from core.memory.graphiti_memory import FinancialKnowledgeMemory

__all__ = ["FinancialKnowledgeMemory"]
