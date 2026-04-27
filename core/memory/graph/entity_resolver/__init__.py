"""entity_resolver package — entity and relationship endpoint resolution.

Public API (backward-compatible with the old entity_resolver.py module):

    from core.memory.graph.entity_resolver import (
        EntityResolver,
        EntityResolution,
        ResolvedEdgeBatch,
        ResolverThresholds,
    )
"""

from .resolver import EntityResolver
from .types import EntityResolution, ResolvedEdgeBatch, ResolverThresholds

__all__ = [
    "EntityResolver",
    "EntityResolution",
    "ResolvedEdgeBatch",
    "ResolverThresholds",
]
