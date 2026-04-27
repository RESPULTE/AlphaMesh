"""Shared dataclasses and constants for entity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Ordinal ranking used when merging textual confidence labels.
_CONFIDENCE_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class ResolverThresholds:
    """Immutable bundle of all similarity thresholds used during resolution."""

    neo4j_fuzzy_threshold: float
    """Normalised [0, 1] threshold passed to Neo4j's fuzzy lookup."""

    rapidfuzz_threshold: float
    """Raw [0, 1] threshold used with rapidfuzz; multiplied by 100 at comparison time."""

    vector_distance_threshold: float
    """Maximum cosine distance (lower = more similar) for a vector match to be accepted."""

    strong_fuzzy_threshold: float
    """Minimum fuzzy similarity score [0, 1] to accept a Neo4j fuzzy candidate directly."""


@dataclass(frozen=True)
class EntityResolution:
    """Result of resolving a single entity name + type."""

    entity_id: Optional[str]
    match_stage: str
    score: Optional[float] = None
    created: bool = False

    @property
    def resolved(self) -> bool:
        return bool(self.entity_id)


@dataclass(frozen=True)
class ResolvedEdgeBatch:
    """Output of resolving and deduplicating a batch of relationship edges."""

    relationships: List[dict]
    entity_cache: Dict[Tuple[str, str], str]
    skipped_relationships: int
