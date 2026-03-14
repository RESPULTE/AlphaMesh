"""Composite re-ranker for RetrievedChunk results."""

from __future__ import annotations

from typing import Dict, List

from core.memory.retrieval.models import RetrievedChunk


class CompositeReranker:
    def __init__(self, alpha: float, beta: float, top_k: int) -> None:
        self._alpha = alpha
        self._beta = beta
        self._top_k = top_k

    def rank(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        deduped: Dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            current = deduped.get(chunk.chunk_id)
            if current is None or chunk.embedding_score > current.embedding_score:
                deduped[chunk.chunk_id] = chunk

        for chunk in deduped.values():
            depth_bonus = 1.0 / (chunk.graph_depth + 1)
            chunk.composite_score = (
                self._alpha * chunk.embedding_score + self._beta * depth_bonus
            )

        ranked = sorted(
            deduped.values(), key=lambda item: item.composite_score, reverse=True
        )
        return ranked[: self._top_k]
