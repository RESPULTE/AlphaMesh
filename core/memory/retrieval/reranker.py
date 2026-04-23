"""
core/memory/retrieval/reranker.py

Two-stage reranking pipeline for retrieved chunks.

CompositePrefilter  — cheap structural pre-scorer (synchronous, zero API cost).
                     Deduplicates by chunk_id, scores embedding + graph depth,
                     and returns the top-N candidates for the second stage.
                     Also used directly by DualStoreRetriever for intermediate
                     memory-chunk ordering where a Jina call would be redundant.

TwoStageReranker   — full semantic reranker (async).
                     Stage 1: CompositePrefilter prunes the candidate pool.
                     Stage 2: Jina cross-encoder reranks the pruned set against
                              the query as query-document pairs — capturing
                              relevance that bi-encoder retrieval scores cannot.
                     Falls back to stage-1 results when JINA_API_KEY is absent
                     or when the Jina API call fails for any reason, so the
                     pipeline degrades gracefully without operator intervention.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from core.memory.retrieval.models import RetrievedChunk
from core.memory.retrieval.tracing import PrefilterTraceContext, RetrievalTraceEvent

logger = logging.getLogger(__name__)

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class CompositePrefilter:
    """
    Fast, synchronous structural scorer.

    Deduplicates chunks by chunk_id (keeping the highest embedding_score copy),
    applies a composite score of embedding similarity + graph depth bonus, and
    returns the top-N candidates.

    Used as stage 1 of TwoStageReranker, and also directly by
    DualStoreRetriever for intermediate memory-chunk ordering inside
    comprehensive_retrieve — where the Jina call happens later in
    _rendezvous_node on the fully-combined pool, making a second Jina call
    here redundant.
    """

    def __init__(self, alpha: float, beta: float, prefilter_k: int) -> None:
        self._alpha = alpha
        self._beta = beta
        self._prefilter_k = prefilter_k

    def score(
        self,
        chunks: List[RetrievedChunk],
        *,
        trace_context: Optional[PrefilterTraceContext] = None,
    ) -> List[RetrievedChunk]:
        """Deduplicate, apply composite score, and return top prefilter_k chunks."""
        # Deduplicate by chunk_id, keeping the best embedding signal per chunk.
        deduped: Dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            existing = deduped.get(chunk.chunk_id)
            if existing is None or chunk.embedding_score > existing.embedding_score:
                deduped[chunk.chunk_id] = chunk

        for chunk in deduped.values():
            depth_bonus = 1.0 / (chunk.graph_depth + 1)
            chunk.composite_score = (
                self._alpha * chunk.embedding_score + self._beta * depth_bonus
            )

        ranked = sorted(
            deduped.values(),
            key=lambda c: c.composite_score,
            reverse=True,
        )
        selected = ranked[: self._prefilter_k]

        if trace_context is not None:
            ranked_payload = []
            selected_ids = {chunk.chunk_id for chunk in selected}
            for idx, chunk in enumerate(ranked, start=1):
                ranked_payload.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "domain": chunk.domain or "",
                        "embedding_score": chunk.embedding_score,
                        "graph_depth": chunk.graph_depth,
                        "composite_score": chunk.composite_score,
                        "rank": idx,
                        "selected": chunk.chunk_id in selected_ids,
                    }
                )
            try:
                trace_context.sink.record(
                    RetrievalTraceEvent(
                        run_id=trace_context.run_id,
                        parent_run_id=trace_context.parent_run_id,
                        domain=trace_context.domain,
                        stage="prefilter_output",
                        hop=trace_context.hop,
                        layer=trace_context.layer,
                        payload={
                            "total_candidates": len(ranked),
                            "selected_count": len(selected),
                            "ranked_chunks": ranked_payload,
                        },
                    )
                )
            except Exception:
                logger.warning("Prefilter trace emission failed", exc_info=True)

        return selected


class TwoStageReranker:
    """
    Composite prefilter followed by Jina cross-encoder semantic reranking.

    Stage 1 (CompositePrefilter) prunes the candidate pool cheaply using
    embedding similarity and graph depth priors.  Stage 2 (Jina) reranks
    the pruned set as query-document pairs — capturing true semantic
    relevance that retrieval-time bi-encoder scores cannot.

    Graceful fallback behaviour:
    - JINA_API_KEY absent  → composite order, truncated to top_k.
    - Pool already ≤ top_k → no Jina call needed; composite order returned.
    - Jina API failure     → logs a warning, returns composite order.
    """

    def __init__(
        self,
        prefilter: CompositePrefilter,
        top_k: int,
        jina_api_key: Optional[str] = None,
        jina_model: str = "jina-reranker-v2-base-multilingual",
    ) -> None:
        self._prefilter = prefilter
        self._top_k = top_k
        self._jina_api_key = jina_api_key or ""
        self._jina_model = jina_model

    async def rank(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        Run two-stage ranking and return at most top_k chunks.

        Stage 1 always runs (free).  Stage 2 (Jina) is skipped when the API
        key is absent or when the prefiltered pool is already small enough
        that no further selection is needed.
        """
        if not chunks:
            return []

        prefiltered = self._prefilter.score(chunks)

        # Skip Jina when unconfigured or when the pool fits the budget already.
        if not self._jina_api_key or len(prefiltered) <= self._top_k:
            return prefiltered[: self._top_k]

        return await self._jina_rerank(query, prefiltered)

    # ── Private ───────────────────────────────────────────────────────────────

    async def _jina_rerank(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        POST to the Jina reranker API and reorder chunks by cross-encoder score.

        Uses top_n to let Jina return only the final budget — the response
        index values map back to positions in the input list.
        Falls back to composite order on any network or API error.
        """
        payload = {
            "model": self._jina_model,
            "query": query,
            "documents": [c.text for c in chunks],
            "top_n": self._top_k,
        }
        headers = {
            "Authorization": f"Bearer {self._jina_api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    _JINA_RERANK_URL, json=payload, headers=headers
                )
            response.raise_for_status()
            results = response.json().get("results", [])
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Jina reranker returned HTTP %d — falling back to composite order",
                exc.response.status_code,
            )
            return chunks[: self._top_k]
        except Exception as exc:
            logger.warning(
                "Jina reranker call failed (%s) — falling back to composite order", exc
            )
            return chunks[: self._top_k]

        # results is [{index: int, relevance_score: float, ...}] sorted by score.
        # The index refers to the position in the input `chunks` list.
        reranked: List[RetrievedChunk] = []
        for item in results:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(chunks):
                reranked.append(chunks[idx])

        if not reranked:
            logger.warning(
                "Jina response contained no valid indices — falling back to composite order"
            )
            return chunks[: self._top_k]

        return reranked
