"""Memory retrieval service - fans out DualStoreRetriever calls per active domain."""

from __future__ import annotations

import asyncio
from typing import List

from core.logger import get_logger
from core.memory.retrieval.dual_store_retriever import DualStoreRetriever
from core.memory.retrieval.models import (
    MemoryContext,
    RetrievedChunk,
    RewrittenQueries,
)
from core.memory.retrieval.reranker import CompositeReranker


class MemoryRetrievalService:
    def __init__(
        self, retriever: DualStoreRetriever, reranker: CompositeReranker
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._logger = get_logger(__name__)

    async def retrieve(self, rewritten_queries: RewrittenQueries) -> MemoryContext:
        domain_map = {
            "company": rewritten_queries.company_query,
            "sector": rewritten_queries.sector_query,
            "market": rewritten_queries.market_query,
            "knowledge": rewritten_queries.knowledge_query,
        }
        active_queries = {
            domain: query
            for domain, query in domain_map.items()
            if domain in rewritten_queries.active_domains and query is not None
        }

        tasks = [
            self._retriever.comprehensive_retrieve(query)
            for query in active_queries.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_chunks: List[RetrievedChunk] = []
        for (domain, _query), result in zip(active_queries.items(), results):
            if isinstance(result, Exception):
                self._logger.error(
                    "Memory retrieval failed for domain %s: %s", domain, result
                )
                continue
            if not isinstance(result, list):
                continue
            for chunk in result:
                if isinstance(chunk, RetrievedChunk):
                    all_chunks.append(RetrievedChunk.with_domain(chunk, domain))

        ranked = self._reranker.rank(all_chunks)
        return MemoryContext(chunks=ranked, rewritten_queries=rewritten_queries)


