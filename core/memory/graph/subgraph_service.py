"""
core/memory/graph/subgraph_service.py

Compatibility shim.  All graph-write logic has moved to:
  - core.memory.graph.graph_queue.GraphQueueManager  (queue + persistence)
  - core.memory.graph.relationship_extractor.RelationshipExtractor  (LLM extraction)
  - core.memory.graph.entity_resolver.EntityResolver  (entity resolution)
  - core.memory.graph.graph_writer.GraphWriter  (Neo4j edge writing)

This file keeps the original SubgraphExtractionService class name and schedule()
API intact so that any remaining call sites continue to work without change.
Callers should migrate to GraphQueueManager directly over time.

NOTE: build_graph() and persist_graph() are removed — they have no callers after
the refactor and nx.DiGraph is no longer used anywhere in the pipeline.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from core.config import settings
from core.logger import get_logger
from core.memory.graph.graph_queue import GraphQueueManager, GraphTask
from core.memory.graph.relationship_extractor import RelationshipExtractor

logger = get_logger(__name__)


class SubgraphExtractionService:
    """
    Compatibility shim — delegates to GraphQueueManager + RelationshipExtractor.

    Injected at construction by service_manager; never fetches from service_manager
    itself to keep it testable.
    """

    def __init__(
        self,
        queue_manager: GraphQueueManager,
        extractor: RelationshipExtractor,
    ) -> None:
        self._queue_manager = queue_manager
        self._extractor = extractor

    # ──────────────────────────────────────────────────────────────────────────
    # Primary API — preserved for call-site compatibility
    # ──────────────────────────────────────────────────────────────────────────

    async def schedule(
        self,
        *,
        agent_name: str,
        conversation_id: str,
        analysis_text: str,
        llm,
        system_prompt: str,
        relationships: Optional[List[dict]] = None,
        bypass_guards: bool = False,
    ) -> Optional[str]:
        """
        Schedule a graph write.  Preserves the original schedule() contract:
          - Returns a task_id string for tracing (replaces the old subgraph_id).
          - bypass_guards=True → write_immediate (taxonomy / user signals bypass).
          - relationships=None + llm provided → LLM extraction before enqueue.
          - relationships=[] → no-op (nothing to write).
          - EXTRACTION_ENABLED=False and no bypass → no-op.

        The caller does NOT need to await graph persistence — this is fire-and-forget.
        """
        if not bypass_guards and not settings.EXTRACTION_ENABLED:
            return None

        if not conversation_id and not bypass_guards:
            return None

        # Resolve relationships: extract if not pre-supplied
        if relationships is None:
            if llm and analysis_text and analysis_text.strip():
                relationships = await self._extractor.extract(
                    text=analysis_text,
                    llm=llm,
                    system_prompt=system_prompt,
                )
            else:
                relationships = []

        if not relationships:
            return None

        task_id = str(uuid4())

        if bypass_guards:
            # System tasks bypass the queue — direct write
            await self._queue_manager.write_immediate(
                relationships=relationships,
                conversation_id=conversation_id or "system",
                source_agent=agent_name,
            )
            return task_id

        # Normal path — enqueue for batched processing
        task = GraphTask(
            task_id=task_id,
            turn_id=conversation_id,  # turn_id = conversation_id for shim compat
            conversation_id=conversation_id,
            source_agent=agent_name,
            relationships=relationships,
        )
        return await self._queue_manager.enqueue(task)

    # ──────────────────────────────────────────────────────────────────────────
    # Extraction-only helper (preserved for callers that call extract directly)
    # ──────────────────────────────────────────────────────────────────────────

    async def extract_relationships(
        self,
        *,
        text: str,
        llm,
        system_prompt: str,
        max_attempts: int = settings.EXTRACTION_LLM_RETRY_ATTEMPTS,
    ) -> List[dict]:
        """Delegates to RelationshipExtractor.extract()."""
        return await self._extractor.extract(
            text=text,
            llm=llm,
            system_prompt=system_prompt,
            max_attempts=max_attempts,
        )
