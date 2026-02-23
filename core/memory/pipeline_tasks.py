"""
core/memory/pipeline_tasks.py

Custom Cognee pipeline task and pipeline builder for the financial memory system.

Pipeline insertion order:
    1. classify_documents
    2. extract_chunks_from_documents
    3. extract_graph_from_data      ← LLM sets target_nodeset on each entity
    4. assign_nodeset_from_target   ← OUR TASK: validates & resolves belongs_to_set
    5. summarize_text
    6. add_data_points

`assign_nodeset_from_target` is the enforcement layer that:
  - Raises MissingTargetNodeSetError if target_nodeset is absent
  - Raises InvalidTargetNodeSetError if value is not GLOBAL or USER
  - Resolves target_nodeset to the actual Cognee NodeSet DataPoint
  - Assigns belongs_to_set = [resolved_nodeset_datapoint]
  - Never silently passes invalid data downstream
"""

from __future__ import annotations

import logging
from typing import List, Optional

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.engine.models.node_set import NodeSet

from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.summarization import summarize_text
from cognee.tasks.storage import add_data_points
from cognee.modules.cognify.config import get_cognify_config

from core.memory.exceptions import (
    InvalidTargetNodeSetError,
    MissingTargetNodeSetError,
    NodeSetResolutionError,
)
from core.memory.graph_models import (
    FinancialBaseDataPoint,
    FinancialKnowledgeGraph,
    NodeSetTarget,
)
from core.memory.nodeset_manager import (
    get_or_create_global_nodeset,
    GLOBAL_NODESET_NAME,
)
from core.memory.prompts import FINANCIAL_COGNIFY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-processing task: assign_nodeset_from_target
# ---------------------------------------------------------------------------


async def assign_nodeset_from_target(
    data_chunks: List[DocumentChunk],
    global_nodeset: NodeSet,
) -> List[DocumentChunk]:
    """
    Post-processing pipeline task.

    Runs AFTER extract_graph_from_data, BEFORE summarize_text.

    For every FinancialBaseDataPoint entity in each chunk's `contains` list:
      1. Read target_nodeset; raise MissingTargetNodeSetError if absent
      2. Validate value is GLOBAL or USER; raise InvalidTargetNodeSetError otherwise
      3. Resolve to actual NodeSet DataPoint:
          - If GLOBAL: use the provided global_nodeset
          - If USER: use the document-level User NodeSet that was stored on the
                     chunk's document during ingestion.
      4. Assign belongs_to_set = [resolved_nodeset]

    Args:
        data_chunks:    List of DocumentChunk objects populated by extract_graph_from_data.
        global_nodeset: The shared GLOBAL NodeSet.

    Returns:
        The same data_chunks list with `belongs_to_set` populated on all entities.

    Raises:
        MissingTargetNodeSetError:  If any entity is missing target_nodeset.
        InvalidTargetNodeSetError:  If any entity has an illegal target_nodeset value.
        NodeSetResolutionError:     Safety net if NodeSet object is None or not found on document.
        TypeError: If data_chunks is not a list.
    """
    if not isinstance(data_chunks, list):
        raise TypeError(
            f"assign_nodeset_from_target: expected list[DocumentChunk], got {type(data_chunks)}"
        )

    total_entities = 0
    global_count = 0
    user_count = 0

    for chunk_idx, chunk in enumerate(data_chunks):
        # Determine the user nodeset from the document's belongs_to_set (set during ingestion)
        # The document was placed in a node_set via update_node_set() during classify_documents
        document_nodesets = getattr(chunk.is_part_of, "belongs_to_set", []) or []
        user_nodeset_candidates = [
            ns for ns in document_nodesets if isinstance(ns, NodeSet) and ns.name != GLOBAL_NODESET_NAME
        ]
        doc_user_nodeset: Optional[NodeSet] = user_nodeset_candidates[0] if user_nodeset_candidates else None
        doc_is_global = any(isinstance(ns, NodeSet) and ns.name == GLOBAL_NODESET_NAME for ns in document_nodesets)

        entities = getattr(chunk, "contains", None)
        if not entities:
            logger.debug("Chunk %d has no entities — skipping.", chunk_idx)
            continue

        # Unpack FinancialKnowledgeGraph if necessary
        # Cognee's extract_graph_from_data sets chunk.contains = FinancialKnowledgeGraph(...)
        # We need chunk.contains to be a list of the actual data points.
        if isinstance(entities, FinancialKnowledgeGraph):
            entities = getattr(entities, "entities", [])
            chunk.contains = entities

        if not isinstance(entities, list):
            entities = [entities]
            chunk.contains = entities

        for entity in entities:
            if not isinstance(entity, FinancialBaseDataPoint):
                logger.debug(
                    "Skipping non-financial entity of type %s.", type(entity).__name__
                )
                continue

            entity_type = type(entity).__name__
            total_entities += 1

            # Step 1: Check target_nodeset is present
            raw_target = entity.target_nodeset
            if raw_target is None:
                raise MissingTargetNodeSetError(entity_type)

            # Step 2: Coerce raw string from LLM to enum (LLM may return "GLOBAL"/"USER")
            if isinstance(raw_target, str):
                try:
                    raw_target = NodeSetTarget(raw_target.strip().upper())
                except ValueError:
                    raise InvalidTargetNodeSetError(entity_type, str(raw_target))

            # Step 3: Resolve to actual NodeSet object
            if raw_target == NodeSetTarget.GLOBAL:
                resolved: NodeSet = global_nodeset
                global_count += 1
            elif raw_target == NodeSetTarget.USER:
                if doc_user_nodeset is not None:
                    resolved = doc_user_nodeset
                    user_count += 1
                elif doc_is_global:
                    # An LLM hallucination: it flagged a GLOBAL document's entity as USER.
                    # We gracefully fallback to GLOBAL to prevent exposing it to a non-existent user.
                    logger.warning(
                        "LLM flagged entity %s as USER, but document is GLOBAL. "
                        "Overriding to GLOBAL NodeSet.", entity_type
                    )
                    resolved = global_nodeset
                    global_count += 1
                    entity.target_nodeset = NodeSetTarget.GLOBAL
                else:
                    raise NodeSetResolutionError(
                        f"Cannot resolve USER NodeSet for entity {entity_type} in chunk {chunk.id}: "
                        "parent document has no user NodeSet assigned."
                    )
            else:
                raise InvalidTargetNodeSetError(entity_type, str(raw_target))

            if resolved is None:
                raise NodeSetResolutionError(str(raw_target))

            # Step 4: Assign belongs_to_set
            entity.belongs_to_set = [resolved]
            logger.debug(
                "Entity %s (id=%s) → belongs_to_set='%s'.",
                entity_type, entity.id, resolved.name,
            )

    logger.info(
        "assign_nodeset_from_target: %d entities (%d GLOBAL, %d USER) across %d chunks.",
        total_entities, global_count, user_count, len(data_chunks),
    )
    return data_chunks


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


async def build_financial_pipeline(
    chunks_per_batch: int = 100,
    chunk_size: Optional[int] = None,
) -> list[Task]:
    """
    Build the custom cognify task list for the financial memory system.

    Inserts `assign_nodeset_from_target` between extract_graph_from_data and
    summarize_text.

    Task order:
        1. classify_documents
        2. extract_chunks_from_documents
        3. extract_graph_from_data  (FinancialKnowledgeGraph + custom_prompt)
        4. assign_nodeset_from_target  ← OUR STEP
        5. summarize_text
        6. add_data_points

    Args:
        chunks_per_batch: Batch size for tasks that support batching.
        chunk_size:       Max tokens per chunk (auto-detected if None).

    Returns:
        List of Task objects in execution order.
    """
    # Pre-fetch global nodeset to ensure it's in cache
    global_nodeset = await get_or_create_global_nodeset()

    # Cognee config for embed_triplets
    cognify_config = get_cognify_config()
    embed_triplets = cognify_config.triplet_embedding

    tasks = [
        Task(classify_documents),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=TextChunker,
        ),
        Task(
            extract_graph_from_data,
            graph_model=FinancialKnowledgeGraph,
            custom_prompt=FINANCIAL_COGNIFY_SYSTEM_PROMPT,
            task_config={"batch_size": chunks_per_batch},
        ),
        # Our custom post-processing task
        Task(
            assign_nodeset_from_target,
            global_nodeset=global_nodeset,
        ),
        Task(
            summarize_text,
            task_config={"batch_size": chunks_per_batch},
        ),
        Task(
            add_data_points,
            embed_triplets=embed_triplets,
            task_config={"batch_size": chunks_per_batch},
        ),
    ]

    logger.info(
        "Built global financial pipeline with %d tasks.",
        len(tasks),
    )
    return tasks
