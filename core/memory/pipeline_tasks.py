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
    get_or_create_user_nodeset,
    GLOBAL_NODESET_NAME,
)
from core.memory.prompts import build_cognify_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-processing task: assign_nodeset_from_target
# ---------------------------------------------------------------------------


async def assign_nodeset_from_target(
    data_chunks: List[DocumentChunk],
    user_nodeset: NodeSet,
    global_nodeset: NodeSet,
) -> List[DocumentChunk]:
    """
    Post-processing pipeline task.

    Runs AFTER extract_graph_from_data, BEFORE summarize_text.

    For every FinancialBaseDataPoint entity in each chunk's `contains` list:
      1. Read target_nodeset; raise MissingTargetNodeSetError if absent
      2. Validate value is GLOBAL or USER; raise InvalidTargetNodeSetError otherwise
      3. Resolve to actual NodeSet DataPoint
      4. Assign belongs_to_set = [resolved_nodeset]

    Args:
        data_chunks:    List of DocumentChunk objects populated by extract_graph_from_data.
        user_nodeset:   The current user's NodeSet (USER_<hash>).
        global_nodeset: The shared GLOBAL NodeSet.

    Returns:
        The same data_chunks list with `belongs_to_set` populated on all entities.

    Raises:
        MissingTargetNodeSetError:  If any entity is missing target_nodeset.
        InvalidTargetNodeSetError:  If any entity has an illegal target_nodeset value.
        NodeSetResolutionError:     Safety net if NodeSet object is None.
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
                resolved = user_nodeset
                user_count += 1
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
    user_email: str,
    chunks_per_batch: int = 100,
    chunk_size: Optional[int] = None,
) -> list[Task]:
    """
    Build the custom cognify task list for the financial memory system.

    Inserts `assign_nodeset_from_target` between extract_graph_from_data and
    summarize_text. NodeSets are pre-created here so they're in cache.

    Task order:
        1. classify_documents
        2. extract_chunks_from_documents
        3. extract_graph_from_data  (FinancialKnowledgeGraph + custom_prompt)
        4. assign_nodeset_from_target  ← OUR STEP
        5. summarize_text
        6. add_data_points

    Args:
        user_email:       Authenticated user's email for NodeSet resolution.
        chunks_per_batch: Batch size for tasks that support batching.
        chunk_size:       Max tokens per chunk (auto-detected if None).

    Returns:
        List of Task objects in execution order.
    """
    # Pre-fetch nodesets — ensures they exist in graph and cache
    global_nodeset = await get_or_create_global_nodeset()
    _, user_nodeset = await get_or_create_user_nodeset(user_email)

    # Build per-user extraction prompt
    custom_prompt = build_cognify_prompt(user_email)

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
            custom_prompt=custom_prompt,
            task_config={"batch_size": chunks_per_batch},
        ),
        # Our custom post-processing task
        Task(
            assign_nodeset_from_target,
            user_nodeset=user_nodeset,
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
        "Built financial pipeline for user '%s' with %d tasks.",
        user_email, len(tasks),
    )
    return tasks
