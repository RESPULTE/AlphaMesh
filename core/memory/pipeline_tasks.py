"""
core/memory/pipeline_tasks.py

Custom Cognee pipeline task and pipeline builder for the financial memory system.

Pipeline insertion order:
    1. classify_documents
    2. extract_chunks_from_documents
    3. extract_graph_from_data
    4. assign_nodesets              ← OUR TASK: validates & resolves belongs_to_set
    5. summarize_text
    6. add_data_points

`assign_nodesets` is the enforcement layer that:
  - Automatically identifies USER vs GLOBAL entities via USER_SPECIFIC_ENTITIES
  - Resolves to the actual Cognee NodeSet DataPoint
  - Assigns belongs_to_set = [resolved_nodeset_datapoint]
  - Never silently passes invalid data downstream
"""

from __future__ import annotations
import uuid

import logging
from typing import List, Optional
from core.memory.graph_models import FinancialEntity

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

from cognee.infrastructure.engine import Edge
from cognee.infrastructure.databases.graph import get_graph_engine
from core.memory.exceptions import (
    NodeSetResolutionError,
)
from core.memory.graph_models import (
    FinancialKnowledgeGraph,
    GlobalInfluence,
    USER_SPECIFIC_ENTITIES,
)
from core.memory.nodeset_manager import (
    get_or_create_global_nodeset,
    GLOBAL_NODESET_NAME,
)
from core.memory.prompts import FINANCIAL_COGNIFY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

import hashlib

# ---------------------------------------------------------------------------
# Post-processing task: process_global_influences
# ---------------------------------------------------------------------------


async def process_global_influences(
    data_chunks: List[DocumentChunk],
) -> tuple[List[DocumentChunk], list]:
    """
    Extracts GlobalInfluence models, turns them into edges, and removes them from the parsed entities list
    so they don't get saved as standard node entities. Returns the modified chunks and a list of custom edges.
    """

    all_custom_edges = []

    for chunk in data_chunks:
        entities = getattr(chunk, "contains", None)

        # Unpack FinancialKnowledgeGraph if necessary
        if isinstance(entities, FinancialKnowledgeGraph):
            entities = getattr(entities, "entities", [])
            chunk.contains = entities

        if not entities or not isinstance(entities, list):
            continue

        influences = [e for e in entities if isinstance(e, GlobalInfluence)]
        logger.info(
            "Found %d global influences in chunk %s.", len(influences), chunk.id
        )

        # Keep only non-influences
        chunk.contains = [e for e in entities if not isinstance(e, GlobalInfluence)]

        if influences:
            for inf in influences:
                props = {}
                if inf.weight is not None:
                    try:
                        props["weight"] = float(inf.weight)
                    except (ValueError, TypeError):
                        pass

                if inf.evidence is not None:
                    props["evidence"] = (
                        str(inf.evidence)
                        if not isinstance(inf.evidence, str)
                        else inf.evidence
                    )

                logger.info(
                    "Processing global influence: %s -> %s",
                    inf.source_id,
                    inf.target_id,
                )
                all_custom_edges.append(
                    (
                        next(
                            (
                                e.id
                                for e in entities
                                if getattr(e, "name", None) == inf.source_id
                            ),
                            str(inf.source_id),
                        ),
                        next(
                            (
                                e.id
                                for e in entities
                                if getattr(e, "name", None) == inf.target_id
                            ),
                            str(inf.target_id),
                        ),
                        str(inf.relationship_name),
                        props,
                    )
                )

    logger.info("Extracted %d global influence custom edges.", len(all_custom_edges))

    return (data_chunks, all_custom_edges)


# ---------------------------------------------------------------------------
# Post-processing task: assign_nodeset_from_target
# ---------------------------------------------------------------------------


def get_canonical_id(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, name.upper())


async def assign_nodesets(
    data_chunks: List[DocumentChunk],
    global_nodeset: NodeSet,
) -> List[DocumentChunk]:
    """
    Post-processing pipeline task.

    Runs AFTER extract_graph_from_data, BEFORE summarize_text.

    For every entity in each chunk's `contains` list:
      1. Check if the entity type belongs to USER_SPECIFIC_ENTITIES.
      2. Resolve to actual NodeSet DataPoint:
          - If GLOBAL: use the provided global_nodeset
          - If USER: use the document-level User NodeSet that was stored on the
                     chunk's document during ingestion.
      3. Assign belongs_to_set = [resolved_nodeset]

    Args:
        data_chunks: List of DocumentChunk objects.
        global_nodeset: The shared GLOBAL NodeSet.

    Returns:
        List[DocumentChunk]: The same data_chunks list with `belongs_to_set` populated on all entities.

    Raises:
        NodeSetResolutionError: Safety net if NodeSet object is None or not found on document.
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
            ns
            for ns in document_nodesets
            if isinstance(ns, NodeSet) and ns.name != GLOBAL_NODESET_NAME
        ]
        doc_user_nodeset: Optional[NodeSet] = (
            user_nodeset_candidates[0] if user_nodeset_candidates else None
        )
        doc_is_global = any(
            isinstance(ns, NodeSet) and ns.name == GLOBAL_NODESET_NAME
            for ns in document_nodesets
        )

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

            entity_type = type(entity).__name__
            total_entities += 1

            # Determine Target based on entity type
            is_user_entity = entity_type in USER_SPECIFIC_ENTITIES

            # Resolve to actual NodeSet object
            if not is_user_entity:
                resolved: NodeSet = global_nodeset
                global_count += 1
            else:
                if doc_user_nodeset is not None:
                    resolved = doc_user_nodeset
                    user_count += 1
                elif doc_is_global:
                    logger.warning(
                        "Entity %s is USER specific, but document is GLOBAL. "
                        "Overriding to GLOBAL NodeSet.",
                        entity_type,
                    )
                    resolved = global_nodeset
                    global_count += 1
                else:
                    raise NodeSetResolutionError(
                        f"Cannot resolve USER NodeSet for entity {entity_type} in chunk {chunk.id}: "
                        "parent document has no user NodeSet assigned."
                    )

            if resolved is None:
                raise NodeSetResolutionError(
                    f"Failed to resolve NodeSet for {entity_type}"
                )

            # Assign belongs_to_set
            entity.belongs_to_set = [resolved]
            logger.debug(
                "Entity %s (id=%s) → belongs_to_set='%s'.",
                entity_type,
                entity.id,
                resolved.name,
            )

            if hasattr(entity, "name"):
                entity.id = get_canonical_id(entity.name)

    logger.info(
        "assign_nodesets: %d entities (%d GLOBAL, %d USER) across %d chunks.",
        total_entities,
        global_count,
        user_count,
        len(data_chunks),
    )
    return data_chunks


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


async def add_data_points_with_custom_edges(
    payload: tuple[List[DocumentChunk], list],
    embed_triplets: bool = False,
):
    """
    Wrapper task for `add_data_points` that unpacks the combined payload (data_chunks, custom_edges)
    from previous tasks and passes `custom_edges` correctly.
    """

    data_chunks, custom_edges = payload
    return await add_data_points(
        data_points=data_chunks,
        custom_edges=custom_edges,
        embed_triplets=embed_triplets,
    )


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
        4. assign_nodesets  ← OUR STEP
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
        # Process global influences and remove them from entities
        # Our custom post-processing task
        Task(
            assign_nodesets,
            global_nodeset=global_nodeset,
        ),
        Task(
            process_global_influences,
        ),
        # Task(
        #     summarize_text,
        #     task_config={"batch_size": chunks_per_batch},
        # ),
        Task(
            add_data_points_with_custom_edges,
            embed_triplets=embed_triplets,
            task_config={"batch_size": chunks_per_batch},
        ),
    ]
    logger.info(
        "Built global financial pipeline with %d tasks.",
        len(tasks),
    )
    return tasks
