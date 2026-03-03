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

import logging
import uuid
from typing import Any, List, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points

from core.memory.entity_merger import find_and_merge_candidates
from core.memory.exceptions import (
    NodeSetResolutionError,
)
from core.memory.graph_models import (
    ALL_ENTITIES,
    ALL_MAIN_SECTORS,
    GLOBAL_FINANCIAL_EVENT_NODESET,
    GLOBAL_FINANCIAL_WISDOM_NODESET,
    USER_SPECIFIC_ENTITIES,
    FinancialKnowledgeGraph,
)
from core.memory.nodeset_manager import (
    GLOBAL_NODESET_NAME,
    get_or_create_all_global_entity_nodesets,
    get_or_create_global_nodeset,
    get_or_create_nodeset,
)
from core.memory.prompts import FINANCIAL_COGNIFY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-processing task: assign_nodeset_from_target
# ---------------------------------------------------------------------------


def get_canonical_id(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, name.upper())


async def assign_nodesets(
    data_chunks: List[DocumentChunk],
    global_nodeset: NodeSet,
    financial_wisdom_nodeset: NodeSet,
    financial_event_nodeset: NodeSet,
) -> List[DocumentChunk]:
    """
    Post-processing pipeline task.

    Runs AFTER extract_graph_from_data, BEFORE summarize_text.

    For every entity in each chunk's `contains` list:
      1. Check if the entity type belongs to USER_SPECIFIC_ENTITIES.
      2. Resolve to actual NodeSet DataPoint:
          - FinancialConcept  → financial_wisdom_nodeset ("Global Financial Wisdom")
          - FinancialEvent    → financial_event_nodeset  ("Global Financial Event")
          - Other GLOBAL      → global_nodeset ("Market")
          - USER              → the document-level user NodeSet
      3. Assign belongs_to_set = [resolved_nodeset]

    Args:
        data_chunks:               List of DocumentChunk objects.
        global_nodeset:            The shared GLOBAL (Market) NodeSet.
        financial_wisdom_nodeset:  NodeSet for FinancialConcept entities.
        financial_event_nodeset:   NodeSet for FinancialEvent entities.

    Returns:
        List[DocumentChunk]: The same data_chunks list with `belongs_to_set` populated on all entities.

    Raises:
        NodeSetResolutionError: Safety net if NodeSet object is None or not found on document.
        TypeError: If data_chunks is not a list.
    """

    async def _classify_company(company: Any, sector_or_industry: str):
        try:
            sector_nodeset = await get_or_create_nodeset(sector_or_industry)
            if sector_nodeset not in company.belongs_to_set:
                company.belongs_to_set.append(sector_nodeset)
            logger.info(
                f"Resolved sector/industry nodeset {sector_or_industry} for Company {company.name}"
            )
        except Exception as e:
            logger.info(
                f"Failed to resolve sector/industry nodeset {sector_or_industry} for Company {company.name}: {e}"
            )

    if not isinstance(data_chunks, list):
        raise TypeError(
            f"assign_nodeset_from_target: expected list[DocumentChunk], got {type(data_chunks)}"
        )

    if not isinstance(global_nodeset, NodeSet):
        raise TypeError(
            f"assign_nodeset_from_target: expected NodeSet, got {type(global_nodeset)}"
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
            if isinstance(ns, NodeSet)
            and ns.name != GLOBAL_NODESET_NAME
            and ns.name.startswith("USER_")
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
                # Route specific global entity types to their dedicated nodesets
                if entity_type == "FinancialConcept":
                    resolved: NodeSet = financial_wisdom_nodeset
                elif entity_type == "FinancialEvent":
                    resolved = financial_event_nodeset
                else:
                    resolved = global_nodeset
                global_count += 1
            else:
                if doc_user_nodeset is not None:
                    resolved = doc_user_nodeset
                    user_count += 1
                elif doc_is_global:
                    logger.info(
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
            entity.belongs_to_set = getattr(entity, "belongs_to_set", []) or []
            if resolved not in entity.belongs_to_set and entity_type != "Company":
                entity.belongs_to_set.append(resolved)

            # Special business logic for specific global entities
            if entity_type == "Company":
                if hasattr(entity, "sector") and entity.sector:
                    await _classify_company(entity, entity.sector)
                elif hasattr(entity, "industry") and entity.industry:
                    await _classify_company(entity, entity.industry)

            elif entity_type == "FinancialEvent" and (
                entity.positively_impacted or entity.negatively_impacted
            ):
                to_check = {
                    "positive": entity.positively_impacted,
                    "negative": entity.negatively_impacted,
                }
                positive_imp_sectors = []
                negative_imp_sectors = []
                for impact_type, impacted_entities in to_check.items():
                    if impacted_entities is None:
                        continue
                    for impacted_entity in impacted_entities:
                        if (
                            impacted_entity.name in ALL_MAIN_SECTORS.keys()
                            or impacted_entity.name == GLOBAL_NODESET_NAME
                        ):
                            sector = impacted_entity.name
                            try:
                                sector_nodeset = await get_or_create_nodeset(sector)
                                if impact_type == "positive":
                                    positive_imp_sectors.append(sector_nodeset)
                                    entity.positively_impacted.remove(impacted_entity)
                                elif impact_type == "negative":
                                    negative_imp_sectors.append(sector_nodeset)
                                    entity.negatively_impacted.remove(impacted_entity)
                                logger.info(
                                    f"Resolved sector nodeset {sector} for FinancialEvent {entity.name}"
                                )
                            except Exception as e:
                                logger.info(
                                    f"Failed to resolve sector nodeset {sector} for FinancialEvent {entity.name}: {e}"
                                )
                if positive_imp_sectors and entity.positively_impacted:
                    entity.positively_impacted.extend(positive_imp_sectors)
                if negative_imp_sectors and entity.negatively_impacted:
                    entity.negatively_impacted.extend(negative_imp_sectors)
                # logger.debug(
                #     "Entity %s (id=%s) → belongs_to_set='%s'.",
                #     entity_type,
                #     entity.id,
                #     ", ".join([ns.name for ns in entity.belongs_to_set]),
                # )

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
    data_chunks: List[DocumentChunk],
    embed_triplets: bool = False,
):
    """
    Wrapper task for `add_data_points` that unpacks the combined payload (data_chunks, custom_edges)
    from previous tasks and passes `custom_edges` correctly.
    """

    return await add_data_points(
        data_points=data_chunks,
        embed_triplets=embed_triplets,
    )


async def merge_entities(data_points: List) -> List:
    """
    Pipeline task: runs after add_data_points_with_custom_edges.

    Receives the List[DataPoint] returned by add_data_points (DocumentChunks).
    Extracts the graph-node level entities from each chunk's `.contains` list,
    filters to global mergeable types, and calls find_and_merge_candidates to
    selectively detect and merge near-duplicate nodes using APOC fuzzy search
    + vector engine semantic validation.

    Returns the original data_points unchanged (pass-through).
    """
    # Flatten graph-level entity nodes from all chunk.contains lists
    graph_nodes = []
    for item in data_points:
        entities = getattr(item, "contains", None)
        if not entities:
            continue
        if isinstance(entities, list):
            graph_nodes.extend(e for e in entities if type(e).__name__ in ALL_ENTITIES)
        elif type(entities).__name__ in ALL_ENTITIES:
            graph_nodes.append(entities)

    if graph_nodes:
        graph_engine = await get_graph_engine()
        await find_and_merge_candidates(graph_engine, graph_nodes)
    else:
        logger.debug(
            "merge_entities: no global entity nodes found in batch — skipping."
        )

    return data_points


async def build_financial_pipeline(
    chunks_per_batch: int = 100,
    chunk_size: Optional[int] = None,
) -> list[Task]:
    """
    Build the custom cognify task list for the financial memory system.

    Task order:
        1. classify_documents
        2. extract_chunks_from_documents
        3. extract_graph_from_data  (FinancialKnowledgeGraph + custom_prompt)
        4. assign_nodesets
        5. process_global_influences
        6. add_data_points_with_custom_edges
        7. merge_entities             ← selective APOC fuzzy + vector dedup

    Args:
        chunks_per_batch: Batch size for tasks that support batching.
        chunk_size:       Max tokens per chunk (auto-detected if None).

    Returns:
        List of Task objects in execution order.
    """
    # Pre-fetch global nodeset and dedicated entity nodesets to ensure they're in cache
    global_nodeset = await get_or_create_global_nodeset()
    await get_or_create_all_global_entity_nodesets()
    financial_wisdom_nodeset = await get_or_create_nodeset(
        GLOBAL_FINANCIAL_WISDOM_NODESET
    )
    financial_event_nodeset = await get_or_create_nodeset(
        GLOBAL_FINANCIAL_EVENT_NODESET
    )

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
            financial_wisdom_nodeset=financial_wisdom_nodeset,
            financial_event_nodeset=financial_event_nodeset,
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
        Task(merge_entities),
    ]
    logger.info(
        "Built global financial pipeline with %d tasks.",
        len(tasks),
    )
    return tasks
