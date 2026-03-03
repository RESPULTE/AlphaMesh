"""
core/memory/nodeset_manager.py

Manages the lifecycle of the single shared Cognee dataset and all NodeSets.

Design:
  - One dataset: DATASET_NAME (alphamese_financial)
  - Two NodeSet types:
      * GLOBAL   — one shared NodeSet for public financial data
      * USER_<hash> — one private NodeSet per user, derived from email SHA-256
  - Uses Cognee's built-in NodeSet DataPoint (cognee.modules.engine.models.node_set)
  - All operations are idempotent (safe to call multiple times)

Privacy guarantee:
  - `get_user_nodeset_names(email)` returns ["GLOBAL", "USER_<hash>"] and nothing else.
  - These are the ONLY nodeset names a user may ever query.
  - search() uses node_type=NodeSet, node_name=["GLOBAL","USER_<hash>"] for filtering.
"""

from __future__ import annotations

import hashlib
import logging
from asyncio import Lock
from typing import Type
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.node_set import NodeSet
from cognee.tasks.storage.add_data_points import add_data_points as cognee_add_dp

from core.memory.exceptions import (
    NodeSetCreationError,
)
from core.memory.graph_models import (
    ALL_MAIN_SECTORS,
    GLOBAL_ENTITY_NODESETS,
    GLOBAL_NODESET_NAME,
    Sector,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantss
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Email hashing
# ---------------------------------------------------------------------------


def hash_user_email(email: str) -> str:
    """
    Deterministically hash a user email to a stable short identifier.

    Uses SHA-256 of the lowercase-stripped email and returns the first
    16 hex characters. Same email → same result forever.

    Args:
        email: Raw user email string.

    Returns:
        16-character lowercase hex string.

    Raises:
        ValueError: If email is empty or not a string.
    """
    if not email or not isinstance(email, str):
        raise ValueError(f"Invalid email for hashing: {email!r}")

    normalized = email.strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:16]


def get_user_nodeset_name(user_email: str) -> str:
    """Return the canonical NodeSet name for a given user email."""
    return f"USER_{hash_user_email(user_email)}"


# ---------------------------------------------------------------------------
# In-memory cache — avoid redundant graph writes within one process run
# ---------------------------------------------------------------------------

_nodeset_cache: dict[str, NodeSet] = {}
_nodeset_lock = Lock()


def _normalize_nodeset_name(name: str) -> str:
    """
    Canonicalize NodeSet names so all code paths refer to one logical node.

    Rules:
      - Strip whitespace
      - "global" (any case) -> "GLOBAL"
      - "user_<hash>" (any case) -> "USER_<hash>"
      - Everything else kept as-is after stripping
    """
    if not isinstance(name, str):
        raise ValueError(f"Invalid nodeset name type: {type(name)!r}")

    normalized = name.strip()
    if not normalized:
        raise ValueError("NodeSet name must be a non-empty string.")

    upper = normalized.upper()
    if upper == GLOBAL_NODESET_NAME:
        return GLOBAL_NODESET_NAME

    if upper.startswith("USER_"):
        return f"USER_{normalized[5:].lower()}"

    return normalized


def _cognee_nodeset_id(name: str):
    """
    Build NodeSet id with Cognee's exact algorithm used in classify_documents:
      generate_node_id(f"NodeSet:{name}")

    This guarantees our manually-created NodeSet IDs match NodeSet IDs created
    by `cognee.add(..., node_set=[...])`, preventing duplicate GLOBAL/USER nodes.
    """
    normalized_input = f"NodeSet:{name}".lower().replace(" ", "_").replace("'", "")
    return uuid5(NAMESPACE_OID, normalized_input)


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NodeSet management
# ---------------------------------------------------------------------------


async def get_or_create_nodeset(
    name: str, nodeset_type: Type[NodeSet] = NodeSet, **kwargs
) -> NodeSet:
    """
    Idempotently retrieve or create a NodeSet by canonical name.

    Cognee's built-in NodeSet is a DataPoint subclass with a `name` field.
    The node is persisted via add_data_points and given a deterministic UUID
    so re-runs always produce the same graph node.

    Args:
        name: Canonical NodeSet name, e.g. "GLOBAL" or "USER_abc123def456".

    Returns:
        NodeSet DataPoint for the given name.

    Raises:
        NodeSetCreationError: On any failure.
    """
    canonical_name = _normalize_nodeset_name(name)

    if canonical_name in _nodeset_cache:
        return _nodeset_cache[canonical_name]
    logger.info("NodeSet '%s' not found in cache, checking DB...", canonical_name)
    # Lock prevents duplicate writes in concurrent ingestion startup paths.
    async with _nodeset_lock:
        if canonical_name in _nodeset_cache:
            return _nodeset_cache[canonical_name]

        stable_id = _cognee_nodeset_id(canonical_name)
        try:
            # Query the graph to see if this NodeSet already exists
            graph_engine = await get_graph_engine()
            logger.info(
                "Querying graph for NodeSet '%s' (id=%s).", canonical_name, stable_id
            )

            # Need to lookup exact node syntax depending on DB, assume typical neo4j:
            query = "MATCH (n:NodeSet {id: $id}) RETURN n"
            params = {"id": str(stable_id)}
            results = await graph_engine.query(query, params)
            if results and len(results) > 0:
                # NodeSet already exists in the graph, hydrate it into the cache
                logger.info(
                    "NodeSet '%s' (id=%s) found in graph DB, loading to cache.",
                    canonical_name,
                    stable_id,
                )
                nodeset = nodeset_type(id=stable_id, name=canonical_name, **kwargs)
                _nodeset_cache[canonical_name] = nodeset
                return nodeset

            # nodeset_type does not exist, create it
            nodeset = nodeset_type(id=stable_id, name=canonical_name, **kwargs)
            await cognee_add_dp(data_points=[nodeset])
            _nodeset_cache[canonical_name] = nodeset
            logger.info(
                "NodeSet '%s' created and persisted (id=%s).", canonical_name, stable_id
            )
            return nodeset
        except Exception as exc:
            raise NodeSetCreationError(canonical_name, str(exc)) from exc


async def get_or_create_global_nodeset() -> NodeSet:
    """Return the GLOBAL NodeSet, creating it if necessary."""
    return await get_or_create_nodeset(
        GLOBAL_NODESET_NAME,
        Sector,
        description=GLOBAL_ENTITY_NODESETS[GLOBAL_NODESET_NAME],
    )


async def get_or_create_user_nodeset(user_email: str) -> tuple[str, NodeSet]:
    """
    Return the private NodeSet for a user, creating it if it doesn't exist.

    Args:
        user_email: The user's email address (will be normalized).

    Returns:
        Tuple of (nodeset_name: str, nodeset: NodeSet).

    Raises:
        NodeSetCreationError: If creation fails.
        ValueError: If email is invalid.
    """
    nodeset_name = get_user_nodeset_name(user_email)
    nodeset = await get_or_create_nodeset(nodeset_name, NodeSet)
    return nodeset_name, nodeset


def get_user_nodeset_names(user_email: str) -> list[str]:
    """
    Return the two NodeSet NAMES a user is authorized to access.

    Privacy guarantee: this function is the ONLY way to determine which
    NodeSets a user may query. It always returns exactly two names —
    the GLOBAL one and the user's own — never any other user's NodeSet.

    This is a synchronous function because it only derives names from the
    email hash — no DB calls needed.

    Args:
        user_email: The authenticated user's email.

    Returns:
        ["GLOBAL", "USER_<hash>"]
    """
    return [GLOBAL_NODESET_NAME, get_user_nodeset_name(user_email)]


async def get_or_create_all_sector_nodesets() -> None:
    """
    Ensure all predefined Sector NodeSets exist in the graph and cache.
    Queries the graph once to find missing sectors, creates them in a single batch.

    Raises:
        NodeSetCreationError: On any failure.
    """
    global_nodeset = await get_or_create_global_nodeset()
    await _create_predefined_nodesets(ALL_MAIN_SECTORS, Sector, global_nodeset)


async def get_or_create_all_global_entity_nodesets() -> None:
    """
    Ensure the dedicated global entity NodeSets exist in the graph and cache.

    Bootstraps:
      - "Global Financial Wisdom"  — home for all FinancialConcept entities
      - "Global Financial Event"   — home for all FinancialEvent entities

    Raises:
        NodeSetCreationError: On any failure.
    """
    await _create_predefined_nodesets(GLOBAL_ENTITY_NODESETS, NodeSet, None)


async def _create_predefined_nodesets(
    registry: dict[str, str],
    nodeset_type: Type[NodeSet],
    parent_nodeset: NodeSet,
) -> None:
    """
    Generic batch bootstrap for any predefined {name: description} registry.

    Checks the in-memory cache, queries the graph for what already exists,
    then creates all missing entries in a single write batch.

    Args:
        registry:       Mapping of nodeset name -> description.
        nodeset_type:   NodeSet subclass to instantiate (e.g. Sector, NodeSet).
        parent_nodeset: The parent NodeSet to link new nodes to via belongs_to_set.

    Raises:
        NodeSetCreationError: On any graph or persistence failure.
    """
    # Build the set of names missing from cache (fast path before locking)
    missing_from_cache = {
        _normalize_nodeset_name(name): desc
        for name, desc in registry.items()
        if _normalize_nodeset_name(name) not in _nodeset_cache
    }
    if not missing_from_cache:
        return

    async with _nodeset_lock:
        # Re-check inside the lock
        missing_from_cache = {
            _normalize_nodeset_name(name): desc
            for name, desc in registry.items()
            if _normalize_nodeset_name(name) not in _nodeset_cache
        }
        if not missing_from_cache:
            return

        ids_to_check = {
            name: str(_cognee_nodeset_id(name)) for name in missing_from_cache
        }

        try:
            graph_engine = await get_graph_engine()
            label = nodeset_type.__name__

            query = f"MATCH (n:{label}) WHERE n.id IN $ids RETURN n.id AS id, n.name AS name"
            params = {"ids": list(ids_to_check.values())}
            results = await graph_engine.query(query, params)

            existing_in_db: set[str] = set()
            if results:
                for row in results:
                    data = (
                        row.data()
                        if hasattr(row, "data")
                        else (dict(row) if isinstance(row, dict) else {})
                    )
                    db_id = data.get("id")
                    db_name = data.get("name")
                    if db_id and db_name:
                        existing_in_db.add(db_name)
                        logger.debug(
                            "NodeSet '%s' (id=%s) found in graph DB, loading to cache.",
                            db_name,
                            db_id,
                        )
                        nodeset = nodeset_type(
                            id=str(db_id),
                            name=db_name,
                            description=missing_from_cache.get(db_name, ""),
                        )
                        if parent_nodeset:
                            nodeset.belongs_to_set = [parent_nodeset]
                        _nodeset_cache[db_name] = nodeset

            to_create = []
            for name, desc in missing_from_cache.items():
                if (
                    name not in existing_in_db
                    and name not in GLOBAL_ENTITY_NODESETS.keys()
                ):
                    stable_id = ids_to_check[name]
                    nodeset = nodeset_type(id=stable_id, name=name, description=desc)
                    if parent_nodeset:
                        nodeset.belongs_to_set = [parent_nodeset]
                    to_create.append(nodeset)
                    _nodeset_cache[name] = nodeset

            if to_create:
                await cognee_add_dp(data_points=to_create)
                logger.info(
                    "NodeSets %s created and persisted in batch.",
                    [n.name for n in to_create],
                )

        except Exception as exc:
            raise NodeSetCreationError(f"Batch {label} NodeSets", str(exc)) from exc
