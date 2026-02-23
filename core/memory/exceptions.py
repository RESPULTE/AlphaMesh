"""
core/memory/exceptions.py

Typed exception hierarchy for the AlphaMesh financial memory system.
All exceptions inherit from MemorySystemError for easy catch-all handling.
"""


class MemorySystemError(Exception):
    """Base exception for all memory system failures."""


class NodeSetCreationError(MemorySystemError):
    """Raised when a NodeSet DataPoint cannot be created or persisted."""

    def __init__(self, nodeset_name: str, reason: str = "") -> None:
        self.nodeset_name = nodeset_name
        super().__init__(
            f"Failed to create NodeSet '{nodeset_name}'"
            + (f": {reason}" if reason else "")
        )


class NodeSetResolutionError(MemorySystemError):
    """Raised when a NodeSet name cannot be resolved to a DataPoint object."""

    def __init__(self, nodeset_name: str) -> None:
        self.nodeset_name = nodeset_name
        super().__init__(
            f"Cannot resolve NodeSet '{nodeset_name}' — it may not exist in the graph"
        )


class InvalidTargetNodeSetError(MemorySystemError):
    """
    Raised during post-processing when an entity's target_nodeset value is not
    one of the allowed values: 'GLOBAL' or 'USER'.
    """

    def __init__(self, entity_type: str, actual_value: str) -> None:
        self.entity_type = entity_type
        self.actual_value = actual_value
        super().__init__(
            f"Entity '{entity_type}' has invalid target_nodeset='{actual_value}'. "
            f"Allowed values are 'GLOBAL' and 'USER'."
        )


class MissingTargetNodeSetError(MemorySystemError):
    """
    Raised during post-processing when an entity is missing the required
    target_nodeset field entirely.
    """

    def __init__(self, entity_type: str) -> None:
        self.entity_type = entity_type
        super().__init__(
            f"Entity '{entity_type}' is missing required field 'target_nodeset'. "
            f"The LLM must set this field to 'GLOBAL' or 'USER' during cognify."
        )


class DatasetInitError(MemorySystemError):
    """Raised when the shared Cognee dataset cannot be initialized."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Failed to initialize Cognee dataset: {reason}")


class IngestionError(MemorySystemError):
    """Raised when data ingestion (cognee.add) fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Ingestion failed: {reason}")


class QueryError(MemorySystemError):
    """Raised when a search query fails or violates privacy constraints."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Query error: {reason}")
