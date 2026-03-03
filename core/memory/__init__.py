"""

core/memory/__init__.py


Public API for the AlphaMesh multi-tenant financial memory system (Cognee-based).
"""

from cognee.modules.engine.models.node_set import NodeSet

from core.memory.exceptions import (
    DatasetInitError,
    IngestionError,
    MemorySystemError,
    NodeSetCreationError,
    NodeSetResolutionError,
    QueryError,
)
from core.memory.graph_models import (
    Company,
    FinancialConcept,
    FinancialKnowledgeGraph,
)
from core.memory.memory_system import (
    FinancialMemorySystem,
    IngestionItem,
    UserMemoryContext,
)
from core.memory.nodeset_manager import (
    GLOBAL_NODESET_NAME,
    get_or_create_global_nodeset,
    get_or_create_user_nodeset,
    get_user_nodeset_name,
    get_user_nodeset_names,
    hash_user_email,
)
from core.memory.pipeline_tasks import assign_nodesets, build_financial_pipeline
from core.memory.prompts import FINANCIAL_COGNIFY_SYSTEM_PROMPT

__all__ = [
    # Main system
    "FinancialMemorySystem",
    "UserMemoryContext",
    "IngestionItem",
    # NodeSet management
    "hash_user_email",
    "get_user_nodeset_name",
    "get_or_create_global_nodeset",
    "get_or_create_user_nodeset",
    "get_user_nodeset_names",
    "GLOBAL_NODESET_NAME",
    # Cognee built-in NodeSet
    "NodeSet",
    # Graph models
    "Company",
    "FinancialConcept",
    "FinancialKnowledgeGraph",
    # Prompts
    "FINANCIAL_COGNIFY_SYSTEM_PROMPT",
    # Pipeline
    "assign_nodesets",
    "build_financial_pipeline",
    # Exceptions
    "MemorySystemError",
    "NodeSetCreationError",
    "NodeSetResolutionError",
    "DatasetInitError",
    "IngestionError",
    "QueryError",
]
