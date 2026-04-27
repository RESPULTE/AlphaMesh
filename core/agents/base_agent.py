from abc import ABC, abstractmethod
from typing import Any, Dict, Type, get_args, get_origin

from pydantic import BaseModel, Field

from core.logger import get_logger

logger = get_logger(__name__)


class AgentInput(BaseModel):
    """Base schema for agent inputs."""

    raw_input: str = Field(description="The original user query.")


class AgentOutput(BaseModel):
    """Base schema for agent outputs."""

    agent_name: str = Field(
        description="The name of the agent that produced the output."
    )
    output: Any = Field(description="The output from the agent.")


class AbstractAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the agent."""
        pass

    @staticmethod
    @abstractmethod
    def description() -> str:
        """A description of what the agent is good for."""
        pass

    @staticmethod
    @abstractmethod
    def get_output_schema_class() -> Type[BaseModel]:
        """The Pydantic model for the agent's specific input."""
        pass

    @abstractmethod
    def run(self, input_data: BaseModel) -> AgentOutput:
        """
        The main entry point for the agent to perform its task.

        Args:
            input_data: A Pydantic model instance matching the agent's `input_schema`.

        Returns:
            An AgentOutput instance containing the results.
        """
        pass

    @classmethod
    def get_input_schema(cls) -> str:
        lines = []
        for name, field in cls.get_input_schema_class().model_fields.items():
            dtype = field.annotation
            desc = field.description or ""
            default = field.default

            # Format type nicely (handles Optional, List, etc.)
            origin = get_origin(dtype)
            if origin is list:
                arg = get_args(dtype)[0]
                type_str = f"List[{arg.__name__}]"
            else:
                type_str = getattr(dtype, "__name__", str(dtype))

            # Default display
            default_str = ""
            if default is None:
                default_str = " (optional)"
            elif default is ...:
                default_str = " (required)"

            lines.append(f"- {name}: {type_str}{default_str} — {desc}")

        return "\n".join(lines)

    @staticmethod
    def render_memory_summary(memory_summary: Dict[str, Any]) -> str:
        """
        Render a compact single-line summary used to feed this agent's prior-turn
        memory back into future orchestrator planning/execution.

        Sub-agents should override this to keep formatting logic domain-local.
        """
        if not memory_summary:
            return ""
        return str(memory_summary)
