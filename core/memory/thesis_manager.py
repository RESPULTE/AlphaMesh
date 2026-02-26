"""
core/memory/thesis_manager.py

Handles operations related to InvestmentTheses, including
preventing thesis bloat (scope evaluation / forking) and managing conviction levels.
"""

from enum import Enum
from typing import Any
import logging

from pydantic import BaseModel, Field

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from core.memory.exceptions import MemorySystemError

logger = logging.getLogger(__name__)

class ThesisAction(str, Enum):
    APPEND = "APPEND"
    FORK = "FORK"

class ThesisScopeDecision(BaseModel):
    action: ThesisAction = Field(description="Action to take: APPEND to existing thesis or FORK a new one.")
    summary: str = Field(description="The core summary of the reasoning or the new thesis.")
    reasoning: str = Field(description="Explanation of why this decision was made.")

class InvestmentThesisManager:
    """
    Manager for InvestmentThesis operations.
    """
    def __init__(self, memory_system: Any):
        """
        Args:
            memory_system: An instance of FinancialMemorySystem.
        """
        self.memory_system = memory_system

    async def evaluate_thesis_scope(
        self,
        new_target_node: Any,
        new_reasoning: str,
        existing_thesis: Any
    ) -> ThesisScopeDecision:
        """
        Evaluates whether a new target node with its reasoning should be appended 
        to an existing thesis or if a new thesis should be forked.

        Args:
            llm_client: The LLM client or adapter.
            new_target_node: The node being added to the thesis (e.g., Company or Sector).
            new_reasoning: The user's reasoning for adding this target.
            existing_thesis: The existing InvestmentThesis node.

        Returns:
            A ThesisScopeDecision structured response containing the action (APPEND or FORK),
            the extracted summary, and the reasoning behind the decision.
        """
        system_prompt = (
            "You are an expert financial AI assistant. "
            "Your task is to determine whether a user's new reasoning for adding a target to an investment "
            "thesis aligns with the existing thesis summary, or if it represents a fundamentally different thesis.\n\n"
            "If the reasoning is semantically similar or supports the existing thesis (e.g., adding a semiconductor "
            "company for the same 'AI chip demand' reason), choose 'APPEND'.\n"
            "If the reasoning introduces a distinct investment logic (e.g., different driver, theme, or catalyst like "
            "'Consumer AR adoption'), choose 'FORK' to create a new thesis.\n\n"
            "Provide a concise summary of the reasoning, and a brief explanation of your decision."
        )

        target_name = getattr(new_target_node, "name", getattr(new_target_node, "ticker", str(new_target_node)))
        thesis_summary = getattr(existing_thesis, "summary", str(existing_thesis))
        
        user_prompt = (
            f"Existing Thesis Summary: {thesis_summary}\n"
            f"New Target Node: {target_name}\n"
            f"New Reasoning: {new_reasoning}\n"
        )
        
        try:
            decision: ThesisScopeDecision = await LLMGateway.acreate_structured_output(
                text_input=user_prompt,
                system_prompt=system_prompt,
                response_model=ThesisScopeDecision,
            )
            return decision
        except Exception as exc:
            logger.error("Failed to evaluate thesis scope: %s", exc)
            # Default to FORK to prevent unintended bloat if evaluation fails
            return ThesisScopeDecision(
                action=ThesisAction.FORK,
                summary=new_reasoning,
                reasoning=f"Error evaluating scope, defaulted to FORK: {exc}"
            )

    async def manage_thesis_conviction(
        self,
        user_nodeset_id: str,
        thesis_id: str,
        delta: float,
        threshold: float = 0.1,
    ) -> float:
        """
        Specific state trigger logic for an InvestmentThesis.

        1. Calls memory_system.adjust_edge_property to update the conviction_level.
        2. Evaluates the new conviction level against a hard threshold.
        3. Automatically transitions the Thesis node to 'Dormant' status if it drops below the threshold.

        Args:
            user_nodeset_id: Node ID denoting the user (source).
            thesis_id: Node ID of the InvestmentThesis (target).
            delta: Float delta to modify the conviction level.
            threshold: Lower bound trigger. Should default to 0.1 in standard logic.
            
        Returns:
            The new conviction level as a float.
        """
        self.memory_system._require_initialized()
        graph_client = self.memory_system.graph_client

        new_conviction = await self.memory_system.adjust_edge_property(
            source_id=user_nodeset_id,
            target_id=thesis_id,
            edge_label="HoldsThesis",
            property_name="conviction_level",
            delta=delta,
            min_val=0.0,
            max_val=1.0,
        )

        # Evaluate state trigger
        if new_conviction is not None and new_conviction < threshold:
            logger.info(
                "Conviction level for thesis '%s' dropped to %.2f (< %.2f). Triggering status update to Dormant.",
                thesis_id,
                new_conviction,
                threshold,
            )

            query = """
            MATCH (t {id: $thesis_id})
            SET t.status = 'Dormant'
            RETURN t.id
            """

            try:
                execute_fn = None
                if hasattr(graph_client, "graph") and hasattr(graph_client.graph, "execute"):
                    execute_fn = graph_client.graph.execute
                elif hasattr(graph_client, "execute"):
                    execute_fn = graph_client.execute

                if execute_fn:
                    await execute_fn(query, {"thesis_id": thesis_id})
                else:
                    logger.error("No execute method found on graph_client to run state trigger.")
            except Exception as exc:
                logger.error(
                    "Failed to execute state change to Dormant for thesis %s: %s",
                    thesis_id,
                    exc,
                )
                raise MemorySystemError(f"Thesis state update to Dormant failed: {exc}") from exc

        return new_conviction
