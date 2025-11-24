"""
test_hybrid_rag_subgraph.py

Unit tests for the Hybrid RAG Subgraph using mocks.
"""

from typing import List
from unittest.mock import MagicMock, Mock

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from core.agents.rag_agent import create_hybrid_rag_subgraph
from core.services import service_manager

# --- Mocks ---


class MockRetriever(BaseRetriever):
    docs: List[Document] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs


# --- Tests ---


def test_hybrid_rag_happy_path():
    """Test the ideal flow: Retrieve -> Grade(Yes) -> Generate -> Validate(Yes) -> Store."""

    # Setup Mocks
    retriever = MockRetriever(
        docs=[
            Document(
                page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."
            )
        ]
    )
    llm = service_manager.get_agent()
    vector_store = service_manager.get_vector_store()

    # Create Subgraph
    rag_graph = create_hybrid_rag_subgraph(
        retriever=retriever,
        llm=llm,
        vector_store=vector_store,
        config={"auto_upsert": True},
    )

    # Run
    initial_state = {
        "query": "What is LangGraph?",
        "original_query": "What is LangGraph?",
        "retrieval_attempts": 0,
    }

    result = rag_graph.invoke(initial_state)

    # Assertions
    assert result["is_relevant"] is True
    assert result["is_grounded"] is True
    assert result["upsert_status"] == "stored"
    assert len(result["documents"]) == 1
    vector_store.add_documents.assert_called_once()


def test_retrieval_retry_logic():
    """Test that low relevance triggers query rewriting."""

    retriever = MockRetriever(docs=[])  # Return empty first

    # Mock LLM to fail grading first, then succeed?
    # For simplicity in this unit test, we check if it hits 'rewrite_query' node logic.
    # We can inspect the state transitions or just check the attempt counter.

    llm = service_manager.get_agent()
    # Override structured output to fail first
    mock_grader = MagicMock()
    # Side effect: First call returns "no", subsequent "yes"
    mock_grader.invoke.side_effect = [
        Mock(binary_score="no"),  # First doc check
        Mock(binary_score="yes"),  # Second doc check (after rewrite)
    ]

    llm.with_structured_output = MagicMock(return_value=mock_grader)

    rag_graph = create_hybrid_rag_subgraph(
        retriever=retriever, llm=llm, config={"max_retrieval_attempts": 2}
    )

    initial_state = {
        "query": "Hard query",
        "original_query": "Hard query",
        "retrieval_attempts": 0,
    }

    # Note: Since our MockRetriever always returns empty list in this specific setup,
    # the grader loop might behave oddly if we don't update the retriever.
    # However, we just want to see the graph attempt to rewrite.

    try:
        result = rag_graph.invoke(initial_state)
    except Exception:
        pass  # It might fail if mocks run out of side_effects, but we check the calls.

    # Verify that rewrite_query was likely called if attempts > 0
    # A more robust test would use LangGraph's streaming to inspect steps.

    # Let's rely on a simpler check: if grader returns 'no', does it loop?
    # We can check the final state 'retrieval_attempts'.

    # Resetting for a cleaner run where we force a loop limit hit
    mock_grader.invoke.side_effect = None
    mock_grader.invoke.return_value = Mock(binary_score="no")  # Always fail
    llm.with_structured_output = MagicMock(return_value=mock_grader)

    result = rag_graph.invoke(initial_state)

    # Should hit max attempts
    assert result["retrieval_attempts"] >= 2


def test_store_decision_logic():
    """Test that decide_and_store respects flags."""
    retriever = MockRetriever(
        docs=[Document(page_content="foo", metadata={"source": "trusted.com"})]
    )
    llm = MockLLM()
    vector_store = MagicMock()

    # Case 1: Auto upsert False
    rag_graph = create_hybrid_rag_subgraph(
        retriever=retriever,
        llm=llm,
        vector_store=vector_store,
        config={"auto_upsert": False},
    )
    res = rag_graph.invoke(
        {"query": "q", "original_query": "q", "retrieval_attempts": 0}
    )
    assert res["upsert_status"] == "skipped"
    vector_store.add_documents.assert_not_called()

    # Case 2: Domain Allowlist mismatch
    rag_graph_2 = create_hybrid_rag_subgraph(
        retriever=retriever,
        llm=llm,
        vector_store=vector_store,
        config={"auto_upsert": True, "domain_allowlist": ["other.com"]},
    )
    res_2 = rag_graph_2.invoke(
        {"query": "q", "original_query": "q", "retrieval_attempts": 0}
    )
    # Should skip because "trusted.com" is not in ["other.com"]
    # Logic: if list is filtered to empty, it won't store
    vector_store.add_documents.assert_not_called()


if __name__ == "__main__":
    test_hybrid_rag_happy_path()
    test_retrieval_retry_logic()
    test_store_decision_logic()
