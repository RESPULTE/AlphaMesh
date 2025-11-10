# tests/test_api_keys.py
import pytest
from core.services import get_llm, get_graph, get_vector_store


@pytest.mark.skipif(
    get_llm() is None, reason="Google LLM not configured or API key missing"
)
def test_google_llm_connection():
    """Test that Google Gemini LLM works with API key."""
    llm = get_llm()
    response = llm.invoke("Reply with OK if API works.")
    assert response.content.strip().upper() == "OK"


@pytest.mark.skipif(
    get_graph() is None, reason="Neo4j not configured or connection failed"
)
def test_neo4j_connection():
    """Test Neo4j connection."""
    graph = get_graph()
    result = graph.query("RETURN 1 AS ok")
    assert result[0]["ok"] == 1


@pytest.mark.skipif(
    get_vector_store() is None, reason="Chroma vector store not configured"
)
def test_chroma_connection():
    """Test Chroma vector store connectivity."""
    vector_store = get_vector_store()
    # Add a test text
    vector_store.add_texts(["test sentence"])
    results = vector_store.similarity_search("test")
    assert len(results) > 0
