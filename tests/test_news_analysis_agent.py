"""Unit tests for NewsAnalysisAgent."""
import pytest

from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.models import BaseAgentInput
from core.graph.models import ChunkExtractionResult
from core.services import service_manager


class DummyNewsAPI:
    def get_everything(self, **kwargs):
        return {
            "status": "ok",
            "articles": [
                {
                    "title": "Test",
                    "url": "http://example.com",
                    "publishedAt": "2026-03-10T00:00:00Z",
                    "description": "desc",
                    "content": "content text",
                    "source": {"name": "Example"},
                }
            ],
        }


class DummyIngestor:
    async def ingest_articles(self, articles, companies_involved):
        return ["chunk-1"]


class DummyEmbeddingFunc:
    async def aembed_query(self, query):
        return [0.1, 0.2]

    async def aembed_documents(self, docs):
        return [[0.1, 0.2] for _ in docs]


class DummyChromaAdapter:
    async def query(self, _embedding, n_results, where=None):
        return {
            "ids": [["chunk-1"]],
            "documents": [["chunk text"]],
            "metadatas": [
                [
                    {
                        "companies_involved": ["Test"],
                        "article_title": "Test",
                        "source_url": "http://example.com",
                    }
                ]
            ],
            "distances": [[0.1]],
        }

    async def update_metadata(self, ids, metadatas):
        self.updated = list(zip(ids, metadatas))


class DummyNeo4jAdapter:
    async def get_chunk_extraction_status(self, chunk_ids):
        return {cid: "PENDING" for cid in chunk_ids}

    async def merge_entity_node(self, node):
        self.entities = getattr(self, "entities", []) + [node]

    async def merge_relationship(self, source_id, target_id, rel_type, props):
        self.relationships = getattr(self, "relationships", []) + [
            (source_id, target_id, rel_type, props)
        ]

    async def update_chunk_extraction_status(self, chunk_id, status):
        self.status = (chunk_id, status)


class DummyLLM:
    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, *args, **kwargs):
        return type("Resp", (), {"content": "analysis"})()


class DummyExtractionPrompt:
    def __or__(self, other):
        return self

    async def ainvoke(self, *args, **kwargs):
        return ChunkExtractionResult(chunk_id="chunk-1", entities=[], relationships=[])


@pytest.mark.asyncio
async def test_news_agent_pipeline(monkeypatch):
    monkeypatch.setattr(service_manager, "get_news_api", lambda: DummyNewsAPI())
    monkeypatch.setattr(service_manager, "get_ingestor", lambda: DummyIngestor())
    monkeypatch.setattr(service_manager, "get_embedding_func", lambda: DummyEmbeddingFunc())
    monkeypatch.setattr(service_manager, "get_chroma_adapter", lambda: DummyChromaAdapter())
    monkeypatch.setattr(service_manager, "get_neo4j_adapter", lambda: DummyNeo4jAdapter())
    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0: DummyLLM())
    monkeypatch.setattr(
        "core.graph.extraction_prompts.build_extraction_prompt",
        lambda: DummyExtractionPrompt(),
    )

    agent = NewsAnalysisAgent()
    input_data = BaseAgentInput(
        query="test",
        vector_query="vector test",
        ticker="TEST",
        start_date="2026-03-01",
        end_date="2026-03-02",
    )
    output = await agent.run(input_data)
    assert output.analysis == "analysis"
    assert output.sources
