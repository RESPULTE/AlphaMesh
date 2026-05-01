from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import core.memory.graph.queue.relationship_extractor as extractor_module
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from core.memory.graph.queue.relationship_extractor import RelationshipExtractor
from core.memory.graph.models import (
    BatchEntityExtractionResult,
    ChunkEntityExtractionResult,
    EntityNode,
)
from tenacity.wait import wait_none


class _FakeResponse:
    def __init__(self, content: object, text: object = "") -> None:
        self.content = content
        self.text = text


class _FakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, _messages: object) -> _FakeResponse:
        self.calls += 1
        if not self._responses:
            raise RuntimeError("No responses configured")
        next_response = self._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        if isinstance(next_response, _FakeResponse):
            return next_response
        return _FakeResponse(str(next_response))


class _FakeChunkLLM:
    def __init__(self, result: BatchEntityExtractionResult) -> None:
        self._result = result

    def with_structured_output(self, _schema: object) -> RunnableLambda:
        return RunnableLambda(lambda _payload: self._result)


class _PromptAwareChunkLLM:
    def __init__(self, result: BatchEntityExtractionResult) -> None:
        self._result = result
        self.system_prompts: list[str] = []

    def with_structured_output(self, _schema: object) -> RunnableLambda:
        def _run(payload: object) -> BatchEntityExtractionResult:
            messages = payload.to_messages() if hasattr(payload, "to_messages") else []
            if messages:
                self.system_prompts.append(str(messages[0].content))
            return self._result

        return RunnableLambda(_run)


def test_extract_skips_blank_text_without_llm_call() -> None:
    extractor = RelationshipExtractor(retry_attempts=3)
    llm = _FakeLLM(
        ['<relationships>[{"from_name":"A"}]</relationships>']
    )

    result = asyncio.run(
        extractor.extract(
            mode="relationships",
            text="   ",
            llm=llm,
            system_prompt="system",
        )
    )

    assert result == []
    assert llm.calls == 0


def test_extract_parses_relationship_array() -> None:
    extractor = RelationshipExtractor(retry_attempts=1)
    llm = _FakeLLM(
        [
            '<relationships>[{"from_name":"A","from_type":"Company","to_name":"B","to_type":"Sector"}]</relationships>'
        ]
    )

    result = asyncio.run(
        extractor.extract(
            mode="relationships",
            text="input",
            llm=llm,
            system_prompt="system",
        )
    )

    assert result == [
        {
            "from_name": "A",
            "from_type": "Company",
            "to_name": "B",
            "to_type": "Sector",
        }
    ]
    assert llm.calls == 1


def test_extract_returns_empty_for_missing_or_invalid_blocks() -> None:
    extractor = RelationshipExtractor(retry_attempts=1)
    missing_block_llm = _FakeLLM(["no xml block"])
    invalid_json_llm = _FakeLLM(["<relationships>{not-json}</relationships>"])

    missing = asyncio.run(
        extractor.extract(
            mode="relationships",
            text="input",
            llm=missing_block_llm,
            system_prompt="system",
        )
    )
    invalid = asyncio.run(
        extractor.extract(
            mode="relationships",
            text="input",
            llm=invalid_json_llm,
            system_prompt="system",
        )
    )

    assert missing == []
    assert invalid == []
    assert missing_block_llm.calls == 1
    assert invalid_json_llm.calls == 1


def test_extract_retries_until_success(monkeypatch) -> None:
    monkeypatch.setattr(
        extractor_module,
        "wait_exponential",
        lambda **_kwargs: wait_none(),
    )
    extractor = RelationshipExtractor(retry_attempts=3)
    llm = _FakeLLM(
        [
            RuntimeError("transient-1"),
            RuntimeError("transient-2"),
            '<relationships>[{"id":"ok"}]</relationships>',
        ]
    )

    result = asyncio.run(
        extractor.extract(
            mode="relationships",
            text="input",
            llm=llm,
            system_prompt="system",
        )
    )

    assert result == [{"id": "ok"}]
    assert llm.calls == 3


def test_extract_with_retry_budget_one_does_not_retry() -> None:
    extractor = RelationshipExtractor(retry_attempts=1)
    llm = _FakeLLM(
        [
            RuntimeError("first failure"),
            '<relationships>[{"id":"would-have-succeeded"}]</relationships>',
        ]
    )

    result = asyncio.run(
        extractor.extract(
            mode="relationships",
            text="input",
            llm=llm,
            system_prompt="system",
        )
    )

    assert result == []
    assert llm.calls == 1


def test_extract_parses_from_response_text_when_content_unusable() -> None:
    extractor = RelationshipExtractor(retry_attempts=1)
    llm = _FakeLLM(
        [
            _FakeResponse(
                content="[{'type': 'text', 'text': 'not-parseable-json-like-content'}]",
                text='<relationships>[{"id":"from-text"}]</relationships>',
            )
        ]
    )

    result = asyncio.run(
        extractor.extract(
            mode="relationships",
            text="input",
            llm=llm,
            system_prompt="system",
        )
    )

    assert result == [{"id": "from-text"}]
    assert llm.calls == 1


def test_extract_uses_content_when_valid_even_if_text_exists() -> None:
    extractor = RelationshipExtractor(retry_attempts=1)
    llm = _FakeLLM(
        [
            _FakeResponse(
                content='<relationships>[{"id":"from-content"}]</relationships>',
                text='<relationships>[{"id":"from-text"}]</relationships>',
            )
        ]
    )

    result = asyncio.run(
        extractor.extract(
            mode="relationships",
            text="input",
            llm=llm,
            system_prompt="system",
        )
    )

    assert result == [{"id": "from-content"}]
    assert llm.calls == 1


def test_chunk_entity_extraction_filters_pending_and_updates_status() -> None:
    neo4j = AsyncMock()
    neo4j.get_chunk_extraction_status.return_value = {
        "c1": "PENDING",
        "c2": "EXTRACTED",
    }
    neo4j.get_entities_for_chunks.return_value = [
        {
            "entity_id": "entity-1",
            "entity_name": "Apple",
            "entity_type": "Company",
            "source_chunk_id": "c1",
        }
    ]
    chroma = AsyncMock()
    chroma.get_documents_by_ids.side_effect = [
        [Document(id="c1", page_content="chunk text", metadata={"chunk_id": "c1"})],
        [Document(id="c1", page_content="chunk text", metadata={"chunk_id": "c1"})],
    ]
    nodeset_manager = AsyncMock()
    nodeset_manager.get_global_financial_events_id.return_value = "nodeset-1"
    entity_resolver = AsyncMock()
    entity_resolver.resolve_entity.return_value = SimpleNamespace(entity_id="entity-1")
    llm = _FakeChunkLLM(
        BatchEntityExtractionResult(
            results=[
                ChunkEntityExtractionResult(
                    chunk_id="c1",
                    entities=[
                        EntityNode(
                            id="",
                            name="Apple",
                            entity_type="Company",
                            description="Apple description",
                        )
                    ],
                )
            ]
        )
    )
    extractor = RelationshipExtractor(
        neo4j_adapter=neo4j,
        chroma_adapter=chroma,
        nodeset_manager=nodeset_manager,
        entity_resolver=entity_resolver,
        llm=llm,
        retry_attempts=1,
    )

    entities = asyncio.run(
        extractor.extract(
            mode="chunk_entities",
            chunk_ids=["c1", "c2"],
            llm=llm,
        )
    )

    assert len(entities) == 1
    assert entities[0].id == "entity-1"
    neo4j.get_chunk_extraction_status.assert_awaited_once_with(["c1", "c2"])
    chroma.get_documents_by_ids.assert_any_await(["c1"])
    neo4j.merge_relationship.assert_awaited_once()
    neo4j.get_entities_for_chunks.assert_awaited_once_with(["c1"])
    neo4j.update_chunk_extraction_status.assert_awaited_once_with("c1", "EXTRACTED")
    chroma.update_metadata.assert_awaited_once()


def test_chunk_entity_extraction_treats_missing_status_rows_as_pending() -> None:
    neo4j = AsyncMock()
    neo4j.get_chunk_extraction_status.return_value = {}
    neo4j.get_entities_for_chunks.return_value = [
        {
            "entity_id": "entity-1",
            "entity_name": "Apple",
            "entity_type": "Company",
            "source_chunk_id": "c1",
        }
    ]
    chroma = AsyncMock()
    chroma.get_documents_by_ids.side_effect = [
        [Document(id="c1", page_content="chunk text", metadata={"chunk_id": "c1"})],
        [Document(id="c1", page_content="chunk text", metadata={"chunk_id": "c1"})],
    ]
    nodeset_manager = AsyncMock()
    nodeset_manager.get_global_financial_events_id.return_value = "nodeset-1"
    entity_resolver = AsyncMock()
    entity_resolver.resolve_entity.return_value = SimpleNamespace(entity_id="entity-1")
    llm = _FakeChunkLLM(
        BatchEntityExtractionResult(
            results=[
                ChunkEntityExtractionResult(
                    chunk_id="c1",
                    entities=[
                        EntityNode(
                            id="",
                            name="Apple",
                            entity_type="Company",
                            description="Apple description",
                        )
                    ],
                )
            ]
        )
    )
    extractor = RelationshipExtractor(
        neo4j_adapter=neo4j,
        chroma_adapter=chroma,
        nodeset_manager=nodeset_manager,
        entity_resolver=entity_resolver,
        llm=llm,
        retry_attempts=1,
    )

    entities = asyncio.run(
        extractor.extract(
            mode="chunk_entities",
            chunk_ids=["c1"],
            llm=llm,
        )
    )

    assert len(entities) == 1
    neo4j.get_chunk_extraction_status.assert_awaited_once_with(["c1"])
    neo4j.merge_relationship.assert_awaited_once()
    neo4j.get_entities_for_chunks.assert_awaited_once_with(["c1"])
    neo4j.update_chunk_extraction_status.assert_awaited_once_with("c1", "EXTRACTED")


def test_chunk_entity_extraction_verification_failure_marks_chunk_pending() -> None:
    neo4j = AsyncMock()
    neo4j.get_chunk_extraction_status.return_value = {"c1": "PENDING"}
    neo4j.get_entities_for_chunks.return_value = []
    chroma = AsyncMock()
    chroma.get_documents_by_ids.side_effect = [
        [Document(id="c1", page_content="chunk text", metadata={"chunk_id": "c1"})],
        [Document(id="c1", page_content="chunk text", metadata={"chunk_id": "c1"})],
    ]
    nodeset_manager = AsyncMock()
    nodeset_manager.get_global_financial_events_id.return_value = "nodeset-1"
    entity_resolver = AsyncMock()
    entity_resolver.resolve_entity.return_value = SimpleNamespace(entity_id="entity-1")
    llm = _FakeChunkLLM(
        BatchEntityExtractionResult(
            results=[
                ChunkEntityExtractionResult(
                    chunk_id="c1",
                    entities=[
                        EntityNode(
                            id="",
                            name="Apple",
                            entity_type="Company",
                            description="Apple description",
                        )
                    ],
                )
            ]
        )
    )
    extractor = RelationshipExtractor(
        neo4j_adapter=neo4j,
        chroma_adapter=chroma,
        nodeset_manager=nodeset_manager,
        entity_resolver=entity_resolver,
        llm=llm,
        retry_attempts=1,
    )

    entities = asyncio.run(
        extractor.extract(
            mode="chunk_entities",
            chunk_ids=["c1"],
            llm=llm,
        )
    )

    assert entities == []
    neo4j.merge_relationship.assert_awaited_once()
    assert neo4j.get_entities_for_chunks.await_count >= 1
    for call in neo4j.get_entities_for_chunks.await_args_list:
        assert call.args == (["c1"],)
    status_calls = [
        call.args
        for call in neo4j.update_chunk_extraction_status.await_args_list
    ]
    assert ("c1", "EXTRACTED") not in status_calls
    assert ("c1", "PENDING") in status_calls
    chroma.update_metadata.assert_awaited()


def test_chunk_entity_extraction_force_ignores_status_filter() -> None:
    neo4j = AsyncMock()
    neo4j.get_entities_for_chunks.return_value = [
        {
            "entity_id": "entity-2",
            "entity_name": "TSMC",
            "entity_type": "Company",
            "source_chunk_id": "c2",
        }
    ]
    chroma = AsyncMock()
    chroma.get_documents_by_ids.side_effect = [
        [Document(id="c2", page_content="chunk text", metadata={"chunk_id": "c2"})],
        [Document(id="c2", page_content="chunk text", metadata={"chunk_id": "c2"})],
    ]
    nodeset_manager = AsyncMock()
    nodeset_manager.get_global_financial_events_id.return_value = "nodeset-1"
    entity_resolver = AsyncMock()
    entity_resolver.resolve_entity.return_value = SimpleNamespace(entity_id="entity-2")
    llm = _FakeChunkLLM(
        BatchEntityExtractionResult(
            results=[
                ChunkEntityExtractionResult(
                    chunk_id="c2",
                    entities=[
                        EntityNode(
                            id="",
                            name="TSMC",
                            entity_type="Company",
                            description="TSMC description",
                        )
                    ],
                )
            ]
        )
    )
    extractor = RelationshipExtractor(
        neo4j_adapter=neo4j,
        chroma_adapter=chroma,
        nodeset_manager=nodeset_manager,
        entity_resolver=entity_resolver,
        llm=llm,
        retry_attempts=1,
    )

    entities = asyncio.run(
        extractor.extract(
            mode="chunk_entities",
            chunk_ids=["c2"],
            llm=llm,
            force=True,
        )
    )

    assert len(entities) == 1
    neo4j.get_chunk_extraction_status.assert_not_called()
    neo4j.get_entities_for_chunks.assert_awaited_once_with(["c2"])


def test_upsert_with_retry_marks_chunk_pending_on_failure() -> None:
    extractor = RelationshipExtractor(retry_attempts=1)
    extractor._mark_chunk_pending = AsyncMock()  # type: ignore[method-assign]

    async def _raise() -> None:
        raise RuntimeError("boom")

    ok = asyncio.run(
        extractor._upsert_with_retry(  # type: ignore[attr-defined]
            _raise,
            chunk_id="c-fail",
            max_attempts=1,
        )
    )

    assert ok is False
    extractor._mark_chunk_pending.assert_awaited_once_with("c-fail")


def test_chunk_entity_extraction_uses_runtime_system_prompt_override() -> None:
    neo4j = AsyncMock()
    neo4j.get_chunk_extraction_status.return_value = {"c3": "PENDING"}
    neo4j.get_entities_for_chunks.return_value = [
        {
            "entity_id": "entity-3",
            "entity_name": "Intel",
            "entity_type": "Company",
            "source_chunk_id": "c3",
        }
    ]
    chroma = AsyncMock()
    chroma.get_documents_by_ids.side_effect = [
        [Document(id="c3", page_content="chunk text", metadata={"chunk_id": "c3"})],
        [Document(id="c3", page_content="chunk text", metadata={"chunk_id": "c3"})],
    ]
    nodeset_manager = AsyncMock()
    nodeset_manager.get_global_financial_events_id.return_value = "nodeset-1"
    entity_resolver = AsyncMock()
    entity_resolver.resolve_entity.return_value = SimpleNamespace(entity_id="entity-3")
    llm = _PromptAwareChunkLLM(
        BatchEntityExtractionResult(
            results=[
                ChunkEntityExtractionResult(
                    chunk_id="c3",
                    entities=[
                        EntityNode(
                            id="",
                            name="Intel",
                            entity_type="Company",
                            description="Intel description",
                        )
                    ],
                )
            ]
        )
    )
    extractor = RelationshipExtractor(
        neo4j_adapter=neo4j,
        chroma_adapter=chroma,
        nodeset_manager=nodeset_manager,
        entity_resolver=entity_resolver,
        retry_attempts=1,
    )

    custom_prompt = "CUSTOM CHUNK PROMPT"
    entities = asyncio.run(
        extractor.extract(
            mode="chunk_entities",
            chunk_ids=["c3"],
            llm=llm,
            system_prompt=custom_prompt,
        )
    )

    assert len(entities) == 1
    assert llm.system_prompts
    assert llm.system_prompts[0] == custom_prompt
    neo4j.get_entities_for_chunks.assert_awaited_once_with(["c3"])
