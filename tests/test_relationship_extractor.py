from __future__ import annotations

import asyncio

import core.memory.graph.queue.relationship_extractor as extractor_module
from core.memory.graph.queue.relationship_extractor import RelationshipExtractor
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


def test_extract_skips_blank_text_without_llm_call() -> None:
    extractor = RelationshipExtractor(retry_attempts=3)
    llm = _FakeLLM(
        ['<relationships>[{"from_name":"A"}]</relationships>']
    )

    result = asyncio.run(
        extractor.extract(text="   ", llm=llm, system_prompt="system")
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
        extractor.extract(text="input", llm=llm, system_prompt="system")
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
        extractor.extract(text="input", llm=missing_block_llm, system_prompt="system")
    )
    invalid = asyncio.run(
        extractor.extract(text="input", llm=invalid_json_llm, system_prompt="system")
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
        extractor.extract(text="input", llm=llm, system_prompt="system")
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
        extractor.extract(text="input", llm=llm, system_prompt="system")
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
        extractor.extract(text="input", llm=llm, system_prompt="system")
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
        extractor.extract(text="input", llm=llm, system_prompt="system")
    )

    assert result == [{"id": "from-content"}]
    assert llm.calls == 1
