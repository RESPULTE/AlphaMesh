from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from core.agents.models.orchestrator_models import OrchestratorPlan, OrchestratorState
from core.agents.orchestrator_agent import OrchestratorAgent
from core.services import service_manager


def test_user_portfolio_path_resolution_uses_sanitized_email() -> None:
    path = OrchestratorAgent._get_user_portfolio_path(
        "data/portfolio.json",
        "Demo+User@AlphaMesh.Local",
    )
    assert str(path).replace("\\", "/").endswith(
        "data/portfolio_demo_user_alphamesh.local.json"
    )


def test_get_portfolio_for_user_returns_empty_for_missing_file(tmp_path) -> None:
    base_path = str(tmp_path / "portfolio.json")
    holdings = OrchestratorAgent.get_portfolio_for_user(
        base_path=base_path,
        user_email="missing@example.com",
    )
    assert holdings == []


def test_get_portfolio_for_user_returns_empty_for_non_list_json(tmp_path) -> None:
    base_path = str(tmp_path / "portfolio.json")
    user_email = "demo@example.com"
    path = OrchestratorAgent._get_user_portfolio_path(base_path, user_email)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{\"ticker\": \"AAPL\"}", encoding="utf-8")

    holdings = OrchestratorAgent.get_portfolio_for_user(
        base_path=base_path,
        user_email=user_email,
    )
    assert holdings == []


def test_get_portfolio_for_user_returns_empty_for_no_user_email(tmp_path) -> None:
    base_path = str(tmp_path / "portfolio.json")
    holdings = OrchestratorAgent.get_portfolio_for_user(
        base_path=base_path,
        user_email=None,
    )
    assert holdings == []


def test_run_populates_portfolio_block_from_per_user_file(monkeypatch, tmp_path) -> None:
    class FakeGraph:
        async def ainvoke(self, state):
            captured["state"] = state
            return {"summary": "ok"}

    class FakeUserContextService:
        def get_formatted_context(self, _user_email: str) -> str:
            return "USER CONTEXT: from test cache"

    captured: dict = {}
    base_path = str(tmp_path / "portfolio.json")
    monkeypatch.setattr("core.config.settings.PORTFOLIO_JSON_PATH", base_path, raising=False)
    monkeypatch.setattr(service_manager, "get_user_context_service", lambda: FakeUserContextService())

    portfolio_path = OrchestratorAgent._get_user_portfolio_path(
        base_path,
        "demo@alphamesh.local",
    )
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_path.write_text(
        json.dumps(
            [
                {
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "asset_type": "equity",
                    "shares": 125,
                }
            ]
        ),
        encoding="utf-8",
    )

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._graph = FakeGraph()

    result = asyncio.run(
        agent.run(
            messages=[HumanMessage(content="How is my portfolio doing?")],
            conversation_id=None,
            user_email="demo@alphamesh.local",
        )
    )

    assert result.summary == "ok"
    assert "AAPL" in captured["state"].portfolio_block
    assert captured["state"].portfolio_block != "[]"


def test_plan_node_receives_portfolio_context_message() -> None:
    captured: dict = {}

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return OrchestratorPlan(query="route this")

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._llm = FakeLLM()

    state = OrchestratorState(
        messages=[HumanMessage(content="Should I add more AAPL?")],
        user_context_block="USER CONTEXT: interested in large-cap tech",
        portfolio_block='[{"ticker":"AAPL","shares":25}]',
    )

    payload = asyncio.run(agent._plan_node(state))

    assert "plan" in payload
    assert any(
        isinstance(message, SystemMessage)
        and "PORTFOLIO HOLDINGS" in str(message.content)
        and "AAPL" in str(message.content)
        for message in captured["messages"]
    )


def test_synthesize_node_uses_state_portfolio_block() -> None:
    captured: dict = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="synthesised output")

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._llm = FakeLLM()

    state = OrchestratorState(
        messages=[HumanMessage(content="Summarise this for me.")],
        user_context_block="USER CONTEXT: None",
        portfolio_block='[{"ticker":"TSLA","shares":10}]',
    )

    payload = asyncio.run(agent._synthesize_node(state))

    assert payload["summary"] == "synthesised output"
    system_messages = [
        m for m in captured["messages"] if isinstance(m, SystemMessage)
    ]
    assert any("TSLA" in str(m.content) for m in system_messages)
