from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.agents.models.orchestrator_models import (
    OrchestratorPlan,
    OrchestratorState,
)
from core.agents.orchestrator_agent import OrchestratorAgent
from core.agents.ticker_validation import TickerInfo
from core.agents.utility.orchestrator_helpers import (
    _get_user_portfolio_path,
    get_portfolio_for_user,
)
from core.memory.user_interest_models import UserInterestQuerySpec
from core.services import service_manager


def test_user_portfolio_path_resolution_uses_sanitized_email() -> None:
    path = _get_user_portfolio_path(
        "data/portfolio.json",
        "Demo+User@AlphaMesh.Local",
    )
    assert str(path).replace("\\", "/").endswith(
        "data/portfolio_demo_user_alphamesh.local.json"
    )


def test_get_portfolio_for_user_returns_empty_for_missing_file(tmp_path) -> None:
    base_path = str(tmp_path / "portfolio.json")
    holdings = get_portfolio_for_user(
        base_path=base_path,
        user_email="missing@example.com",
    )
    assert holdings == []


def test_get_portfolio_for_user_returns_empty_for_non_list_json(tmp_path) -> None:
    base_path = str(tmp_path / "portfolio.json")
    user_email = "demo@example.com"
    path = _get_user_portfolio_path(base_path, user_email)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{\"ticker\": \"AAPL\"}", encoding="utf-8")

    holdings = get_portfolio_for_user(
        base_path=base_path,
        user_email=user_email,
    )
    assert holdings == []


def test_get_portfolio_for_user_returns_empty_for_no_user_email(tmp_path) -> None:
    base_path = str(tmp_path / "portfolio.json")
    holdings = get_portfolio_for_user(
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
        async def load_for_user(self, _user_email: str):
            return None

        def get_formatted_context(self, _user_email: str) -> str:
            return "USER CONTEXT: from test cache"

    captured: dict = {}
    base_path = str(tmp_path / "portfolio.json")
    monkeypatch.setattr("core.config.settings.PORTFOLIO_JSON_PATH", base_path, raising=False)
    monkeypatch.setattr(service_manager, "get_user_context_service", lambda: FakeUserContextService())

    portfolio_path = _get_user_portfolio_path(
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


def test_plan_node_receives_portfolio_context_message(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return OrchestratorPlan(query="route this")

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    fake_llm = FakeLLM()
    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0.0: fake_llm)

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {
        "news_agent": SimpleNamespace(
            build_memory_context_from_history=lambda turns, window=8: (
                "- [2026-04-26T00:01:00+00:00] actions=newsapi; sources=3"
            )
        )
    }

    state = OrchestratorState(
        messages=[HumanMessage(content="Should I add more AAPL?")],
        user_context_block="USER CONTEXT: interested in large-cap tech",
        portfolio_block='[{"ticker":"AAPL","shares":25}]',
        conversation_memory_block="1. [2026-04-26T00:00:00+00:00] turn=t-1 score=0.91\n   User asked about AAPL weighting.",
        history_turns=[
            {
                "created_at": "2026-04-26T00:01:00+00:00",
                "agent_memory_summaries": {
                    "news_agent": {
                        "research_actions": ["newsapi"],
                        "source_count": 3,
                        "main_catalyst": "Guidance",
                    }
                },
            }
        ],
    )

    payload = asyncio.run(agent._plan_node(state))

    assert "plan" in payload
    assert any(
        isinstance(message, HumanMessage)
        and "PORTFOLIO HOLDINGS" in str(message.content)
        and "AAPL" in str(message.content)
        for message in captured["messages"]
    )
    assert any(
        isinstance(message, HumanMessage)
        and "Agent-provided memory contexts from prior turn summaries" in str(message.content)
        and "news_agent" in str(message.content)
        for message in captured["messages"]
    )
    assert any(
        isinstance(message, HumanMessage)
        and "Retrieved private conversation memory chunks" in str(message.content)
        and "AAPL weighting" in str(message.content)
        for message in captured["messages"]
    )


def test_synthesize_node_uses_state_portfolio_block(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "core.config.settings.ENABLE_ANALYSIS_TOKEN_STREAMING",
        False,
        raising=False,
    )

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(text="synthesised output")

    monkeypatch.setattr(
        service_manager,
        "get_agent",
        lambda temperature=0.0: FakeLLM(),
    )

    agent = OrchestratorAgent.__new__(OrchestratorAgent)

    state = OrchestratorState(
        messages=[HumanMessage(content="Summarise this for me.")],
        user_context_block="USER CONTEXT: None",
        portfolio_block='[{"ticker":"TSLA","shares":10}]',
        conversation_memory_block="1. [2026-04-26T00:00:00+00:00] turn=t-1 score=0.87\n   User prefers concise summaries.",
    )

    payload = asyncio.run(agent._synthesize_node(state))

    assert payload["summary"] == "synthesised output"
    system_messages = [
        m for m in captured["messages"] if isinstance(m, SystemMessage)
    ]
    human_messages = [m for m in captured["messages"] if isinstance(m, HumanMessage)]
    assert any("TSLA" in str(m.content) for m in system_messages)
    assert any(
        "Retrieved private conversation memory chunks" in str(m.content)
        and "prefers concise summaries" in str(m.content)
        for m in human_messages
    )


def test_plan_node_sets_clarification_on_invalid_ticker(monkeypatch) -> None:
    class FakeStructuredLLM:
        async def ainvoke(self, _messages):
            return OrchestratorPlan(
                query="check this ticker",
                target_agents=["news_agent"],
                tickers=["AAPL"],
            )

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    class FakeTickerValidator:
        async def validate_and_enrich(self, tickers):
            assert tickers == ["AAPL"]
            return {
                "AAPL": TickerInfo(
                    ticker="AAPL",
                    is_valid=False,
                    is_equity=False,
                    quote_type=None,
                    needs_confirmation=True,
                    suggestions=["AAPL"],
                )
            }

    monkeypatch.setattr(
        service_manager, "get_ticker_validator", lambda: FakeTickerValidator()
    )

    monkeypatch.setattr(
        service_manager,
        "get_agent",
        lambda temperature=0.0: FakeLLM(),
    )

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {}

    state = OrchestratorState(
        messages=[HumanMessage(content="Analyze AAPL please")],
        user_context_block="USER CONTEXT: None",
        portfolio_block="[]",
    )

    payload = asyncio.run(agent._plan_node(state))

    assert payload["plan"] is not None
    assert payload["plan"].final_answer is not None
    assert "confirm the securities" in payload["plan"].final_answer
    assert payload["company_context_blocks"] == {}
    assert payload["ticker_metadata"] == {}


def test_plan_node_populates_enrichment_for_valid_equity(monkeypatch) -> None:
    class FakeStructuredLLM:
        async def ainvoke(self, _messages):
            return OrchestratorPlan(
                query="analyze apple",
                target_agents=["news_agent"],
                tickers=["AAPL"],
            )

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    class FakeTickerValidator:
        async def validate_and_enrich(self, tickers):
            assert tickers == ["AAPL"]
            return {
                "AAPL": TickerInfo(
                    ticker="AAPL",
                    is_valid=True,
                    is_equity=True,
                    quote_type="EQUITY",
                    long_name="Apple Inc.",
                    description="Consumer electronics company.",
                    sector="Technology",
                    industry="Consumer Electronics",
                )
            }

    events: list[tuple[str, str, dict]] = []

    monkeypatch.setattr(
        service_manager, "get_ticker_validator", lambda: FakeTickerValidator()
    )
    monkeypatch.setattr(
        "core.agents.utility.orchestrator_helpers.publish_frontend_event",
        lambda source, event_type, payload: events.append((source, event_type, payload)),
    )

    monkeypatch.setattr(
        service_manager,
        "get_agent",
        lambda temperature=0.0: FakeLLM(),
    )

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {}

    state = OrchestratorState(
        messages=[HumanMessage(content="Analyze AAPL please")],
        user_context_block="USER CONTEXT: None",
        portfolio_block="[]",
    )

    payload = asyncio.run(agent._plan_node(state))

    assert payload["plan"] is not None
    assert payload["plan"].final_answer is None
    assert "AAPL" in payload["company_context_blocks"]
    assert "COMPANY BACKGROUND: Apple Inc. (AAPL)" in payload["company_context_blocks"]["AAPL"]
    assert payload["ticker_metadata"]["AAPL"]["long_name"] == "Apple Inc."
    assert events == [
        (
            "orchestrator",
            "ticker_resolved",
            {"ticker": "AAPL", "tickers": ["AAPL"]},
        )
    ]


def test_build_graph_runs_preplanner_before_planner() -> None:
    order: list[str] = []

    async def _preplanner(_state):
        order.append("preplanner")
        return {"user_interest_graph_context_block": "(none)"}

    async def _planner(_state):
        order.append("planner")
        return {"plan": OrchestratorPlan(query="x", final_answer="done")}

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {}
    agent._prepare_user_interest_context_node = _preplanner  # type: ignore[method-assign]
    agent._plan_node = _planner  # type: ignore[method-assign]
    graph = agent._build_graph()

    result = asyncio.run(
        graph.ainvoke(
            OrchestratorState(messages=[HumanMessage(content="hello")])
        )
    )

    assert order == ["preplanner", "planner"]
    assert result["summary"] == "done"


def test_prepare_user_interest_context_delegates_to_user_context_service(
    monkeypatch,
) -> None:
    captured: dict = {}

    sentinel_llm = object()

    class FakeUserContextService:
        async def build_targeted_orchestrator_context(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                query_spec=UserInterestQuerySpec(broad_fallback=True),
                context_block="Domain Summary:\n1. investment:Technology",
                debug_payload={"mode": "fallback_domains_only"},
            )

    monkeypatch.setattr(
        service_manager, "get_user_context_service", lambda: FakeUserContextService()
    )
    monkeypatch.setattr(
        service_manager, "get_agent", lambda temperature=0.0: sentinel_llm
    )

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {}

    state = OrchestratorState(
        user_email="demo@alphamesh.local",
        messages=[HumanMessage(content="what are my interests these days?")],
        user_context_block="USER CONTEXT: baseline",
        portfolio_block='[{"ticker":"AAPL"}]',
    )

    payload = asyncio.run(agent._prepare_user_interest_context_node(state))

    assert captured["user_email"] == "demo@alphamesh.local"
    assert captured["latest_user_message"] == "what are my interests these days?"
    assert captured["baseline_user_context_block"] == "USER CONTEXT: baseline"
    assert captured["portfolio_block"] == '[{"ticker":"AAPL"}]'
    assert captured["llm"] is sentinel_llm
    assert payload["user_interest_graph_context_block"].startswith("Domain Summary")
    assert payload["user_interest_query_debug"]["mode"] == "fallback_domains_only"


def test_prepare_user_interest_context_service_error_returns_none_block(
    monkeypatch,
) -> None:
    class FakeUserContextService:
        async def build_targeted_orchestrator_context(self, **kwargs):
            raise RuntimeError("service failed")

    monkeypatch.setattr(
        service_manager, "get_user_context_service", lambda: FakeUserContextService()
    )
    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0.0: object())

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {}

    state = OrchestratorState(
        user_email="demo@alphamesh.local",
        messages=[HumanMessage(content="check apple")],
        user_context_block="USER CONTEXT: baseline",
        portfolio_block="[]",
    )
    payload = asyncio.run(agent._prepare_user_interest_context_node(state))
    assert payload["user_interest_graph_context_block"] == "(none)"
    assert payload["user_interest_query_debug"]["mode"] == "error"


def test_plan_node_receives_targeted_user_interest_context_message(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return OrchestratorPlan(query="route this")

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    fake_llm = FakeLLM()
    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0.0: fake_llm)

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {}

    state = OrchestratorState(
        messages=[HumanMessage(content="How about my risk setup?")],
        user_context_block="USER CONTEXT: baseline",
        portfolio_block="[]",
        user_interest_graph_context_block="Matched Domain:\n- investment:Technology",
    )

    _ = asyncio.run(agent._plan_node(state))

    assert any(
        isinstance(message, HumanMessage)
        and "TARGETED USER-INTEREST GRAPH CONTEXT" in str(message.content)
        and "investment:Technology" in str(message.content)
        for message in captured["messages"]
    )


def test_plan_node_receives_canonical_sector_names_context_message(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return OrchestratorPlan(query="route this")

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    fake_llm = FakeLLM()
    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0.0: fake_llm)

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {}

    state = OrchestratorState(
        messages=[HumanMessage(content="What sector should this be?")],
        user_context_block="USER CONTEXT: baseline",
        portfolio_block="[]",
    )

    _ = asyncio.run(agent._plan_node(state))

    assert any(
        isinstance(message, HumanMessage)
        and "CANONICAL SECTOR NAMES" in str(message.content)
        and "Technology" in str(message.content)
        and "Healthcare" in str(message.content)
        for message in captured["messages"]
    )


def test_plan_node_injects_raw_turn_history_messages(monkeypatch) -> None:
    captured: dict = {}

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return OrchestratorPlan(query="route this")

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0.0: FakeLLM())

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent._agents = {}

    state = OrchestratorState(
        messages=[HumanMessage(content="What should I do now?")],
        user_context_block="USER CONTEXT: baseline",
        portfolio_block="[]",
        history_turns=[
            {
                "user_message": "How is AAPL doing?",
                "assistant_synthesis": "AAPL is stable with mixed catalysts.",
            }
        ],
    )

    _ = asyncio.run(agent._plan_node(state))

    message_types = [type(m) for m in captured["messages"]]
    assert message_types.count(SystemMessage) == 1
    assert any(
        isinstance(m, HumanMessage) and m.content == "How is AAPL doing?"
        for m in captured["messages"]
    )
    assert any(
        isinstance(m, AIMessage) and m.content == "AAPL is stable with mixed catalysts."
        for m in captured["messages"]
    )
