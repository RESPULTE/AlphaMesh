"""Tests for OrchestratorAgent flow."""

from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage

import core.agents.orchestrator_agent as orchestrator_module
from core.agents.models import CitedSource, NewsAgentOutput
from core.agents.orchestrator_agent import (
    FinalResponse,
    OrchestratorAgent,
    OrchestratorPlan,
)
from core.services import service_manager


class StubNewsAgent:
    async def run(self, input_data):
        return NewsAgentOutput(
            agent_name="news_agent",
            analysis="analysis",
            sources=[CitedSource(source_id=1, title="t", url="u", page_content="c")],
            entities_enriched=[],
        )


@pytest.mark.asyncio
async def test_orchestrator_returns_final_answer(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "AVAILABLE_AGENTS", [])
    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0: object())
    agent = OrchestratorAgent()
    plan = OrchestratorPlan(
        query="hi",
        vector_query="hi",
        ticker="",
        start_date=datetime.now(timezone.utc),
        end_date=datetime.now(timezone.utc),
        target_agents=[],
        request_requires_agents=False,
        final_answer="hello",
    )

    async def fake_plan(self, state):
        return {"plan": plan}

    monkeypatch.setattr(OrchestratorAgent, "_plan_node", fake_plan)
    result = await agent.run([HumanMessage(content="hi")])
    assert isinstance(result, FinalResponse)
    assert result.summary == "hello"


@pytest.mark.asyncio
async def test_orchestrator_runs_agent_and_synthesizes(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "AVAILABLE_AGENTS", [])
    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0: object())
    agent = OrchestratorAgent()
    plan = OrchestratorPlan(
        query="test",
        vector_query="test vector",
        ticker="TEST",
        start_date=datetime.now(timezone.utc),
        end_date=datetime.now(timezone.utc),
        target_agents=["news_agent"],
        request_requires_agents=True,
        final_answer=None,
    )

    async def fake_plan(self, state):
        return {"plan": plan}

    async def fake_synthesize(self, state):
        return {
            "final_response": FinalResponse(summary="analysis", sources=[]),
            "writeback_relationships": [],
            "writeback_entities": [],
        }

    monkeypatch.setattr(OrchestratorAgent, "_plan_node", fake_plan)
    monkeypatch.setattr(OrchestratorAgent, "_synthesize_node", fake_synthesize)
    agent._agents = {"news_agent": StubNewsAgent()}

    result = await agent.run([HumanMessage(content="test")])
    assert result.summary == "analysis"
