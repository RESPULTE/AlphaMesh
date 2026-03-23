"""Tests for OrchestratorAgent flow."""

from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

import core.agents.orchestrator_agent as orchestrator_module
from core.agents.models.news_agent_models import CitedSource, NewsAgentOutput
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

    # Patch service_manager.get_agent to return a dummy LLM with with_structured_output
    class DummyLLM:
        def with_structured_output(self, schema):
            if schema.__name__ == "OrchestratorPlan":

                async def fake_plan_invoke(x):
                    return plan

                return RunnableLambda(fake_plan_invoke)
            else:

                async def fake_synth_invoke(x):
                    return schema(relationships=[], response="hello")

                return RunnableLambda(fake_synth_invoke)

        async def ainvoke(self, *args, **kwargs):
            pass

    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0: DummyLLM())
    agent = OrchestratorAgent()

    # Patch agent._llm to a mock with with_structured_output
    class DummyLLM2:
        def with_structured_output(self, schema):
            if schema.__name__ == "OrchestratorPlan":

                async def fake_plan_invoke(x):
                    return plan

                return RunnableLambda(fake_plan_invoke)
            else:

                async def fake_synth_invoke(x):
                    return schema(relationships=[], response="hello")

                return RunnableLambda(fake_synth_invoke)

        async def ainvoke(self, *args, **kwargs):
            pass

    agent._llm = DummyLLM2()
    plan = OrchestratorPlan(
        query="hi",
        vector_query="hi",
        ticker="",
        start_date=datetime.now(timezone.utc),
        end_date=datetime.now(timezone.utc),
        target_agents=[],
        needs_memory=False,
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

    # Patch service_manager.get_agent to return a dummy LLM with with_structured_output
    class DummyLLM:
        def with_structured_output(self, schema):
            if schema.__name__ == "OrchestratorPlan":

                async def fake_plan_invoke(x):
                    return plan

                return RunnableLambda(fake_plan_invoke)
            else:

                async def fake_synth_invoke(x):
                    return schema(relationships=[], response="analysis")

                return RunnableLambda(fake_synth_invoke)

        async def ainvoke(self, *args, **kwargs):
            pass

    monkeypatch.setattr(service_manager, "get_agent", lambda temperature=0: DummyLLM())
    agent = OrchestratorAgent()
    plan = OrchestratorPlan(
        query="test",
        vector_query="test vector",
        ticker="TEST",
        start_date=datetime.now(timezone.utc),
        end_date=datetime.now(timezone.utc),
        target_agents=["news_agent"],
        needs_memory=False,
        final_answer=None,
    )

    async def fake_plan(self, state):
        return {"plan": plan}

    async def fake_synthesize(self, state):
        return FinalResponse(summary="analysis", sources=[], fundamental_data=None)

    monkeypatch.setattr(OrchestratorAgent, "_plan_node", fake_plan)
    monkeypatch.setattr(OrchestratorAgent, "_synthesize_node", fake_synthesize)
    agent._agents = {"news_agent": StubNewsAgent()}

    result = await agent.run([HumanMessage(content="test")])
    assert result.summary == "analysis"


@pytest.mark.asyncio
async def test_load_context_node_called_when_needs_memory_true(monkeypatch):
    """When planner returns needs_memory=True, load_context node must run."""
    load_context_called = []

    async def fake_load_context(self, state):
        load_context_called.append(True)
        return {
            "user_context": None,
            "user_context_block": "USER INVESTMENT PROFILE:\n1. [Bought] AAPL",
            "user_context_loaded": True,
        }

    async def fake_synthesize(self, state):
        return FinalResponse(summary="done", sources=[], fundamental_data=None)

    monkeypatch.setattr(OrchestratorAgent, "_load_context_node", fake_load_context)
    monkeypatch.setattr(OrchestratorAgent, "_synthesize_node", fake_synthesize)

    agent = OrchestratorAgent()
    plan = OrchestratorPlan(
        query="how is my portfolio?",
        vector_query="portfolio performance",
        ticker=None,
        start_date=datetime.now(timezone.utc),
        end_date=datetime.now(timezone.utc),
        target_agents=[],
        needs_memory=True,
        final_answer=None,
    )

    async def fake_plan(self, state):
        return {"plan": plan}

    monkeypatch.setattr(OrchestratorAgent, "_plan_node", fake_plan)
    result = await agent.run([HumanMessage(content="how is my portfolio?")])
    assert load_context_called, "load_context must be called when needs_memory=True"
    assert isinstance(result, FinalResponse)
