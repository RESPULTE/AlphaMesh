"""Integration test for full AlphaMesh pipeline using live services."""

import logging
from datetime import datetime, timedelta

import pytest
from langchain_core.messages import HumanMessage

from core.agents.models import BaseAgentInput
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.agents.orchestrator_agent import OrchestratorAgent


@pytest.mark.asyncio
async def test_full_orchestrator_pipeline_live() -> None:
    logging.basicConfig(level=logging.INFO)

    agent = OrchestratorAgent()
    messages = [HumanMessage(content="What is the recent news of Apple?")]

    try:
        output = await agent.run(messages, conversation_id="test-e2e-1234")
    except Exception as exc:
        exc_str = str(exc).lower()
        exc_type = str(type(exc).__name__).lower()
        if any(
            keyword in exc_str or keyword in exc_type
            for keyword in [
                "connection",
                "refused",
                "chroma",
                "neo4j",
                "could not connect",
                "api key",
                "apikey",
                "unauthorized",
                "permission",
                "forbidden",
                "quota",
                "rate",
                "timeout",
                "newsapi",
            ]
        ):
            pytest.skip(f"Skipping; live service call failed: {exc}")
        raise

    assert output.summary is not None
    assert isinstance(output.summary, str)
    assert len(output.summary) > 0

    logging.info(f"Agent summary output: {output.summary}")
    logging.info(f"Agent cited sources: {output.sources}")


@pytest.mark.asyncio
async def test_news_agent_pipeline_live() -> None:
    logging.basicConfig(level=logging.INFO)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)

    agent = NewsAnalysisAgent()
    input_data = BaseAgentInput(
        query="Recent Apple stock price",
        vector_query="Apple stock price",
        ticker="AAPL",
        start_date=start_date,
        end_date=end_date,
        target_agents=["news_agent"],
        request_requires_agents=True,
    )

    try:
        output = await agent.run(input_data)
    except Exception as exc:
        exc_str = str(exc).lower()
        exc_type = str(type(exc).__name__).lower()
        if any(
            keyword in exc_str or keyword in exc_type
            for keyword in [
                "connection",
                "refused",
                "chroma",
                "neo4j",
                "could not connect",
                "api key",
                "apikey",
                "unauthorized",
                "permission",
                "forbidden",
                "quota",
                "rate",
                "timeout",
                "newsapi",
            ]
        ):
            pytest.skip(f"Skipping; live service call failed: {exc}")
        raise

    assert output.agent_name == "news_agent"
    assert isinstance(output.analysis, str)
    if output.sources:
        assert all(source.title for source in output.sources)
        assert all(source.url for source in output.sources)
