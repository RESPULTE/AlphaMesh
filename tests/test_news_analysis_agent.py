"""Integration test for NewsAnalysisAgent using live services."""

import logging
from datetime import datetime, timedelta, timezone

import pytest

from core.agents.models import BaseAgentInput
from core.agents.news_analysis_agent import NewsAnalysisAgent


@pytest.mark.asyncio
async def test_news_agent_pipeline() -> None:
    logging.basicConfig(level=logging.INFO)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=3)

    agent = NewsAnalysisAgent()
    input_data = BaseAgentInput(
        query="Recent Apple stock price",
        vector_query="Apple stock price",
        ticker="AAPL",
        start_date=start_date,
        end_date=end_date,
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
