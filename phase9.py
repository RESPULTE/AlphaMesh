import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

# Mock dependencies before importing agents to avoid DB/API issues
mock_db = MagicMock()
mock_db.initialize = AsyncMock()
mock_db.update_financials = AsyncMock()
mock_db.search_label = AsyncMock(return_value=MagicMock(empty=True))
mock_db.get_labels = AsyncMock(return_value=[])

with patch(
    "core.agents.fundamental_analysis_agent.FinancialDatabase", return_value=mock_db
):
    from langchain_core.messages import HumanMessage

    from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
    from core.agents.models import BaseAgentInput
    from core.agents.news_analysis_agent import NewsAnalysisAgent
    from core.agents.orchestrator_agent import FinalResponse, OrchestratorAgent
    from core.memory.memory_system import FinancialMemorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_integration_tests():
    logger.info("=========================================")
    logger.info("  Phase 9 Integration Smoke Tests (FIXED)  ")
    logger.info("=========================================")

    # 1. Test Ingest Document Lean
    logger.info("\n--- TEST 9.1: ingest_document_lean ---")
    try:
        memory = FinancialMemorySystem()
        memory.initialized = True  # Force initialized

        with patch.object(memory, "_add_to_cognee", new_callable=AsyncMock) as mock_add:
            with patch(
                "core.memory.memory_system.run_custom_pipeline", new_callable=AsyncMock
            ) as mock_pipeline:
                mock_pipeline.return_value = {"status": "success"}

                result = await memory.ingest_document_lean(
                    ticker="AAPL",
                    report_type="test-report",
                    content="This is a lean test.",
                    include_summaries=False,
                )

                assert mock_add.called, "_add_to_cognee was not called"
                assert mock_pipeline.called, "run_custom_pipeline was not called"
                logger.info(
                    "PASS: ingest_document_lean executed pipeline successfully."
                )
    except Exception as e:
        logger.error(f"FAIL 9.1: {e}")

    # 2. Test ingest_conversation no-op
    logger.info("\n--- TEST 9.2: ingest_conversation no-op ---")
    try:
        memory = FinancialMemorySystem()
        memory.initialized = True
        await memory.ingest_conversation(
            "user@test.com", [{"role": "user", "content": "hi"}]
        )
        logger.info("PASS: ingest_conversation executed as safe no-op.")
    except Exception as e:
        logger.error(f"FAIL 9.2: {e}")

    # 3. Test Agent outputs
    logger.info("\n--- TEST 9.3: Agent Outputs (Fundamental & News) ---")
    try:
        input_data = BaseAgentInput(
            ticker="MSFT",
            vector_query="Microsoft",
            query="test",
            start_date="2025-01-01",
            end_date="2025-01-02",
        )

        # Fundamental Agent
        with patch(
            "core.agents.fundamental_analysis_agent.FinancialDatabase",
            return_value=mock_db,
        ):
            f_agent = FundamentalAnalysisAgent()
            # We want to test the 'run' method logic which now includes entities_enriched
            # So we mock the graph to return the expected state keys
            f_agent._graph.ainvoke = AsyncMock(
                return_value={
                    "financial_data": None,
                    "analysis": "Test fundamental analysis.",
                    "entities_enriched": [MagicMock(ticker="MSFT")],
                }
            )
            f_out = await f_agent.run(input_data)
            assert hasattr(
                f_out, "entities_enriched"
            ), "Fundamental output missing entities_enriched"
            assert (
                len(f_out.entities_enriched) == 1
            ), f"Expected 1 enriched company entity, got {len(f_out.entities_enriched)}"
            assert (
                f_out.entities_enriched[0].ticker == "MSFT"
            ), "Ticker mismatch in entity"
            logger.info("PASS: FundamentalAnalysisAgent output contract verified.")

        # News Agent
        with patch("core.services.service_manager.get_agent", MagicMock()):
            n_agent = NewsAnalysisAgent()
            n_agent._graph.ainvoke = AsyncMock(
                return_value={
                    "analysis": "Test news analysis.",
                    "news_context": [],
                    "entities_enriched": [MagicMock(ticker="MSFT")],
                }
            )
            n_out = await n_agent.run(input_data)
            assert hasattr(
                n_out, "entities_enriched"
            ), "News output missing entities_enriched"
            assert (
                len(n_out.entities_enriched) == 1
            ), "Expected 1 enriched company entity"
            logger.info("PASS: NewsAnalysisAgent output contract verified.")

    except Exception as e:
        logger.error(f"FAIL 9.3: {e}")

    # 4. Test Orchestrator flow
    logger.info("\n--- TEST 9.4: Orchestrator flow (Synthesiser & Writeback) ---")
    try:
        with patch("core.services.service_manager.get_agent", MagicMock()):
            o_agent = OrchestratorAgent()
            # Mock the graph to return a state that includes the response
            o_agent._graph.ainvoke = AsyncMock(
                return_value={
                    "final_response": FinalResponse(
                        summary="Orchestrator mocked summary", sources=[]
                    )
                }
            )

            res = await o_agent.run(
                [HumanMessage(content="test query")], conversation_id="test_conv_id"
            )
            assert res is not None, "Failed to get FinalResponse"
            logger.info(
                "PASS: Orchestrator flow returned final response without crashing."
            )
    except Exception as e:
        logger.error(f"FAIL 9.4: {e}")

    logger.info("\n=========================================")
    logger.info("  All tests completed.")
    logger.info("=========================================")


if __name__ == "__main__":
    asyncio.run(run_integration_tests())
