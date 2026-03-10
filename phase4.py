import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory.pipeline_tasks import (
    build_financial_pipeline,
    build_lean_document_pipeline,
    summarise_chunks_lean,
)

print("--- TASK 4.1: CHECK summarise_chunks_lean ---")
try:

    async def test41():
        class MockChunk:
            def __init__(self, text):
                self.id = "test-id"
                self.text = text
                self.text_summary = None

        chunks = [
            MockChunk(
                "Apple Inc reported revenue of $90B in Q3 2024, up 5% year over year."
                * 3
            ),
            MockChunk("short"),
            MockChunk(None),
        ]

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "AAPL Q3 2024 revenue $90B, up 5% YoY."
        )

        with patch("core.memory.pipeline_tasks.LLMGateway") as MockLLMGateway:
            MockLLMGateway.acreate = AsyncMock(return_value=mock_response)
            result = await summarise_chunks_lean(chunks)

        assert result is chunks, "Must return same list object"
        assert chunks[0].text_summary == "AAPL Q3 2024 revenue $90B, up 5% YoY."
        assert chunks[1].text_summary is None, "Short chunk should not have summary"
        assert chunks[2].text_summary is None, "None-text chunk should not have summary"
        print("PASS: summarise_chunks_lean works correctly")

    asyncio.run(test41())
except Exception as e:
    print(f"FAIL task 4.1: {e}")

print("--- TASK 4.2: CHECK build_lean_document_pipeline ---")
try:

    async def test42():
        tasks_with = await build_lean_document_pipeline(include_summaries=True)
        task_names_with = [t.executable.__name__ for t in tasks_with]
        assert "classify_documents" in task_names_with
        assert "extract_chunks_from_documents" in task_names_with
        assert "summarise_chunks_lean" in task_names_with
        assert "add_data_points_with_custom_edges" in task_names_with
        assert (
            "extract_financial_graph" not in task_names_with
        ), "Must NOT include graph extraction"
        assert (
            "assign_nodesets" not in task_names_with
        ), "Must NOT include assign_nodesets"
        assert (
            "merge_entities" not in task_names_with
        ), "Must NOT include merge_entities"

        idx_summarise = task_names_with.index("summarise_chunks_lean")
        idx_add = task_names_with.index("add_data_points_with_custom_edges")
        assert idx_summarise < idx_add, "summarise must come before add_data_points"

        tasks_without = await build_lean_document_pipeline(include_summaries=False)
        task_names_without = [t.executable.__name__ for t in tasks_without]
        assert "summarise_chunks_lean" not in task_names_without

        print(f"PASS: lean pipeline with summaries = {task_names_with}")
        print(f"PASS: lean pipeline without summaries = {task_names_without}")

    asyncio.run(test42())
except Exception as e:
    print(f"FAIL task 4.2: {e}")

print("--- TASK 4.3: CHECK build_financial_pipeline ---")
try:

    async def test43():
        tasks = await build_financial_pipeline()
        task_names = [t.executable.__name__ for t in tasks]
        required = [
            "classify_documents",
            "extract_chunks_from_documents",
            "extract_financial_graph",
            "assign_nodesets",
            "add_data_points_with_custom_edges",
        ]
        for name in required:
            assert name in task_names, f"build_financial_pipeline missing: {name}"
        print(f"PASS: build_financial_pipeline unchanged: {task_names}")

    asyncio.run(test43())
except Exception as e:
    print(f"FAIL task 4.3: {e}")
