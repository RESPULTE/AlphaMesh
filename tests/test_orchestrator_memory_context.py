from __future__ import annotations

import asyncio

from core.agents.models.news_agent_models import NewsAgentOutput
from core.agents.models.orchestrator_models import OrchestratorPlan, OrchestratorState
from core.agents.orchestrator_agent import OrchestratorAgent
from core.services import service_manager


def test_runtime_agent_memory_context_uses_agent_adapter_window() -> None:
    class _HistoryAdapterAgent:
        @staticmethod
        def build_memory_context_from_history(turns, window=8):
            rows = []
            for turn in turns:
                payload = (turn.get("agent_memory_summaries") or {}).get("news_agent")
                if not isinstance(payload, dict):
                    continue
                ts = turn.get("created_at") or "unknown_time"
                rows.append(
                    f"- [{ts}] sources={payload.get('source_count')}; "
                    f"catalyst={payload.get('main_catalyst')}"
                )
            return "\n".join(rows[-window:])

    turns = []
    for i in range(10):
        turns.append(
            {
                "created_at": f"2026-04-26T00:{i:02d}:00+00:00",
                "agent_memory_summaries": {
                    "news_agent": {
                        "source_count": i + 1,
                        "main_catalyst": f"catalyst-{i + 1}",
                    }
                },
            }
        )

    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator._agents = {"news_agent": _HistoryAdapterAgent()}
    contexts = orchestrator._build_runtime_agent_memory_contexts(turns, window=8)
    news_context = contexts["news_agent"]

    assert "sources=1;" not in news_context
    assert "sources=2;" not in news_context
    assert "sources=10;" in news_context
    assert "catalyst=catalyst-10" in news_context


def test_execute_node_propagates_turn_id_and_agent_memory_context() -> None:
    class _CaptureAgent:
        def __init__(self) -> None:
            self.last_input = None

        @staticmethod
        def build_memory_context_from_history(turns, window=8):
            rows = []
            for turn in turns:
                payload = (turn.get("agent_memory_summaries") or {}).get("news_agent")
                if not isinstance(payload, dict):
                    continue
                ts = turn.get("created_at") or "unknown_time"
                rows.append(
                    f"- [{ts}] sources={payload.get('source_count')}; "
                    f"catalyst={payload.get('main_catalyst')}"
                )
            return "\n".join(rows[-window:])

        async def run(self, input_data):
            self.last_input = input_data
            return NewsAgentOutput(analysis="ok")

    capture_agent = _CaptureAgent()

    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator._agents = {"news_agent": capture_agent}

    state = OrchestratorState(
        plan=OrchestratorPlan(
            query="original query",
            target_agents=["news_agent"],
            per_agent_goals={"news_agent": "Assess AAPL near-term catalyst strength and risks."},
            tickers=["AAPL"],
        ),
        conversation_id="conv-1",
        turn_id="turn-123",
        history_turns=[
            {
                "created_at": "2026-04-26T01:00:00+00:00",
                "agent_memory_summaries": {
                    "news_agent": {
                        "source_count": 4,
                        "main_catalyst": "guidance raise",
                    }
                },
            }
        ],
        company_context_blocks={"AAPL": "Company Context Block"},
    )

    payload = asyncio.run(orchestrator._execute_node(state))

    assert "news_agent" in payload["agent_outputs"]
    assert capture_agent.last_input is not None
    assert capture_agent.last_input.turn_id == "turn-123"
    assert capture_agent.last_input.agent_memory_context.startswith("- [2026-04-26T01:00:00+00:00]")
    assert capture_agent.last_input.query == ""
    assert capture_agent.last_input.goal == "Assess AAPL near-term catalyst strength and risks."
    assert capture_agent.last_input.company_context == "Company Context Block"


def test_service_manager_orchestrator_singleton(monkeypatch) -> None:
    import core.agents.orchestrator_agent as orchestrator_module

    class _FakeOrchestrator:
        created = 0

        def __init__(self) -> None:
            _FakeOrchestrator.created += 1

    service_manager._orchestrator_agent = None
    monkeypatch.setattr(orchestrator_module, "OrchestratorAgent", _FakeOrchestrator)

    first = service_manager.get_orchestrator_agent()
    second = service_manager.get_orchestrator_agent()

    assert first is second
    assert _FakeOrchestrator.created == 1

    service_manager._orchestrator_agent = None
