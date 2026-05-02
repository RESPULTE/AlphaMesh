from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agents.models.base_agent_models import BaseAgentOutput
from core.agents.models.orchestrator_models import FinalResponse, OrchestratorPlan
from core.agents.ticker_validation import TickerInfo
from core.config import settings
from core.event_queue import publish_frontend_event, publish_progress
from core.logger import get_logger
from core.memory.retrieval.models import CitedSource
from core.services import service_manager

logger = get_logger(__name__)


def _build_clarification_message(needs_confirmation: Dict[str, TickerInfo]) -> str:
    """Format a user-facing message asking for ticker confirmation."""
    lines = ["Before proceeding, I want to confirm the securities you're asking about:"]
    for ticker, info in needs_confirmation.items():
        if not info.is_valid and info.suggestions:
            suggestions_str = ", ".join(f"**{s}**" for s in info.suggestions[:3])
            lines.append(
                f"- **{ticker}** wasn't recognised as a valid ticker symbol. "
                f"Did you mean one of: {suggestions_str}?"
            )
        elif not info.is_valid:
            lines.append(
                f"- **{ticker}** wasn't recognised as a valid ticker symbol. "
                "Please double-check the symbol and try again."
            )
        else:
            quote_type = info.quote_type or "unknown type"
            lines.append(
                f"- **{ticker}** appears to be a `{quote_type}` rather than a common equity. "
                "Is this correct, or did you mean a different symbol?"
            )
    lines.append("\nPlease reply with the correct ticker symbol(s) and I'll proceed.")
    return "\n".join(lines)


def _sanitize_portfolio_user_email(user_email: str) -> str:
    value = (user_email or "").strip().lower()
    safe = re.sub(r"[^a-z0-9._-]+", "_", value).strip("._-")
    if not safe:
        raise ValueError("Invalid user_email")
    return safe


def _get_user_portfolio_path(base_path: str, user_email: str) -> Path:
    base = Path(base_path)
    safe_user = _sanitize_portfolio_user_email(user_email)
    return base.parent / f"{base.stem}_{safe_user}.json"


def get_portfolio_for_user(base_path: str, user_email: Optional[str]) -> List[dict]:
    if not user_email:
        return []
    try:
        path = _get_user_portfolio_path(base_path, user_email)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        logger.warning("Portfolio file not found for user '%s' at %s", user_email, path)
        return []
    except ValueError:
        logger.warning(
            "Invalid portfolio user_email for path resolution: %s", user_email
        )
        return []
    except Exception:
        logger.exception("Failed to load portfolio for user '%s'", user_email)
        return []


def _collect_latest_agent_memory_summaries(
    turns: List[dict],
) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for turn in turns:
        summaries = turn.get("agent_memory_summaries") or {}
        if not isinstance(summaries, dict):
            continue
        for agent_name, payload in summaries.items():
            if isinstance(payload, dict):
                latest[str(agent_name)] = payload
    return latest


def _load_portfolio_block(user_email: Optional[str]) -> str:
    portfolio = get_portfolio_for_user(settings.PORTFOLIO_JSON_PATH, user_email)
    return json.dumps(portfolio, indent=2) if portfolio else "[]"


async def _open_graph_session(conversation_id: Optional[str]) -> None:
    if not conversation_id:
        return
    try:
        await service_manager.get_graph_queue_manager().open_session(conversation_id)
    except Exception:
        logger.exception(
            "run: failed to open graph queue session for '%s'", conversation_id
        )


async def _flush_graph_turn(conversation_id: Optional[str], turn_id: str) -> None:
    if not conversation_id:
        return
    try:
        await service_manager.get_graph_queue_manager().flush_turn(
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
    except Exception:
        logger.exception("run: failed to flush graph queue turn_id='%s'", turn_id)


async def _warm_user_context_cache(user_email: Optional[str]) -> None:
    if not user_email:
        return
    try:
        svc = service_manager.get_user_context_service()
        await svc.load_for_user(user_email)
    except Exception:
        logger.exception("run: failed to warm user context cache for %s", user_email)


def _coerce_final_response(raw: Any) -> FinalResponse:
    if raw is None:
        return FinalResponse(summary="")
    if isinstance(raw, FinalResponse):
        return raw
    if isinstance(raw, dict):
        return FinalResponse(
            summary=raw.get("summary") or "",
            fundamental_data=raw.get("fundamental_data"),
            fundamentals_visualization=raw.get("fundamentals_visualization"),
            fundamentals_raw_display_data=raw.get("fundamentals_raw_display_data"),
            fundamentals_row_semantics=_normalize_row_semantics(
                raw.get("fundamentals_row_semantics") or {}
            ),
            fundamentals_task_completed=raw.get("fundamentals_task_completed", True),
            fundamentals_task_completion_reason=raw.get(
                "fundamentals_task_completion_reason", ""
            ),
            sources=raw.get("sources") or [],
            agent_analyses=raw.get("agent_analyses") or {},
            agent_memory_summaries=raw.get("agent_memory_summaries") or {},
            tickers=raw.get("tickers") or [],
            turn_id=raw.get("turn_id") or "",
        )
    return FinalResponse(
        summary=getattr(raw, "summary", "") or "",
        fundamental_data=getattr(raw, "fundamental_data", None),
        fundamentals_visualization=getattr(raw, "fundamentals_visualization", None),
        fundamentals_raw_display_data=getattr(
            raw, "fundamentals_raw_display_data", None
        ),
        fundamentals_row_semantics=_normalize_row_semantics(
            getattr(raw, "fundamentals_row_semantics", {}) or {}
        ),
        fundamentals_task_completed=getattr(raw, "fundamentals_task_completed", True),
        fundamentals_task_completion_reason=getattr(
            raw, "fundamentals_task_completion_reason", ""
        ),
        sources=getattr(raw, "sources", []) or [],
        agent_analyses=getattr(raw, "agent_analyses", {}) or {},
        agent_memory_summaries=getattr(raw, "agent_memory_summaries", {}) or {},
        tickers=getattr(raw, "tickers", []) or [],
        turn_id=getattr(raw, "turn_id", "") or "",
    )


def _extract_response_text(raw: str) -> str:
    return raw.strip()


def _normalize_row_semantics(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for row_label, meta in raw.items():
        key = str(row_label or "").strip()
        if not key:
            continue
        meta_dict: Optional[Dict[str, Any]] = None
        if isinstance(meta, dict):
            meta_dict = dict(meta)
        elif hasattr(meta, "model_dump"):
            try:
                dumped = meta.model_dump()
                if isinstance(dumped, dict):
                    meta_dict = dumped
            except Exception:
                meta_dict = None
        if meta_dict:
            normalized[key] = meta_dict
    return normalized


def _collect_synthesis_inputs(agent_outputs: Dict[str, BaseAgentOutput]) -> Dict[str, Any]:
    context_parts: List[str] = []
    fundamental_df = None
    fundamentals_visualization = None
    fundamentals_raw_display_data = None
    fundamentals_row_semantics = {}
    fundamentals_task_completed = True
    fundamentals_task_completion_reason = ""
    news_sources: List[CitedSource] = []
    agent_memory_summaries: Dict[str, Dict[str, Any]] = {}
    per_agent_analyses: Dict[str, str] = {}

    for name, output in agent_outputs.items():
        try:
            context_parts.append(output.get_llm_context_str())
        except Exception:
            logger.exception("_synthesize_node: get_llm_context_str failed for '%s'", name)
        if name == "fundamentals_agent":
            fundamental_df = getattr(output, "financial_data", None)
            fundamentals_visualization = getattr(output, "visualization_plan", None)
            fundamentals_raw_display_data = getattr(output, "raw_display_data", None)
            fundamentals_row_semantics = _normalize_row_semantics(
                getattr(output, "row_semantics", {}) or {}
            )
            fundamentals_task_completed = bool(getattr(output, "task_completed", True))
            fundamentals_task_completion_reason = (
                getattr(output, "task_completion_reason", "") or ""
            )
        if name == "news_agent":
            news_sources = getattr(output, "sources", []) or []
        memory_summary = getattr(output, "memory_summary", {}) or {}
        if isinstance(memory_summary, dict) and memory_summary:
            agent_memory_summaries[name] = memory_summary
        per_agent_analyses[name] = getattr(output, "analysis", "") or ""

    return {
        "context_parts": context_parts,
        "fundamental_df": fundamental_df,
        "fundamentals_visualization": fundamentals_visualization,
        "fundamentals_raw_display_data": fundamentals_raw_display_data,
        "fundamentals_row_semantics": fundamentals_row_semantics,
        "fundamentals_task_completed": fundamentals_task_completed,
        "fundamentals_task_completion_reason": fundamentals_task_completion_reason,
        "news_sources": news_sources,
        "agent_memory_summaries": agent_memory_summaries,
        "per_agent_analyses": per_agent_analyses,
    }


async def validate_and_enrich_plan_tickers(
    plan: Optional[OrchestratorPlan],
) -> Dict[str, Any]:
    tickers: List[str] = getattr(plan, "tickers", []) if plan else []
    if not tickers:
        return {"company_context_blocks": {}, "ticker_metadata": {}}

    try:
        publish_progress("orchestrator", f"Validating ticker(s): {', '.join(tickers)}")
        validator = service_manager.get_ticker_validator()
        results: Dict[str, TickerInfo] = await validator.validate_and_enrich(tickers)
    except Exception:
        logger.exception("validate_and_enrich_plan_tickers: validation failed")
        return {"company_context_blocks": {}, "ticker_metadata": {}}

    needing_confirmation = {
        ticker: info
        for ticker, info in results.items()
        if info.needs_confirmation or not info.is_valid
    }
    if needing_confirmation and plan is not None:
        clarification = _build_clarification_message(needing_confirmation)
        updated_plan = plan.model_copy(update={"final_answer": clarification})
        return {
            "plan": updated_plan,
            "company_context_blocks": {},
            "ticker_metadata": {},
        }

    confirmed_tickers = [
        ticker for ticker, info in results.items() if info.is_valid and info.is_equity
    ]
    if confirmed_tickers:
        publish_frontend_event(
            "orchestrator",
            "ticker_resolved",
            {"ticker": confirmed_tickers[0], "tickers": confirmed_tickers},
        )

    company_context_blocks: Dict[str, str] = {}
    ticker_metadata: Dict[str, dict] = {}
    for ticker, info in results.items():
        if not info.is_valid or not info.is_equity:
            continue
        context_block = info.to_context_block()
        if context_block:
            company_context_blocks[ticker] = context_block
        ticker_metadata[ticker] = {
            "long_name": info.long_name,
            "sector": info.sector,
            "industry": info.industry,
            "description": info.description,
        }

    return {
        "company_context_blocks": company_context_blocks,
        "ticker_metadata": ticker_metadata,
    }
