"""
Chat Panel — the core interaction hub of Nexus AI.

Responsibilities:
  1. Render chat history with cited text + source chips
  2. Assemble context from toggled portfolio / chart / log items
  3. Drive agent_graph animation state transitions
  4. Stream agent responses with live log updates
  5. Display parallel log columns during execution
  6. Render fundamental data table when returned
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui.components.agent_graph import render_agent_graph
from ui.components.log_panel import (
    add_log,
    clear_logs,
    get_selected_log_context,
    render_log_panel,
)
from ui.components.source_modal import normalise_sources, render_cited_text
from ui.config import FONTS as F
from ui.config import GRAPH
from ui.config import THEME as T

# ── Message bubble helpers ────────────────────────────────────────


def _user_bubble(text: str, context_items: List[str]) -> None:
    ctx_html = ""
    if context_items:
        ctx_html = (
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px">'
            + "".join(
                [
                    f'<span style="font-family:\'{F["mono"]}\';font-size:0.62rem;'
                    f'color:{T["accent_gold"]};background:rgba(245,166,35,0.08);'
                    f"border:1px solid rgba(245,166,35,0.25);border-radius:4px;"
                    f'padding:1px 6px">✦ {c[:40]}</span>'
                    for c in context_items[:6]
                ]
            )
            + "</div>"
        )

    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin:10px 0">'
        f'<div style="max-width:78%;background:{T["bg_elevated"]};'
        f'border:1px solid {T["border_gold"]};border-radius:16px 16px 4px 16px;'
        f'padding:10px 14px">'
        f'<p style="font-family:\'{F["ui"]}\';font-size:0.88rem;'
        f'color:{T["text_primary"]};margin:0;line-height:1.6">{text}</p>'
        f"{ctx_html}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _assistant_header(agents_used: List[str]) -> None:
    badge_html = " ".join(
        [
            f'<span class="nx-agent-badge {a.replace("_agent","").replace("_","")}">'
            f'{a.replace("_agent","").replace("_"," ").title()}</span>'
            for a in agents_used
        ]
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        f'<span style="font-family:\'{F["display"]}\',serif;font-size:0.82rem;'
        f'font-weight:700;color:{T["agent_synthesiser"]}">◉ Nexus</span>'
        f"{badge_html}</div>",
        unsafe_allow_html=True,
    )


def _fundamental_table(df: pd.DataFrame) -> None:
    """Renders the fundamentals DataFrame with toggle for context inclusion."""
    if df is None or df.empty:
        return

    ctx_key = "fundamentals_in_context"
    is_in = st.session_state.get(ctx_key, False)
    border = T["accent_gold"] if is_in else T["border"]

    with st.expander("📊 Fundamental Data — click to expand", expanded=False):
        toggled = st.toggle(
            "✦ Include this data in agent context",
            value=is_in,
            key=f"fund_ctx_{id(df)}",
        )
        st.session_state[ctx_key] = toggled
        st.session_state["fundamentals_df"] = df if toggled else None

        fmt_df = df.copy()
        st.dataframe(
            fmt_df,
            use_container_width=True,
            hide_index=False,
        )


# ── Agent graph animation driver ─────────────────────────────────


def _show_graph(
    state: str,
    active_agents: List[str],
    progress: Dict[str, str],
    placeholder,
):
    html = render_agent_graph(
        state=state,
        active_agents=active_agents,
        agent_progress=progress,
        height=GRAPH["graph_height"],
    )
    with placeholder:
        components.html(html, height=GRAPH["graph_height"], scrolling=False)


# ── Main chat render ──────────────────────────────────────────────


def render_chat_panel(
    orchestrator,  # OrchestratorAgent instance
    portfolio_context: Optional[str] = None,
    chart_context: Optional[Dict] = None,
) -> None:
    """
    Renders the full chat interface including:
    • Graph animation area
    • Chat history
    • Live log columns during execution
    • Streaming response
    """

    # ── Agent graph placeholder ───────────────────────────────────
    graph_placeholder = st.empty()
    graph_state = st.session_state.get("graph_state", "idle")
    active_agents = st.session_state.get("active_agents", [])
    agent_progress = st.session_state.get("agent_progress", {})
    _show_graph(graph_state, active_agents, agent_progress, graph_placeholder)

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────
    history = st.session_state.get("chat_history", [])

    for msg in history:
        if msg["role"] == "user":
            _user_bubble(msg["content"], msg.get("context_items", []))
        else:
            _assistant_header(msg.get("agents_used", []))
            sources = normalise_sources(msg.get("sources", []))
            if sources:
                render_cited_text(msg["content"], sources)
            else:
                st.markdown(
                    f'<div style="font-family:\'{F["ui"]}\';font-size:0.88rem;'
                    f'color:{T["text_primary"]};line-height:1.75;'
                    f'padding:0 0 6px 0">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            if msg.get("fundamental_data") is not None:
                _fundamental_table(msg["fundamental_data"])

        st.markdown('<div style="margin-bottom:4px"></div>', unsafe_allow_html=True)

    # ── Log panel (shown during/after execution) ──────────────────
    if st.session_state.get("agent_logs"):
        st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="nx-section-header">'
            f'<div class="nx-section-dot" style="background:{T["accent_violet"]}"></div>'
            f'<span class="nx-section-title">Agent Execution Logs</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        render_log_panel()

    # ── Context items summary ─────────────────────────────────────
    ctx_sources = st.session_state.get("ctx_sources", set())
    ctx_logs = get_selected_log_context()
    fund_df = st.session_state.get("fundamentals_df")
    port_ctx = portfolio_context
    ctx_items = []

    if port_ctx:
        ctx_items.append("Portfolio holdings")
    if fund_df is not None:
        ctx_items.append("Fundamental data")
    if ctx_logs:
        ctx_items += [f"Log: {l[:30]}…" for l in ctx_logs[:3]]
    if ctx_sources:
        ctx_items += [f"Source [{s}]" for s in ctx_sources]
    if chart_context:
        ctx_items.append(
            f"Chart: {chart_context.get('ticker')} "
            f"{chart_context.get('start_date',''):%b %d %Y} → "
            f"{chart_context.get('end_date',''):%b %d %Y}"
            if isinstance(chart_context.get("start_date"), datetime)
            else "Chart selection"
        )

    if ctx_items:
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px;'
            f'padding:8px 10px;background:{T["bg_elevated"]};'
            f'border:1px solid {T["border_gold"]};border-radius:10px;">'
            f'<span style="font-family:\'{F["ui"]}\';font-size:0.65rem;'
            f'color:{T["accent_gold"]};font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;align-self:center;margin-right:4px">'
            f"Context ({len(ctx_items)}):</span>"
            + "".join(
                [
                    f'<span style="font-family:\'{F["ui"]}\';font-size:0.68rem;'
                    f'color:{T["text_secondary"]};background:{T["bg_surface"]};'
                    f'border:1px solid {T["border"]};border-radius:12px;'
                    f'padding:2px 8px">✦ {item}</span>'
                    for item in ctx_items
                ]
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    # ── Chat input ────────────────────────────────────────────────
    prefill = st.session_state.pop("chat_prefill", "")
    user_input = st.chat_input(
        "Ask about stocks, markets, your portfolio…",
        key="chat_input_box",
    )

    # Handle prefilled query from chart selection
    if prefill and not user_input:
        user_input = prefill

    if not user_input:
        return

    # ── Run agent ─────────────────────────────────────────────────
    clear_logs()
    st.session_state["agent_logs"] = {}

    # Assemble system context
    context_parts = []
    context_labels = []

    if port_ctx:
        context_parts.append(port_ctx)
        context_labels.append("Portfolio holdings")

    if fund_df is not None:
        context_parts.append(
            "FUNDAMENTAL DATA (from previous analysis):\n"
            + fund_df.to_string(max_rows=15)
        )
        context_labels.append("Fundamental data")

    if ctx_logs:
        context_parts.append("PREVIOUS AGENT LOG CONTEXT:\n" + "\n".join(ctx_logs))
        context_labels.append(f"{len(ctx_logs)} log entries")

    if chart_context and isinstance(chart_context.get("start_date"), datetime):
        cd = chart_context
        context_parts.append(
            f"CHART SELECTION: {cd['ticker']} from "
            f"{cd['start_date']:%Y-%m-%d} to {cd['end_date']:%Y-%m-%d}"
        )
        context_labels.append(f"Chart: {cd['ticker']}")

    # Append user message to history immediately
    history.append(
        {
            "role": "user",
            "content": user_input,
            "context_items": context_labels,
        }
    )
    st.session_state.chat_history = history

    # ── Phase 1: Planning animation ───────────────────────────────
    st.session_state.update(
        {
            "graph_state": "planning",
            "active_agents": [],
            "agent_progress": {"orchestrator": "planning…"},
        }
    )
    _show_graph("planning", [], {"orchestrator": "planning…"}, graph_placeholder)
    add_log("orchestrator", "Received query — planning agent dispatch", "info")
    add_log(
        "orchestrator",
        (
            f"Query: {user_input[:80]}…"
            if len(user_input) > 80
            else f"Query: {user_input}"
        ),
        "info",
    )

    # Rebuild messages for LangChain
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc_messages = []

    if context_parts:
        combined_ctx = "\n\n".join(context_parts)
        lc_messages.append(SystemMessage(content=combined_ctx))

    for msg in history[:-1]:  # history before current user message
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    lc_messages.append(HumanMessage(content=user_input))

    # ── Phase 2: Run orchestrator (async via asyncio) ─────────────
    response_placeholder = st.empty()
    response_placeholder.markdown(
        f'<div class="nx-skeleton" style="height:18px;width:60%;margin:8px 0"></div>'
        f'<div class="nx-skeleton" style="height:14px;width:80%;margin:4px 0"></div>'
        f'<div class="nx-skeleton" style="height:14px;width:50%;margin:4px 0"></div>',
        unsafe_allow_html=True,
    )

    try:
        add_log("orchestrator", "Calling LLM planner for agent selection", "info")

        # ── Run the async orchestrator ────────────────────────────
        # asyncio.new_event_loop() is used because Streamlit runs in a
        # synchronous thread; we create a fresh loop per request.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                orchestrator.run(
                    messages=lc_messages,
                    conversation_id=st.session_state.get("conversation_id"),
                    user_email=st.session_state.get("user_email"),
                )
            )
        finally:
            loop.close()

        # ── Normalise result ──────────────────────────────────────
        # LangGraph with output_schema=FinalResponse may return:
        #   (a) a FinalResponse instance directly, or
        #   (b) a dict  {"summary":…, "fundamental_data":…, "sources":…}, or
        #   (c) None when the graph short-circuits to END via final_answer
        #
        # We normalise all three cases into a plain FinalResponse.
        from core.agents.orchestrator_agent import FinalResponse as _FR

        if result is None:
            # Planner returned final_answer → no agents ran
            # The summary was already set in the plan; we don't have access
            # to it here so we gracefully fall back to a generic message.
            result = _FR(summary="", sources=[], fundamental_data=None)
            add_log("orchestrator", "Direct answer (no agents needed)", "info")

        elif isinstance(result, dict):
            # ainvoke returned the raw state dict — construct FinalResponse
            result = _FR(
                summary=result.get("summary") or "",
                fundamental_data=result.get("fundamental_data"),
                sources=result.get("sources") or [],
            )

        elif not isinstance(result, _FR):
            # Unexpected type — try best-effort attribute access
            result = _FR(
                summary=getattr(result, "summary", "") or "",
                fundamental_data=getattr(result, "fundamental_data", None),
                sources=getattr(result, "sources", []) or [],
            )

        # Guard: ensure all fields are safe to use
        summary = result.summary or ""
        sources = normalise_sources(result.sources or [])
        fundamental_df = result.fundamental_data

        # ── Infer which agents ran ────────────────────────────────
        # The orchestrator doesn't expose plan_agents on FinalResponse,
        # so we infer from the output content.
        plan_agents: List[str] = []
        if fundamental_df is not None and not (
            hasattr(fundamental_df, "empty") and fundamental_df.empty
        ):
            plan_agents.append("fundamentals_agent")
        if sources:
            plan_agents.append("news_agent")

        direct_answer = not plan_agents  # True when final_answer path was taken

        # ── Animation phases ──────────────────────────────────────
        if plan_agents:
            # Phase 3 — Spawning
            st.session_state.update(
                {
                    "graph_state": "spawning",
                    "active_agents": plan_agents,
                    "agent_progress": {a: "starting…" for a in plan_agents},
                }
            )
            _show_graph(
                "spawning",
                plan_agents,
                {a: "starting…" for a in plan_agents},
                graph_placeholder,
            )
            add_log(
                "orchestrator", f"Spawning agents: {', '.join(plan_agents)}", "node"
            )
            for ag in plan_agents:
                add_log(ag, "Agent initialised", "info")
                add_log(ag, "Processing query…", "info")

            # Phase 4 — Executing
            st.session_state.update(
                {
                    "graph_state": "executing",
                    "agent_progress": {a: "running…" for a in plan_agents},
                }
            )
            _show_graph(
                "executing",
                plan_agents,
                {a: "running…" for a in plan_agents},
                graph_placeholder,
            )
            for ag in plan_agents:
                add_log(ag, "Analysis complete", "success")

            # Phase 5 — Merging (only when >1 agent)
            if len(plan_agents) > 1:
                st.session_state.update(
                    {
                        "graph_state": "merging",
                        "agent_progress": {a: "merging…" for a in plan_agents},
                    }
                )
                _show_graph(
                    "merging",
                    plan_agents,
                    {a: "merging…" for a in plan_agents},
                    graph_placeholder,
                )
                add_log("synthesiser", "Receiving agent outputs", "info")
                add_log("synthesiser", "Synthesising final response…", "info")

        # Phase 6 — Complete
        st.session_state.update(
            {
                "graph_state": "complete",
                "active_agents": plan_agents,
                "agent_progress": {a: "done ✓" for a in plan_agents},
            }
        )
        _show_graph(
            "complete",
            plan_agents,
            {a: "done ✓" for a in plan_agents},
            graph_placeholder,
        )
        add_log("orchestrator", "Response ready ✓", "success")

        # ── Render response ───────────────────────────────────────
        response_placeholder.empty()

        # When the planner answered directly (greeting / trivial query)
        # the summary may be empty — show a fallback.
        if not summary:
            summary = (
                "I'm here and ready to help. Ask me about any stock, "
                "company, or your portfolio."
            )

        _assistant_header(plan_agents if plan_agents else ["orchestrator"])

        if sources:
            render_cited_text(summary, sources)
        else:
            st.markdown(
                f'<div style="font-family:\'{F["ui"]}\';font-size:0.88rem;'
                f'color:{T["text_primary"]};line-height:1.75">{summary}</div>',
                unsafe_allow_html=True,
            )

        if fundamental_df is not None and not (
            hasattr(fundamental_df, "empty") and fundamental_df.empty
        ):
            _fundamental_table(fundamental_df)

        # Persist to history
        history.append(
            {
                "role": "assistant",
                "content": summary,
                "agents_used": plan_agents,
                "sources": sources,  # already normalised to List[dict]
                "fundamental_data": fundamental_df,
            }
        )
        st.session_state.chat_history = history

    except Exception as exc:
        import traceback

        response_placeholder.empty()
        tb = traceback.format_exc()
        add_log("orchestrator", f"Error: {exc}", "error")
        st.session_state.update({"graph_state": "idle", "active_agents": []})
        _show_graph("idle", [], {}, graph_placeholder)

        st.markdown(
            f'<div style="padding:12px 16px;background:rgba(240,82,82,0.08);'
            f'border:1px solid {T["danger"]};border-radius:10px;'
            f'font-family:\'{F["mono"]}\';font-size:0.78rem;color:{T["danger"]}">'
            f"⚠ Agent error: <b>{str(exc)}</b></div>"
            f'<details style="margin-top:6px"><summary style="font-size:0.7rem;'
            f'color:{T["text_muted"]};cursor:pointer">Traceback</summary>'
            f'<pre style="font-size:0.65rem;color:{T["text_muted"]};'
            f'white-space:pre-wrap;padding:8px">{tb}</pre></details>',
            unsafe_allow_html=True,
        )

        history.append(
            {
                "role": "assistant",
                "content": f"⚠ An error occurred: {str(exc)}",
                "agents_used": [],
                "sources": [],
                "fundamental_data": None,
            }
        )
        st.session_state.chat_history = history

    st.rerun()
