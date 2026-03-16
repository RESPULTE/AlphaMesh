"""
Agent Log Panel Component.

Renders parallel side-by-side log columns — one per active agent —
with colour-coded log levels, agent badges, and per-log context toggles.
Logs stream in via st.session_state["agent_logs"].
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import streamlit as st

from ui.config import THEME as T, FONTS as F, ANIMATION as A


# ── Log level → colour + prefix ──────────────────────────────────
LOG_STYLES = {
    "info":    {"color": T["accent_blue"],     "icon": "·",  "css": "info"},
    "success": {"color": T["success"],         "icon": "✓",  "css": "success"},
    "warn":    {"color": T["warning"],         "icon": "⚠",  "css": "warn"},
    "error":   {"color": T["danger"],          "icon": "✗",  "css": "error"},
    "node":    {"color": T["accent_violet"],   "icon": "⬡",  "css": "info"},
    "data":    {"color": T["accent_teal"],     "icon": "◈",  "css": "success"},
}

AGENT_COLORS = {
    "orchestrator":       T["agent_orchestrator"],
    "fundamentals_agent": T["agent_fundamentals"],
    "news_agent":         T["agent_news"],
    "synthesiser":        T["agent_synthesiser"],
}

AGENT_LABELS = {
    "orchestrator":       "Orchestrator",
    "fundamentals_agent": "Fundamentals",
    "news_agent":         "News Intel",
    "synthesiser":        "Synthesiser",
}


def _agent_header(agent_id: str) -> str:
    color = AGENT_COLORS.get(agent_id, T["text_secondary"])
    label = AGENT_LABELS.get(agent_id, agent_id)
    icon_map = {
        "orchestrator": "⬡", "fundamentals_agent": "◆",
        "news_agent": "◈", "synthesiser": "◉",
    }
    icon = icon_map.get(agent_id, "◈")
    return (
        f'<div style="display:flex;align-items:center;gap:7px;'
        f'padding:6px 10px 6px 10px;'
        f'border-bottom:1px solid {T["border"]};'
        f'background:{T["bg_elevated"]};border-radius:10px 10px 0 0;">'
        f'<span style="font-size:13px;color:{color}">{icon}</span>'
        f'<span style="font-family:\'{F["ui"]}\';font-size:0.7rem;font-weight:700;'
        f'letter-spacing:0.07em;text-transform:uppercase;color:{color}">{label}</span>'
        f'<span id="status-{agent_id}" style="margin-left:auto;font-family:\'{F["mono"]}\';'
        f'font-size:0.62rem;color:{T["text_muted"]}">idle</span>'
        f'</div>'
    )


def _log_entry_html(log: dict, idx: int, agent_id: str) -> str:
    """Render a single log entry as an HTML div."""
    level   = log.get("level", "info")
    message = log.get("message", "")
    ts      = log.get("ts", "")
    style   = LOG_STYLES.get(level, LOG_STYLES["info"])
    color   = style["color"]
    icon    = style["icon"]
    css     = style["css"]

    # Context toggle state
    ctx_key = f"log_ctx_{agent_id}_{idx}"
    in_ctx  = st.session_state.get(ctx_key, False)
    badge   = (
        f'<span style="float:right;font-size:0.58rem;font-weight:700;'
        f'color:{T["accent_gold"]};letter-spacing:0.04em;'
        f'font-family:\'{F["ui"]}\'">✦</span>'
        if in_ctx else ""
    )
    return (
        f'<div class="nx-log {css}" title="{ts}">'
        f'<span style="color:{color};font-weight:600;margin-right:4px">{icon}</span>'
        f'{message}{badge}'
        f'</div>'
    )


def add_log(agent_id: str, message: str, level: str = "info"):
    """
    Convenience: push a log entry into session state.
    Call this from agent callbacks / async wrappers.
    """
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = {}
    if agent_id not in st.session_state.agent_logs:
        st.session_state.agent_logs[agent_id] = []
    st.session_state.agent_logs[agent_id].append({
        "level":   level,
        "message": message,
        "ts":      time.strftime("%H:%M:%S"),
    })


def clear_logs():
    """Clear all agent logs (call at start of each agent run)."""
    st.session_state.agent_logs = {}
    # Also clear per-log context selections
    keys_to_remove = [k for k in st.session_state if k.startswith("log_ctx_")]
    for k in keys_to_remove:
        del st.session_state[k]


def get_selected_log_context() -> List[str]:
    """
    Returns all log messages that the user has toggled into context.
    Called by the chat panel before building the agent's system prompt.
    """
    result = []
    logs = st.session_state.get("agent_logs", {})
    for agent_id, entries in logs.items():
        for idx, log in enumerate(entries):
            if st.session_state.get(f"log_ctx_{agent_id}_{idx}", False):
                result.append(f"[{agent_id}] {log['message']}")
    return result


def render_log_panel(
    agents: Optional[List[str]] = None,
    max_height_px: int = 320,
):
    """
    Renders parallel log columns for each agent.
    agents: list of agent IDs to show; if None, inferred from session state.
    """
    logs: Dict[str, List[dict]] = st.session_state.get("agent_logs", {})

    if agents is None:
        agents = list(logs.keys())
        # Always show orchestrator first if present
        if "orchestrator" in agents:
            agents = ["orchestrator"] + [a for a in agents if a != "orchestrator"]

    if not agents:
        st.markdown(
            f'<div style="text-align:center;padding:1.5rem;'
            f'color:{T["text_muted"]};font-size:0.78rem;font-family:\'{F["ui"]}\'">'
            f'Agent logs will appear here when a query is running.</div>',
            unsafe_allow_html=True,
        )
        return

    n_cols = min(len(agents), 3)  # max 3 side-by-side on desktop
    cols   = st.columns(n_cols)

    for col, agent_id in zip(cols, agents[:n_cols]):
        agent_logs = logs.get(agent_id, [])
        color      = AGENT_COLORS.get(agent_id, T["text_secondary"])

        with col:
            # Agent column header
            st.markdown(_agent_header(agent_id), unsafe_allow_html=True)

            # Scrollable log body
            log_html = (
                f'<div style="height:{max_height_px}px;overflow-y:auto;'
                f'background:{T["bg_surface"]};border:1px solid {T["border"]};'
                f'border-top:none;border-radius:0 0 10px 10px;padding:6px 4px;">'
            )

            if not agent_logs:
                log_html += (
                    f'<div style="padding:12px 10px;font-family:\'{F["mono"]}\';'
                    f'font-size:0.68rem;color:{T["text_muted"]}">Waiting…</div>'
                )
            else:
                for idx, log in enumerate(agent_logs):
                    log_html += _log_entry_html(log, idx, agent_id)

            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

            # Per-log context toggles (below the scroll area)
            if agent_logs:
                with st.expander(
                    f"Select logs for context ({len(agent_logs)} entries)",
                    expanded=False,
                ):
                    for idx, log in enumerate(agent_logs):
                        ctx_key = f"log_ctx_{agent_id}_{idx}"
                        level   = log.get("level", "info")
                        icon    = LOG_STYLES.get(level, LOG_STYLES["info"])["icon"]
                        msg     = log.get("message", "")
                        short   = (msg[:55] + "…") if len(msg) > 55 else msg
                        st.checkbox(
                            f"{icon} {short}",
                            key=ctx_key,
                            value=st.session_state.get(ctx_key, False),
                        )

    # If more agents than columns, render remaining stacked below
    if len(agents) > n_cols:
        for agent_id in agents[n_cols:]:
            st.markdown(f'<div style="margin-top:0.5rem"></div>', unsafe_allow_html=True)
            render_log_panel(agents=[agent_id], max_height_px=200)
