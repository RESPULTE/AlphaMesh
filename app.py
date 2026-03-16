"""
AlphaMesh — Streamlit Chat Front-End
======================================
Design Philosophy: Claude.ai-inspired comfort
  • Soft warm-ivory background, not pure white
  • Rounded cards and bubbles
  • DM Serif Display (headings) + DM Sans (body)
  • Accent: warm amber/gold for AI, soft slate for user
  • Subtle pulse animation while agent is running
  • Citations: inline [N] badges with hover tooltip showing title + URL
  • Tables: rendered via st.dataframe with custom styling
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Optional

import markdown as md_lib
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# ── project imports ────────────────────────────────────────────────────────────
from core.agents.orchestrator_agent import FinalResponse, OrchestratorAgent
from core.event_queue import EventLevel, StreamlitSink, get_queue

# ══════════════════════════════════════════════════════════════════════════════
# Page config  (must be the FIRST Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AlphaMesh · Financial Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# Global CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap" rel="stylesheet">

    <style>
    /* ── Reset & root vars ─────────────────────────────────────────── */
    :root {
        --bg:           #faf9f7;
        --surface:      #ffffff;
        --surface-2:    #f5f3ef;
        --border:       #e8e4de;
        --text-primary: #1a1814;
        --text-secondary: #6b6560;
        --text-muted:   #9e9890;
        --accent:       #c8954a;
        --accent-light: #f5e6cc;
        --accent-dark:  #9e6e2e;
        --user-bubble:  #eef2f8;
        --user-border:  #d0d8e8;
        --ai-bubble:    #fdf8f2;
        --ai-border:    #e8d9c0;
        --radius-lg:    18px;
        --radius-md:    12px;
        --radius-sm:    8px;
        --shadow-sm:    0 1px 4px rgba(0,0,0,0.06);
        --shadow-md:    0 4px 16px rgba(0,0,0,0.08);
        --font-body:    'DM Sans', system-ui, sans-serif;
        --font-display: 'DM Serif Display', Georgia, serif;
    }

    /* ── Global background ─────────────────────────────────────────── */
    .stApp {
        background-color: var(--bg) !important;
        font-family: var(--font-body) !important;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 0 !important;
        max-width: 860px !important;
        margin: 0 auto !important;
    }

    /* ── Top header bar ────────────────────────────────────────────── */
    .am-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 24px 0 16px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 24px;
    }
    .am-logo {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; color: white;
        box-shadow: var(--shadow-sm);
        flex-shrink: 0;
    }
    .am-title {
        font-family: var(--font-display);
        font-size: 22px;
        color: var(--text-primary);
        margin: 0;
        line-height: 1;
    }
    .am-subtitle {
        font-size: 12px;
        color: var(--text-muted);
        letter-spacing: 0.04em;
        font-weight: 400;
        margin-top: 2px;
    }
    .am-badge {
        margin-left: auto;
        background: var(--accent-light);
        color: var(--accent-dark);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(200,149,74,0.25);
    }

    /* ── Chat messages ─────────────────────────────────────────────── */
    .chat-row {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
        animation: fadeSlideIn 0.35s ease both;
    }
    .chat-row.user { flex-direction: row-reverse; }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .chat-avatar {
        width: 34px; height: 34px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px;
        font-weight: 600;
        margin-top: 2px;
    }
    .avatar-ai {
        background: linear-gradient(135deg, var(--accent-light), #f0d9b0);
        color: var(--accent-dark);
        border: 1.5px solid var(--accent-light);
    }
    .avatar-user {
        background: var(--user-bubble);
        color: #4a5568;
        border: 1.5px solid var(--user-border);
    }

    .chat-bubble {
        max-width: 78%;
        padding: 14px 18px;
        border-radius: var(--radius-lg);
        line-height: 1.65;
        font-size: 14.5px;
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
        word-break: break-word;
    }
    .bubble-ai {
        background: var(--ai-bubble);
        border: 1px solid var(--ai-border);
        border-top-left-radius: 4px;
    }
    .bubble-user {
        background: var(--user-bubble);
        border: 1px solid var(--user-border);
        border-top-right-radius: 4px;
    }

    /* ── Markdown typography inside bubbles ────────────────────────── */
    .chat-bubble p          { margin: 0 0 10px; }
    .chat-bubble p:last-child { margin-bottom: 0; }
    .chat-bubble h1,
    .chat-bubble h2,
    .chat-bubble h3         { font-family: var(--font-display);
                               color: var(--text-primary);
                               margin: 14px 0 6px;
                               line-height: 1.3; }
    .chat-bubble h1         { font-size: 18px; }
    .chat-bubble h2         { font-size: 16px; }
    .chat-bubble h3         { font-size: 14.5px; }
    .chat-bubble strong     { font-weight: 600; color: var(--text-primary); }
    .chat-bubble em         { font-style: italic; color: var(--text-secondary); }
    .chat-bubble ul,
    .chat-bubble ol         { margin: 6px 0 10px 20px; padding: 0; }
    .chat-bubble li         { margin-bottom: 4px; }
    .chat-bubble code       { font-family: 'Fira Code', monospace;
                               font-size: 12.5px;
                               background: var(--surface-2);
                               border: 1px solid var(--border);
                               border-radius: 4px;
                               padding: 1px 5px; }
    .chat-bubble pre        { background: var(--surface-2);
                               border: 1px solid var(--border);
                               border-radius: var(--radius-sm);
                               padding: 10px 14px;
                               overflow-x: auto;
                               margin: 8px 0; }
    .chat-bubble pre code   { background: none; border: none; padding: 0; }
    .chat-bubble blockquote { border-left: 3px solid var(--accent);
                               margin: 8px 0;
                               padding: 4px 12px;
                               color: var(--text-secondary); }
    .chat-bubble hr         { border: none;
                               border-top: 1px solid var(--border);
                               margin: 12px 0; }
    .chat-bubble a          { color: var(--accent-dark); text-decoration: underline; }

    /* ── Citation badges ───────────────────────────────────────────── */
    .cite {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--accent-light);
        color: var(--accent-dark);
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.02em;
        padding: 1px 6px;
        border-radius: 20px;
        border: 1px solid rgba(200,149,74,0.35);
        cursor: help;
        position: relative;
        text-decoration: none !important;
        vertical-align: middle;
        margin: 0 1px;
        transition: background 0.15s, transform 0.15s;
        white-space: nowrap;
    }
    .cite:hover {
        background: var(--accent);
        color: white;
        transform: translateY(-1px);
    }
    .cite .tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: var(--text-primary);
        color: #f5f3ef;
        border-radius: var(--radius-md);
        padding: 10px 14px;
        width: 280px;
        font-size: 12px;
        font-weight: 400;
        line-height: 1.5;
        box-shadow: var(--shadow-md);
        z-index: 9999;
        pointer-events: none;
        transition: opacity 0.2s ease, visibility 0.2s ease;
        text-align: left;
        letter-spacing: 0;
    }
    .cite .tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: var(--text-primary);
    }
    .cite .tooltip .t-title {
        font-weight: 600;
        color: #ffffff;
        display: block;
        margin-bottom: 4px;
    }
    .cite .tooltip .t-url {
        color: var(--accent-light);
        font-size: 11px;
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 252px;
    }
    .cite:hover .tooltip { visibility: visible; opacity: 1; }

    /* ── Thinking / running animation ─────────────────────────────── */
    .am-thinking {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 16px;
        background: var(--ai-bubble);
        border: 1px solid var(--ai-border);
        border-radius: var(--radius-lg);
        border-top-left-radius: 4px;
        font-size: 13px;
        color: var(--text-secondary);
        animation: fadeSlideIn 0.3s ease both;
    }
    .dots { display: flex; gap: 4px; align-items: center; }
    .dot {
        width: 6px; height: 6px;
        background: var(--accent);
        border-radius: 50%;
        animation: dotBounce 1.4s infinite ease-in-out both;
    }
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }
    .dot:nth-child(3) { animation-delay: 0s; }

    @keyframes dotBounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
        40%            { transform: scale(1.0); opacity: 1.0; }
    }

    /* ── Data table wrapper ────────────────────────────────────────── */
    .am-table-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.07em;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 6px;
        padding-left: 2px;
    }

    /* ── Sources list below message ────────────────────────────────── */
    .sources-block {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid var(--border);
    }
    .sources-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.07em;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .source-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 12.5px;
    }
    .source-num {
        min-width: 20px; height: 20px;
        background: var(--accent-light);
        color: var(--accent-dark);
        font-weight: 700; font-size: 10.5px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; margin-top: 1px;
    }
    .source-link {
        color: var(--accent-dark);
        text-decoration: none;
        line-height: 1.4;
    }
    .source-link:hover { text-decoration: underline; }

    /* ── Input area ────────────────────────────────────────────────── */
    .stChatInput > div {
        border-radius: var(--radius-lg) !important;
        border-color: var(--border) !important;
        background: var(--surface) !important;
        box-shadow: var(--shadow-sm) !important;
        font-family: var(--font-body) !important;
    }
    .stChatInput textarea {
        font-family: var(--font-body) !important;
        font-size: 14.5px !important;
        color: var(--text-primary) !important;
    }

    /* ── Empty state ───────────────────────────────────────────────── */
    .am-empty {
        text-align: center;
        padding: 60px 20px;
        color: var(--text-muted);
    }
    .am-empty-icon  { font-size: 40px; margin-bottom: 16px; opacity: 0.6; }
    .am-empty-title {
        font-family: var(--font-display);
        font-size: 22px;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .am-empty-sub {
        font-size: 14px; line-height: 1.6;
        max-width: 420px; margin: 0 auto;
    }
    .am-suggestions {
        display: flex; flex-wrap: wrap; gap: 8px;
        justify-content: center; margin-top: 24px;
    }
    .am-suggestion {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 7px 14px;
        font-size: 13px;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.15s;
    }
    .am-suggestion:hover {
        border-color: var(--accent);
        color: var(--accent-dark);
        background: var(--accent-light);
    }

    /* ── Scrollbar ─────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = OrchestratorAgent()
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "running" not in st.session_state:
        st.session_state.running = False
    if "pending_query" not in st.session_state:
        # Holds a suggestion chip text until the next render loop picks it up
        st.session_state.pending_query = None


_init_state()

# ══════════════════════════════════════════════════════════════════════════════
# DataFrame formatting helpers
# ══════════════════════════════════════════════════════════════════════════════

_RATIO_LABEL_RE = re.compile(
    r"(margin|ratio|cagr|yield|growth|rate|return|roe|roa|roic|roc|coverage"
    r"|turnover|efficiency|payout|pe_ratio|pb_ratio|ps_ratio|ev_|price_to"
    r"|gross_margin|net_margin|operating_margin|debt_to|leverage)",
    re.IGNORECASE,
)


def _is_ratio_row(label: str, series: pd.Series) -> bool:
    if _RATIO_LABEL_RE.search(str(label)):
        return True
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool((numeric.abs() < 20).all() and (numeric.abs() > 0).any())


def _fmt_financial(value: float) -> str:
    if pd.isna(value):
        return "—"
    abs_v = abs(value)
    if abs_v >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs_v >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_v >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.4g}"


def _fmt_ratio(value: float) -> str:
    if pd.isna(value):
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) <= 1.5:
        return f"{v * 100:.2f}%"
    return f"{v:.2f}%"


def _format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy().astype(object)
    for label in display.index:
        row = df.loc[label]
        if _is_ratio_row(label, row):
            display.loc[label] = [
                (
                    _fmt_ratio(x)
                    if isinstance(x, (int, float)) and not pd.isna(x)
                    else ("—" if pd.isna(x) else x)
                )
                for x in row
            ]
        else:
            display.loc[label] = [
                (
                    _fmt_financial(x)
                    if isinstance(x, (int, float)) and not pd.isna(x)
                    else ("—" if pd.isna(x) else x)
                )
                for x in row
            ]
    return display


# ══════════════════════════════════════════════════════════════════════════════
# Citation + Markdown helpers
# ══════════════════════════════════════════════════════════════════════════════

# Matches [N] citation markers that are NOT already inside an HTML tag.
# Uses a negative look-behind for '>' to avoid double-processing.
_CITE_RE = re.compile(r"(?<![>])\[(\d+)\]")


def _citation_badge(num: int, src) -> str:
    """Build the HTML for a single hoverable citation badge."""
    safe_title = src.title.replace('"', "&quot;").replace("'", "&#39;")
    display_url = src.url[:55] + "…" if len(src.url) > 58 else src.url
    safe_url = display_url.replace('"', "&quot;").replace("'", "&#39;")
    return (
        f'<span class="cite">[{num}]'
        f'<span class="tooltip">'
        f'<span class="t-title">{safe_title}</span>'
        f'<span class="t-url">{safe_url}</span>'
        f"</span></span>"
    )


def _render_ai_content(text: str, sources) -> str:
    """
    Convert an AI response to display HTML:
      1. Protect [N] citation markers from the markdown parser by replacing
         them with unique placeholders.
      2. Run the full text through the markdown library (bold, italic, headers,
         lists, code blocks, blockquotes all get converted to HTML).
      3. Restore citation markers as hoverable badge HTML.
    """
    src_map = {s.source_id: s for s in (sources or [])}

    # Step 1 — stash citation markers so markdown doesn't mangle them
    placeholders: dict[str, str] = {}

    def _stash(m: re.Match) -> str:
        num = int(m.group(1))
        token = f"\x00CITE{num}\x00"
        placeholders[token] = num
        return token

    protected = _CITE_RE.sub(_stash, text)

    # Step 2 — convert markdown → HTML
    html = md_lib.markdown(
        protected,
        extensions=["extra", "nl2br"],  # tables, fenced code, smart line-breaks
    )

    # Step 3 — replace placeholder tokens with badge HTML
    for token, num in placeholders.items():
        src = src_map.get(num)
        if src:
            badge = _citation_badge(num, src)
        else:
            # Source not in map: render a plain badge with no tooltip
            badge = f'<span class="cite">[{num}]<span class="tooltip"><span class="t-title">Source {num}</span></span></span>'
        html = html.replace(token, badge)

    return html


def _render_user_content(text: str) -> str:
    """Convert user message markdown to HTML (no citations)."""
    return md_lib.markdown(text, extensions=["extra", "nl2br"])


# ══════════════════════════════════════════════════════════════════════════════
# Message renderer
# ══════════════════════════════════════════════════════════════════════════════


def _render_message(msg: dict):
    role = msg["role"]
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    df: Optional[pd.DataFrame] = msg.get("df")

    is_user = role == "user"
    row_cls = "chat-row user" if is_user else "chat-row"
    bubble_cls = "chat-bubble bubble-user" if is_user else "chat-bubble bubble-ai"
    avatar_cls = "chat-avatar avatar-user" if is_user else "chat-avatar avatar-ai"
    avatar_icon = "U" if is_user else "✦"

    inner_html = (
        _render_user_content(content)
        if is_user
        else _render_ai_content(content, sources)
    )

    # Sources footer (AI only)
    sources_html = ""
    if sources and not is_user:
        items = "".join(
            f'<div class="source-item">'
            f'<span class="source-num">{s.source_id}</span>'
            f'<a class="source-link" href="{s.url}" target="_blank">{s.title}</a>'
            f"</div>"
            for s in sources
        )
        sources_html = (
            f'<div class="sources-block">'
            f'<div class="sources-label">Sources</div>'
            f"{items}</div>"
        )

    st.markdown(
        f'<div class="{row_cls}">'
        f'  <div class="{avatar_cls}">{avatar_icon}</div>'
        f'  <div class="{bubble_cls}">{inner_html}{sources_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # DataFrame below the bubble
    if df is not None and not df.empty:
        st.markdown(
            '<div class="am-table-label">Financial Data</div>', unsafe_allow_html=True
        )
        display_df = _format_dataframe(df)
        with st.container():
            st.dataframe(
                display_df.style.set_properties(
                    **{"font-size": "12.5px", "font-family": "'DM Sans', sans-serif"}
                ).set_table_styles(
                    [
                        {
                            "selector": "thead th",
                            "props": [
                                ("background-color", "#fdf3e3"),
                                ("color", "#6b4c1e"),
                                ("font-weight", "600"),
                                ("font-size", "11.5px"),
                                ("letter-spacing", "0.03em"),
                                ("border-bottom", "2px solid #e8d9c0"),
                                ("padding", "8px 12px"),
                            ],
                        },
                        {
                            "selector": "tbody tr:nth-child(odd)",
                            "props": [("background-color", "#faf9f7")],
                        },
                        {
                            "selector": "tbody tr:nth-child(even)",
                            "props": [("background-color", "#ffffff")],
                        },
                        {
                            "selector": "tbody td",
                            "props": [
                                ("padding", "7px 12px"),
                                ("color", "#1a1814"),
                                ("border-bottom", "1px solid #eeebe6"),
                            ],
                        },
                    ]
                ),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Agent runner
# ══════════════════════════════════════════════════════════════════════════════


def _run_agent_sync(query: str) -> FinalResponse:
    queue = get_queue()
    queue.start_response("orchestrator")
    lc_msgs = st.session_state.lc_history + [HumanMessage(content=query)]
    loop = asyncio.new_event_loop()
    try:
        result: FinalResponse = loop.run_until_complete(
            st.session_state.agent.run(
                messages=lc_msgs,
                conversation_id=st.session_state.conversation_id,
            )
        )
    finally:
        queue.end_response()
        loop.close()
    return result


_THINKING_HTML = """
<div class="chat-row">
  <div class="chat-avatar avatar-ai">✦</div>
  <div class="am-thinking">
    <div class="dots">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>
    {label}
  </div>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="am-header">
      <div class="am-logo">✦</div>
      <div>
        <div class="am-title">AlphaMesh</div>
        <div class="am-subtitle">Financial Intelligence · Multi-Agent Research</div>
      </div>
      <div class="am-badge">BETA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

_SUGGESTIONS = [
    "AAPL revenue growth over 5 years",
    "NVIDIA latest earnings sentiment",
    "DCF valuation for MSFT",
    "Tesla news this month",
]

if not st.session_state.messages:
    st.markdown(
        """
        <div class="am-empty">
          <div class="am-empty-icon">✦</div>
          <div class="am-empty-title">What would you like to explore?</div>
          <div class="am-empty-sub">
            Ask me about stocks, earnings, news sentiment, valuations, or
            portfolio insights. I'll analyse multiple sources and synthesise
            a grounded answer for you.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Render suggestion chips as real Streamlit buttons inside a flex row.
    # CSS below overrides the default button appearance to match .am-suggestion.
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 20px !important;
            padding: 7px 14px !important;
            font-size: 13px !important;
            font-family: var(--font-body) !important;
            color: var(--text-secondary) !important;
            transition: all 0.15s !important;
            box-shadow: none !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            border-color: var(--accent) !important;
            color: var(--accent-dark) !important;
            background: var(--accent-light) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(_SUGGESTIONS))
    for col, suggestion in zip(cols, _SUGGESTIONS):
        with col:
            if st.button(suggestion, key=f"sug_{suggestion}", use_container_width=True):
                st.session_state.pending_query = suggestion
                st.rerun()
else:
    for msg in st.session_state.messages:
        _render_message(msg)

# Single thinking + event placeholders — always after history, never duplicated
thinking_placeholder = st.empty()
event_placeholder = st.empty()

user_input = st.chat_input(
    placeholder="Ask about a ticker, financials, news, or your portfolio…",
    disabled=st.session_state.running,
)

# ══════════════════════════════════════════════════════════════════════════════
# Handle submission
# ══════════════════════════════════════════════════════════════════════════════

# Resolve the query from either the chat input or a suggestion chip click
_submitted = user_input or st.session_state.pending_query
if _submitted:
    st.session_state.pending_query = None  # consume it

if _submitted and not st.session_state.running:
    st.session_state.messages.append({"role": "user", "content": _submitted})
    st.session_state.running = True
    st.rerun()

if (
    st.session_state.running
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    query = st.session_state.messages[-1]["content"]

    thinking_placeholder.markdown(
        _THINKING_HTML.format(label="Analysing with multi-agent pipeline…"),
        unsafe_allow_html=True,
    )

    queue = get_queue()
    queue.add_sink(
        StreamlitSink(placeholder=event_placeholder, min_level=EventLevel.INFO)
    )

    result: FinalResponse = _run_agent_sync(query)

    thinking_placeholder.empty()
    event_placeholder.empty()

    st.session_state.lc_history.append(HumanMessage(content=query))
    st.session_state.lc_history.append(AIMessage(content=result.summary))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.summary,
            "sources": result.sources or [],
            "df": result.fundamental_data,
        }
    )

    st.session_state.running = False
    st.rerun()
