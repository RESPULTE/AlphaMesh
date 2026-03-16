"""
╔══════════════════════════════════════════════════════════════════╗
║  NEXUS AI — Main Application                                    ║
║  Entry: streamlit run app.py                                    ║
║  Retheme: edit ui/config.py                                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys

# Insert the project root (parent of ui/) so that `from ui.X import` resolves.
# Works whether you run:  streamlit run ui/app.py   (from project root)
#                     or: streamlit run app.py       (from inside ui/)
_HERE = os.path.dirname(os.path.abspath(__file__))  # …/project/ui
_PROJECT_ROOT = os.path.dirname(_HERE)  # …/project
for _p in (_PROJECT_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import streamlit.components.v1 as components

from ui.auth import render_logout, require_auth
from ui.config import APP_ICON, APP_NAME, APP_TAGLINE
from ui.config import FONTS as F
from ui.config import GRAPH
from ui.config import THEME as T
from ui.services.market_data import MarketDataService
from ui.services.portfolio_service import PortfolioService
from ui.styles import get_global_css

# Must be FIRST st call
st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_global_css(), unsafe_allow_html=True)

# ── Auth ─────────────────────────────────────────────────────────
authenticated, user_email = require_auth()
if not authenticated:
    st.stop()


# ── Services ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _market_data():
    return MarketDataService()


@st.cache_resource(show_spinner=False)
def _portfolio_svc(_md):
    return PortfolioService(_md)


@st.cache_resource(show_spinner=False)
def _orchestrator():
    try:
        from core.agents.orchestrator_agent import OrchestratorAgent

        return OrchestratorAgent()
    except Exception:
        return None


market_data = _market_data()
portfolio_svc = _portfolio_svc(market_data)

# ── Session defaults ──────────────────────────────────────────────
for k, v in {
    "chat_history": [],
    "graph_state": "idle",
    "active_agents": [],
    "agent_progress": {},
    "agent_logs": {},
    "active_tab": "chat",
    "portfolio_in_context": False,
    "portfolio_ctx_tickers": set(),
    "ctx_sources": set(),
    "conversation_id": f"conv_{user_email}",
    "user_email": user_email,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

NAV = [
    ("chat", "💬", "AI Chat"),
    ("portfolio", "📊", "Portfolio"),
    ("chart", "📈", "Market Chart"),
    ("settings", "⚙", "Settings"),
]

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'<div style="padding:0.4rem 0 1rem;border-bottom:1px solid {T["border"]};'
        f'margin-bottom:1rem">'
        f'<div style="font-family:\'{F["display"]}\',serif;font-size:1.4rem;'
        f'font-weight:700;color:{T["text_primary"]};letter-spacing:-0.01em">'
        f"{APP_ICON} {APP_NAME}</div>"
        f'<div style="font-family:\'{F["ui"]}\';font-size:0.62rem;color:{T["text_muted"]};'
        f'letter-spacing:0.09em;text-transform:uppercase;margin-top:2px">{APP_TAGLINE}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    render_logout()

    st.markdown(
        f'<div class="nx-section-title" style="margin-bottom:8px">Navigation</div>',
        unsafe_allow_html=True,
    )
    for tid, icon, label in NAV:
        active = st.session_state.active_tab == tid
        if st.button(f"{icon}  {label}", key=f"nav_{tid}", use_container_width=True):
            st.session_state.active_tab = tid
            st.rerun()

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

    # System status
    st.markdown(
        f'<div class="nx-section-title" style="margin-bottom:6px">System</div>',
        unsafe_allow_html=True,
    )
    for label, ok in [
        ("Redis cache", market_data.redis_ok),
        ("Orchestrator", _orchestrator() is not None),
    ]:
        dot = T["success"] if ok else T["warning"]
        txt = f'{label}: {"online" if ok else "offline"}'
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;'
            f'font-family:\'{F["mono"]}\';font-size:0.67rem;color:{T["text_muted"]}">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{dot};'
            f'display:inline-block;flex-shrink:0"></span>{txt}</div>',
            unsafe_allow_html=True,
        )

    # Context count pill
    ctx_n = (
        len(st.session_state.get("ctx_sources", set()))
        + len(st.session_state.get("portfolio_ctx_tickers", set()))
        + (1 if st.session_state.get("portfolio_in_context") else 0)
        + (1 if st.session_state.get("fundamentals_in_context") else 0)
    )
    if ctx_n:
        st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="padding:8px 10px;background:rgba(245,166,35,0.08);'
            f'border:1px solid rgba(245,166,35,0.28);border-radius:10px">'
            f'<span style="font-family:\'{F["ui"]}\';font-size:0.68rem;'
            f'color:{T["accent_gold"]};font-weight:700">✦ {ctx_n} context items</span><br>'
            f'<span style="font-family:\'{F["ui"]}\';font-size:0.62rem;'
            f'color:{T["text_muted"]}">Sent with your next query</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear context", key="clr_ctx_sb"):
            for k in ["ctx_sources", "portfolio_ctx_tickers", "fundamentals_df"]:
                st.session_state[k] = set() if k != "fundamentals_df" else None
            st.session_state["portfolio_in_context"] = False
            st.session_state["fundamentals_in_context"] = False
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════
tab = st.session_state.active_tab

# Mobile tab bar
st.markdown(
    f'<div class="nx-mobile" style="display:flex;gap:5px;margin-bottom:10px;'
    f'overflow-x:auto;padding-bottom:2px;-webkit-overflow-scrolling:touch">'
    + "".join(
        [
            f'<a onclick="void(0)" style="flex-shrink:0;padding:5px 13px;border-radius:20px;'
            f'font-size:0.7rem;font-family:\'{F["ui"]}\';font-weight:600;text-decoration:none;'
            f'border:1px solid {""+T["accent_gold"] if tab==tid else T["border"]};'
            f'background:{"rgba(245,166,35,0.1)" if tab==tid else T["bg_elevated"]};'
            f'color:{T["accent_gold"] if tab==tid else T["text_secondary"]}">'
            f"{icon} {label}</a>"
            for tid, icon, label in NAV
        ]
    )
    + "</div>",
    unsafe_allow_html=True,
)

# ── CHAT ─────────────────────────────────────────────────────────
if tab == "chat":
    from ui.components.chat_panel import render_chat_panel

    port_ctx = None
    if st.session_state.get("portfolio_in_context"):
        df = st.session_state.get("portfolio_df")
        if df is not None and not df.empty:
            try:
                port_ctx = portfolio_svc.to_context_string(portfolio_svc.valuate(df))
            except Exception:
                pass

    orch = _orchestrator()
    if not orch:
        # Show idle graph + warning
        from ui.components.agent_graph import render_agent_graph

        components.html(
            render_agent_graph("idle", height=GRAPH["graph_height"]),
            height=GRAPH["graph_height"],
            scrolling=False,
        )
        st.markdown(
            f'<div style="margin-top:1rem;padding:14px 18px;border:1px solid {T["warning"]};'
            f"border-radius:12px;background:rgba(245,166,35,0.06);"
            f'font-family:\'{F["ui"]}\';font-size:0.84rem;color:{T["warning"]}">'
            f"⚠ Backend orchestrator is not available. "
            f"Ensure your environment variables, Neo4j, and LLM API keys are configured, "
            f"then reload the page.</div>",
            unsafe_allow_html=True,
        )
    else:
        render_chat_panel(
            orchestrator=orch,
            portfolio_context=port_ctx,
            chart_context=st.session_state.get("pending_chart_query"),
        )

# ── PORTFOLIO ─────────────────────────────────────────────────────
elif tab == "portfolio":
    from ui.components.portfolio_panel import render_portfolio_panel

    render_portfolio_panel(portfolio_svc, market_data, user_email)

# ── CHART ─────────────────────────────────────────────────────────
elif tab == "chart":
    from ui.components.stock_chart import render_stock_chart

    st.markdown(
        f'<div class="nx-section-header">'
        f'<div class="nx-section-dot" style="background:{T["accent_blue"]}"></div>'
        f'<span class="nx-section-title">Market Chart</span>'
        f"</div>"
        f'<p style="font-family:\'{F["ui"]}\';font-size:0.76rem;color:{T["text_muted"]};'
        f'margin-bottom:10px">Drag across the chart to select a period → '
        f'<b style="color:{T["accent_gold"]}">Ask Agent About This Period</b></p>',
        unsafe_allow_html=True,
    )
    render_stock_chart(market_data=market_data, key_prefix="main_chart")

# ── SETTINGS ──────────────────────────────────────────────────────
elif tab == "settings":
    st.markdown(
        f'<div class="nx-section-header">'
        f'<div class="nx-section-dot" style="background:{T["text_muted"]}"></div>'
        f'<span class="nx-section-title">Settings</span></div>',
        unsafe_allow_html=True,
    )
    ca, cb = st.columns(2)
    with ca:
        st.markdown(
            f'<div class="nx-card"><div style="font-family:\'{F["ui"]}\';font-size:0.82rem;'
            f'font-weight:700;color:{T["text_primary"]};margin-bottom:8px">🗄 Cache</div>',
            unsafe_allow_html=True,
        )
        stats = market_data.cache_stats()
        if stats.get("available"):
            st.markdown(
                f'<span style="font-family:\'{F["mono"]}\';font-size:0.73rem;'
                f'color:{T["text_secondary"]}">Keys: {stats["keys"]} &nbsp;|&nbsp; '
                f'Mem: {stats["used_memory_human"]}</span>',
                unsafe_allow_html=True,
            )
            if st.button("Flush cache", key="flush_c"):
                market_data.flush_cache()
                st.success("Flushed ✓")
        else:
            st.markdown(
                f'<span style="font-family:\'{F["mono"]}\';font-size:0.73rem;'
                f'color:{T["warning"]}">Redis offline</span>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="nx-card" style="margin-top:0.75rem">'
            f'<div style="font-family:\'{F["ui"]}\';font-size:0.82rem;'
            f'font-weight:700;color:{T["text_primary"]};margin-bottom:8px">💬 Conversation</div>',
            unsafe_allow_html=True,
        )
        n = len(st.session_state.get("chat_history", []))
        st.markdown(
            f'<span style="font-family:\'{F["mono"]}\';font-size:0.73rem;'
            f'color:{T["text_secondary"]}">{n} messages</span>',
            unsafe_allow_html=True,
        )
        if st.button("Clear history", key="clr_hist"):
            st.session_state.chat_history = []
            st.session_state.agent_logs = {}
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with cb:
        st.markdown(
            f'<div class="nx-card">'
            f'<div style="font-family:\'{F["ui"]}\';font-size:0.82rem;'
            f'font-weight:700;color:{T["text_primary"]};margin-bottom:10px">🎨 Theme</div>'
            + "".join(
                [
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">'
                    f'<span style="width:13px;height:13px;border-radius:3px;background:{c};'
                    f'border:1px solid rgba(255,255,255,0.1)"></span>'
                    f'<span style="font-family:\'{F["mono"]}\';font-size:0.65rem;'
                    f'color:{T["text_muted"]}">{n}</span></div>'
                    for n, c in [
                        ("accent_gold", T["accent_gold"]),
                        ("accent_blue", T["accent_blue"]),
                        ("accent_teal", T["accent_teal"]),
                        ("accent_violet", T["accent_violet"]),
                        ("bg_surface", T["bg_surface"]),
                        ("bg_elevated", T["bg_elevated"]),
                    ]
                ]
            )
            + f'<div style="margin-top:8px;font-family:\'{F["ui"]}\';font-size:0.7rem;'
            f'color:{T["text_muted"]}">Edit <code style="color:{T["accent_teal"]}">'
            f"ui/config.py</code> to retheme.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="nx-card" style="margin-top:0.75rem">'
            f'<div style="font-family:\'{F["ui"]}\';font-size:0.82rem;'
            f'font-weight:700;color:{T["text_primary"]};margin-bottom:8px">🔤 Fonts</div>'
            f'<div style="font-family:\'{F["display"]}\',serif;font-size:1rem;'
            f'color:{T["text_primary"]}">{F["display"]}</div>'
            f'<div style="font-family:\'{F["ui"]}\';font-size:0.82rem;'
            f'color:{T["text_secondary"]};margin-top:2px">{F["ui"]}</div>'
            f'<div style="font-family:\'{F["mono"]}\';font-size:0.78rem;'
            f'color:{T["text_muted"]};margin-top:2px">$AAPL 184.23 +1.42%</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
