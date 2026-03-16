"""
Stock Chart Component — interactive Plotly chart with drag-to-select.

Features:
  • Live ticker search (yfinance)
  • Timeframe buttons 1W → 5Y
  • Candlestick / Line toggle
  • Drag to select date range → offers "Ask Agent" CTA
  • Dynamic interval selection based on selected range
  • All data cached via MarketDataService / Redis
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.config import THEME as T, FONTS as F, MARKET, ANIMATION as A
from ui.services.market_data import MarketDataService

_UP   = T["success"]
_DOWN = T["danger"]


def _build_fig(
    df: pd.DataFrame,
    ticker: str,
    chart_type: str,
    sel_start: Optional[datetime],
    sel_end:   Optional[datetime],
) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(
            paper_bgcolor=T["bg_surface"], plot_bgcolor=T["bg_surface"],
            margin=dict(l=0,r=0,t=12,b=0),
            annotations=[dict(
                text=f"No data for <b>{ticker}</b>",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=T["text_muted"], size=14),
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    x = df.index
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=x, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing=dict(line=dict(color=_UP, width=1), fillcolor=_UP),
            decreasing=dict(line=dict(color=_DOWN, width=1), fillcolor=_DOWN),
            name=ticker, hoverinfo="x+y",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=df["Close"],
            mode="lines",
            line=dict(color=T["accent_gold"], width=2),
            fill="tozeroy",
            fillcolor="rgba(245,166,35,0.07)",
            name=ticker,
            hovertemplate="%{x|%b %d, %Y}<br><b>$%{y:,.2f}</b><extra></extra>",
        ))

    # Volume bars
    if "Volume" in df.columns:
        colors = [
            _UP if (df["Close"].iloc[i] >= df["Open"].iloc[i] if "Open" in df.columns else True)
            else _DOWN
            for i in range(len(df))
        ]
        fig.add_trace(go.Bar(
            x=x, y=df["Volume"], marker_color=colors, marker_opacity=0.28,
            name="Volume", yaxis="y2", hoverinfo="skip",
        ))

    # Highlight selected range
    if sel_start and sel_end:
        fig.add_vrect(
            x0=sel_start, x1=sel_end,
            fillcolor=f"rgba(77,142,245,0.1)", layer="below", line_width=0,
        )
        for d in (sel_start, sel_end):
            fig.add_vline(x=d, line_dash="dot", line_color=T["accent_blue"],
                          line_width=1, opacity=0.55)

    fig.update_layout(
        paper_bgcolor=T["bg_surface"],
        plot_bgcolor=T["bg_surface"],
        margin=dict(l=0, r=4, t=12, b=0),
        font=dict(color=T["text_secondary"], family=F["mono"], size=11),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=T["bg_elevated"], bordercolor=T["border_active"],
            font=dict(family=F["mono"], size=11, color=T["text_primary"]),
        ),
        showlegend=False,
        dragmode="select",
        selectdirection="h",
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=10, color=T["text_secondary"]),
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis=dict(
            showgrid=True, gridcolor=T["border"], gridwidth=0.5,
            zeroline=False, tickprefix="$",
            tickfont=dict(size=10, color=T["text_secondary"]),
            side="right", domain=[0.18, 1.0],
        ),
        yaxis2=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            domain=[0.0, 0.16],
        ),
    )
    return fig


def _company_header(info: Dict, ticker: str, price: Optional[float]):
    name = info.get("shortName", ticker)
    currency = info.get("currency", "USD")
    mktcap = info.get("marketCap")
    pe = info.get("trailingPE")
    h52 = info.get("fiftyTwoWeekHigh")
    l52 = info.get("fiftyTwoWeekLow")

    def fmt_cap(v):
        if not v: return "—"
        return f"${v/1e12:.2f}T" if v>=1e12 else f"${v/1e9:.2f}B" if v>=1e9 else f"${v/1e6:.1f}M"

    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:6px">'
        f'<span style="font-family:\'{F["display"]}\',serif;font-size:1.1rem;'
        f'font-weight:700;color:{T["text_primary"]}">{name}</span>'
        f'<span style="font-family:\'{F["mono"]}\';font-size:0.7rem;'
        f'color:{T["text_muted"]};border:1px solid {T["border"]};'
        f'border-radius:4px;padding:1px 6px">{ticker}</span>'
        f'{"<span style=\\"font-family:\'" + F["mono"] + "\';font-size:1rem;font-weight:700;color:" + T["accent_gold"] + "\\">$" + f"{price:,.2f}" + "</span>" if price else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )
    stat_cols = st.columns(4)
    stats = [
        ("Mkt Cap", fmt_cap(mktcap), T["text_primary"]),
        ("P/E", f"{pe:.1f}" if pe else "—", T["text_primary"]),
        ("52W High", f"${h52:.2f}" if h52 else "—", T["success"]),
        ("52W Low",  f"${l52:.2f}" if l52 else "—", T["danger"]),
    ]
    for col, (label, val, color) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{T["text_muted"]};font-family:\'{F["ui"]}\'">'
                f'{label}</div>'
                f'<div style="font-family:\'{F["mono"]}\';font-size:0.82rem;color:{color}">'
                f'{val}</div>',
                unsafe_allow_html=True,
            )


def render_stock_chart(
    market_data: MarketDataService,
    key_prefix: str = "chart",
) -> Optional[Tuple[datetime, datetime]]:
    """
    Main entry point. Renders the full chart panel.
    Returns (start, end) if user drag-selected a range, else None.
    """
    # ── Ticker search ────────────────────────────────────────────
    sc1, sc2 = st.columns([5, 2])
    with sc1:
        raw_query = st.text_input(
            "ticker",
            value=st.session_state.get(f"{key_prefix}_ticker", MARKET["default_ticker"]),
            key=f"{key_prefix}_q",
            placeholder="Search ticker or company…",
            label_visibility="collapsed",
        )
    with sc2:
        chart_type = st.selectbox(
            "type", ["line", "candlestick"], index=0,
            key=f"{key_prefix}_type", label_visibility="collapsed",
        )

    ticker = st.session_state.get(f"{key_prefix}_ticker", MARKET["default_ticker"])

    # Show autocomplete if query changed
    if raw_query and raw_query != ticker:
        results = market_data.search_ticker(raw_query)
        if results:
            opts = [f'{r["ticker"]} — {r["name"]}' for r in results[:6]]
            chosen = st.selectbox("Pick", opts, key=f"{key_prefix}_pick", label_visibility="collapsed")
            if chosen:
                new_ticker = chosen.split(" — ")[0].strip()
                if new_ticker != ticker:
                    st.session_state[f"{key_prefix}_ticker"] = new_ticker
                    ticker = new_ticker
                    st.rerun()
        else:
            st.session_state[f"{key_prefix}_ticker"] = raw_query.upper()
            ticker = raw_query.upper()

    # ── Timeframe buttons ────────────────────────────────────────
    active_tf = st.session_state.get(f"{key_prefix}_tf", "1Y")
    tf_cols = st.columns(len(MARKET["timeframes"]))
    for col, tf in zip(tf_cols, MARKET["timeframes"]):
        with col:
            active_style = "primary" if tf == active_tf else "secondary"
            if st.button(tf, key=f"{key_prefix}_tf_{tf}",
                         type=active_style, use_container_width=True):
                st.session_state[f"{key_prefix}_tf"] = tf
                st.rerun()

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

    # ── Fetch ────────────────────────────────────────────────────
    tf_cfg = MARKET["timeframes"].get(active_tf, MARKET["timeframes"]["1Y"])
    with st.spinner(""):
        df   = market_data.get_ohlcv(ticker, period=tf_cfg["period"], interval=tf_cfg["interval"])
        info = market_data.get_company_info(ticker)
        price = market_data.get_current_price(ticker)

    # ── Company header ───────────────────────────────────────────
    _company_header(info, ticker, price)
    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

    # ── Chart ────────────────────────────────────────────────────
    sel_start = st.session_state.get(f"{key_prefix}_sel_start")
    sel_end   = st.session_state.get(f"{key_prefix}_sel_end")

    fig = _build_fig(df, ticker, chart_type, sel_start, sel_end)

    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{key_prefix}_plt",
        on_select="rerun",
        config={
            "modeBarButtonsToAdd":    ["select2d"],
            "modeBarButtonsToRemove": ["lasso2d", "autoScale2d", "zoomIn2d", "zoomOut2d"],
            "displaylogo": False,
        },
    )

    # ── Parse drag selection ─────────────────────────────────────
    returned_range = None
    try:
        box_list = selection.selection.get("box", []) if selection and hasattr(selection, "selection") else []
        if box_list:
            xs = box_list[0].get("x", [])
            if len(xs) >= 2:
                start_dt = pd.to_datetime(xs[0]).to_pydatetime()
                end_dt   = pd.to_datetime(xs[-1]).to_pydatetime()
                if start_dt > end_dt:
                    start_dt, end_dt = end_dt, start_dt
                st.session_state[f"{key_prefix}_sel_start"] = start_dt
                st.session_state[f"{key_prefix}_sel_end"]   = end_dt
                sel_start, sel_end = start_dt, end_dt
                returned_range = (start_dt, end_dt)
    except Exception:
        pass

    # ── Selection info bar ───────────────────────────────────────
    if sel_start and sel_end:
        delta = (sel_end - sel_start).days
        period_label = (
            f"{delta}d" if delta < 30 else
            f"{delta//30}mo" if delta < 365 else
            f"{delta/365:.1f}y"
        )
        sel_df = market_data.get_ohlcv_range(ticker, sel_start, sel_end)

        ic1, ic2, ic3, ic4 = st.columns([2, 1, 1, 2])
        with ic1:
            st.markdown(
                f'<span style="font-family:\'{F["mono"]}\';font-size:0.72rem;'
                f'color:{T["text_muted"]}">📅 '
                f'<b style="color:{T["accent_blue"]}">'
                f'{sel_start.strftime("%b %d %Y")} → {sel_end.strftime("%b %d %Y")}'
                f'</b> ({period_label})</span>',
                unsafe_allow_html=True,
            )
        if not sel_df.empty:
            p0, p1 = sel_df["Close"].iloc[0], sel_df["Close"].iloc[-1]
            chg = p1 - p0
            chg_pct = (chg / p0 * 100) if p0 else 0
            color = T["success"] if chg >= 0 else T["danger"]
            sign  = "+" if chg >= 0 else ""
            with ic2:
                st.markdown(
                    f'<span style="font-family:\'{F["mono"]}\';font-size:0.72rem;'
                    f'color:{T["text_muted"]}">Δ '
                    f'<b style="color:{color}">{sign}${chg:.2f}</b></span>',
                    unsafe_allow_html=True,
                )
            with ic3:
                st.markdown(
                    f'<span style="font-family:\'{F["mono"]}\';font-size:0.72rem;'
                    f'color:{T["text_muted"]}">Ret '
                    f'<b style="color:{color}">{sign}{chg_pct:.2f}%</b></span>',
                    unsafe_allow_html=True,
                )
        with ic4:
            if st.button(
                "✦ Ask Agent About This Period",
                key=f"{key_prefix}_ask",
                use_container_width=True,
            ):
                st.session_state["chat_prefill"] = (
                    f"Analyse {ticker} from {sel_start.strftime('%b %d %Y')} "
                    f"to {sel_end.strftime('%b %d %Y')}"
                )
                st.session_state["active_tab"] = "chat"
                st.rerun()

        if st.button("✕ Clear", key=f"{key_prefix}_clr"):
            st.session_state.pop(f"{key_prefix}_sel_start", None)
            st.session_state.pop(f"{key_prefix}_sel_end", None)
            st.rerun()

    return returned_range
