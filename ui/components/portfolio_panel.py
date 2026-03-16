"""
Portfolio Panel Component.

Features:
  • Manual position entry (ticker, qty, avg cost)
  • CSV upload & broker API placeholder
  • Live P&L and market value via MarketDataService
  • Interactive Plotly donut allocation chart
  • Per-ticker and per-row click-to-include-in-context toggles
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.config import THEME as T, FONTS as F
from ui.services.market_data import MarketDataService
from ui.services.portfolio_service import PortfolioService

_PALETTE = [
    T["agent_orchestrator"], T["agent_fundamentals"], T["agent_news"],
    T["agent_synthesiser"],  T["agent_portfolio"],
    "#A3E635", "#FB923C", "#38BDF8", "#F472B6", "#A78BFA",
]


def _donut(valued_df: pd.DataFrame) -> go.Figure:
    df = valued_df[valued_df.get("market_value", pd.Series(dtype=float)) > 0] if not valued_df.empty else valued_df
    if df.empty or "market_value" not in df.columns:
        return go.Figure()
    total = df["market_value"].sum()
    fig = go.Figure(go.Pie(
        labels=df["ticker"], values=df["market_value"],
        hole=0.62,
        marker=dict(colors=_PALETTE[:len(df)], line=dict(color=T["bg_surface"], width=2)),
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
        textinfo="label+percent",
        textfont=dict(family=F["mono"], size=10, color=T["text_primary"]),
        insidetextorientation="radial",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        annotations=[dict(
            text=f"<b>${total:,.0f}</b>",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(family=F["mono"], size=15, color=T["text_primary"]),
        )],
    )
    return fig


def _fmt_dollar(v):
    return f"${v:,.2f}" if pd.notna(v) else "—"

def _fmt_pct(v):
    if pd.isna(v): return "—"
    s = "+" if v >= 0 else ""
    return f"{s}{v:.2f}%"


def render_portfolio_panel(
    portfolio_svc: PortfolioService,
    market_data:   MarketDataService,
    user_email:    str,
) -> Optional[str]:
    """
    Renders the full portfolio panel.
    Returns a context string if the user has toggled portfolio into agent context.
    """
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = portfolio_svc.load(user_email)
    df: pd.DataFrame = st.session_state.portfolio_df

    # ── Panel header + global context toggle ─────────────────────
    hcol, tcol = st.columns([5, 3])
    with hcol:
        st.markdown(
            f'<div class="nx-section-header">'
            f'<div class="nx-section-dot" style="background:{T["accent_gold"]}"></div>'
            f'<span class="nx-section-title">My Portfolio</span></div>',
            unsafe_allow_html=True,
        )
    with tcol:
        include_all = st.toggle(
            "Include portfolio in context",
            value=st.session_state.get("portfolio_in_context", False),
            key="port_global_ctx",
        )
        st.session_state["portfolio_in_context"] = include_all

    # ── Tabs ─────────────────────────────────────────────────────
    t_hold, t_add, t_import = st.tabs(["📊 Holdings", "➕ Add Position", "📂 Import"])

    # ═══════════════════════════════════════════
    # HOLDINGS TAB
    # ═══════════════════════════════════════════
    with t_hold:
        if df.empty:
            st.markdown(
                f'<div style="text-align:center;padding:2rem 1rem;'
                f'color:{T["text_muted"]};font-family:\'{F["ui"]}\';font-size:0.85rem">'
                f'No holdings yet.<br>Use <b>Add Position</b> or <b>Import</b> to get started.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Refreshing live prices…"):
                valued = portfolio_svc.valuate(df)
            sm = portfolio_svc.summary(valued)

            # Summary metrics row
            mc1, mc2, mc3, mc4 = st.columns(4)
            pnl_sign = "+" if sm["total_pnl"] >= 0 else ""
            with mc1:
                st.metric("Market Value", f'${sm["total_value"]:,.2f}')
            with mc2:
                st.metric("Cost Basis", f'${sm["total_cost"]:,.2f}')
            with mc3:
                st.metric(
                    "Unrealised P&L",
                    f'{pnl_sign}${sm["total_pnl"]:,.2f}',
                    delta=f'{pnl_sign}{sm["total_pnl_pct"]:.2f}%',
                )
            with mc4:
                st.metric("Positions", str(sm["n_positions"]))

            st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

            # Donut + Holdings table
            dcol, tcol2 = st.columns([2, 3])
            with dcol:
                st.plotly_chart(_donut(valued), use_container_width=True, key="port_donut")

            with tcol2:
                display = valued[[
                    c for c in [
                        "ticker", "quantity", "avg_cost", "current_price",
                        "market_value", "unrealized_pnl", "pnl_pct", "weight_pct",
                    ] if c in valued.columns
                ]].copy()

                def colour_cell(val):
                    if pd.isna(val): return f"color:{T['text_muted']}"
                    return (f"color:{T['success']};font-weight:600"
                            if val >= 0 else f"color:{T['danger']};font-weight:600")

                styled = (
                    display.style
                    .format({
                        "quantity":      "{:,.3f}",
                        "avg_cost":      _fmt_dollar,
                        "current_price": lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
                        "market_value":  _fmt_dollar,
                        "unrealized_pnl": _fmt_dollar,
                        "pnl_pct":       _fmt_pct,
                        "weight_pct":    lambda v: f"{v:.1f}%",
                    })
                    .applymap(colour_cell, subset=["unrealized_pnl", "pnl_pct"])
                    .set_properties(**{
                        "font-family": F["mono"],
                        "font-size": "0.73rem",
                        "color": T["text_primary"],
                    })
                    .set_table_styles([
                        {"selector": "thead th", "props": [
                            ("background", T["bg_elevated"]),
                            ("color", T["text_muted"]),
                            ("font-size", "0.62rem"),
                            ("text-transform", "uppercase"),
                            ("letter-spacing", "0.07em"),
                            ("padding", "5px 8px"),
                            ("white-space", "nowrap"),
                        ]},
                        {"selector": "tbody td", "props": [("padding", "4px 8px")]},
                        {"selector": "table", "props": [
                            ("border-collapse", "collapse"),
                            ("width", "100%"),
                        ]},
                    ])
                )
                st.markdown(
                    f'<div style="overflow-x:auto;border-radius:10px;'
                    f'border:1px solid {T["border"]}">'
                    + styled.to_html(escape=False) + "</div>",
                    unsafe_allow_html=True,
                )

            # ── Per-ticker context selection ─────────────────
            st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="nx-section-title" style="margin-bottom:8px">'
                f'Click tickers to toggle in agent context</div>',
                unsafe_allow_html=True,
            )
            ctx_set: set = st.session_state.get("portfolio_ctx_tickers", set())
            tickers = valued["ticker"].tolist()
            btn_cols = st.columns(min(len(tickers), 8))
            for col, tk in zip(btn_cols, tickers):
                with col:
                    is_in = tk in ctx_set
                    icon  = "✦" if is_in else "○"
                    color_style = (
                        f"background:rgba(245,166,35,0.15);border:1px solid {T['accent_gold']};"
                        f"color:{T['accent_gold']}"
                        if is_in else
                        f"background:{T['bg_elevated']};border:1px solid {T['border']};"
                        f"color:{T['text_muted']}"
                    )
                    if st.button(
                        f"{icon} {tk}", key=f"pctx_{tk}",
                        help=f"{'Remove' if is_in else 'Add'} {tk} from context",
                    ):
                        if is_in: ctx_set.discard(tk)
                        else: ctx_set.add(tk)
                        st.session_state["portfolio_ctx_tickers"] = ctx_set
                        st.rerun()

            # Remove buttons
            st.markdown(
                f'<div class="nx-section-title" style="margin-top:12px;margin-bottom:6px">'
                f'Remove positions</div>',
                unsafe_allow_html=True,
            )
            rm_cols = st.columns(min(len(tickers), 8))
            for col, tk in zip(rm_cols, tickers):
                with col:
                    if st.button(f"✕ {tk}", key=f"prm_{tk}"):
                        st.session_state.portfolio_df = portfolio_svc.remove_position(
                            user_email, df, tk,
                        )
                        st.rerun()

    # ═══════════════════════════════════════════
    # ADD POSITION TAB
    # ═══════════════════════════════════════════
    with t_add:
        with st.form("add_pos_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                add_ticker = st.text_input("Ticker symbol *", placeholder="AAPL")
                add_qty    = st.number_input("Quantity (shares) *", min_value=0.0,
                                             step=0.0001, format="%.4f")
            with c2:
                add_name = st.text_input("Company name", placeholder="Auto-fetched if blank")
                add_cost = st.number_input("Avg cost per share ($) *", min_value=0.0,
                                           step=0.01, format="%.2f")
            add_ccy = st.selectbox(
                "Currency",
                ["USD","EUR","GBP","SGD","MYR","HKD","JPY","CAD","AUD","CHF"],
            )
            submitted = st.form_submit_button("➕ Add Position", use_container_width=True)
            if submitted:
                add_ticker = add_ticker.upper().strip()
                if not add_ticker:
                    st.error("Ticker symbol is required.")
                else:
                    if not add_name:
                        with st.spinner("Looking up company…"):
                            info = market_data.get_company_info(add_ticker)
                        add_name = info.get("shortName", add_ticker)
                    st.session_state.portfolio_df = portfolio_svc.add_position(
                        user_email, st.session_state.portfolio_df,
                        add_ticker, add_name, add_qty, add_cost, add_ccy,
                    )
                    st.success(f"✓ {add_ticker} added to portfolio")
                    st.rerun()

    # ═══════════════════════════════════════════
    # IMPORT TAB
    # ═══════════════════════════════════════════
    with t_import:
        csv_tab, broker_tab = st.tabs(["CSV Upload", "Broker API"])

        with csv_tab:
            st.markdown(
                f'<p style="font-size:0.78rem;color:{T["text_secondary"]};'
                f'font-family:\'{F["ui"]}\'">'
                f'Required columns: <code style="color:{T["accent_teal"]}">ticker</code>, '
                f'<code style="color:{T["accent_teal"]}">quantity</code>, '
                f'<code style="color:{T["accent_teal"]}">avg_cost</code>.<br>'
                f'Optional: <code style="color:{T["text_muted"]}">name</code>, '
                f'<code style="color:{T["text_muted"]}">currency</code>.</p>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Drop your CSV here", type=["csv"],
                label_visibility="collapsed",
            )
            if uploaded:
                with st.spinner("Parsing CSV…"):
                    parsed = portfolio_svc.from_csv(uploaded.read())
                if parsed.empty:
                    st.error("Could not parse CSV — check column names.")
                else:
                    st.dataframe(parsed, use_container_width=True)
                    if st.button("✓ Import these positions", use_container_width=True,
                                 key="csv_import_btn"):
                        combined = pd.concat(
                            [st.session_state.portfolio_df, parsed],
                            ignore_index=True,
                        ).drop_duplicates(subset=["ticker"], keep="last")
                        portfolio_svc.save(user_email, combined)
                        st.session_state.portfolio_df = combined
                        st.success(f"Imported {len(parsed)} positions ✓")
                        st.rerun()

        with broker_tab:
            st.markdown(
                f'<div style="border:1px solid {T["border"]};border-radius:12px;'
                f'padding:1.25rem;background:{T["bg_elevated"]}">'
                f'<div style="font-family:\'{F["display"]}\',serif;font-size:1rem;'
                f'color:{T["text_primary"]};margin-bottom:0.5rem">🔗 Broker Connectivity</div>'
                f'<p style="font-size:0.8rem;color:{T["text_secondary"]};'
                f'font-family:\'{F["ui"]}\';line-height:1.65">'
                f'Connect your brokerage account to automatically sync positions '
                f'and transactions. Supported platforms below — contact your account '
                f'manager to enable API access.</p>'
                f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:1rem">'
                + "".join([
                    f'<span style="padding:4px 12px;border:1px solid {T["border"]};'
                    f'border-radius:20px;font-size:0.72rem;color:{T["text_muted"]};'
                    f'font-family:\'{F["ui"]}\'">{b}</span>'
                    for b in ["Interactive Brokers", "Alpaca", "Robinhood", "TD Ameritrade",
                              "E*TRADE", "Fidelity", "Schwab", "Webull"]
                ])
                + f'</div>'
                f'<div style="margin-top:1rem;padding:0.75rem;background:{T["bg_surface"]};'
                f'border-radius:8px;border-left:3px solid {T["accent_blue"]}">'
                f'<span style="font-size:0.72rem;color:{T["accent_blue"]};'
                f'font-family:\'{F["mono"]}\'">ℹ Broker integration is configured '
                f'server-side. Contact your administrator to set up OAuth credentials.</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ── Return context string ─────────────────────────────────────
    if include_all and not df.empty:
        try:
            valued = portfolio_svc.valuate(df)
            ctx_tickers = st.session_state.get("portfolio_ctx_tickers")
            if ctx_tickers:
                filtered = valued[valued["ticker"].isin(ctx_tickers)]
                return portfolio_svc.to_context_string(filtered) if not filtered.empty else None
            return portfolio_svc.to_context_string(valued)
        except Exception:
            pass
    return None
