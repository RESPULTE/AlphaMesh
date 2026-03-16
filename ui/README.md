# Nexus AI — UI Module

> Personalised AI Investment Intelligence · Streamlit hybrid + D3.js + Plotly + Redis

---

## Quick Start

```bash
pip install -r requirements_ui.txt
docker run -d -p 6379:6379 redis:alpine   # optional but recommended
streamlit run ui/app.py
```
**Demo login:** `demo@nexus.ai` / `nexus2024`

---

## File Structure

```
ui/
├── app.py                   ← Entrypoint: streamlit run ui/app.py
├── config.py                ← ⭐ ALL colours, fonts, animations — edit here to retheme
├── styles.py                ← CSS generator (reads config.py)
├── auth.py                  ← Multi-user cookie auth (streamlit-authenticator)
├── requirements_ui.txt
├── .streamlit/config.toml   ← Streamlit server config
├── services/
│   ├── market_data.py       ← yfinance + Redis caching
│   └── portfolio_service.py ← Portfolio CRUD, live valuation, CSV import
└── components/
    ├── agent_graph.py       ← D3.js v7 agent spawn & merge animation
    ├── stock_chart.py       ← Plotly OHLCV + drag-to-select period
    ├── portfolio_panel.py   ← Holdings table, donut chart, context toggles
    ├── chat_panel.py        ← Streaming chat, animation driver, context assembly
    ├── log_panel.py         ← Side-by-side agent execution logs
    └── source_modal.py      ← Cited source chips + pop-up cards
```

---

## Retheme in One File

`ui/config.py` controls **everything**:

| Key | Controls |
|---|---|
| `THEME` | All colour tokens (background, accents, text, borders) |
| `FONTS` | Display / UI / monospace typefaces |
| `RADIUS` | Border-radius at xs → xl → pill |
| `SHADOW` | Box-shadows and glow effects |
| `ANIMATION` | All ms durations, easing, particle count |
| `GRAPH` | D3 node sizes, edge width, force charge |
| `MARKET` | Redis TTLs, default ticker, timeframe maps |
| `AUTH` | Cookie name/key/expiry, allowed emails |

---

## Agent Graph Animation States

```
idle → planning → spawning → executing → merging → complete
```

| State | Visual |
|---|---|
| `idle` | Single orchestrator node, dim pulse |
| `planning` | Spinning dashed ring around orchestrator |
| `spawning` | Edges animate out to child nodes (staggered) |
| `executing` | All nodes pulse their brand colour in parallel |
| `merging` | Particle streams fly to synthesiser; nodes drift in |
| `complete` | Synthesiser glows; "✓ ANALYSIS COMPLETE" banner |

---

## Context System

Every UI component is toggleable into agent context:

- **Portfolio** — global toggle + per-ticker buttons
- **Log entries** — per-entry checkboxes in expander
- **Sources** — "○ Add to Context" on each source card  
- **Fundamentals** — toggle above the data table
- **Chart selection** — auto-sent when clicking "Ask Agent"

Active items show as `✦ tag` chips above the chat input and are injected into the orchestrator's system prompt.

---

## Source Citations

Responses with `[1]`, `[2]` markers render:
1. Inline blue clickable chips
2. Click → source card with title, domain chip, excerpt, URL
3. "Add to Context" button per source

---

## Redis Cache Key Schema

| Pattern | Data | TTL |
|---|---|---|
| `nx:px:{TICKER}` | Live price | 5 min |
| `nx:ohlcv:{TICKER}:{PERIOD}:{INTERVAL}` | OHLCV history | 1 hr |
| `nx:ohlcv_range:{T}:{START}:{END}:{I}` | Date-range OHLCV | 1 hr |
| `nx:info:{TICKER}` | Company metadata | 24 hr |
| `nx:portfolio:{EMAIL}` | User portfolio | Persistent |

Redis is optional — degrades gracefully to live fetches.

---

## Responsive Breakpoints

- **≥769px (desktop)**: Sidebar nav, multi-column layouts, side-by-side logs
- **<768px (mobile)**: Sidebar collapses, horizontal scroll top-nav, columns stack

Breakpoints in `styles.py` `.nx-desktop` / `.nx-mobile` CSS classes.
