"""
╔══════════════════════════════════════════════════════════════╗
║  AlphaMesh — Central UI Configuration                        ║
║  Edit this file to retheme colours, fonts, and animations.  ║
║  All UI modules import from here — one source of truth.     ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── IDENTITY ────────────────────────────────────────────────────────
APP_NAME = "AlphaMesh"
APP_TAGLINE = "Personalised Investment Intelligence"
APP_ICON = "◈"  # Used in browser tab + logo

# ── COLOUR PALETTE ──────────────────────────────────────────────────
# Deep-space navy meets warm amber — Bloomberg precision, boutique warmth
THEME = {
    # Backgrounds  (darkest → lightest)
    "bg_base": "#070C18",  # Page canvas
    "bg_surface": "#0D1525",  # Cards, panels
    "bg_elevated": "#162035",  # Hover, modals, tooltips
    "bg_input": "#0F1A2E",  # Form fields
    # Brand accents
    "accent_gold": "#F5A623",  # Primary CTA, selected state, gold shine
    "accent_gold_dim": "#B87B12",  # Muted gold for hover
    "accent_gold_glow": "rgba(245,166,35,0.18)",
    "accent_blue": "#4D8EF5",  # Info, links, active nav
    "accent_blue_glow": "rgba(77,142,245,0.18)",
    "accent_teal": "#00CBA8",  # Positive P&L, success
    "accent_teal_glow": "rgba(0,203,168,0.18)",
    "accent_violet": "#9B72F5",  # Synthesiser agent
    # Per-agent brand colours
    "agent_orchestrator": "#F5A623",
    "agent_fundamentals": "#4D8EF5",
    "agent_news": "#00CBA8",
    "agent_synthesiser": "#9B72F5",
    "agent_portfolio": "#F56A6A",
    # Text
    "text_primary": "#EDF2FF",
    "text_secondary": "#7A90B8",
    "text_muted": "#3A4D6B",
    "text_inverse": "#070C18",
    # Semantic
    "success": "#10C98A",
    "danger": "#F05252",
    "warning": "#F5A623",
    "info": "#4D8EF5",
    # Borders
    "border": "#1A2B45",
    "border_active": "#4D8EF5",
    "border_gold": "#F5A623",
    # Context-toggle (included in agent context)
    "context_on": "#F5A623",
    "context_off": "#1A2B45",
}

# ── TYPOGRAPHY ──────────────────────────────────────────────────────
FONTS = {
    "display": "Playfair Display",  # Elegant serif — headlines, agent names
    "ui": "Sora",  # Geometric sans — all UI chrome
    "mono": "Fira Code",  # Monospaced — tickers, logs, prices
    "google_url": (
        "https://fonts.googleapis.com/css2?"
        "family=Playfair+Display:wght@400;600;700&"
        "family=Sora:wght@300;400;500;600;700&"
        "family=Fira+Code:wght@400;500&"
        "display=swap"
    ),
}

# ── SHAPE & SPACING ──────────────────────────────────────────────────
RADIUS = {
    "xs": "6px",
    "sm": "10px",
    "md": "16px",
    "lg": "22px",
    "xl": "30px",
    "pill": "9999px",
}

SHADOW = {
    "sm": "0 2px 8px  rgba(0,0,0,0.45)",
    "md": "0 4px 20px rgba(0,0,0,0.55)",
    "lg": "0 8px 40px rgba(0,0,0,0.65)",
    "glow_gold": "0 0 24px rgba(245,166,35,0.35)",
    "glow_blue": "0 0 24px rgba(77,142,245,0.30)",
    "glow_teal": "0 0 24px rgba(0,203,168,0.30)",
    "glow_violet": "0 0 24px rgba(155,114,245,0.30)",
}

# ── ANIMATION TIMINGS (milliseconds) ────────────────────────────────
ANIMATION = {
    "fast": 150,  # Micro-interactions
    "normal": 350,  # Standard transitions
    "slow": 700,  # Page-level reveals
    "easing": "cubic-bezier(0.4, 0, 0.2, 1)",
    "easing_spring": "cubic-bezier(0.34, 1.56, 0.64, 1)",
    "easing_out": "cubic-bezier(0.0, 0, 0.2, 1)",
    # Agent graph
    "spawn_stagger": 250,  # Delay between each agent node appearing
    "spawn_duration": 700,  # Animated edge draw duration
    "pulse_period": 2200,  # Node heartbeat animation
    "merge_duration": 1400,  # Agents flowing into synthesiser
    "node_appear": 450,  # Node fade+scale in
    "particle_count": 16,  # Particles per agent on merge
    # Log panel
    "log_slide": 180,
    # Chart crossfade
    "chart_fade": 300,
}

# ── AGENT GRAPH GEOMETRY ─────────────────────────────────────────────
GRAPH = {
    "node_radius": 30,
    "orchestrator_radius": 40,
    "synthesiser_radius": 36,
    "edge_width": 2,
    "force_charge": -260,
    "force_link_distance": 160,
    "graph_height": 320,
}

# ── MARKET DATA ──────────────────────────────────────────────────────
MARKET = {
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_db": 0,
    "cache_ttl_price": 300,  # 5 min — live prices
    "cache_ttl_ohlcv": 3600,  # 1 hr  — OHLCV history
    "cache_ttl_info": 86400,  # 24 hr  — company metadata
    "default_ticker": "AAPL",
    # Maps UI label → yfinance params
    "timeframes": {
        "1W": {"period": "5d", "interval": "1h"},
        "1M": {"period": "1mo", "interval": "1d"},
        "3M": {"period": "3mo", "interval": "1d"},
        "6M": {"period": "6mo", "interval": "1d"},
        "YTD": {"period": "ytd", "interval": "1d"},
        "1Y": {"period": "1y", "interval": "1wk"},
        "2Y": {"period": "2y", "interval": "1wk"},
        "5Y": {"period": "5y", "interval": "1mo"},
    },
}

# ── AUTH ─────────────────────────────────────────────────────────────
AUTH = {
    "cookie_name": "nexus_auth",
    "cookie_key": "CHANGE_ME_IN_PRODUCTION_32_CHARS_",  # 32-char secret
    "cookie_expiry": 30,  # days
    "preauthorised": [],  # emails allowed to self-register
}

# ── CONTEXT SELECTION ────────────────────────────────────────────────
CONTEXT = {
    "max_items": 12,  # Max context items per query
    "label_on": "✦ In Context",
    "label_off": "○ Add to Context",
}
