"""
Global CSS generator for Nexus AI.
All values sourced from config.py — edit that file to retheme.
"""
from ui.config import THEME as T, FONTS as F, RADIUS as R, SHADOW as S, ANIMATION as A


def get_global_css() -> str:
    """Returns the full <style> block to inject into Streamlit."""
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{F['google_url']}" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════
   CSS VARIABLES — driven by config.py
═══════════════════════════════════════════ */
:root {{
  --bg-base:        {T['bg_base']};
  --bg-surface:     {T['bg_surface']};
  --bg-elevated:    {T['bg_elevated']};
  --bg-input:       {T['bg_input']};
  --accent-gold:    {T['accent_gold']};
  --accent-gold-dim:{T['accent_gold_dim']};
  --accent-gold-glow:{T['accent_gold_glow']};
  --accent-blue:    {T['accent_blue']};
  --accent-teal:    {T['accent_teal']};
  --accent-violet:  {T['accent_violet']};
  --text-primary:   {T['text_primary']};
  --text-secondary: {T['text_secondary']};
  --text-muted:     {T['text_muted']};
  --success:        {T['success']};
  --danger:         {T['danger']};
  --warning:        {T['warning']};
  --border:         {T['border']};
  --border-active:  {T['border_active']};
  --radius-xs:      {R['xs']};
  --radius-sm:      {R['sm']};
  --radius-md:      {R['md']};
  --radius-lg:      {R['lg']};
  --radius-xl:      {R['xl']};
  --radius-pill:    {R['pill']};
  --font-display:   '{F['display']}', Georgia, serif;
  --font-ui:        '{F['ui']}', system-ui, sans-serif;
  --font-mono:      '{F['mono']}', 'Courier New', monospace;
  --shadow-sm:      {S['sm']};
  --shadow-md:      {S['md']};
  --anim-fast:      {A['fast']}ms;
  --anim-normal:    {A['normal']}ms;
  --easing:         {A['easing']};
}}

/* ═══════════════════════════════════════════
   RESET & STREAMLIT BASE
═══════════════════════════════════════════ */
*, *::before, *::after {{ box-sizing: border-box; }}

.stApp {{
  background: var(--bg-base) !important;
  font-family: var(--font-ui) !important;
  color: var(--text-primary) !important;
}}
#MainMenu, footer, .stDeployButton,
header[data-testid="stHeader"] {{ display: none !important; }}

.block-container {{
  padding: 1rem 1.25rem 2rem !important;
  max-width: 100% !important;
}}

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
[data-testid="stSidebar"] {{
  background: {T['bg_surface']} !important;
  border-right: 1px solid {T['border']} !important;
}}
[data-testid="stSidebar"] > div {{
  padding: 1rem 0.75rem !important;
}}
[data-testid="stSidebarNav"] {{ display: none; }}

/* ═══════════════════════════════════════════
   INPUTS
═══════════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
  background: var(--bg-input) !important;
  border: 1px solid {T['border']} !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-ui) !important;
  caret-color: var(--accent-gold) !important;
  transition: border-color var(--anim-fast) var(--easing),
              box-shadow   var(--anim-fast) var(--easing);
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 3px {T['accent_blue_glow']} !important;
  outline: none !important;
}}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stNumberInput label, .stFileUploader label {{
  color: var(--text-secondary) !important;
  font-family: var(--font-ui) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}}

/* ═══════════════════════════════════════════
   CHAT INPUT
═══════════════════════════════════════════ */
[data-testid="stChatInput"] textarea {{
  background: var(--bg-input) !important;
  border: 1px solid {T['border']} !important;
  border-radius: var(--radius-lg) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-ui) !important;
  transition: border-color var(--anim-fast) var(--easing) !important;
}}
[data-testid="stChatInput"]:focus-within textarea {{
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 3px {T['accent_blue_glow']} !important;
}}
[data-testid="stChatInput"] button {{
  background: linear-gradient(135deg, {T['accent_gold']}, {T['accent_gold_dim']}) !important;
  border-radius: var(--radius-sm) !important;
  border: none !important;
}}

/* ═══════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════ */
.stButton > button {{
  background: linear-gradient(135deg, {T['accent_gold']}, {T['accent_gold_dim']}) !important;
  color: {T['text_inverse']} !important;
  border: none !important;
  border-radius: var(--radius-pill) !important;
  font-family: var(--font-ui) !important;
  font-weight: 700 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.04em !important;
  padding: 0.45rem 1.4rem !important;
  transition: transform var(--anim-fast) {A['easing_spring']},
              box-shadow var(--anim-fast) var(--easing) !important;
  cursor: pointer !important;
}}
.stButton > button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: {S['glow_gold']} !important;
}}
.stButton > button:active {{
  transform: translateY(0px) !important;
}}
.stButton > button[kind="secondary"] {{
  background: transparent !important;
  border: 1px solid {T['border']} !important;
  color: var(--text-secondary) !important;
}}
.stButton > button[kind="secondary"]:hover {{
  border-color: var(--accent-blue) !important;
  color: var(--accent-blue) !important;
  box-shadow: none !important;
}}

/* ═══════════════════════════════════════════
   SELECTBOX & NUMBER INPUT
═══════════════════════════════════════════ */
.stSelectbox > div > div {{
  background: var(--bg-input) !important;
  border: 1px solid {T['border']} !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
}}
.stSelectbox [data-baseweb="select"] > div {{
  background: transparent !important;
  border: none !important;
  color: var(--text-primary) !important;
}}

/* ═══════════════════════════════════════════
   DATA EDITOR / TABLE
═══════════════════════════════════════════ */
[data-testid="stDataEditor"],
[data-testid="stDataFrame"] {{
  border-radius: var(--radius-md) !important;
  border: 1px solid {T['border']} !important;
  overflow: hidden;
}}
.stDataFrame thead th {{
  background: {T['bg_elevated']} !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-ui) !important;
  font-size: 0.72rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  border-bottom: 1px solid {T['border']} !important;
}}

/* ═══════════════════════════════════════════
   METRICS
═══════════════════════════════════════════ */
[data-testid="stMetric"] {{
  background: var(--bg-surface);
  border: 1px solid {T['border']};
  border-radius: var(--radius-md);
  padding: 0.9rem 1.1rem;
  transition: border-color var(--anim-fast) var(--easing);
}}
[data-testid="stMetric"]:hover {{
  border-color: {T['border_active']};
}}
[data-testid="stMetricLabel"] span {{
  color: var(--text-secondary) !important;
  font-family: var(--font-ui) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
}}
[data-testid="stMetricValue"] {{
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 1.35rem !important;
}}
[data-testid="stMetricDelta"] {{
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
}}

/* ═══════════════════════════════════════════
   TABS
═══════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  background: transparent !important;
  gap: 0.25rem;
  border-bottom: 1px solid {T['border']};
  padding: 0 0 0 0.25rem;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-ui) !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.04em !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  padding: 0.5rem 1rem !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  transition: all var(--anim-fast) var(--easing) !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {{
  color: var(--text-primary) !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
  color: {T['accent_gold']} !important;
  border-bottom-color: {T['accent_gold']} !important;
}}

/* ═══════════════════════════════════════════
   EXPANDER
═══════════════════════════════════════════ */
[data-testid="stExpander"] {{
  background: var(--bg-surface) !important;
  border: 1px solid {T['border']} !important;
  border-radius: var(--radius-md) !important;
}}
[data-testid="stExpander"] summary {{
  color: var(--text-secondary) !important;
  font-family: var(--font-ui) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
}}

/* ═══════════════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════════════ */
[data-testid="stFileUploader"] {{
  border: 1px dashed {T['border']} !important;
  border-radius: var(--radius-md) !important;
  background: var(--bg-input) !important;
  transition: border-color var(--anim-fast) var(--easing) !important;
}}
[data-testid="stFileUploader"]:hover {{
  border-color: var(--accent-blue) !important;
}}

/* ═══════════════════════════════════════════
   CHAT MESSAGES
═══════════════════════════════════════════ */
[data-testid="stChatMessage"] {{
  background: transparent !important;
}}

/* ═══════════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════════ */
.stProgress > div > div > div {{
  background: linear-gradient(90deg, {T['accent_blue']}, {T['accent_teal']}) !important;
  border-radius: var(--radius-pill) !important;
}}
.stProgress > div > div {{
  background: {T['bg_elevated']} !important;
  border-radius: var(--radius-pill) !important;
}}

/* ═══════════════════════════════════════════
   CUSTOM NEXUS COMPONENTS
═══════════════════════════════════════════ */

/* -- Nexus Card ------------------------------------------ */
.nx-card {{
  background: var(--bg-surface);
  border: 1px solid {T['border']};
  border-radius: var(--radius-md);
  padding: 1.1rem 1.25rem;
  transition: border-color var(--anim-fast) var(--easing),
              box-shadow   var(--anim-fast) var(--easing);
  position: relative;
  overflow: hidden;
  animation: nx-fadeInUp 0.35s ease both;
}}
.nx-card:hover {{
  border-color: {T['border_active']};
  box-shadow: var(--shadow-sm);
}}
.nx-card.nx-included {{
  border-color: {T['accent_gold']} !important;
  box-shadow: {S['glow_gold']} !important;
}}
.nx-card.nx-excluded {{
  opacity: 0.48;
}}

/* -- Context Toggle Badge -------------------------------- */
.nx-ctx-badge {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  font-size: 0.65rem;
  font-weight: 700;
  font-family: var(--font-ui);
  letter-spacing: 0.07em;
  text-transform: uppercase;
  cursor: pointer;
  transition: all var(--anim-fast) var(--easing);
  user-select: none;
  border: 1px solid;
  position: absolute;
  top: 8px;
  right: 10px;
}}
.nx-ctx-badge.on {{
  color: {T['accent_gold']};
  border-color: {T['accent_gold']};
  background: {T['accent_gold_glow']};
}}
.nx-ctx-badge.off {{
  color: {T['text_muted']};
  border-color: {T['border']};
  background: transparent;
}}

/* -- Agent Badge ----------------------------------------- */
.nx-agent-badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 9px;
  border-radius: var(--radius-pill);
  font-size: 0.68rem;
  font-weight: 700;
  font-family: var(--font-ui);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  border: 1px solid;
}}
.nx-agent-badge.orchestrator {{ color:{T['agent_orchestrator']}; border-color:{T['agent_orchestrator']}; background:rgba(245,166,35,0.1); }}
.nx-agent-badge.fundamentals  {{ color:{T['agent_fundamentals']}; border-color:{T['agent_fundamentals']}; background:rgba(77,142,245,0.1); }}
.nx-agent-badge.news          {{ color:{T['agent_news']};         border-color:{T['agent_news']};         background:rgba(0,203,168,0.1); }}
.nx-agent-badge.synthesiser   {{ color:{T['agent_synthesiser']};  border-color:{T['agent_synthesiser']};  background:rgba(155,114,245,0.1); }}

/* -- Source Chip ----------------------------------------- */
.nx-source-chip {{
  display: inline-flex;
  align-items: center;
  gap: 3px;
  background: rgba(77,142,245,0.1);
  border: 1px solid rgba(77,142,245,0.28);
  color: {T['accent_blue']};
  border-radius: var(--radius-pill);
  padding: 1px 7px;
  font-size: 0.7rem;
  font-weight: 700;
  font-family: var(--font-mono);
  cursor: pointer;
  transition: all var(--anim-fast) var(--easing);
  margin: 0 2px;
  text-decoration: none;
}}
.nx-source-chip:hover {{
  background: rgba(77,142,245,0.2);
  border-color: {T['accent_blue']};
  transform: translateY(-1px);
}}

/* -- Log Entry ------------------------------------------- */
.nx-log {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  line-height: 1.55;
  padding: 3px 8px;
  border-left: 2px solid transparent;
  border-radius: 0 {R['xs']} {R['xs']} 0;
  margin-bottom: 2px;
  animation: nx-slideLeft {A['log_slide']}ms ease forwards;
  word-break: break-word;
}}
.nx-log.info    {{ border-color:{T['accent_blue']};  color:{T['text_secondary']}; }}
.nx-log.success {{ border-color:{T['success']};      color:{T['success']}; }}
.nx-log.warn    {{ border-color:{T['warning']};      color:{T['warning']}; }}
.nx-log.error   {{ border-color:{T['danger']};       color:{T['danger']}; background:rgba(240,82,82,0.06); }}

/* -- Divider --------------------------------------------- */
.nx-divider {{
  height: 1px;
  background: linear-gradient(90deg, transparent, {T['border']}, transparent);
  margin: 0.85rem 0;
  border: none;
}}

/* -- Section Header -------------------------------------- */
.nx-section-header {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 0.75rem;
}}
.nx-section-title {{
  font-family: var(--font-ui);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: {T['text_muted']};
}}
.nx-section-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}}

/* -- Timeframe Button ------------------------------------ */
.nx-tf-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 0.6rem;
}}
.nx-tf-btn {{
  background: {T['bg_elevated']};
  border: 1px solid {T['border']};
  color: {T['text_secondary']};
  border-radius: var(--radius-pill);
  padding: 3px 11px;
  font-size: 0.72rem;
  font-weight: 600;
  font-family: var(--font-ui);
  cursor: pointer;
  transition: all var(--anim-fast) var(--easing);
  outline: none;
}}
.nx-tf-btn:hover {{ border-color:{T['accent_blue']}; color:{T['accent_blue']}; }}
.nx-tf-btn.active {{
  background: rgba(77,142,245,0.15);
  border-color: {T['accent_blue']};
  color: {T['accent_blue']};
}}

/* -- Tooltip (native title override) --------------------- */
.nx-tooltip {{
  position: fixed;
  background: {T['bg_elevated']};
  border: 1px solid {T['border_active']};
  border-radius: var(--radius-sm);
  padding: 0.6rem 0.85rem;
  font-size: 0.78rem;
  color: var(--text-primary);
  font-family: var(--font-ui);
  box-shadow: {S['md']};
  z-index: 9999;
  max-width: 300px;
  pointer-events: none;
  animation: nx-fadeInUp 0.15s ease;
}}

/* -- Source Modal ---------------------------------------- */
.nx-source-modal {{
  background: {T['bg_elevated']};
  border: 1px solid {T['border_active']};
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  box-shadow: {S['lg']};
  animation: nx-fadeInUp 0.25s ease;
  max-width: 480px;
  width: 100%;
}}
.nx-source-title {{
  font-family: var(--font-display);
  font-size: 1.05rem;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
  line-height: 1.4;
}}
.nx-source-url {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: {T['accent_blue']};
  word-break: break-all;
  text-decoration: none;
}}
.nx-source-url:hover {{ text-decoration: underline; }}
.nx-source-excerpt {{
  font-family: var(--font-ui);
  font-size: 0.8rem;
  color: {T['text_secondary']};
  line-height: 1.6;
  margin-top: 0.75rem;
  border-left: 2px solid {T['border_active']};
  padding-left: 0.75rem;
  font-style: italic;
}}

/* ═══════════════════════════════════════════
   KEYFRAME ANIMATIONS
═══════════════════════════════════════════ */
@keyframes nx-fadeInUp {{
  from {{ opacity: 0; transform: translateY(10px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes nx-slideLeft {{
  from {{ opacity: 0; transform: translateX(-6px); }}
  to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes nx-pulse-glow {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(245,166,35,0); }}
  50%       {{ box-shadow: {S['glow_gold']}; }}
}}
@keyframes nx-spin {{
  to {{ transform: rotate(360deg); }}
}}
@keyframes nx-shimmer {{
  0%   {{ background-position: -200% 0; }}
  100% {{ background-position:  200% 0; }}
}}
@keyframes nx-blink {{
  0%, 100% {{ opacity: 1; }}
  50%       {{ opacity: 0.3; }}
}}

/* -- Skeleton loading ------------------------------------ */
.nx-skeleton {{
  background: linear-gradient(
    90deg,
    {T['bg_surface']} 25%,
    {T['bg_elevated']} 50%,
    {T['bg_surface']} 75%
  );
  background-size: 200% 100%;
  animation: nx-shimmer 1.6s infinite;
  border-radius: var(--radius-sm);
}}

/* ═══════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════ */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {T['bg_base']}; }}
::-webkit-scrollbar-thumb {{
  background: {T['border']};
  border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {T['text_muted']}; }}

/* ═══════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════ */
@media (max-width: 768px) {{
  .block-container {{ padding: 0.75rem 0.75rem 2rem !important; }}
  .nx-desktop {{ display: none !important; }}
  [data-testid="column"] {{ min-width: 100% !important; }}
}}
@media (min-width: 769px) {{
  .nx-mobile {{ display: none !important; }}
}}
</style>
"""
