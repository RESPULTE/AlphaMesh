"""
Agent Graph Visualisation Component.

Returns a self-contained HTML string (D3.js v7) for use with
st.components.v1.html(). State transitions drive all animations.

Valid states:
    idle        → System ready, single orchestrator node
    planning    → Orchestrator pulses "thinking"
    spawning    → Animated edges grow outward; child nodes appear
    executing   → Nodes pulse with their agent colour in parallel
    merging     → Child nodes fly toward synthesiser with particles
    complete    → Synthesiser glows; all nodes settle
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ui.config import THEME, FONTS, ANIMATION, GRAPH


def render_agent_graph(
    state: str = "idle",
    active_agents: Optional[List[str]] = None,
    agent_progress: Optional[Dict[str, str]] = None,
    height: Optional[int] = None,
) -> str:
    active_agents  = active_agents or []
    agent_progress = agent_progress or {}
    h = height or GRAPH["graph_height"]

    cfg = json.dumps({
        "state":        state,
        "activeAgents": active_agents,
        "progress":     agent_progress,
        "h":            h,
        "colors": {
            "orchestrator":       THEME["agent_orchestrator"],
            "fundamentals_agent": THEME["agent_fundamentals"],
            "news_agent":         THEME["agent_news"],
            "synthesiser":        THEME["agent_synthesiser"],
            "textSec":            THEME["text_secondary"],
            "textPri":            THEME["text_primary"],
            "textMuted":          THEME["text_muted"],
        },
        "anim": {
            "spawnStagger":  ANIMATION["spawn_stagger"],
            "spawnDuration": ANIMATION["spawn_duration"],
            "pulse":         ANIMATION["pulse_period"],
            "merge":         ANIMATION["merge_duration"],
            "appear":        ANIMATION["node_appear"],
            "particles":     ANIMATION["particle_count"],
        },
        "gr": {
            "nr":  GRAPH["node_radius"],
            "or_": GRAPH["orchestrator_radius"],
            "sr":  GRAPH["synthesiser_radius"],
            "ew":  GRAPH["edge_width"],
        },
        "fontMono": FONTS["mono"],
        "fontUi":   FONTS["ui"],
    })

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ background:transparent; overflow:hidden; width:100%; height:{h}px; }}
  #gc {{ width:100%; height:{h}px; }}
  svg {{ display:block; width:100%; height:100%; }}
  @keyframes dash-flow {{ to {{ stroke-dashoffset: -24; }} }}
  @keyframes spin-ring {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<div id="gc"><svg id="gsvg"></svg></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"
  integrity="sha512-M7nHCiNp39/HwB9d4lBRfyP7RGdWGIVMoIZmvNgBm7i7rDWVFpWlcm7YBHi3HWWXE5SZXI1OwWMN4pODL3gw=="
  crossorigin="anonymous"></script>
<script>
const CFG = {cfg};
const S = CFG.state;
const ACTIVE = new Set(CFG.activeAgents);

const gc = document.getElementById('gc');
const W  = gc.offsetWidth || 360;
const H  = CFG.h;
const cx = W / 2, cy = H / 2;

const svg  = d3.select('#gsvg').attr('viewBox', `0 0 ${{W}} ${{H}}`);
const defs = svg.append('defs');

// Glow filters
function addGlow(id, std) {{
  const f = defs.append('filter').attr('id', id)
    .attr('x','-80%').attr('y','-80%').attr('width','260%').attr('height','260%');
  f.append('feGaussianBlur').attr('stdDeviation', std).attr('result','blur');
  const m = f.append('feMerge');
  m.append('feMergeNode').attr('in','blur');
  m.append('feMergeNode').attr('in','SourceGraphic');
}}
addGlow('glow-s', 3);
addGlow('glow-h', 7);

// Radial gradients per agent
['orchestrator','fundamentals_agent','news_agent','synthesiser'].forEach(id => {{
  const c = CFG.colors[id];
  if (!c) return;
  const g = defs.append('radialGradient')
    .attr('id',`rg-${{id}}`).attr('cx','35%').attr('cy','32%').attr('r','68%');
  g.append('stop').attr('offset','0%')
    .attr('stop-color', d3.color(c)?.brighter(0.6)?.formatHex() || c);
  g.append('stop').attr('offset','100%')
    .attr('stop-color', d3.color(c)?.darker(1)?.formatHex() || c);
}});

// Node positions
function pos() {{
  const n = CFG.activeAgents.length;
  if (S === 'idle' || S === 'planning' || n === 0) {{
    return {{
      orchestrator:       {{x: cx,       y: cy - 15}},
      fundamentals_agent: {{x: cx - 115, y: cy + 115}},
      news_agent:         {{x: cx + 115, y: cy + 115}},
      synthesiser:        {{x: cx,       y: cy + 115}},
    }};
  }}
  if (n === 1) {{
    const a = CFG.activeAgents[0];
    const r = {{
      orchestrator: {{x: cx, y: cy - 85}},
      synthesiser:  {{x: cx, y: cy + 95}},
    }};
    r[a] = {{x: cx, y: cy + 15}};
    ['fundamentals_agent','news_agent'].forEach(id => {{ if (!(id in r)) r[id] = {{x:-999,y:cy}}; }});
    return r;
  }}
  return {{
    orchestrator:       {{x: cx,       y: cy - 95}},
    fundamentals_agent: {{x: cx - 118, y: cy + 30}},
    news_agent:         {{x: cx + 118, y: cy + 30}},
    synthesiser:        {{x: cx,       y: cy + 135}},
  }};
}}
const P = pos();

// Layers
const lEdge = svg.append('g');
const lPart = svg.append('g');
const lNode = svg.append('g');
const lLbl  = svg.append('g');
const lUI   = svg.append('g');

// Edges
const EDGES = [
  {{s:'orchestrator',t:'fundamentals_agent'}},
  {{s:'orchestrator',t:'news_agent'}},
  {{s:'fundamentals_agent',t:'synthesiser'}},
  {{s:'news_agent',t:'synthesiser'}},
];

function eVisible(e) {{
  if (S==='idle'||S==='planning') return false;
  if (S==='spawning'||S==='executing') return e.s==='orchestrator' && ACTIVE.has(e.t);
  if (S==='merging') return (e.s==='orchestrator'&&ACTIVE.has(e.t))||(ACTIVE.has(e.s)&&e.t==='synthesiser');
  if (S==='complete') return true;
  return false;
}}

EDGES.filter(eVisible).forEach((e,i) => {{
  const sp = P[e.s], tp = P[e.t];
  const animated = (S==='spawning'&&e.s==='orchestrator'&&ACTIVE.has(e.t)) ||
                   (S==='executing'&&e.s==='orchestrator'&&ACTIVE.has(e.t)) ||
                   (S==='merging'&&ACTIVE.has(e.s)&&e.t==='synthesiser');
  const col = CFG.colors[e.s] || CFG.colors.orchestrator;

  const line = lEdge.append('line')
    .attr('x1',sp.x).attr('y1',sp.y)
    .attr('x2',sp.x).attr('y2',sp.y)
    .attr('stroke',col).attr('stroke-width',CFG.gr.ew)
    .attr('stroke-linecap','round').attr('opacity',0.7);

  if (animated) line.attr('stroke-dasharray','7 4')
    .style('animation','dash-flow 1.1s linear infinite');

  line.transition().duration(CFG.anim.spawnDuration).delay(i*85)
    .attr('x2',tp.x).attr('y2',tp.y);
}});

// Nodes
const NODES = [
  {{id:'orchestrator',       label:'Orchestrator',  icon:'⬡', r:CFG.gr.or_}},
  {{id:'fundamentals_agent', label:'Fundamentals',  icon:'◆', r:CFG.gr.nr}},
  {{id:'news_agent',         label:'News Intel',    icon:'◈', r:CFG.gr.nr}},
  {{id:'synthesiser',        label:'Synthesiser',   icon:'◉', r:CFG.gr.sr}},
];

function nVisible(n) {{
  if (n.id==='orchestrator') return true;
  if (n.id==='synthesiser')  return S==='merging'||S==='complete';
  return S !== 'idle';
}}
function nOpacity(n) {{
  if (S==='idle') return n.id==='orchestrator' ? 0.5 : 0;
  if (n.id==='orchestrator') return 1;
  if (n.id==='synthesiser' && S!=='merging' && S!=='complete') return 0;
  if (!ACTIVE.has(n.id) && n.id!=='synthesiser') return 0.22;
  return 1;
}}
function nStatus(n) {{
  const p = CFG.progress[n.id];
  if (p) return p;
  if (n.id==='orchestrator') {{
    return {{idle:'ready',planning:'planning…',spawning:'routing',
             executing:'waiting',merging:'merging',complete:'done ✓'}}[S]||'';
  }}
  if (ACTIVE.has(n.id)) {{
    if (S==='executing') return 'running…';
    if (S==='merging')   return '→ synth';
    if (S==='complete')  return 'done ✓';
  }}
  if (n.id==='synthesiser') {{
    if (S==='merging')  return 'merging…';
    if (S==='complete') return 'done ✓';
  }}
  return '';
}}

NODES.filter(nVisible).forEach((n,i) => {{
  const p   = P[n.id];
  const col = CFG.colors[n.id] || CFG.colors.orchestrator;
  const isActive = ACTIVE.has(n.id) || n.id==='orchestrator';
  const delay = i * (CFG.anim.spawnStagger * 0.5);

  const g = lNode.append('g')
    .attr('transform',`translate(${{p.x}},${{p.y}})`)
    .attr('opacity',0);

  // Pulse ring
  if (isActive && S!=='idle' && S!=='complete') {{
    const ring = g.append('circle').attr('r',n.r).attr('fill','none')
      .attr('stroke',col).attr('stroke-width',1).attr('opacity',0);
    const pulse = () => ring.attr('r',n.r).attr('opacity',0.8)
      .transition().duration(CFG.anim.pulse)
      .attr('r',n.r+20).attr('opacity',0)
      .on('end',pulse);
    setTimeout(pulse, delay+300);
  }}

  // Shadow
  g.append('circle').attr('r',n.r+3).attr('fill','rgba(0,0,0,0.3)')
    .attr('transform','translate(2,4)');
  // Main circle
  g.append('circle').attr('r',n.r)
    .attr('fill',`url(#rg-${{n.id}})`).attr('stroke',col).attr('stroke-width',1.5)
    .attr('filter', isActive && S!=='idle' ? 'url(#glow-s)' : 'none');
  // Icon
  g.append('text').attr('font-size',n.r>32?'18':'14').attr('fill','rgba(255,255,255,0.92)')
    .attr('text-anchor','middle').attr('dominant-baseline','central')
    .attr('dy',n.r>32?'-4':'-2').attr('pointer-events','none').text(n.icon);

  g.transition().duration(CFG.anim.appear).delay(delay).attr('opacity',nOpacity(n));

  // Labels
  const lx = p.x, ly = p.y + n.r + 16;
  lLbl.append('text').attr('x',lx).attr('y',ly)
    .attr('text-anchor','middle').attr('font-size','10.5')
    .attr('fill',CFG.colors.textSec).attr('font-family',CFG.fontUi+',sans-serif')
    .attr('letter-spacing','0.04em').attr('opacity',0)
    .text(n.label)
    .transition().duration(300).delay(delay+CFG.anim.appear)
    .attr('opacity',nOpacity(n)*0.8);

  const st = nStatus(n);
  if (st) {{
    lLbl.append('text').attr('x',lx).attr('y',ly+15)
      .attr('text-anchor','middle').attr('font-size','9')
      .attr('font-weight','700').attr('text-transform','uppercase')
      .attr('fill',col).attr('font-family',CFG.fontMono+',monospace')
      .attr('letter-spacing','0.08em').attr('opacity',0)
      .text(st)
      .transition().duration(300).delay(delay+CFG.anim.appear+80)
      .attr('opacity',0.9);
  }}
}});

// Merge: particles + node drift
if (S==='merging') {{
  const sp = P['synthesiser'];
  CFG.activeAgents.forEach(agentId => {{
    const src = P[agentId];
    const col = CFG.colors[agentId]||CFG.colors.orchestrator;
    const cnt = Math.ceil(CFG.anim.particles / Math.max(CFG.activeAgents.length,1));
    for (let k=0; k<cnt; k++) {{
      const jit = () => (Math.random()-0.5)*16;
      lPart.append('circle')
        .attr('r',3.5).attr('fill',col).attr('filter','url(#glow-s)')
        .attr('cx',src.x+jit()).attr('cy',src.y+jit()).attr('opacity',0.9)
        .transition().duration(CFG.anim.merge*0.7).delay(k*50).ease(d3.easeCubicIn)
        .attr('cx',sp.x+jit()*0.2).attr('cy',sp.y).attr('r',1).attr('opacity',0)
        .on('end',function(){{d3.select(this).remove();}});
    }}
    // Drift agent node
    lNode.selectAll('g').each(function() {{
      const xf = d3.select(this).attr('transform')||'';
      const m  = xf.match(/translate\(([^,]+),([^)]+)\)/);
      if (!m) return;
      const nx=+m[1], ny=+m[2];
      const pp = P[agentId];
      if (pp && Math.abs(pp.x-nx)<5 && Math.abs(pp.y-ny)<5) {{
        d3.select(this).transition().duration(CFG.anim.merge*0.8).delay(350)
          .ease(d3.easeCubicInOut)
          .attr('transform',`translate(${{sp.x}},${{sp.y}})`).attr('opacity',0);
      }}
    }});
  }});
}}

// Planning: spinning dashed ring around orchestrator
if (S==='planning') {{
  const op = P['orchestrator'];
  lNode.append('circle')
    .attr('cx',op.x).attr('cy',op.y)
    .attr('r',CFG.gr.or_+8)
    .attr('fill','none')
    .attr('stroke',CFG.colors.orchestrator)
    .attr('stroke-width',1.5)
    .attr('stroke-dasharray','5 4')
    .attr('opacity',0.5)
    .style('transform-origin',`${{op.x}}px ${{op.y}}px`)
    .style('animation','spin-ring 3.5s linear infinite');
}}

// State banner
const BANNERS = {{
  idle:'SYSTEM READY', planning:'⬡ ORCHESTRATOR PLANNING',
  spawning:'⬡ SPAWNING AGENTS', executing:'⬡ PARALLEL EXECUTION',
  merging:'⬡ SYNTHESISING RESULTS', complete:'✓ ANALYSIS COMPLETE',
}};
lUI.append('text')
  .attr('x',W/2).attr('y',H-11)
  .attr('text-anchor','middle').attr('font-size','10')
  .attr('font-weight','700').attr('letter-spacing','0.12em')
  .attr('fill',S==='complete'?CFG.colors.synthesiser:CFG.colors.textMuted)
  .attr('font-family',CFG.fontMono+',monospace')
  .attr('opacity',0)
  .text(BANNERS[S]||'')
  .transition().duration(500).attr('opacity',0.65);
</script>
</body>
</html>"""
