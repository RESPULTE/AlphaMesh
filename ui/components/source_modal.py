"""
Source Modal & Citation Renderer.

Parses [1], [2] citation markers in agent text, replaces them with
clickable chips, and renders a pop-up card when clicked.

Works entirely in Streamlit session_state — no JS bridge needed.
"""

from __future__ import annotations

import re
from typing import Any, List

import streamlit as st

from ui.config import FONTS as F
from ui.config import THEME as T

# ── Normalisation ─────────────────────────────────────────────────


def _to_dict(src: Any) -> dict:
    """
    Convert a source to a plain dict regardless of origin:
      • CitedSource Pydantic model  (live orchestrator result)
      • dict                         (retrieved from chat history)
      • any other object             (best-effort getattr fallback)
    This is the single choke-point that fixes 'CitedSource is not subscriptable'.
    """
    if isinstance(src, dict):
        return src
    if hasattr(src, "model_dump"):  # Pydantic v2
        return src.model_dump()
    if hasattr(src, "dict"):  # Pydantic v1
        return src.dict()
    return {  # plain object / dataclass
        "source_id": getattr(src, "source_id", "?"),
        "title": getattr(src, "title", "Untitled"),
        "url": getattr(src, "url", "#"),
        "page_content": getattr(src, "page_content", ""),
    }


def normalise_sources(sources: List[Any]) -> List[dict]:
    """Normalise an entire source list to List[dict] in one call."""
    return [_to_dict(s) for s in (sources or [])]


# ── Source card ───────────────────────────────────────────────────


def _source_card(src: Any, included: bool, card_key: str) -> None:
    """
    Renders a source metadata card with title, URL chip, excerpt,
    context-toggle button, and dismiss button.
    Accepts both CitedSource objects and plain dicts.
    """
    d = _to_dict(src)
    idx = d.get("source_id", "?")
    title = d.get("title", "Untitled")
    url = d.get("url", "#") or "#"
    body = d.get("page_content", "") or ""
    excerpt = (body[:280] + "…") if len(body) > 280 else body
    domain = url.replace("https://", "").replace("http://", "").split("/")[0]

    st.markdown(
        f'<div class="nx-source-modal">'
        f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px">'
        f'<span style="font-family:\'{F["mono"]}\';font-size:0.7rem;font-weight:700;'
        f'color:{T["accent_blue"]};background:rgba(77,142,245,0.12);'
        f"border:1px solid rgba(77,142,245,0.3);border-radius:4px;"
        f'padding:2px 7px;flex-shrink:0">[{idx}]</span>'
        f'<span class="nx-source-title">{title}</span>'
        f"</div>"
        f'<a class="nx-source-url" href="{url}" target="_blank" rel="noopener">'
        f"🔗 {domain}</a>"
        f'<div class="nx-source-excerpt">{excerpt}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
        ctx_label = "✦ In Context" if included else "○ Add to Context"
        if st.button(ctx_label, key=f"{card_key}_ctx", use_container_width=True):
            ctx_set: set = st.session_state.get("ctx_sources", set())
            if included:
                ctx_set.discard(str(idx))
            else:
                ctx_set.add(str(idx))
            st.session_state["ctx_sources"] = ctx_set
            st.rerun()
    with b2:
        if url and url != "#":
            st.markdown(
                f'<a href="{url}" target="_blank" rel="noopener" style="display:block;'
                f"text-align:center;padding:5px 0;font-size:0.75rem;"
                f'font-family:\'{F["ui"]}\';color:{T["accent_blue"]};'
                f'border:1px solid {T["border"]};border-radius:20px;'
                f'text-decoration:none">↗ Open Article</a>',
                unsafe_allow_html=True,
            )
    with b3:
        if st.button("✕", key=f"{card_key}_close", use_container_width=True):
            st.session_state.pop("active_source_id", None)
            st.rerun()


# ── Main renderer ─────────────────────────────────────────────────


def render_cited_text(text: str, sources: List[Any]) -> None:
    """
    Render agent text with [N] citations as clickable chips.
    Accepts sources as CitedSource objects, dicts, or a mix.
    """
    if not sources:
        st.markdown(text)
        return

    # Normalise once — everything below uses plain dicts
    srcs: List[dict] = normalise_sources(sources)
    source_map = {str(s["source_id"]): s for s in srcs}
    ctx_sources: set = st.session_state.get("ctx_sources", set())

    st.markdown(
        "<style>.nx-citation-block { line-height: 1.75; }</style>",
        unsafe_allow_html=True,
    )

    def make_chip(m):
        n = m.group(1)
        inc = n in ctx_sources
        gold = T["accent_gold"] if inc else T["accent_blue"]
        bg = "rgba(245,166,35,0.12)" if inc else "rgba(77,142,245,0.10)"
        bd = "rgba(245,166,35,0.35)" if inc else "rgba(77,142,245,0.28)"
        return (
            f'<span class="nx-source-chip" '
            f'style="background:{bg};border-color:{bd};color:{gold}">[{n}]</span>'
        )

    html_text = re.sub(r"\[(\d+)\]", make_chip, text)
    st.markdown(
        f'<div class="nx-citation-block" style="font-family:\'{F["ui"]}\';'
        f'font-size:0.88rem;color:{T["text_primary"]};line-height:1.75">'
        f"{html_text}</div>",
        unsafe_allow_html=True,
    )

    # Sources row
    st.markdown(
        f'<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px">'
        f'<span style="font-family:\'{F["ui"]}\';font-size:0.65rem;'
        f'color:{T["text_muted"]};letter-spacing:0.07em;text-transform:uppercase;'
        f'align-self:center;margin-right:4px">Sources:</span>'
        + "".join(
            [
                f'<span style="font-family:\'{F["mono"]}\';font-size:0.7rem;'
                f'color:{T["accent_blue"]};background:rgba(77,142,245,0.08);'
                f"border:1px solid rgba(77,142,245,0.25);border-radius:4px;"
                f'padding:1px 6px;cursor:default">'
                f'[{s["source_id"]}] {str(s["title"])[:40]}…</span>'
                for s in srcs
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # Chip selector buttons
    chip_cols = st.columns(min(len(srcs), 4))
    for col, s in zip(chip_cols, srcs):
        sid = s["source_id"]
        active = st.session_state.get("active_source_id") == sid
        icon = "▼" if active else "▶"
        is_in = str(sid) in ctx_sources
        badge = " ✦" if is_in else ""
        with col:
            if st.button(
                f"{icon} [{sid}]{badge}",
                key=f"src_chip_{sid}",
                use_container_width=True,
                help=str(s.get("title", "")),
            ):
                if active:
                    st.session_state.pop("active_source_id", None)
                else:
                    st.session_state["active_source_id"] = sid
                st.rerun()

    # Active source card
    active_id = st.session_state.get("active_source_id")
    if active_id is not None:
        active_src = source_map.get(str(active_id))
        if active_src:
            st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)
            _source_card(
                active_src,
                included=str(active_id) in ctx_sources,
                card_key=f"src_modal_{active_id}",
            )


# ── Context helper ────────────────────────────────────────────────


def get_context_source_strings(sources: List[Any]) -> List[str]:
    """
    Return formatted strings for sources the user toggled into context.
    Injected into the agent system prompt.
    """
    ctx_sources: set = st.session_state.get("ctx_sources", set())
    result = []
    for s in normalise_sources(sources):
        if str(s["source_id"]) in ctx_sources:
            result.append(
                f"[Source {s['source_id']}] {s['title']}\n"
                f"URL: {s['url']}\n"
                f"{s.get('page_content', '')[:400]}"
            )
    return result
