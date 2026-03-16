"""
Authentication Module for Nexus AI.
Multi-user cookie-based auth via streamlit-authenticator.
Falls back to simple session auth if library unavailable.

Install: pip install streamlit-authenticator pyyaml bcrypt
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import streamlit as st

from ui.config import AUTH, THEME as T, FONTS as F, APP_NAME, APP_ICON

logger = logging.getLogger(__name__)

try:
    import yaml
    import streamlit_authenticator as stauth
    from yaml.loader import SafeLoader
    _AUTH_OK = True
except ImportError:
    _AUTH_OK = False

_DEMO_USERS = {
    "demo@nexus.ai":  "nexus2024",
    "admin@nexus.ai": "admin2024",
}

_CRED_FILE = os.path.join(os.path.dirname(__file__), "credentials.yaml")
_DEFAULT_CREDS = {
    "credentials": {"usernames": {
        "demo":  {"email": "demo@nexus.ai",  "name": "Demo User", "password": "$2b$12$YcmYNfR7s.c/H3OvH5YVeOyIFEqhF8iW5.lDeJjMqgN.n/GZSB0Iy"},
        "admin": {"email": "admin@nexus.ai", "name": "Admin",     "password": "$2b$12$jVTMlEYb.pzPX4fFUolYqeXkuVCLrPFJyHQmVqyoH3kFXeK3I/sEy"},
    }},
    "cookie": {"expiry_days": AUTH["cookie_expiry"], "key": AUTH["cookie_key"], "name": AUTH["cookie_name"]},
    "preauthorized": {"emails": AUTH["preauthorised"]},
}

def _login_shell(extra_html: str = ""):
    st.markdown(
        f'<div style="text-align:center;padding:2rem 0 1rem">'
        f'<div style="font-family:\'{F["display"]}\',serif;font-size:2.4rem;'
        f'font-weight:700;color:{T["text_primary"]};letter-spacing:-0.02em">'
        f'{APP_ICON} {APP_NAME}</div>'
        f'<div style="font-family:\'{F["ui"]}\';font-size:0.82rem;'
        f'color:{T["text_muted"]};margin-top:6px">Personalised AI Investment Intelligence</div>'
        f'</div>{extra_html}',
        unsafe_allow_html=True,
    )

def _demo_login() -> Tuple[bool, Optional[str]]:
    _login_shell()
    _, fc, _ = st.columns([1, 2, 1])
    with fc:
        with st.form("login_form"):
            email    = st.text_input("Email",    placeholder="demo@nexus.ai")
            password = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign In →", use_container_width=True)
        if ok:
            if _DEMO_USERS.get(email) == password:
                st.session_state.update({
                    "authenticated": True,
                    "user_email":    email,
                    "user_name":     email.split("@")[0].title(),
                })
                st.rerun()
                return True, email
            st.error("Invalid credentials")
        st.markdown(
            f'<div style="margin-top:10px;padding:10px 14px;border-radius:10px;'
            f'background:rgba(77,142,245,0.08);border:1px solid rgba(77,142,245,0.2)">'
            f'<span style="font-family:\'{F["mono"]}\';font-size:0.72rem;'
            f'color:{T["accent_blue"]}">Demo → demo@nexus.ai / nexus2024</span></div>',
            unsafe_allow_html=True,
        )
    return False, None

def _full_login() -> Tuple[bool, Optional[str]]:
    if not os.path.exists(_CRED_FILE):
        try:
            with open(_CRED_FILE, "w") as f:
                yaml.dump(_DEFAULT_CREDS, f)
        except Exception:
            pass
    try:
        with open(_CRED_FILE) as f:
            config = yaml.load(f, Loader=SafeLoader)
    except Exception:
        config = _DEFAULT_CREDS

    auth = stauth.Authenticate(
        config["credentials"], config["cookie"]["name"],
        config["cookie"]["key"], config["cookie"]["expiry_days"],
        config.get("preauthorized", {}),
    )
    _login_shell()
    _, fc, _ = st.columns([1, 2, 1])
    with fc:
        name, status, username = auth.login("Login", "main")
    if status:
        email = config["credentials"]["usernames"].get(username, {}).get("email", f"{username}@nexus.ai")
        st.session_state.update({
            "authenticated": True, "user_email": email,
            "user_name": name or username.title(), "authenticator": auth,
        })
        return True, email
    elif status is False:
        st.error("Incorrect username or password.")
    return False, None

def require_auth() -> Tuple[bool, Optional[str]]:
    if st.session_state.get("authenticated"):
        return True, st.session_state.get("user_email")
    return _full_login() if _AUTH_OK else _demo_login()

def render_logout():
    user  = st.session_state.get("user_name", "User")
    email = st.session_state.get("user_email", "")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:8px 10px;'
        f'background:{T["bg_elevated"]};border:1px solid {T["border"]};'
        f'border-radius:10px;margin-bottom:12px">'
        f'<div style="width:28px;height:28px;border-radius:50%;'
        f'background:linear-gradient(135deg,{T["accent_gold"]},{T["accent_blue"]});'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:12px;color:{T["text_inverse"]};font-weight:700">{user[0].upper()}</div>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-family:\'{F["ui"]}\';font-size:0.78rem;font-weight:700;'
        f'color:{T["text_primary"]};white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{user}</div>'
        f'<div style="font-family:\'{F["mono"]}\';font-size:0.62rem;color:{T["text_muted"]};'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{email}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign Out", key="logout_btn", use_container_width=True):
        for k in ["authenticated","user_email","user_name","chat_history",
                  "portfolio_df","agent_logs","graph_state","active_agents"]:
            st.session_state.pop(k, None)
        st.rerun()
