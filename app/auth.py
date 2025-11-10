# components/auth.py

import streamlit as st
from supabase import Client


def set_auth_mode_and_show_modal(mode: str):
    """
    Callback function to set the authentication mode ('Login' or 'Sign Up')
    and trigger the native st.dialog to be displayed on the next rerun.
    """
    st.session_state.auth_mode = mode
    st.session_state.show_auth_dialog = True
    st.rerun()


def authenticate_user(supabase: Client, email: str, password: str):
    """
    Authenticates the user with Supabase, handling both sign-up and login.
    The mode is determined by st.session_state.auth_mode.
    """
    mode = st.session_state.get("auth_mode", "Login")

    try:
        if mode == "Sign Up":
            response = supabase.auth.sign_up({"email": email, "password": password})
            st.toast("Account created successfully! Welcome.", icon="🎉")
        elif mode == "Login":
            response = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.toast(f"Welcome back, {email}!", icon="👋")

        if response.user and response.session:
            st.session_state.is_authenticated = True
            st.session_state.show_auth_dialog = False
            st.session_state.user_id = response.user.user_metadata["email"]
            # This is the redirect to the dashboard after successful auth
            st.switch_page("pages/dashboard.py")
        else:
            st.error("Authentication failed. Please try again.", icon="🚨")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}", icon="🚨")


@st.dialog("Authentication", width="small")
def auth_dialog(supabase: Client):
    """
    Renders a unified authentication dialog with tabs for Login and Sign Up.
    """
    # Use st.radio to create a tab-like switcher.
    modes = ["Login", "Sign Up"]
    current_mode_index = modes.index(st.session_state.get("auth_mode", "Login"))

    selected_mode = st.radio(
        "Select Action",
        modes,
        index=current_mode_index,
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_mode != st.session_state.auth_mode:
        st.session_state.auth_mode = selected_mode
        st.rerun()

    # --- Display Form Fields based on selected mode ---
    st.markdown("---")

    email = st.text_input("Email", placeholder="you@gmail.com", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")

    if selected_mode == "Sign Up":
        confirm_password = st.text_input(
            "Confirm Password", type="password", key="auth_confirm_password"
        )

    st.write("")  # Spacer

    if st.button(
        selected_mode,
        key="dialog_form_submit",
        use_container_width=True,
        type="primary",
    ):
        if not email or not password:
            st.warning("Please enter both email and password.", icon="⚠️")
        elif selected_mode == "Sign Up" and password != confirm_password:
            st.error("Passwords do not match.", icon="🚨")
        else:
            # The authenticate_user function already knows the mode from session_state
            authenticate_user(supabase, email, password)

    st.markdown(
        '<div style="text-align: center; margin: 1.5rem 0 1.5rem 0; color: var(--text-color-light);">— OR —</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Sign in with Google", key="dialog_google_signin", use_container_width=True
    ):
        st.info("Google Sign-In is not yet configured.", icon="ℹ️")

    st.markdown(
        """
        <div style="text-align: center; font-size: 0.8rem; color: var(--text-color-light); margin-top: 2rem;">
            By continuing, you agree to our <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>.
        </div>
    """,
        unsafe_allow_html=True,
    )


def handle_auth_dialog(supabase: Client):
    """
    Checks session state and shows the dialog if triggered.
    """
    if (
        st.session_state.get("show_auth_dialog", False)
        and not st.session_state.is_authenticated
    ):
        auth_dialog(supabase)
