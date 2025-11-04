import streamlit as st
from utils.ui import load_css
from components.landing_page import render_landing_page
from components.auth import handle_auth_dialog

def main():
    """
    Main function to run the Streamlit application.
    Initializes the app, loads styles, manages session state, and renders pages.
    """
    # --- Page Configuration (must be the first Streamlit command) ---
    st.set_page_config(
        page_title="AlphaMesh",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # --- Load CSS ---
    # Load global styles first, then page-specific styles
    load_css("styles/style.css")
    load_css("styles/landing_page.css")
    load_css("styles/header.css") 

    # --- Initialize Session State ---
    # Use a single initialization block for clarity
    if 'show_auth_dialog' not in st.session_state:
        st.session_state.show_auth_dialog = False
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'Sign Up' # Default to Sign Up
    # 🆕 New state for authentication status
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False

    # --- Render Page Content ---
    if st.session_state.is_authenticated:
        # ⚠️ This is the traditional way to handle page rendering *without* the built-in
        # multi-page app runner. For the official multi-page structure (using 'pages/' folder),
        # Streamlit handles the routing. We rely on st.switch_page() in auth.py.
        # This app.py simply renders the landing page content as the default entry.
        # The switch_page call will take precedence.
        pass # Do nothing, st.switch_page() handles the redirect.
    else:
        render_landing_page()

    # --- Conditionally Display Dialog ---
    handle_auth_dialog()

if __name__ == "__main__":
    main()