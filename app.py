import streamlit as st
from utils.ui import load_css
from components.auth import handle_auth_dialog
from components.auth import set_auth_mode_and_show_modal
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["google"]["api_key"]

def _display_header():
    """
    Displays the header using native Streamlit columns and text elements.
    """
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 5, 1, 1.2])

        with col1:
            # USE: st.title for the main logo/title. It's semantically correct.
            st.title("AlphaMesh")



def _display_hero_section():
    """Renders the main hero section using native components."""
    with st.container():
        col1, col2 = st.columns([1.1, 0.9], gap="large", vertical_alignment="center")
        with col1:
            # USE: st.title and st.write for hero text instead of HTML tags.
            st.header("AI Agents, Human Insight.")
            st.write("AlphaMesh uses a team of specialized AI agents to analyze markets, debate strategies, and deliver clear, actionable investment intelligence. No code, just results.")
            if st.button("Get Started for Free", key="hero_get_started", type="primary"):
                set_auth_mode_and_show_modal('Sign Up')
        with col2:
            # USE: A simple container to act as a placeholder.
            with st.container(border=True, height=400):
                 st.write("Product Animation")


def _display_social_proof_section():
    """Displays the 'Powered By' logos section."""
    with st.container(width=800, horizontal=False, horizontal_alignment="center"):
        st.caption("POWERED BY LEADING-EDGE TECHNOLOGY", width="content")

        cols = st.columns(4, gap="medium")
        logos = [
            {"path": "https://upload.wikimedia.org/wikipedia/commons/d/d9/Google_Gemini_logo_2025.svg", "caption": "Google Gemini"},
            {"path": "https://upload.wikimedia.org/wikipedia/commons/6/60/LangChain_Logo.svg", "caption": "LangChain"},
            {"path": "https://upload.wikimedia.org/wikipedia/commons/c/ca/Interactive_Brokers_Logo_%282014%29.svg", "caption": "Interactive Brokers"},
            {"path": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Yahoo%21_%282019%29.svg", "caption": "Yahoo Finance", }
        ]
        for i, logo in enumerate(logos):
            with cols[i]:
                st.image(logo["path"], caption=logo["caption"], use_container_width="always")


def _display_how_it_works_section():
    """Displays the 'How It Works' section with three info cards."""
    with st.container():
        # USE: st.header with a divider is a clean way to create a section title.
        st.header("From Market Noise to Actionable Thesis in 3 Steps", divider="rainbow")

        cards_data = [
            {"icon": "1.", "title": "Connect Your Goals", "text": "Tell AlphaMesh your investment style and risk tolerance. Your AI team calibrates to your specific needs."},
            {"icon": "2.", "title": "Agents Analyze & Debate", "text": "Our AI agents—a Risk Analyst, a News Sentinel, a Fundamental Analyst—gather data and debate the best course of action."},
            {"icon": "3.", "title": "Receive Your Briefing", "text": "Get a clear, synthesized report with bull and bear cases, key data points, and a final recommendation. All transparent, all for you."}
        ]

        cols = st.columns(3, gap="large")

        for i, data in enumerate(cards_data):
            with cols[i]:
                # USE: st.container(border=True) creates a card effect natively.
                with st.container(border=True):
                    st.subheader(f'{data["icon"]} {data["title"]}')
                    st.write(data["text"])

def _display_features_section():
    """Displays key features with an alternating text/image layout."""
    with st.container():
        st.header("An Unfair Advantage, Built For You", divider="rainbow")
        st.space(2) # USE: st.space() for vertical spacing instead of <br> tags.

        # Feature 1
        col1, col2 = st.columns([1, 1], gap="large", vertical_alignment='center')
        with col1:
            st.header("The Daily Briefing")
            st.write("Start your day with a mission-critical summary. Your agent team works 24/7, so you wake up with insights, not just alerts. Get a curated look at your portfolio's overnight news, market sentiment, and the top opportunity identified by the mesh.")
        with col2:
            st.image("https://placehold.co/500x300/1a1a1a/FFFFFF?text=Dashboard+Mockup")

        st.space(2)

        # Feature 2
        col1, col2 = st.columns([1, 1], gap="large", vertical_alignment='center')
        with col1:
            st.image("https://placehold.co/500x300/1a1a1a/FFFFFF?text=Transparent+Analysis")
        with col2:
            st.header("Deep Dive with Full Transparency")
            st.write("Every insight is backed by transparent reasoning. Understand the 'why' behind each recommendation by exploring the Bull Case from the Fundamental Agent, the Bear Case from the Risk Manager, and citations from the News Sentinel.")

def _display_final_cta():
    """Displays the final call-to-action section."""
    # USE: A bordered container makes the CTA section stand out.
    with st.container(border=True, horizontal_alignment="center", ):
        st.header("The Future of Investing is Collaborative Intelligence.", width='content')
        st.text("Stop guessing. Start making data-driven decisions with your personal AI investment committee.")

        _, col, _ = st.columns([1, 0.6, 1])
        with col:
            if st.button("Sign Up Now - It's Free", key="final_cta_button", width='stretch', type="primary"):
                set_auth_mode_and_show_modal('Sign Up')

def _display_footer():
    """Displays the page footer."""
    with st.container():
        # USE: st.divider() for a clean horizontal line.
        st.divider()
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("© 2024 AlphaMesh. All rights reserved.")
        with c2:
            # Using markdown for links is acceptable and standard.
            st.markdown('<div style="text-align: right;"><a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></div>', unsafe_allow_html=True)

def render_landing_page():
    """Renders all the sections of the landing page in order."""
    with st.container(horizontal_alignment="center"):
        _display_header()
        _display_hero_section()

        st.space(40)


        _display_social_proof_section()

        st.space(40)

        _display_how_it_works_section()
        _display_features_section()
        _display_final_cta()
        _display_footer()

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