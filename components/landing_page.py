# components/landing_page.py

import streamlit as st
from components.auth import set_auth_mode_and_show_modal

def _display_header():
    """
    Displays the sticky header using a native Streamlit container and columns.
    The stickiness and styling are applied via CSS in header.css.
    """
    # This container will be targeted by CSS to become the sticky header.
    with st.container():
        # Use columns for layout: Logo | Spacer | Login Button | Sign Up Button
        # The ratios create the desired spacing.
        col1, col2, col3, col4 = st.columns([2, 5, 1, 1.2]) # Adjusted ratio for "Sign Up"

        with col1:
            st.markdown('<div class="logo">AlphaMesh</div>', unsafe_allow_html=True)

        # col2 is an empty spacer column

        with col3:
            if st.button("Login", key="header_login", use_container_width=True):
                set_auth_mode_and_show_modal('Login')

        with col4:
            if st.button("Sign Up", key="header_signup", use_container_width=True):
                set_auth_mode_and_show_modal('Sign Up')

def _display_hero_section():
    """Renders the main hero section of the landing page."""
    with st.container():
        st.markdown('<div class="section-container hero-section">', unsafe_allow_html=True)
        col1, col2 = st.columns([1.1, 0.9], gap="large")
        with col1:
            st.markdown('<h1 class="main-title">AI Agents, Human Insight.<br>Smarter Investing.</h1>', unsafe_allow_html=True)
            st.markdown('<p class="sub-headline">AlphaMesh uses a team of specialized AI agents to analyze markets, debate strategies, and deliver clear, actionable investment intelligence. No code, just results.</p>', unsafe_allow_html=True)
            if st.button("Get Started for Free", key="hero_get_started"):
                set_auth_mode_and_show_modal('Sign Up')
        with col2:
            st.markdown('<div class="product-animation-placeholder"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def _display_social_proof_section():
    """Displays the 'Powered By' logos section."""
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="trusted-by-container">
                <span class="trusted-by-text">POWERED BY LEADING-EDGE TECHNOLOGY</span>
                <div class="logos-container">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Google-Gemini-Logo.svg" alt="Google Gemini" title="Google Gemini">
                    <img src="https://python.langchain.com/assets/images/brand_square-7c0f18f6738d8f086878b19a12822a16.png" alt="LangChain" title="LangChain">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b3/IBKR_logo.svg" alt="Interactive Brokers" title="Interactive Brokers">
                    <img src="https://logotyp.us/files/yahoo.svg" alt="Yahoo Finance" title="Yahoo Finance" style="height: 30px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def _display_how_it_works_section():
    """Displays the 'How It Works' section with three info cards."""
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-headline">From Market Noise to Actionable Thesis in 3 Steps</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="large")

        cards_data = [
            {"icon": "1.", "title": "Connect Your Goals", "text": "Tell AlphaMesh your investment style and risk tolerance. Your AI team calibrates to your specific needs."},
            {"icon": "2.", "title": "Agents Analyze & Debate", "text": "Our AI agents—a Risk Analyst, a News Sentinel, a Fundamental Analyst—gather data and debate the best course of action."},
            {"icon": "3.", "title": "Receive Your Briefing", "text": "Get a clear, synthesized report with bull and bear cases, key data points, and a final recommendation. All transparent, all for you."}
        ]

        for col, data in zip([col1, col2, col3], cards_data):
            with col:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="info-card-icon">{data['icon']}</div>
                        <h3>{data['title']}</h3>
                        <p>{data['text']}</p>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def _display_features_section():
    """Displays key features with an alternating text/image layout."""
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-headline">An Unfair Advantage, Built For You</h2>', unsafe_allow_html=True)

        st.write("<br>", unsafe_allow_html=True)
        # Feature 1: Text on left, image on right
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.subheader("The Daily Briefing")
            st.write("Start your day with a mission-critical summary. Your agent team works 24/7, so you wake up with insights, not just alerts. Get a curated look at your portfolio's overnight news, market sentiment, and the top opportunity identified by the mesh.")
        with col2:
            st.image("https://placehold.co/600x400/1a1a1a/FFFFFF?text=Dashboard+Mockup", width='stretch')

        st.write("<br><br><br>", unsafe_allow_html=True) # Spacer

        # Feature 2: Image on left, text on right
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.image("https://placehold.co/600x400/1a1a1a/FFFFFF?text=Transparent+Analysis", width='stretch')
        with col2:
            st.subheader("Deep Dive with Full Transparency")
            st.write("Every insight is backed by transparent reasoning. Understand the 'why' behind each recommendation by exploring the Bull Case from the Fundamental Agent, the Bear Case from the Risk Manager, and citations from the News Sentinel.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def _display_final_cta():
    """Displays the final call-to-action section."""
    with st.container():
        st.markdown('<div class="section-container cta-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-headline">The Future of Investing is Collaborative Intelligence.</h2>', unsafe_allow_html=True)
        st.markdown('<p class="cta-subtext">Stop guessing. Start making data-driven decisions with your personal AI investment committee.</p>', unsafe_allow_html=True)
        
        # Center the button
        _, col, _ = st.columns([1, 0.6, 1])
        with col:
            if st.button("Sign Up Now - It's Free", key="final_cta_button", use_container_width=True):
                set_auth_mode_and_show_modal('Sign Up')
        st.markdown('</div>', unsafe_allow_html=True)

def _display_footer():
    """Displays the page footer."""
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("© 2024 AlphaMesh. All rights reserved.")
        with c2:
            st.markdown('<div style="text-align: right;"><a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_landing_page():
    """
    Renders all the sections of the landing page in order.
    """
    _display_header()
    # The main content div wraps all sections except the header and footer for correct padding
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    _display_hero_section()
    _display_social_proof_section()
    _display_how_it_works_section()
    _display_features_section()
    _display_final_cta()
    _display_footer()
    st.markdown('</div>', unsafe_allow_html=True)