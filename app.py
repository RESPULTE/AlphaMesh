import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AlphaMesh",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Helper Functions ---
def load_css(file_name):
    """Function to load a local CSS file."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_name}")

# --- Modal & Authentication ---
def display_auth_modal():
    """Renders the authentication modal overlay."""
    
    # Determine which toggle button is active
    active_signup_class = "active" if st.session_state.auth_mode == 'Sign Up' else ""
    active_login_class = "active" if st.session_state.auth_mode == 'Login' else ""

    modal_html = f"""
    <div class="modal-overlay">
        <div class="modal-content">
            <div id="close-button-container"></div>
            
            <div class="modal-header">
                <h2>{ 'Create your account' if st.session_state.auth_mode == 'Sign Up' else 'Welcome back' }</h2>
                <p>to continue to AlphaMesh</p>
            </div>

            <div class="auth-toggle-container">
                <div id="signup-toggle-container"></div>
                <div id="login-toggle-container"></div>
            </div>

            <div id="google-signin-container"></div>
            
            <div class="modal-footer">
                By continuing, you agree to our <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>.
            </div>
        </div>
    </div>
    """
    inject_modal_html(modal_html)

    # Use Streamlit buttons within containers for callbacks
    with st.container():
        # This is a hacky way to place Streamlit elements in specific divs
        # We target the containers by ID and then use columns to place the buttons
        # Close Button
        close_container = st.container()
        if close_container.button("X", key="close_modal"):
            st.session_state.show_modal = False
            st.rerun()

        # Auth Toggles
        toggle_cols = st.columns(2)
        if toggle_cols[0].button("Sign Up", key="signup_toggle", use_container_width=True):
            st.session_state.auth_mode = 'Sign Up'
            st.rerun()
        if toggle_cols[1].button("Login", key="login_toggle", use_container_width=True):
            st.session_state.auth_mode = 'Login'
            st.rerun()
        
        # Google Sign-in
        if st.button("Sign in with Google", key="google_signin", use_container_width=True):
            # Placeholder for Google OAuth logic
            pass


def set_auth_mode_and_show_modal(mode):
    """Callback to set the auth mode and show the modal."""
    st.session_state.auth_mode = mode
    st.session_state.show_modal = True

# --- Landing Page Sections ---
def display_header():
    """Displays the sticky header with Logo and Login/Sign Up buttons."""
    st.markdown('<div class="header"><div class="header-content">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="logo">AlphaMesh</div>', unsafe_allow_html=True)
    with col2:
        cols = st.columns([1, 1])
        if cols[0].button("Login", key="header_login", use_container_width=True):
            set_auth_mode_and_show_modal('Login')
        if cols[1].button("Sign Up", key="header_signup", use_container_width=True):
            set_auth_mode_and_show_modal('Sign Up')

    st.markdown('</div></div>', unsafe_allow_html=True)


def display_hero_section():
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

# (Other display functions remain unchanged: display_social_proof_section, display_how_it_works_section, etc.)
def display_social_proof_section():
    """Displays the 'Trusted By' logos."""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="trusted-by-container">
            <span class="trusted-by-text">POWERED BY LEADING-EDGE TECHNOLOGY</span>
            <div class="logos-container">
                <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Google-Gemini-Logo.svg" alt="Google Gemini">
                <img src="https://python.langchain.com/assets/images/brand_square-7c0f18f6738d8f086878b19a12822a16.png" alt="LangChain">
                <img src="https://upload.wikimedia.org/wikipedia/commons/b/b3/IBKR_logo.svg" alt="Interactive Brokers">
                <img src="https://cdn-icons-png.flaticon.com/512/5969/5969192.png" alt="Yahoo Finance">
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_how_it_works_section():
    """Displays the 'How It Works' section with info cards."""
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


def display_features_section():
    """Displays key features with an alternating layout."""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-headline">An Unfair Advantage, Built For You</h2>', unsafe_allow_html=True)

    # Feature 1: Text on left, image on right
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("The Daily Briefing")
        st.write("Start your day with a mission-critical summary. Your agent team works 24/7, so you wake up with insights, not just alerts. Get a curated look at your portfolio's overnight news, market sentiment, and the top opportunity identified by the mesh.")
    with col2:
        st.image("https://placehold.co/600x400/111111/FFFFFF?text=Dashboard+Mockup", use_column_width=True)

    st.write("<br><br>", unsafe_allow_html=True) # Spacer

    # Feature 2: Image on left, text on right
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image("https://placehold.co/600x400/111111/FFFFFF?text=Transparent+Analysis", use_column_width=True)
    with col2:
        st.subheader("Deep Dive with Full Transparency")
        st.write("Every insight is backed by transparent reasoning. Understand the 'why' behind each recommendation by exploring the Bull Case from the Fundamental Agent, the Bear Case from the Risk Manager, and citations from the News Sentinel.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_final_cta():
    """Displays the final call-to-action section."""
    st.markdown('<div class="section-container cta-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-headline">The Future of Investing is Collaborative Intelligence.</h2>', unsafe_allow_html=True)
    st.markdown('<p class="cta-subtext">Stop guessing. Start making data-driven decisions with your personal AI investment committee.</p>', unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 0.6, 1])
    with col:
        if st.button("Sign Up Now - It's Free", key="final_cta_button", use_container_width=True):
            set_auth_mode_and_show_modal('Sign Up')
    st.markdown('</div>', unsafe_allow_html=True)

def display_footer():
    """Displays the page footer."""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns([1,1])
    with c1:
        st.write("© 2024 AlphaMesh. All rights reserved.")
    with c2:
        st.markdown('<div style="text-align: right;"><a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def inject_modal_html(html):
    """
    Injects modal HTML at the very end of Streamlit's root app DOM
    so it overlays the entire page properly.
    """
    js = f"""
    <script>
    const modal = `{html}`;
    const root = window.parent.document.querySelector('.stApp');
    if (root && !root.querySelector('.modal-overlay')) {{
        root.insertAdjacentHTML('beforeend', modal);
    }}
    </script>
    """
    st.components.v1.html(js, height=0)


# --- Main Application ---
def main():
    load_css("styles/style.css")

    # Initialize session state
    if 'show_modal' not in st.session_state:
        st.session_state.show_modal = False
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'Sign Up'

    # Display page content
    display_header()
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    display_hero_section()
    display_social_proof_section()
    display_how_it_works_section()
    display_features_section()
    display_final_cta()
    display_footer()
    st.markdown('</div>', unsafe_allow_html=True)

    # Conditionally display the modal on top of everything
    if st.session_state.show_modal:
        display_auth_modal()

if __name__ == "__main__":
    main()