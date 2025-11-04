import streamlit as st
import base64

# --- Page Configuration ---
# Must be the first Streamlit command.
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

def get_image_as_base64(path):
    """Function to encode a local image to base64."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# --- Landing Page Sections ---

def display_hero_section():
    """Displays the main hero section with title, subtitle, and CTA."""
    st.markdown('<a id="home"></a>', unsafe_allow_html=True) # Anchor for navigation
    
    # Using columns for layout: Left for text, Right for visual
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<h1 class="main-title">AlphaMesh: Your Personal AI Investment Committee.</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <p class="sub-headline">
            Harness a collaborative team of specialized AI agents, powered by Google Gemini, 
            to analyze real-time market data, news, and sentiment—delivering personalized 
            investment strategies, not just data points.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        # CTA Form
        email_col, button_col = st.columns([2, 1])
        with email_col:
            st.text_input(
                "email",
                placeholder="Enter your email to request early access",
                label_visibility="collapsed"
            )
        with button_col:
            st.button("Request Early Access")
            
    with col2:
        # Placeholder for the abstract animation/visual
        # Using a simple SVG as a placeholder to represent the mesh network
        mesh_svg = """
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#00BFFF;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#00F5D4;stop-opacity:1" />
                </linearGradient>
            </defs>
            <g fill="none" stroke="url(#grad1)" stroke-width="0.5">
                <circle cx="50" cy="50" r="10"/>
                <circle cx="20" cy="20" r="5"/>
                <circle cx="80" cy="20" r="5"/>
                <circle cx="20" cy="80" r="5"/>
                <circle cx="80" cy="80" r="5"/>
                <circle cx="50" cy="15" r="4"/>
                <circle cx="50" cy="85" r="4"/>
                <circle cx="15" cy="50" r="4"/>
                <circle cx="85" cy="50" r="4"/>
                <path d="M 50 50 L 20 20 M 50 50 L 80 20 M 50 50 L 20 80 M 50 50 L 80 80"/>
                <path d="M 50 50 L 50 15 M 50 50 L 50 85 M 50 50 L 15 50 M 50 50 L 85 50"/>
                <path d="M 20 20 L 50 15 L 80 20 L 85 50 L 80 80 L 50 85 L 20 80 L 15 50 Z"/>
            </g>
        </svg>
        """
        st.markdown(mesh_svg, unsafe_allow_html=True)


def display_problem_solution_section():
    """Displays the problem & solution section with three info cards."""
    st.markdown('<h2 class="section-headline">Stop Drowning in Data. Start Making Decisions.</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>📉 Information Overload</h3>
                <p>The market is a firehose of conflicting news, complex reports, and endless charts. Traditional tools just add to the noise.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>⬛ The "Black Box" Problem</h3>
                <p>Other AI tools give you recommendations with no explanation. How can you trust a strategy if you don't understand the 'why'?</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>💡 Your AI-Powered Solution</h3>
                <p>AlphaMesh orchestrates a team of AI specialists who analyze, debate, and synthesize a clear, actionable thesis—just for you.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_features_section():
    """Displays key features with mockups."""
    st.markdown('<h2 class="section-headline">An Unfair Advantage, Built Just for You.</h2>', unsafe_allow_html=True)

    # Feature 1
    st.write("---")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("The Daily Briefing")
        st.write("""
        Start your day with a mission-critical summary. Your agent team works 24/7, 
        so you wake up with insights, not just alerts. Get a curated look at your portfolio's
        overnight news, market sentiment, and the top opportunity identified by the mesh.
        """)
    with col2:
        st.image("https://placehold.co/600x400/0A192F/FFFFFF?text=Dashboard+Mockup", use_column_width=True)

    # Feature 2
    st.write("---")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image("https://placehold.co/600x400/0A192F/FFFFFF?text=Transparent+Analysis", use_column_width=True)
    with col2:
        st.subheader("Deep Dive with Full Transparency")
        st.write("""
        Every insight is backed by transparent reasoning. Understand the 'why' behind each 
        recommendation by exploring the Bull Case from the Fundamental Agent, the Bear Case 
        from the Risk Manager, and citations from the News Sentinel.
        """)

def display_tech_stack_section():
    """Displays the technology stack logos."""
    st.markdown('<h2 class="section-headline">Built on a Foundation of Excellence</h2>', unsafe_allow_html=True)
    
    # Using columns for logos
    col1, col2, col3, col4 = st.columns(4)
    logos = {
        "Google Gemini": "https://upload.wikimedia.org/wikipedia/commons/2/2d/Google-Gemini-Logo.svg",
        "LangChain": "https://python.langchain.com/assets/images/brand_square-7c0f18f6738d8f086878b19a12822a16.png",
        "Interactive Brokers": "https://upload.wikimedia.org/wikipedia/commons/b/b3/IBKR_logo.svg",
        "yfinance": "https://avatars.githubusercontent.com/u/36422238?s=200&v=4" # Placeholder yfinance / yahoo logo
    }
    
    cols = [col1, col2, col3, col4]
    for idx, (name, url) in enumerate(logos.items()):
        with cols[idx]:
            st.markdown(
                f'<div style="text-align: center;"><img src="{url}" class="tech-logo" alt="{name} logo"><p>{name}</p></div>',
                unsafe_allow_html=True
            )

def display_final_cta():
    """Displays the final call-to-action section."""
    st.write("---")
    st.markdown('<h2 class="section-headline">The Future of Investing is Collaborative Intelligence.</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="text-align: center; max-width: 600px; margin: auto; color: #bdc5d4;">
        Be the first to experience the power of a multi-agent AI investment committee. 
        Join the AlphaMesh waitlist today. Limited spots available for our initial beta.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    # Centering the final CTA form
    _, form_col, _ = st.columns([1, 1.5, 1])
    with form_col:
        st.text_input(
            "final_email",
            placeholder="Enter your email",
            label_visibility="collapsed"
        )
        st.button("Secure My Spot", key="final_cta_button")

def display_footer():
    """Displays the page footer."""
    st.write("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("© 2024 AlphaMesh. All rights reserved.")
    with c2:
        st.write("[Privacy Policy](https://example.com) | [Terms of Service](https://example.com)")
    with c3:
        st.write("[Twitter](https://x.com) | [LinkedIn](https://linkedin.com)")

# --- Main Application ---
def main():
    """The main function that orchestrates the app's layout."""
    
    # Load custom CSS from the file
    load_css("style.css")

    # Display all the sections
    with st.container():
        display_hero_section()
    
    with st.container():
        display_problem_solution_section()
        
    with st.container():
        display_features_section()

    with st.container():
        display_tech_stack_section()

    with st.container():
        display_final_cta()
        display_footer()


if __name__ == "__main__":
    main()