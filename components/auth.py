# components/auth.py

import streamlit as st

def set_auth_mode_and_show_modal(mode: str):
    """
    Callback function to set the authentication mode ('Login' or 'Sign Up')
    and trigger the native st.dialog to be displayed on the next rerun.
    """
    st.session_state.auth_mode = mode
    st.session_state.show_auth_dialog = True
    st.rerun()  # Trigger a rerun so the dialog appears on next run

def authenticate_user(username: str = None):
    """
    Placeholder function for actual Google OAuth/Login logic.
    Sets the session state to authenticated and triggers page switch.
    """
    # --- ⚠️ FUTURE IMPLEMENTATION CHECK (Simulated) ---
    # In the future, this function will call an authentication service.
    # For now, we simulate a successful login.
    
    auth_successful = True # st.session_state.auth_mode == 'Login' or username == "test@alphamesh.ai"

    if auth_successful:
        # Clear the dialog state
        st.session_state.is_authenticated = True
        st.session_state.show_auth_dialog = False
        
        st.toast(f"Welcome, {username if username else 'User'}!", icon="👋")
        
        # Redirect to the dashboard page file (must be in the 'pages/' folder)
        st.switch_page("pages/dashboard.py")
        
    else:
        st.error("Authentication failed. Please check your credentials.", icon="🚨")

# ✅ Define dialog function using decorator
@st.dialog("Authentication", width="small")
def auth_dialog():
    """
    Renders the internal content of the authentication dialog.
    """
    mode = st.session_state.auth_mode
    
    st.subheader(f"{ 'Create your account' if mode == 'Sign Up' else 'Welcome back' }")
    st.markdown(f"<p style='color: var(--text-color-light);'>to { 'continue' if mode == 'Login' else 'get started' } with AlphaMesh</p>", unsafe_allow_html=True)
    
    st.markdown("---") # Visual separator

    # --- Simple Email/Password Form ---
    if mode == 'Sign Up':
        st.text_input("Email", placeholder="you@example.com", key="dialog_email")
        st.text_input("Password", type="password", key="dialog_password_1")
        st.text_input("Confirm Password", type="password", key="dialog_password_2")
    else:
        st.text_input("Email", placeholder="you@example.com", key="dialog_email_login")
        st.text_input("Password", type="password", key="dialog_password_login")
        
    
    if st.button(mode, key="dialog_form_submit", use_container_width=True, type="primary"):
        # Placeholder for form-based login/signup logic
        email_key = "dialog_email" if mode == 'Sign Up' else "dialog_email_login"
        current_email = st.session_state.get(email_key, 'Placeholder User')
        authenticate_user(current_email)

    st.markdown('<div style="text-align: center; margin: 1.5rem 0 1.5rem 0; color: var(--text-color-light);">— OR —</div>', unsafe_allow_html=True)
    
    # --- Google Sign-in Button (calls the authentication logic) ---
    if st.button("Sign in with Google", key="dialog_google_signin", use_container_width=True):
        authenticate_user('Google User') # Pass a generic user name

    # --- Footer ---
    st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: var(--text-color-light); margin-top: 2rem;">
            By continuing, you agree to our <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>.
        </div>
    """, unsafe_allow_html=True)


def handle_auth_dialog():
    """
    Checks session state and shows the dialog if triggered.
    """
    if st.session_state.get("show_auth_dialog", False) and not st.session_state.is_authenticated:
        # Show the dialog only if triggered and the user is not yet logged in
        auth_dialog()