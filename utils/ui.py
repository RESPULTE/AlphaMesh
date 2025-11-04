# utils/ui.py

import streamlit as st

def load_css(file_name: str):
    """
    Loads a CSS file into the Streamlit application.
    """
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_name}")