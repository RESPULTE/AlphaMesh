# pages/dashboard.py

import streamlit as st

st.set_page_config(
    page_title="AlphaMesh Dashboard",
    layout="wide"
)

st.title("Welcome to your AlphaMesh Dashboard")
st.markdown("This is your personalized investment intelligence hub. Features will be added here soon.")

# You can add a link back to the home page if desired
if st.button("Go back to Home"):
    st.switch_page("app.py")