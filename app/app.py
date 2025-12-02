import os
import sys
import uuid

import streamlit as st

# Add project root to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.storage import ChatStorage
from core.agents import OrchestratorAgent

# --- Configuration ---
PAGE_TITLE = "Financial Research Agent"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")


# --- Initialization (Singleton Pattern) ---
@st.cache_resource
def get_storage():
    return ChatStorage()


@st.cache_resource
def get_agent():
    # Initialize the agent once to avoid reloading models on every interaction
    return OrchestratorAgent()


db = get_storage()
agent = get_agent()

# --- User Identification Logic ---
# In the future, replace this with a real login system.
# For now, we allow the user to simulate a login or generate a random ID.
with st.sidebar:
    st.title("User Settings")

    # Check if a user_id is already in session, otherwise generate one
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())[:8]

    # Allow user to "login" by typing a specific ID to load past history
    user_input_id = st.text_input(
        "User ID (Simulate Login)", value=st.session_state["user_id"]
    )

    if user_input_id != st.session_state["user_id"]:
        st.session_state["user_id"] = user_input_id
        st.rerun()  # Refresh to load new user's history

    st.caption(f"Current Session ID: {st.session_state['user_id']}")

    if st.button("Clear History"):
        db.clear_history(st.session_state["user_id"])
        st.rerun()

# --- Main Interface ---
st.title(f"🤖 {PAGE_TITLE}")

# 1. Load History from Database
# We fetch history from DB every time the app reruns to ensure consistency
history = db.get_history(st.session_state["user_id"])

# 2. Display Chat History
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Handle New Input
if prompt := st.chat_input("Ask about market trends, specific stocks, or news..."):

    # A. Display User Message Immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Save User Message to DB
    db.add_message(st.session_state["user_id"], "user", prompt)

    # C. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing market data..."):
            try:
                # Run the orchestrator
                response = agent.run(prompt)
                st.markdown(response)

                # D. Save Assistant Response to DB
                db.add_message(st.session_state["user_id"], "assistant", response)
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                # Optionally log errors to DB too
                db.add_message(st.session_state["user_id"], "assistant", error_msg)
