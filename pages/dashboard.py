# pages/dashboard.py
import streamlit as st
import uuid
from components.user_profile_rag import UserProfileRAG

st.set_page_config(
    page_title="AlphaMesh Dashboard",
    layout="wide"
)

st.title("Welcome to your AlphaMesh Dashboard")
st.markdown("This is your personalized investment intelligence hub. I can learn about your interests and goals over time.")

# --- Session State Management ---

# 2. Initialize the RAG agent for the user
# Caching the agent to avoid re-initializing on every script rerun
@st.cache_resource
def get_rag_agent(user_id):
    try:
        return UserProfileRAG(user_id=user_id)
    except ConnectionError as e:
        st.error(f"Fatal Error: Could not connect to backend services. Please check your API keys and database connections in st.secrets. Details: {e}")
        return None

rag_agent = get_rag_agent(st.session_state.user_id)

# 3. Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask AlphaMesh anything... (e.g., 'My name is Jane and I'm interested in tech stocks')"):
    if rag_agent is None:
        st.error("The assistant is currently offline due to a connection issue.")
    else:
        # Store and display user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the personalized response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # The single, powerful call to our RAG agent
                response = rag_agent.get_augmented_response(prompt)
                st.markdown(response)

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})