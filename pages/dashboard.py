import os
import streamlit as st
import uuid
# Import the new RAG class
from components.user_profile_rag import UserProfileRAG

st.set_page_config(
    page_title="AlphaMesh Dashboard",
    layout="wide"
)

st.title("Welcome to your AlphaMesh Dashboard")
st.markdown("This is your personalized investment intelligence hub. I learn about you as we chat to provide better answers.")

# --- Integration Start ---

# 1. Ensure a unique user ID for the session
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# 2. Initialize the RAG agent for the user
# st.cache_resource is used to create and cache the object across reruns.
@st.cache_resource
def get_rag_agent(user_id):
    try:
        return UserProfileRAG(user_id=user_id)
    except ConnectionError as e:
        st.error(f"Could not connect to backend services: {e}")
        return None

rag_agent = get_rag_agent(st.session_state.user_id)

# --- Integration End ---


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask AlphaMesh anything..."):
    # Store user input
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display a placeholder for the model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # If the agent failed to initialize, show an error
            if rag_agent is None:
                st.error("The assistant is currently unavailable. Please check the connection settings.")
            else:
                # 3. Call the get_augmented_response method
                response = rag_agent.get_augmented_response(prompt)
                st.markdown(response)

    # Add assistant message to history
    if rag_agent:
        st.session_state.messages.append({"role": "assistant", "content": response})