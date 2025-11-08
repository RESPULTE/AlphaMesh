# pages/dashboard.py
import streamlit as st
from components.rag_handler import UserProfileRAG, generate_augmented_response
from components.services import get_llm, get_graph, get_vector_store

st.set_page_config(
    page_title="AlphaMesh Dashboard",
    layout="wide"
)

st.title("Welcome to your AlphaMesh Dashboard")
st.markdown("This is your personalized investment intelligence hub. I can learn about your interests and goals over time.")

# --- Service and Agent Initialization ---

# Initialize all backend services
llm = get_llm()
graph = get_graph()
vector_store = get_vector_store()

# Check if services initialized correctly before proceeding
if not all([llm, graph, vector_store]):
    st.error("One or more backend services failed to initialize. The application cannot continue.")
    st.stop() # Halts the script execution

# Caching the RAG handler to avoid re-initializing on every script rerun for the same user
@st.cache_resource
def get_rag_handler(user_id, _llm, _graph, _vector_store):
    """Factory function to create and cache the RAG handler per user."""
    return UserProfileRAG(
        user_id=user_id,
        llm=_llm,
        graph=_graph,
        vector_store=_vector_store
    )

# Assume st.session_state.user_id is set upon login
if 'user_id' not in st.session_state:
    st.warning("Please log in to use the dashboard.")
    st.stop()

rag_handler = get_rag_handler(st.session_state.user_id, llm, graph, vector_store)

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask AlphaMesh anything... (e.g., 'My name is Jane and I'm interested in tech stocks')"):
    # Store and display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the personalized response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Get context from the RAG handler
            # This step also updates the user's memory profile
            context = rag_handler.get_context_for_prompt(prompt)

            # 2. Invoke the LLM with the prepared context
            response = generate_augmented_response(
                llm=llm,
                short_term_context=context["short_term_context"],
                long_term_context=context["long_term_context"],
                user_input=prompt
            )
            rag_handler.update_memories_from_turn(prompt, response)
            
            st.markdown(response)

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response})