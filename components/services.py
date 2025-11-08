import streamlit as st
import logging

from langchain_neo4j import Neo4jGraph
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

# --- Constants ---
CHROMA_COLLECTION = "user_profile_short_term"
CHROMA_PERSIST_DIR = "./chroma_db_user"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash-lite" 

# Using Streamlit's caching to initialize services only once
@st.cache_resource
def get_llm():
    """Initializes and returns the Generative AI model."""
    try:
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            google_api_key=st.secrets["google"]["api_key"], 
            temperature=0
        )
    except Exception as e:
        logger.error(f"Failed to initialize Google LLM: {e}")
        st.error("Fatal: Could not connect to Google Generative AI. Please check your API key.")
        return None

@st.cache_resource
def get_graph():
    """Initializes and returns the Neo4j graph connection."""
    try:
        return Neo4jGraph(
            url=st.secrets["neo4j"]["url"],
            username=st.secrets["neo4j"]["username"],
            password=st.secrets["neo4j"]["password"]
        )
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j Graph: {e}")
        st.error("Fatal: Could not connect to Neo4j. Please check your connection details.")
        return None

@st.cache_resource
def get_vector_store():
    """Initializes and returns the Chroma vector store."""
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=st.secrets["google"]["api_key"]
        )
        return Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embedding_function,
            persist_directory=CHROMA_PERSIST_DIR
        )
    except Exception as e:
        logger.error(f"Failed to initialize Chroma Vector Store: {e}")
        st.error("Fatal: Could not connect to ChromaDB.")
        return None