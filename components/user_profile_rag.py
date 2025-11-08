# components/user_profile_rag.py
import streamlit as st
import logging

# Corrected imports for newer LangChain versions
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from . import prompts  # NEW: Import the dedicated prompts module

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserProfileRAG:
    """
    Manages the logic for a Retrieval-Augmented Generation system
    that learns a user's profile over time, with an advanced function
    to consolidate memories into a knowledge graph.
    """
    # NEW: Centralized configuration as class constants
    _CHROMA_COLLECTION = "user_profile_short_term"
    _CHROMA_PERSIST_DIR = "./chroma_db_user"
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._initialize_services()

    # REFACTORED: Initialization logic is moved to a dedicated method
    def _initialize_services(self):
        """Initializes connections to external services (LLM, DBs)."""
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=st.secrets["google"]["api_key"], temperature=0)
            self.graph = Neo4jGraph(url=st.secrets["neo4j"]["url"], username=st.secrets["neo4j"]["username"], password=st.secrets["neo4j"]["password"])
            embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=st.secrets["google"]["api_key"])
            self.vector_store = Chroma(
                collection_name=self._CHROMA_COLLECTION,
                embedding_function=embedding_function,
                persist_directory=self._CHROMA_PERSIST_DIR
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize services. Check credentials in st.secrets. Error: {e}")

    # REFACTORED: Renamed for clarity and to better reflect its purpose
    def _update_memory(self, user_input: str):
        """
        Updates short-term (vector) and long-term (graph) memories
        based on the user's input.
        """
        # Store raw input for short-term recall
        self.vector_store.add_documents([Document(page_content=user_input, metadata={"user_id": self.user_id})])
        
        # Consolidate important facts into the graph for long-term memory
        self._consolidate_memory_to_graph(user_input)

    # REFACTORED: Now uses the abstracted prompt template
    def _consolidate_memory_to_graph(self, user_input: str):
        """
        Extracts entities and relationships from input and stores them in Neo4j.
        """
        # MOVED: The large template string is now imported from prompts.py
        cypher_prompt = PromptTemplate.from_template(prompts.CYPHER_GENERATION_TEMPLATE)
        cypher_chain = cypher_prompt | self.llm

        try:
            generated_cypher = cypher_chain.invoke({"user_id": self.user_id, "user_input": user_input}).content
            if "NO_STATEMENTS" in generated_cypher:
                logger.info("No new profile information to store in the graph.")
                return

            queries = [q.strip() for q in generated_cypher.split(';') if q.strip()]
            for query in queries:
                logger.info(f"Executing Cypher: {query}")
                self.graph.query(query)
            
            if queries:
                logger.info(f"Successfully stored {len(queries)} new facts in the knowledge graph.")
        except Exception as e:
            logger.error(f"Error processing or executing Cypher for long-term memory: {e}")

    # NEW: Dedicated method for short-term memory retrieval
    def _retrieve_short_term_context(self, user_input: str) -> str:
        """Retrieves relevant recent conversation topics from the vector store."""
        try:
            docs = self.vector_store.similarity_search(user_input, k=3, filter={"user_id": self.user_id})
            return "\n".join([doc.page_content for doc in docs]) if docs else "No recent context available."
        except Exception as e:
            logger.warning(f"Could not retrieve from vector store: {e}")
            return "No recent context available."

    # NEW: Dedicated method for long-term memory retrieval
    def _retrieve_long_term_context(self) -> str:
        """Retrieves summarized user facts from the knowledge graph."""
        try:
            cypher_chain = GraphCypherQAChain.from_llm(graph=self.graph, llm=self.llm, validate_cypher=True)
            graph_query = f"What are the known interests, name, profession, and goals for the user with id '{self.user_id}'? Summarize the findings."
            result = cypher_chain.invoke({"query": graph_query})
            return result.get('result', 'Failed to retrieve from graph.')
        except Exception as e:
            logger.warning(f"Could not retrieve from graph database: {e}")
            return "No long-term profile available."

    # REFACTORED: Main method now orchestrates calls to smaller, focused methods
    def get_augmented_response(self, user_input: str) -> str:
        """
        Main method to get a personalized response. It learns from the user's
        input and then generates an augmented response.
        """
        # 1. Learn from the user's latest input
        self._update_memory(user_input)

        # 2. Retrieve context from both memory stores
        short_term_context = self._retrieve_short_term_context(user_input)
        long_term_context = self._retrieve_long_term_context()


        # 3. Generate Augmented Response
        # MOVED: The large template string is now imported from prompts.py
        augmented_prompt = PromptTemplate.from_template(prompts.AUGMENTED_RESPONSE_TEMPLATE)
        generation_chain = augmented_prompt | self.llm
        
        response = generation_chain.invoke({
            "short_term_context": short_term_context,
            "long_term_context": long_term_context,
            "user_input": user_input
        })

        return response.content