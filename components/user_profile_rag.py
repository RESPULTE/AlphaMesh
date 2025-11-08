# components/user_profile_rag.py
import streamlit as st
import logging

# Corrected imports for newer LangChain versions
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Setup logging to see the generated Cypher queries in your terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserProfileRAG:
    """
    Manages the logic for a Retrieval-Augmented Generation system
    that learns a user's profile over time, with an advanced function
    to consolidate memories into a knowledge graph.
    """
    def __init__(self, user_id: str):
        """
        Initializes the RAG system for a specific user session.
        Args:
            user_id: A unique identifier for the current user.
        """
        self.user_id = user_id

        # --- 1. Initialize Connections ---
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=st.secrets["google"]["api_key"],
                temperature=0,
            )
            self.graph = Neo4jGraph(
                url=st.secrets["neo4j"]["url"],
                username=st.secrets["neo4j"]["username"],
                password=st.secrets["neo4j"]["password"]
            )
            embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=st.secrets["google"]["api_key"]
            )
            # Use a persistent directory for Chroma to store data between app runs
            self.vector_store = Chroma(
                collection_name="user_profile_short_term",
                embedding_function=embedding_function,
                persist_directory="./chroma_db_user"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize services. Check credentials in st.secrets. Error: {e}")

    def _extract_and_store_user_profile(self, user_input: str):
        """
        Extracts profile info, stores raw text for short-term memory,
        and calls the function to consolidate structured info for long-term memory.
        """
        # Immediately store raw input for short-term context/recall
        self.vector_store.add_documents(
            [Document(page_content=user_input, metadata={"user_id": self.user_id})]
        )
        
        # Intelligently consolidate important facts from the input into the graph
        self.consolidate_memory_to_graph(user_input)

    def consolidate_memory_to_graph(self, user_input: str):
        """
        NEW FUNCTION: Intelligently extracts entities and relationships from user
        input and stores them in the Neo4j graph database as long-term memory.
        It uses an LLM to generate Cypher queries directly.
        """
        cypher_generation_template = """
        You are an expert data modeler. Your task is to extract user profile information
        from the user's input and convert it into Cypher MERGE statements for a Neo4j graph.
        
        The graph schema is as follows:
        - User nodes: `(:User {{id: string, name: string}})`
        - Interest nodes: `(:Interest {{name: string}})`
        - Profession nodes: `(:Profession {{name: string}})`
        - Goal nodes: `(:Goal {{description: string}})`
        
        Relationships:
        - A User is interested in an Interest: `(u:User)-[:INTERESTED_IN]->(i:Interest)`
        - A User works as a Profession: `(u:User)-[:WORKS_AS]->(p:Profession)`
        - A User has a Goal: `(u:User)-[:HAS_GOAL]->(g:Goal)`
        
        Instructions:
        1. Analyze the user input to identify core, stable facts (name, interests, profession, stated goals).
        2. Use the user ID `{user_id}` to identify the user node.
        3. Generate only Cypher `MERGE` statements to create or update the graph. This is crucial to avoid duplicates.
        4. If the input contains no new, stable profile information, output the string "NO_STATEMENTS".
        5. Separate multiple statements with a semicolon ';'.
        
        Example Input: "My name is Bob and I work as an engineer. I'm really interested in machine learning."
        Example Output:
        MERGE (u:User {{id: '{user_id}'}}) SET u.name = 'Bob';
        MERGE (u:User {{id: '{user_id}'}}) MERGE (p:Profession {{name: 'engineer'}}) MERGE (u)-[:WORKS_AS]->(p);
        MERGE (u:User {{id: '{user_id}'}}) MERGE (i:Interest {{name: 'machine learning'}}) MERGE (u)-[:INTERESTED_IN]->(i)

        User Input: "{user_input}"
        Your Cypher Statements:
        """

        cypher_prompt = PromptTemplate.from_template(cypher_generation_template)
        cypher_chain = cypher_prompt | self.llm

        try:
            generated_cypher = cypher_chain.invoke({
                "user_id": self.user_id,
                "user_input": user_input
            }).content

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

    def get_augmented_response(self, user_input: str) -> str:
        """
        Main method to get a personalized response. It learns from the user's
        input and then generates an augmented response.
        """
        # 1. Learn from the user's latest input
        self._extract_and_store_user_profile(user_input)

        # 2. Retrieve context from memory
        short_term_context = "No recent context available."
        long_term_context = "No long-term profile available."
        try:
            short_term_docs = self.vector_store.similarity_search(user_input, k=3, filter={"user_id": self.user_id})
            if short_term_docs:
                short_term_context = "\n".join([doc.page_content for doc in short_term_docs])
        except Exception as e:
            logger.warning(f"Could not retrieve from vector store: {e}")

        try:
            # Use the more powerful GraphCypherQAChain to query the graph in natural language
            cypher_chain = GraphCypherQAChain.from_llm(graph=self.graph, llm=self.llm, validate_cypher=True)
            graph_query = f"What are the known interests, name, profession, and goals for the user with id '{self.user_id}'? Summarize the findings."
            result = cypher_chain.invoke({"query": graph_query}, allow_dangerous_requests=True)
            long_term_context = result.get('result', 'Failed to retrieve from graph.')
        except Exception as e:
            logger.warning(f"Could not retrieve from graph database: {e}")

        # 3. Generate Augmented Response
        augmented_prompt = PromptTemplate.from_template(
            "You are AlphaMesh, a personalized investment intelligence assistant. "
            "Use the following user profile to personalize your response. "
            "Do not explicitly mention the profile; just use it to tailor your answer naturally.\n\n"
            "--- User Profile ---\n"
            "Recent Conversation Topics: {short_term_context}\n"
            "Long-Term User Facts: {long_term_context}\n"
            "--------------------\n\n"
            "User's Question: {user_input}\n\n"
            "Your Personalized Answer:"
        )
        generation_chain = augmented_prompt | self.llm
        response = generation_chain.invoke({
            "short_term_context": short_term_context,
            "long_term_context": long_term_context,
            "user_input": user_input
        })

        return response.content