# components/user_profile_rag.py
import streamlit as st

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, embeddings

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

class UserProfileRAG:
    """
    Manages the logic for a Retrieval-Augmented Generation system
    that learns a user's profile over time.
    """
    def __init__(self, user_id: str):
        """
        Initializes the RAG system for a specific user session.
        Args:
            user_id: A unique identifier for the current user.
        """
        self.user_id = user_id

        # --- 1. Initialize Connections ---
        # Make sure your API keys and credentials are set as environment variables
        # e.g., OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            self.graph = Neo4jGraph(
                url=st.secrets["neo4j"]["url"],
                username=st.secrets["neo4j"]["username"],
                password=st.secrets["neo4j"]["password"]
            )
            embedding_function = embeddings()
            # Use a persistent directory for Chroma to store data between app runs
            self.vector_store = Chroma(
                collection_name="user_profile_short_term",
                embedding_function=embedding_function,
                persist_directory="./chroma_db_user"
            )
        except Exception as e:
            # Raise a specific error if connections fail
            raise ConnectionError(f"Failed to initialize services. Check credentials. Error: {e}")

    def _extract_and_store_user_profile(self, user_input: str):
        """
        Private method to extract profile info and store it in vector and graph DBs.
        """
        extraction_prompt = PromptTemplate.from_template(
            "From the user input, extract key facts for a user profile (name, interests, profession, goals). "
            "If no personal details are found, output 'No profile information found.'. "
            "Otherwise, format output as a list of simple statements, e.g., 'User's name is Jane. User is interested in AI.'\n"
            "User Input: {user_input}"
        )
        extraction_chain = extraction_prompt | self.llm
        extracted_info_str = extraction_chain.invoke({"user_input": user_input}).content

        if "no profile information found" in extracted_info_str.lower():
            return

        extracted_statements = [s.strip() for s in extracted_info_str.split('.') if s.strip()]

        if extracted_statements:
            # Store in Chroma (Short-Term Memory)
            self.vector_store.add_documents(
                [Document(page_content=statement, metadata={"user_id": self.user_id}) for statement in extracted_statements]
            )

            # Store in Neo4j (Long-Term Memory) - Simplified Parser
            for statement in extracted_statements:
                if "name is" in statement.lower():
                    name = statement.split("name is")[-1].strip().replace("'", "")
                    self.graph.query(f"MERGE (u:User {{id: '{self.user_id}'}}) SET u.name = '{name}'")
                elif "interested in" in statement.lower():
                    interest = statement.split("interested in")[-1].strip().replace("'", "")
                    self.graph.query(f"MERGE (u:User {{id: '{self.user_id}'}}) MERGE (i:Interest {{name: '{interest}'}}) MERGE (u)-[:INTERESTED_IN]->(i)")
                elif "works as" in statement.lower():
                    profession = statement.split("works as")[-1].strip().replace("'", "")
                    self.graph.query(f"MERGE (u:User {{id: '{self.user_id}'}}) MERGE (p:Profession {{name: '{profession}'}}) MERGE (u)-[:WORKS_AS]->(p)")

    def get_augmented_response(self, user_input: str) -> str:
        """
        Main method to get a personalized response. It learns from the user's
        input and then generates an augmented response.
        """
        # 1. Learn from the user's latest input
        self._extract_and_store_user_profile(user_input)

        # 2. Retrieve context from memory
        short_term_context = "No recent context."
        long_term_context = "No long-term profile."
        try:
            short_term_docs = self.vector_store.similarity_search(user_input, k=2, filter={"user_id": self.user_id})
            if short_term_docs:
                short_term_context = "\n".join([doc.page_content for doc in short_term_docs])
        except Exception:
            pass # Fails silently if no context is found

        try:
            cypher_chain = GraphCypherQAChain.from_llm(graph=self.graph, cypher_llm=self.llm, qa_llm=self.llm, validate_cypher=True)
            graph_query = f"What are the known interests, name, and profession for the user with id '{self.user_id}'?"
            long_term_context = cypher_chain.invoke({"query": graph_query})['result']
        except Exception:
            pass # Fails silently if no profile is found

        # 3. Generate Augmented Response
        augmented_prompt = PromptTemplate.from_template(
            "You are AlphaMesh, a personalized investment intelligence assistant. "
            "Use the following user profile to personalize your response. "
            "Do not explicitly mention that you are using this information, just use it to tailor your answer.\n\n"
            "--- User Profile ---\n"
            "Recent context: {short_term_context}\n"
            "Long-term profile: {long_term_context}\n"
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