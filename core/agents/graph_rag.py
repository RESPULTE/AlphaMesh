from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph


class UserMemoryGraph:
    """
    A specialized Graph RAG class designed to build a 'User Profile'
    graph in Neo4j. It extracts preferences, history, and knowledge
    from chat logs and links them to a specific User ID.
    """

    def __init__(self, url: str, username: str, password: str, llm: BaseLanguageModel):
        self.llm = llm

        # Connect to Neo4j
        self.graph = Neo4jGraph(url=url, username=username, password=password)

        # 1. Define a Strict Schema for efficiency
        # This prevents the graph from becoming a mess of random words.
        # We only want to learn specific things about the user.
        self.allowed_nodes = [
            "User",
            "Topic",
            "Preference",
            "Skill",
            "Location",
            "Person",
            "Company",
            "Goal",
        ]

        self.allowed_relationships = [
            "LIKES",
            "DISLIKES",
            "KNOWS",
            "LIVES_IN",
            "WORKS_AT",
            "HAS_GOAL",
            "DISCUSSED",
            "IS_EXPERT_IN",
        ]

        # 2. Initialize the Transformer
        # We allow the LLM to act as a 'profiler'
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.allowed_nodes,
            allowed_relationships=self.allowed_relationships,
        )

        # 3. Initialize QA Chain (for deep queries)
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm, graph=self.graph, verbose=True, allow_dangerous_requests=True
        )

        # Ensure schema is ready
        self.graph.refresh_schema()

    def learn_from_chat(self, user_id: str, user_message: str):
        """
        Parses a user message, extracts facts, and stores them in the graph
        linked to the specific user_id.
        """
        print(f"--- Learning from User {user_id} ---")

        # Contextualize the input so the LLM knows 'I' refers to 'User X'
        # This is the key trick to ensure nodes link to the User node.
        contextualized_text = f"User with ID '{user_id}' said: {user_message}"

        doc = Document(page_content=contextualized_text)

        # Extract graph data
        graph_docs = self.transformer.convert_to_graph_documents([doc])

        if not graph_docs:
            print("No new facts extracted.")
            return

        # Optimization: Enforce 'User' label constraints manually if needed
        # or simply store. The prompt usually handles linking if the text is clear.
        try:
            self.graph.add_graph_documents(
                graph_docs, baseEntityLabel=True, include_source=True
            )
            self.graph.refresh_schema()
            print(
                f"Stored {len(graph_docs[0].nodes)} nodes and {len(graph_docs[0].relationships)} relationships."
            )
        except Exception as e:
            print(f"Error storing memory: {e}")

    def get_user_context(self, user_id: str, limit: int = 10) -> str:
        """
        FAST RETRIEVAL:
        Retrieves the immediate 'Ego Graph' of the user.
        Use this to augment the System Prompt before generating a response.

        Example output: "User LIKES Python, User LIVES_IN London"
        """
        # We use a direct Cypher query here for speed and precision.
        # We don't need the LLM to generate Cypher for this standard look-up.
        cypher = f"""
        MATCH (u:User {{id: '{user_id}'}})-[r]->(n)
        RETURN type(r) as relationship, labels(n) as type, n.id as value
        LIMIT {limit}
        """

        try:
            results = self.graph.query(cypher)
            if not results:
                return "No prior information known about this user."

            # Format into a natural language string for the LLM
            context_strs = []
            for row in results:
                # e.g., "LIKES Topic: Python"
                target_val = row["value"]
                # If the node doesn't have an ID property, try name (fallback)
                if target_val is None:
                    target_val = "Unknown Entity"

                context_strs.append(
                    f"- User {row['relationship']} {row['type'][0]}: {target_val}"
                )

            return "\n".join(context_strs)
        except Exception as e:
            return f"Error retrieving context: {e}"

    def ask_about_user(self, user_id: str, question: str) -> str:
        """
        DEEP RETRIEVAL:
        Allows the agent to ask complex questions about the user's history.
        e.g., "What projects has the user discussed regarding AI?"
        """
        # We modify the question to enforce the user_id scope
        scoped_question = f"Regarding User {user_id}: {question}"

        try:
            result = self.qa_chain.invoke({"query": scoped_question})
            return result.get("result", "No answer found.")
        except Exception as e:
            return f"Error querying graph: {e}"

    def reset_memory(self, user_id: str):
        """
        Clears memory for a specific user (Useful for testing).
        """
        cypher = f"MATCH (u:User {{id: '{user_id}'}}) DETACH DELETE u"
        self.graph.query(cypher)
        print(f"Memory cleared for User {user_id}")


# ==========================================
# Example Workflow
# ==========================================
if __name__ == "__main__":
    from core.config import settings
    from core.services import service_manager

    # Initialize Memory System
    memory_graph = UserMemoryGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        llm=service_manager.get_agent(),
    )

    USER_ID = "User_123"

    # 1. SIMULATE CHAT (Learning Phase)
    # The user chats, and we extract facts in the background.

    print("\n--- User Chatting ---")
    chat_inputs = [
        "Hi, I'm a software engineer living in Berlin.",
        "I really love coding in Python, but I hate Java.",
        "I am currently learning about Graph Databases.",
    ]

    for msg in chat_inputs:
        memory_graph.learn_from_chat(USER_ID, msg)

    # 2. AUGMENT GENERATION (Fast Retrieval Phase)
    # Before the agent replies to a new message, it pulls context.

    print("\n--- Retrieving Context for System Prompt ---")
    user_profile = memory_graph.get_user_context(USER_ID)
    print(f"Found User Profile:\n{user_profile}")

    # (In a real app, you would prepend this to your prompt:
    #  "System: You are helpful. Here is what you know about the user:\n{user_profile}")

    # 3. COMPLEX QUERY (Deep Retrieval Phase)
    # The agent might need to recall specific details logically.

    print("\n--- Complex Query ---")
    question = "What programming languages does the user prefer?"
    answer = memory_graph.ask_about_user(USER_ID, question)
    print(f"Q: {question}\nA: {answer}")
