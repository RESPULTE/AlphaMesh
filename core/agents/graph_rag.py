"""
Evolving Knowledge Graph RAG System
Refactored for:
1. Strict Property vs. Node separation (Fixing the 'John is a node' error).
2. Modeling Learning Progress (Concepts, Skills, Goals).
3. No Raw Chat Storage (Privacy focused).
"""

from typing import Any, Dict, List, Literal, Optional

# Preserving strict imports
from core.services import service_manager
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field

# ============================================================================
# 1. Strict Schema Definitions
# ============================================================================

# ALLOWED NODE TYPES
# We intentionally REMOVED 'Person' to prevent the user from being duplicated.
# If the user mentions "Elon Musk", that can be a 'KeyFigure' or generic 'Entity',
# but the User themselves is handled via properties.
AllowedNodeLabels = Literal[
    "Concept",  # e.g., "Graph Databases", "Python"
    "Skill",  # e.g., "System Design", "Prompt Engineering"
    "Goal",  # e.g., "Build a RAG Agent"
    "Resource",  # e.g., "Neo4j Documentation"
    "Project",  # e.g., "My Chatbot"
    "Inquiry",  # Abstracted question/topic of interest
]

# ALLOWED RELATIONSHIP TYPES
AllowedRelTypes = Literal[
    "INTERESTED_IN",  # User -> Concept
    "LEARNING",  # User -> Skill (implies active study)
    "MASTERED",  # User -> Skill (implies competence)
    "WORKING_ON",  # User -> Project
    "HAS_GOAL",  # User -> Goal
    "RELATED_TO",  # Concept -> Concept
    "REQUIRES",  # Goal -> Skill
]


class UserProfileUpdate(BaseModel):
    """
    Captures INTRINSIC attributes of the user.
    These become PROPERTIES of the :User node, not separate nodes.
    """

    name: Optional[str] = Field(None, description="The user's stated name.")
    role: Optional[str] = Field(
        None, description="Professional role (e.g., 'Backend Engineer')."
    )
    experience_level: Optional[str] = Field(
        None, description="e.g., 'Junior', 'Senior', 'Beginner'."
    )
    learning_style: Optional[str] = Field(
        None, description="e.g., 'Visual', 'Hands-on'."
    )
    current_focus: Optional[str] = Field(
        None, description="What they are currently focused on generally."
    )


class GraphNode(BaseModel):
    """Represents an EXTRINSIC entity (Concept, Skill, etc.)."""

    id: str = Field(description="Unique identifier (lowercase name).")
    label: AllowedNodeLabels = Field(description="The category of the node.")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata (e.g., {'difficulty': 'Hard', 'status': 'Active'}).",
    )


class GraphRelationship(BaseModel):
    """Represents the connection between User and Concepts, or Concept to Concept."""

    source: str = Field(description="Source node ID. Use 'CURRENT_USER' for the user.")
    target: str = Field(description="Target node ID. Use 'CURRENT_USER' for the user.")
    type: AllowedRelTypes = Field(description="Relationship type.")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Edge attributes (e.g., {'confidence': 0.8, 'date': '2023-10-10'}).",
    )


class KnowledgeGraphUpdate(BaseModel):
    """The complete payload to update the User's Knowledge Graph."""

    user_attributes: UserProfileUpdate = Field(
        description="Updates to the User's own profile properties."
    )
    nodes: List[GraphNode] = Field(
        default_factory=list, description="New concepts, skills, or goals."
    )
    relationships: List[GraphRelationship] = Field(
        default_factory=list,
        description="Connections identifying structure and progress.",
    )


# ============================================================================
# 2. Strategic Knowledge Extractor
# ============================================================================


class KnowledgeExtractor:
    """
    Distinguishes between User Attributes (Properties) and Knowledge Entities (Nodes).
    """

    def __init__(self, llm):
        self.llm = llm.with_structured_output(KnowledgeGraphUpdate)

    async def extract_knowledge(self, chat_history: List[Any]) -> KnowledgeGraphUpdate:

        # We only analyze the recent context to keep extraction focused
        recent_messages = chat_history[-4:]
        formatted_history = "\n".join(
            [f"{m.type.upper()}: {m.content}" for m in recent_messages]
        )

        system_prompt = """You are a Knowledge Graph Architect building a 'Learning Profile' for a user.

YOUR GOAL:
Map the user's conversation into a structured graph comprising:
1. **User Properties (Intrinsic)**: Things the user IS (Name, Role, Experience). 
   - Example: "I'm John" -> User Property `name="John"`. Do NOT make a Node.
2. **Concepts & Skills (Extrinsic)**: Things the user interacts WITH.
   - Example: "I'm learning Neo4j" -> Node `Neo4j` (Concept).
3. **Relationships**: How the user relates to concepts (LEARNING, MASTERED, INTERESTED_IN).

CRITICAL RULES:
- **NO CHAT LOGS**: Do not store the raw message text. Abstract it into concepts or goals.
- **The 'John' Rule**: If the user gives their name, it is a PROPERTY of the User node. NOT a separate Person node.
- **Relationship Directions**: 
  - User -> Concept (INTERESTED_IN)
  - Project -> Requires -> Skill
- **Inquiry Handling**: If the user asks a complex question, create an `:Inquiry` node with a summary topic, not the full text.

Analyze the following interaction:
"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", formatted_history)]
        )

        try:
            chain = prompt | self.llm
            return await chain.ainvoke({})
        except Exception as e:
            print(f"Extraction Error: {e}")
            return KnowledgeGraphUpdate(
                user_attributes=UserProfileUpdate(), nodes=[], relationships=[]
            )


# ============================================================================
# 3. Graph Manager (The Storage Engine)
# ============================================================================


class GraphManager:
    """
    Handles the physical storage of nodes and properties.
    Separates 'User Property Updates' from 'Graph Topology Updates'.
    """

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._initialize_schema()

    def _initialize_schema(self):
        """Ensures the graph is optimized for this schema."""
        constraints = [
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE",
            "CREATE INDEX node_id IF NOT EXISTS FOR (n:Entity) ON (n.id)",
        ]
        for q in constraints:
            try:
                self.graph.query(q)
            except Exception:
                pass

    def update_graph(self, user_id: str, knowledge: KnowledgeGraphUpdate):
        """
        Orchestrates the update:
        1. Update User Properties (SET)
        2. Merge Nodes (MERGE)
        3. Merge Relationships (MERGE)
        """

        # 1. Update User Properties
        # Filter out None values to avoid overwriting existing data with nulls
        user_props = {
            k: v
            for k, v in knowledge.user_attributes.model_dump().items()
            if v is not None
        }

        if user_props:
            self.graph.query(
                """
                MERGE (u:User {id: $user_id})
                SET u += $props, u.last_active = timestamp()
                """,
                {"user_id": user_id, "props": user_props},
            )

        # 2. Upsert Nodes
        for node in knowledge.nodes:
            # We explicitly prevent 'User' or 'Person' labels coming from the generic node list
            # to ensure the "John is a node" error never recurs.
            if node.label in ["User", "Person"]:
                continue

            query = f"""
            MERGE (n:{node.label} {{id: $id}})
            ON CREATE SET n += $props, n.created_at = timestamp()
            ON MATCH SET n += $props, n.updated_at = timestamp()
            """
            self.graph.query(query, {"id": node.id, "props": node.properties})

        # 3. Upsert Relationships
        for rel in knowledge.relationships:
            # Resolve 'CURRENT_USER' placeholder to actual ID
            source = user_id if rel.source == "CURRENT_USER" else rel.source
            target = user_id if rel.target == "CURRENT_USER" else rel.target

            # Skip self-loops if extraction failed
            if source == target:
                continue

            query = f"""
            MATCH (s {{id: $source}})
            MATCH (t {{id: $target}})
            MERGE (s)-[r:{rel.type}]->(t)
            ON CREATE SET r += $props, r.created_at = timestamp()
            ON MATCH SET r += $props, r.updated_at = timestamp()
            """
            try:
                self.graph.query(
                    query, {"source": source, "target": target, "props": rel.properties}
                )
            except Exception as e:
                print(f"Rel Error ({rel.type}): {e}")

    def get_user_learning_state(self, user_id: str) -> str:
        """
        Retrieves a semantic summary of the user's graph.
        Focuses on what they are learning, working on, and their goals.
        """
        query = """
        MATCH (u:User {id: $user_id})
        
        // Get User Properties
        WITH u
        
        // Get Active Interests & Skills
        OPTIONAL MATCH (u)-[r1:INTERESTED_IN|LEARNING|MASTERED]->(c)
        WITH u, collect(c.id + ' (' + type(r1) + ')') as interests
        
        // Get Goals
        OPTIONAL MATCH (u)-[:HAS_GOAL]->(g:Goal)
        WITH u, interests, collect(g.id) as goals
        
        // Get Current Projects
        OPTIONAL MATCH (u)-[:WORKING_ON]->(p:Project)
        WITH u, interests, goals, collect(p.id) as projects
        
        RETURN u {.*, id: null} as profile, interests, goals, projects
        """

        data = self.graph.query(query, {"user_id": user_id})
        if not data:
            return "New User."

        record = data[0]

        # Format for LLM Context
        context = "User Profile:\n"
        for k, v in record["profile"].items():
            if k not in ["created_at", "last_active"]:
                context += f"- {k}: {v}\n"

        context += (
            f"\nLearning Journey:\n- Concepts: {', '.join(record['interests'])}\n"
        )
        context += f"- Active Goals: {', '.join(record['goals'])}\n"
        context += f"- Projects: {', '.join(record['projects'])}\n"

        return context


# ============================================================================
# 4. Main Agent Controller
# ============================================================================


class EvolvingGraphRAGAgent:
    def __init__(self, graph: Neo4jGraph, llm, user_id: str = "user_core"):
        self.user_id = user_id
        self.graph = graph
        self.llm = llm

        self.graph_manager = GraphManager(self.graph)
        self.extractor = KnowledgeExtractor(self.llm)
        self.chat_history: List[Any] = []

    async def process_interaction(self, user_input: str) -> str:
        # 1. Get Graph Context (The User's "State")
        user_state = self.graph_manager.get_user_learning_state(self.user_id)

        # 2. Generate Response (Using State + History)
        response = await self._generate_response(user_input, user_state)

        # 3. Update History (In-memory only)
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))

        # 4. Evolve Graph (Extract & Update)
        # We pass the history so the LLM understands "it", "that", etc.
        knowledge = await self.extractor.extract_knowledge(self.chat_history)

        # 5. Post-Process: Anchor relationships to the User
        self._anchor_knowledge_to_user(knowledge)

        # 6. Commit to DB
        self.graph_manager.update_graph(self.user_id, knowledge)

        return response

    async def _generate_response(self, user_input: str, user_state: str) -> str:
        system_msg = f"""You are a personalized AI tutor.
        
USER STATE (Knowledge Graph):
{user_state}

INSTRUCTIONS:
- Adapt your difficulty level based on the user's 'experience_level' and 'mastered' skills.
- If the user has a Goal, help them towards it.
- If the user mentions a Project, ask about its progress.
- Do NOT explicitly say "According to your graph...". Be natural.
"""
        messages = (
            [SystemMessage(content=system_msg)]
            + self.chat_history[-6:]
            + [HumanMessage(content=user_input)]
        )
        response = await self.llm.ainvoke(messages)
        return response.content

    def _anchor_knowledge_to_user(self, knowledge: KnowledgeGraphUpdate):
        """
        Ensures relationships meant for the user are correctly ID'd.
        The LLM is prompted to use 'CURRENT_USER', but we double check.
        """
        # If the LLM inferred "user" or "me" as a source/target, normalize it.
        for rel in knowledge.relationships:
            if rel.source.lower() in ["user", "me", "self", "current_user"]:
                rel.source = "CURRENT_USER"  # Handled in GraphManager
            if rel.target.lower() in ["user", "me", "self", "current_user"]:
                rel.target = "CURRENT_USER"


# ============================================================================
# Example Usage
# ============================================================================


async def main():
    # Setup
    graph = service_manager.get_graph()
    llm = service_manager.get_agent()

    # Initialize Agent
    agent = EvolvingGraphRAGAgent(graph, llm, user_id="user_12345")

    # Scenario: User introducing themselves and their goals
    # interactions = [
    #     "Hi, I'm John. I'm a Senior Backend Dev.",
    #     "I want to build a Chatbot using Neo4j.",
    #     "I already know Python pretty well, but I'm new to Graph Theory.",
    #     "What should I learn first?",
    # ]

    interactions = ["What do you know about me?"]

    print(f"{'='*50}\nSTARTING INTERACTION\n{'='*50}")

    for msg in interactions:
        print(f"\nUser: {msg}")
        resp = await agent.process_interaction(msg)
        print(f"Assistant: {resp}")

    # Verify the Graph Structure
    print(f"\n{'='*50}\nGRAPH STATE VERIFICATION\n{'='*50}")

    # Check User Properties (Should have name='John', role='Senior Backend Dev')
    user_node = graph.query("MATCH (u:User {id: 'user_12345'}) RETURN u")
    print("User Node Properties:", user_node)

    # Check Concepts & Relationships
    # Should see:
    # (User)-[:MASTERED]->(Python)
    # (User)-[:INTERESTED_IN]->(Graph Theory)
    # (User)-[:HAS_GOAL]->(Build Chatbot)
    rels = graph.query(
        """
        MATCH (u:User {id: 'user_12345'})-[r]->(n) 
        RETURN type(r) as relation, n.id as entity, labels(n) as type
    """
    )
    for r in rels:
        print(f"User --[{r['relation']}]--> {r['entity']} ({r['type'][0]})")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
