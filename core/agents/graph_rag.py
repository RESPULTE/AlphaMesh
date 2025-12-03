"""
Evolving Temporal Graph RAG System
Refactored for:
1. Temporal Awareness (Start/End dates on relationships).
2. Dynamic State Management (Handling User preference shifts).
3. Configuration-driven logic (No hardcoded if/else chains).
"""

from typing import Any, Dict, List, Literal, Optional

# Preserving strict imports
from core.services import service_manager
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field

# ============================================================================
# 1. Temporal Schema & Configuration
# ============================================================================

# ALLOWED NODE TYPES
AllowedNodeLabels = Literal[
    "Concept",  # Python, App Development
    "Skill",  # System Design
    "Preference",  # Detailed, Brief, Visual (Abstract nodes for styles)
    "Goal",  # Build a Chatbot
    "Project",  # My Side Project
]

# ALLOWED RELATIONSHIP TYPES
AllowedRelTypes = Literal[
    "HAS_PREFERENCE",  # User -> Preference (Exclusive: User usually has 1 active style)
    "INTERESTED_IN",  # User -> Concept (Stateful: Can change to DISLIKES)
    "DISLIKES",  # User -> Concept
]

# --- DYNAMIC BEHAVIOR CONFIGURATION ---
# This dictates how the Graph Manager handles updates.
# NOTHING is hardcoded in the logic; it follows these rules.
RELATIONSHIP_BEHAVIOR = {
    # Rule: "EXCLUSIVE_CATEGORY"
    # Meaning: User can only have ONE active relationship of this type to ANY node.
    # Ex: If User sets "Brief" preference, close "Detailed" preference.
    "HAS_PREFERENCE": "EXCLUSIVE_CATEGORY",
    # Rule: "ENTITY_STATE"
    # Meaning: User can only have ONE active relationship to a SPECIFIC node.
    # Ex: If User "DISLIKES" App Dev, close "INTERESTED_IN" App Dev.
    "INTERESTED_IN": "ENTITY_STATE",
    "DISLIKES": "ENTITY_STATE",
    "MENTIONED": "STANDARD",
}

# Grouping conflicting types for Entity State logic
# If a new rel is in this group, close ALL other active rels in this group for that target.
CONFLICTING_REL_GROUPS = {"SENTIMENT": ["INTERESTED_IN", "DISLIKES"]}


class UserProfileUpdate(BaseModel):
    """Intrinsic properties of the User node (Name, Role)."""

    name: Optional[str] = None
    role: Optional[str] = None
    experience_level: Optional[str] = None


class GraphNode(BaseModel):
    """Extrinsic Entities."""

    id: str = Field(description="Unique ID (lowercase).")
    label: AllowedNodeLabels
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    """Temporal Relationship Update."""

    source: str = Field(description="Source ID ('CURRENT_USER' for user).")
    target: str = Field(description="Target ID.")
    type: AllowedRelTypes
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphUpdate(BaseModel):
    user_attributes: UserProfileUpdate
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]


# ============================================================================
# 2. Temporal Knowledge Extractor
# ============================================================================


class KnowledgeExtractor:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(KnowledgeGraphUpdate)

    async def extract_knowledge(self, conversation: List[Any]) -> KnowledgeGraphUpdate:
        # We look at the last few messages to catch context changes
        formatted_history = "\n".join(
            [f"{m.type.upper()}: {m.content}" for m in conversation]
        )

        system_prompt = """You are a Temporal Knowledge Graph Architect.
        Your goal is to capture the *current state* of the user's mind.

        RULES:
        1. **Detect Changes**: If the user says "I used to like X, but now I hate it", output a `DISLIKES` relationship. The system will automatically archive the old `INTERESTED_IN` relationship.
        2. **Preferences**: If the user says "Give me brief answers", connect User -> `Brief` (Preference).
        3. **User Anchor**: Use 'CURRENT_USER' as the ID for the user.

        RELATIONSHIP TYPES:
        - HAS_PREFERENCE: For communication styles (Brief, Detailed, Code-Only).
        - INTERESTED_IN / DISLIKES: For concepts/topics.
        - LEARNING / MASTERED: For skills.

        Extract the CURRENT truth. Do not worry about the past; the database handles history.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", formatted_history)]
        )

        try:
            return await (prompt | self.llm).ainvoke({})
        except Exception as e:
            print(f"Extraction Error: {e}")
            return KnowledgeGraphUpdate(
                user_attributes=UserProfileUpdate(), nodes=[], relationships=[]
            )


# ============================================================================
# 3. Temporal Graph Manager (The Dynamic Engine)
# ============================================================================


class TemporalGraphManager:
    """
    Manages graph updates with Time-Travel capabilities.
    Automatically 'closes' old relationships based on behavior configuration.
    """

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._initialize_schema()

    def _initialize_schema(self):
        constraints = [
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE INDEX node_id IF NOT EXISTS FOR (n:Entity) ON (n.id)",
            # Index for performance on checking active relationships
            "CREATE INDEX rel_end_date IF NOT EXISTS FOR ()-[r:HAS_PREFERENCE]-() ON (r.end_date)",
        ]
        for q in constraints:
            try:
                self.graph.query(q)
            except Exception:
                pass

    def update_graph(self, user_id: str, knowledge: KnowledgeGraphUpdate):
        """
        Transactional update wrapper.
        """
        # 1. Update User Properties
        user_props = {
            k: v
            for k, v in knowledge.user_attributes.model_dump().items()
            if v is not None
        }
        if user_props:
            self.graph.query(
                "MERGE (u:User {id: $user_id}) SET u += $props, u.last_active = datetime()",
                {"user_id": user_id, "props": user_props},
            )

        # 2. Upsert Nodes (Standard MERGE)
        for node in knowledge.nodes:
            if node.label == "User":
                continue
            query = f"""
            MERGE (n:{node.label} {{id: $id}})
            ON CREATE SET n += $props, n.created_at = datetime()
            ON MATCH SET n += $props, n.updated_at = datetime()
            """
            self.graph.query(query, {"id": node.id, "props": node.properties})

        # 3. Handle Temporal Relationships
        for rel in knowledge.relationships:
            self._upsert_temporal_relationship(user_id, rel)

    def _upsert_temporal_relationship(self, user_id: str, rel: GraphRelationship):
        """
        The core logic for handling state changes dynamically.
        """
        source = user_id if rel.source == "CURRENT_USER" else rel.source
        target = user_id if rel.target == "CURRENT_USER" else rel.target

        behavior = RELATIONSHIP_BEHAVIOR.get(rel.type, "STANDARD")

        # LOGIC BRANCH 1: Exclusive Category (e.g., Preference Style)
        # "User can have only ONE active 'HAS_PREFERENCE' relationship total."
        if behavior == "EXCLUSIVE_CATEGORY":
            print("Exclusive Category Behavior:")
            print(f"Closing all active '{rel.type}' relationships for {source}.")

            self._close_category_relationships(source, rel.type)
            self._create_new_active_relationship(
                source, target, rel.type, rel.properties
            )

        # LOGIC BRANCH 2: Entity State (e.g., Interest vs Dislike)
        # "User can have only ONE active sentiment towards 'App Dev'."
        elif behavior == "ENTITY_STATE":
            print("Entity State Behavior:")
            print(f"Closing all active '{rel.type}' relationships for {target}.")

            # Determine which types conflict with this one (e.g., INTERESTED_IN conflicts with DISLIKES)
            conflicting_types = self._get_conflicting_types(rel.type)
            self._close_entity_state_relationships(source, target, conflicting_types)
            self._create_new_active_relationship(
                source, target, rel.type, rel.properties
            )

        # LOGIC BRANCH 3: Standard (Accumulative)
        else:
            print("Standard Behavior:")
            print(f"Creating new '{rel.type}' relationship for {source} -> {target}.")
            self._create_new_active_relationship(
                source, target, rel.type, rel.properties
            )

    def _get_conflicting_types(self, rel_type: str) -> List[str]:
        """Finds all relationship types that conflict with the new one."""
        for group in CONFLICTING_REL_GROUPS.values():
            if rel_type in group:
                print(f"Conflict Group: {group}")
                return group
        return [rel_type]  # Default: conflicts with itself

    def _close_category_relationships(self, source_id: str, rel_type: str):
        """Closes ANY active relationship of this type from the source."""
        query = f"""
        MATCH (s {{id: $source}})-[r:{rel_type}]->()
        WHERE r.end_date IS NULL
        SET r.end_date = datetime()
        """
        self.graph.query(query, {"source": source_id})

    def _close_entity_state_relationships(
        self, source_id: str, target_id: str, types: List[str]
    ):
        """Closes active relationships of specific types between specific nodes."""
        # Dynamic type matching in Cypher requires listing or APOC.
        # We'll stick to a WHERE clause for safety without APOC.
        types_str = "|".join([f"`{t}`" for t in types])
        query = f"""
        MATCH (s {{id: $source}})-[r:{types_str}]->(t {{id: $target}})
        WHERE r.end_date IS NULL
        SET r.end_date = datetime()
        """
        self.graph.query(query, {"source": source_id, "target": target_id})

    def _create_new_active_relationship(
        self, source: str, target: str, rel_type: str, props: dict
    ):
        """Creates the new relationship with start_date = now and end_date = null."""
        query = f"""
        MATCH (s {{id: $source}})
        MATCH (t {{id: $target}})
        CREATE (s)-[r:{rel_type}]->(t)
        SET r += $props, r.start_date = datetime(), r.end_date = null
        """
        self.graph.query(query, {"source": source, "target": target, "props": props})

    def get_current_user_state(self, user_id: str) -> str:
        """
        Fetches ONLY active relationships (where end_date is null).
        """
        query = """
        MATCH (u:User {id: $user_id})
        
        // Active Preferences
        OPTIONAL MATCH (u)-[r1:HAS_PREFERENCE]->(p:Preference)
        WHERE r1.end_date IS NULL
        WITH u, collect(p.id) as preferences
        
        // Active Interests
        OPTIONAL MATCH (u)-[r2:INTERESTED_IN]->(c:Concept)
        WHERE r2.end_date IS NULL
        WITH u, preferences, collect(c.id) as interests
        
        // Active Dislikes (Important context!)
        OPTIONAL MATCH (u)-[r3:DISLIKES]->(d:Concept)
        WHERE r3.end_date IS NULL
        WITH u, preferences, interests, collect(d.id) as dislikes
        
        RETURN u {.*, id:null, created_at:null, last_active:null} as profile, 
               preferences, interests, dislikes
        """

        data = self.graph.query(query, {"user_id": user_id})
        if not data:
            return "New User"

        rec = data[0]
        return f"""Current Profile:
Attributes: {rec['profile']}
Active Preferences: {rec['preferences']}
Current Interests: {rec['interests']}
Known Dislikes: {rec['dislikes']}"""


# ============================================================================
# 4. Main Agent
# ============================================================================


class EvolvingGraphRAGAgent:
    def __init__(self, graph: Neo4jGraph, llm, user_id: str = "user_core"):
        self.user_id = user_id
        self.graph = graph
        self.llm = llm
        self.manager = TemporalGraphManager(self.graph)
        self.extractor = KnowledgeExtractor(self.llm)

    async def process_interaction(self, user_input: str) -> str:
        # 1. Get Current State
        state = self.manager.get_current_user_state(self.user_id)

        # 2. Generate Response
        response = await self._generate_response(user_input, state)

        # 3. Extract & Evolve
        knowledge = await self.extractor.extract_knowledge(
            [HumanMessage(content=user_input), AIMessage(content=response)]
        )
        self.manager.update_graph(self.user_id, knowledge)

        return response

    async def _generate_response(self, user_input: str, state: str) -> str:
        system_msg = f"""You are a helpful assistant.
        
        USER'S CURRENT CONTEXT:
        {state}

        NOTE:
        - If the user has 'Dislikes', do NOT suggest those topics.
        - Adapt to their 'Active Preferences' (e.g., if 'brief', be concise).
        """
        messages = [SystemMessage(content=system_msg)] + [
            HumanMessage(content=user_input)
        ]
        return (await self.llm.ainvoke(messages)).content


# ============================================================================
# Example Execution: The "Change of Heart" Scenario
# ============================================================================


async def main():
    # 1. Setup
    graph = service_manager.get_graph()
    llm = service_manager.get_agent()
    agent = EvolvingGraphRAGAgent(graph, llm, user_id="user_evolution_demo")

    # 🛑 RESET GRAPH for a clean demo
    print("🧹 Clearing Database for Demo...")
    graph.query("MATCH (n) DETACH DELETE n")

    # Helper to visualize the graph state
    def print_graph_snapshot(step_name):
        print(f"\n{'='*20} {step_name} {'='*20}")

        # Fetch Active Edges
        active = graph.query(
            """
            MATCH (u:User)-[r]->(n) 
            WHERE r.end_date IS NULL 
            RETURN type(r) as rel, n.id as node, n.label as label
        """
        )

        # Fetch Archived Edges
        archived = graph.query(
            """
            MATCH (u:User)-[r]->(n) 
            WHERE r.end_date IS NOT NULL 
            RETURN type(r) as rel, n.id as node, r.end_date as ended
        """
        )

        print("🟢 ACTIVE STATE:")
        if not active:
            print("   (Empty)")
        for row in active:
            print(f"   (User) --[{row['rel']}]--> ({row['node']}) [{row['label']}]")

        print("\n🔴 HISTORY (Archived):")
        if not archived:
            print("   (None)")
        for row in archived:
            print(
                f"   (User) --[{row['rel']}]--> ({row['node']}) [Ended: {row['ended']}]"
            )
        print("=" * 60)

    # ============================================================
    # 📨 Message 1: Initialization
    # Setting identity, interest, and specific preference.
    # ============================================================
    msg_1 = "Hi, I'm Alex. I really love Python development. Please give me detailed, in-depth answers."
    print(f"\nUser: {msg_1}")
    await agent.process_interaction(msg_1)
    print_graph_snapshot("AFTER MESSAGE 1")
    # EXPECTED:
    # Active: INTERESTED_IN -> python, HAS_PREFERENCE -> detailed
    # History: None

    # ============================================================
    # 📨 Message 2: Expansion
    # Adding a new skill (Accumulative change).
    # ============================================================
    msg_2 = "I am also starting to learn Docker for containerization."
    print(f"\nUser: {msg_2}")
    await agent.process_interaction(msg_2)
    print_graph_snapshot("AFTER MESSAGE 2")
    # EXPECTED:
    # Active: ... + LEARNING -> docker
    # History: None

    # ============================================================
    # 📨 Message 3: Preference Shift (Conflict Type 1)
    # Changing "Detailed" -> "Brief".
    # The 'detailed' edge should close.
    # ============================================================
    msg_3 = (
        "Actually, your answers are too long. Keep them brief and concise from now on."
    )
    print(f"\nUser: {msg_3}")
    await agent.process_interaction(msg_3)
    print_graph_snapshot("AFTER MESSAGE 3")
    # EXPECTED:
    # Active: HAS_PREFERENCE -> brief, INTERESTED_IN -> python, LEARNING -> docker
    # History: HAS_PREFERENCE -> detailed

    # ============================================================
    # 📨 Message 4: Sentiment Shift (Conflict Type 2)
    # Changing "Love Python" -> "Dislike Python".
    # The 'INTERESTED_IN' Python edge should close.
    # ============================================================
    msg_4 = "Also, I'm tired of Python. It's too slow. I dislike it now."
    print(f"\nUser: {msg_4}")
    await agent.process_interaction(msg_4)
    print_graph_snapshot("AFTER MESSAGE 4")
    # EXPECTED:
    # Active: DISLIKES -> python, HAS_PREFERENCE -> brief, LEARNING -> docker
    # History: HAS_PREFERENCE -> detailed, INTERESTED_IN -> python


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
