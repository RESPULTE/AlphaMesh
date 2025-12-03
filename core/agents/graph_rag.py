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
]

# ALLOWED RELATIONSHIP TYPES
AllowedRelTypes = Literal[
    "INTERESTED_IN",  # User -> Concept (Stateful: Can change to DISLIKES)
    "DISLIKES",  # User -> Concept
    "RELATED_TO",  # Concept -> Concept (New: For sub-aspects like Docker -> Containers)
]

# --- DYNAMIC BEHAVIOR CONFIGURATION ---
# This dictates how the Graph Manager handles updates.
# NOTHING is hardcoded in the logic; it follows these rules.
RELATIONSHIP_BEHAVIOR = {
    # Meaning: User can only have ONE active relationship of this type to ANY node.
    # Ex: If User sets "Brief" preference, close "Detailed" preference.
    # Rule: "ENTITY_STATE"
    # Meaning: User can only have ONE active relationship to a SPECIFIC node.
    # Ex: If User "DISLIKES" App Dev, close "INTERESTED_IN" App Dev.
    "INTERESTED_IN": "ENTITY_STATE",
    "DISLIKES": "ENTITY_STATE",
    "RELATED_TO": "STANDARD",
}

# Grouping conflicting types for Entity State logic
# If a new rel is in this group, close ALL other active rels in this group for that target.
CONFLICTING_REL_GROUPS = {"SENTIMENT": ["INTERESTED_IN", "DISLIKES"]}


class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    experience_level: Optional[str] = None


class GraphNode(BaseModel):
    id: str = Field(description="Unique ID (lowercase).")
    label: AllowedNodeLabels
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    source: str = Field(description="Source ID.")
    target: str = Field(description="Target ID.")
    type: AllowedRelTypes
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: int = Field(default=1, description="Importance score. Default is 1.")


class KnowledgeGraphUpdate(BaseModel):
    user_attributes: UserProfileUpdate
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]


# ============================================================================
# 2. Temporal Knowledge Extractor
# ============================================================================

# ============================================================================
# 2. Temporal Knowledge Extractor
# ============================================================================


class KnowledgeExtractor:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(KnowledgeGraphUpdate)

    async def extract_knowledge(
        self, conversation: List[Any], existing_entities: str
    ) -> KnowledgeGraphUpdate:
        """
        Extracts new knowledge and reinforcements of existing knowledge.
        """
        formatted_history = "\n".join(
            [f"{m.type.upper()}: {m.content}" for m in conversation]
        )

        system_prompt = f"""You are a Temporal Knowledge Graph Architect.
    Your goal is to map the conversation to a graph, capturing both User Intent and **Learned Concepts**.

    ### EXISTING GRAPH NODES:
    {existing_entities}

    ### RULES:
    1. **Implicit Interest (Crucial)**: 
    - If the user asks about or mentions a Concept (e.g., "How does Docker work?"), assume they are **INTERESTED_IN** it.
    - Create a relationship: `CURRENT_USER -> INTERESTED_IN -> Docker`.
    - **Exception**: If the user explicitly says they hate/dislike it, use `DISLIKES` instead.

    2. **Analyze the Assistant's Answer**: 
    - If the Assistant explains a specific detail about a topic, extract it as a new Concept and link it using 'RELATED_TO'.
    - Example: AI says "Docker uses Containers." -> 
        Nodes: [Docker, Containers]
        Rels: [Docker -> RELATED_TO -> Containers]

    3. **Strict Directionality**:
    - 'DISLIKES', 'INTERESTED_IN', 'HAS_PREFERENCE' **MUST** always start with 'CURRENT_USER'.
    - NEVER say "Python DISLIKES Docker". 
    - Use 'RELATED_TO' for concept-to-concept links.

    4. **Scoring & Priority**:
    - If a topic is discussed *again*, **OUTPUT THE RELATIONSHIP AGAIN**. The database will increment the score.

    5. **Entity Resolution**:
    - Use the 'EXISTING GRAPH NODES' list to reuse IDs.
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
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._initialize_schema()

    def _initialize_schema(self):
        constraints = [
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE INDEX node_id IF NOT EXISTS FOR (n:Entity) ON (n.id)",
            "CREATE INDEX rel_end_date IF NOT EXISTS FOR ()-[r:HAS_PREFERENCE]-() ON (r.end_date)",
        ]
        for q in constraints:
            try:
                self.graph.query(q)
            except Exception:
                pass

    def update_graph(self, user_id: str, knowledge: KnowledgeGraphUpdate):
        # 1. ALWAYS Ensure User Node Exists
        # We do this outside the 'if user_props' block to guarantee the node exists
        # even if no new attributes (name/role) were extracted in this turn.
        self.graph.query(
            """
            MERGE (u:User {id: $user_id})
            ON CREATE SET u.created_at = datetime(), u.last_active = datetime()
            ON MATCH SET u.last_active = datetime()
            """,
            {"user_id": user_id},
        )

        # 2. Update User Properties (if any)
        user_props = {
            k: v
            for k, v in knowledge.user_attributes.model_dump().items()
            if v is not None
        }
        if user_props:
            self.graph.query(
                "MATCH (u:User {id: $user_id}) SET u += $props",
                {"user_id": user_id, "props": user_props},
            )

        # 3. Upsert Nodes
        for node in knowledge.nodes:
            if node.label == "User":
                continue
            query = f"""
            MERGE (n:{node.label} {{id: $id}})
            ON CREATE SET n += $props, n.created_at = datetime()
            ON MATCH SET n += $props, n.updated_at = datetime()
            """
            self.graph.query(query, {"id": node.id, "props": node.properties})

        # 4. Handle Relationships
        for rel in knowledge.relationships:
            self._upsert_temporal_relationship(user_id, rel)

    def get_known_entities(self) -> str:
        """Retrieves existing nodes for entity resolution."""
        try:
            results = self.graph.query(
                "MATCH (n) WHERE NOT 'User' IN labels(n) RETURN n.id, labels(n) LIMIT 100"
            )
            if not results:
                return "No existing entities."
            return "\n".join([f"- {r['n.id']} ({r['labels(n)'][0]})" for r in results])
        except Exception:
            return ""

    def _upsert_temporal_relationship(self, user_id: str, rel: GraphRelationship):
        # --- FIX: Strict Directionality Enforcement ---
        # These types MUST start from the User. If LLM says "Python DISLIKES Docker", we correct it or ignore it.
        USER_CENTRIC_RELS = {
            "INTERESTED_IN",
            "DISLIKES",
            "HAS_PREFERENCE",
            "LEARNING",
            "MASTERED",
            "WORKING_ON",
        }

        if rel.type in USER_CENTRIC_RELS:
            source = user_id  # Force source to be User
            target = rel.target
        else:
            # For Concept-to-Concept (RELATED_TO), trust the extraction
            source = user_id if rel.source == "CURRENT_USER" else rel.source
            target = user_id if rel.target == "CURRENT_USER" else rel.target

        # Prevent self-loops (User -> User) or Nulls
        if source == target or not target:
            return

        behavior = RELATIONSHIP_BEHAVIOR.get(rel.type, "STANDARD")

        # Check if this exact relationship already exists and is active (Score Boosting)
        if self._is_relationship_active(source, target, rel.type):
            print(
                f"   [Scoring] Boosting weight for existing rel: {source} -[{rel.type}]-> {target}"
            )
            self._increment_weight(source, target, rel.type)
            return

        # LOGIC BRANCH 2: Entity State
        if behavior == "ENTITY_STATE":
            conflicting_types = self._get_conflicting_types(rel.type)
            self._close_entity_state_relationships(source, target, conflicting_types)
            self._create_new_active_relationship(
                source, target, rel.type, rel.properties
            )

        # LOGIC BRANCH 3: Standard (Accumulative)
        else:
            self._create_new_active_relationship(
                source, target, rel.type, rel.properties
            )

    def _is_relationship_active(self, source: str, target: str, rel_type: str) -> bool:
        query = f"""
        MATCH (s {{id: $source}})-[r:{rel_type}]->(t {{id: $target}})
        WHERE r.end_date IS NULL
        RETURN count(r) > 0 as exists
        """
        res = self.graph.query(query, {"source": source, "target": target})
        return res[0]["exists"]

    def _increment_weight(self, source: str, target: str, rel_type: str):
        query = f"""
        MATCH (s {{id: $source}})-[r:{rel_type}]->(t {{id: $target}})
        WHERE r.end_date IS NULL
        SET r.weight = coalesce(r.weight, 1) + 1, r.updated_at = datetime()
        """
        self.graph.query(query, {"source": source, "target": target})

    def _create_new_active_relationship(
        self, source: str, target: str, rel_type: str, props: dict
    ):
        # Initial weight is 1
        query = f"""
        MATCH (s {{id: $source}})
        MATCH (t {{id: $target}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET r += $props, r.start_date = datetime(), r.end_date = null, r.weight = 1
        ON MATCH SET r.end_date = null, r.weight = coalesce(r.weight, 1) + 1
        """
        self.graph.query(query, {"source": source, "target": target, "props": props})

    def _get_conflicting_types(self, rel_type: str) -> List[str]:
        for group in CONFLICTING_REL_GROUPS.values():
            if rel_type in group:
                return group
        return [rel_type]

    def _close_category_relationships(self, source_id: str, rel_type: str):
        query = f"MATCH (s {{id: $source}})-[r:{rel_type}]->() WHERE r.end_date IS NULL SET r.end_date = datetime()"
        self.graph.query(query, {"source": source_id})

    def _close_entity_state_relationships(
        self, source_id: str, target_id: str, types: List[str]
    ):
        types_str = "|".join([f"`{t}`" for t in types])
        query = f"MATCH (s {{id: $source}})-[r:{types_str}]->(t {{id: $target}}) WHERE r.end_date IS NULL SET r.end_date = datetime()"
        self.graph.query(query, {"source": source_id, "target": target_id})

    def get_current_user_state(self, user_id: str) -> str:
        """
        Fetches active relationships, SORTED by weight (Priority).
        """
        query = """
        MATCH (u:User {id: $user_id})
        
        // 1. Active Preferences (Sorted by Weight DESC, then Recency)
        OPTIONAL MATCH (u)-[r1:HAS_PREFERENCE]->(p:Preference) 
        WHERE r1.end_date IS NULL
        WITH u, p, r1 ORDER BY r1.weight DESC, r1.start_date DESC
        WITH u, collect(p.id) as preferences
        
        // 2. Active Interests (Sorted by Weight DESC - Highest Priority First)
        OPTIONAL MATCH (u)-[r2:INTERESTED_IN]->(c:Concept) 
        WHERE r2.end_date IS NULL
        WITH u, preferences, c, r2 ORDER BY r2.weight DESC
        // Format: "ConceptName [Priority: 5]"
        WITH u, preferences, collect(c.id + ' [Priority: ' + coalesce(r2.weight, 1) + ']') as interests
        
        // 3. Known Dislikes
        OPTIONAL MATCH (u)-[r3:DISLIKES]->(d:Concept) 
        WHERE r3.end_date IS NULL
        WITH u, preferences, interests, collect(d.id) as dislikes
        
        RETURN u {.*, id:null, created_at:null, last_active:null} as profile, 
               preferences, interests, dislikes
        """
        data = self.graph.query(query, {"user_id": user_id})

        if not data:
            return "New User (No history yet)."

        rec = data[0]

        # We format this string to be very clear for the LLM
        return f"""User Profile & Context:
- Attributes: {rec['profile']}
- Communication Preferences: {', '.join(rec['preferences'])}
- Active Interests (Sorted by Priority): {', '.join(rec['interests'])}
- Dislikes (Do not mention): {', '.join(rec['dislikes'])}"""


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
        known_entities = self.manager.get_known_entities()
        knowledge = await self.extractor.extract_knowledge(
            [HumanMessage(content=user_input), AIMessage(content=response)],
            known_entities,
        )
        self.manager.update_graph(self.user_id, knowledge)

        return response

    async def _generate_response(self, user_input: str, state: str) -> str:
        system_msg = f"""You are a helpful AI assistant with access to a personalized Knowledge Graph.

USER CONTEXT:
{state}

INSTRUCTIONS:
1. **Prioritize High-Value Topics**: 
   - Look at the 'Active Interests' list. Items with higher `[Priority: X]` scores are more important to the user.
   - Focus your examples and analogies around these high-priority concepts.
   
2. **Respect Constraints**:
   - Never mention topics listed in 'Dislikes'.
   - Adapt your tone based on 'Communication Preferences'.

3. **Be Natural**: 
   - Do not say "I see you have a priority 5 interest in Python". 
   - Instead, say "Since you're deeply into Python..." or use Python code snippets naturally.
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

    # # 🛑 RESET GRAPH for a clean demo
    # print("🧹 Clearing Database for Demo...")
    # graph.query("MATCH (n) DETACH DELETE n")

    # # Helper to visualize the graph state
    # def print_graph_snapshot(step_name):
    #     print(f"\n{'='*20} {step_name} {'='*20}")

    #     # Fetch Active Edges
    #     active = graph.query(
    #         """
    #         MATCH (u:User)-[r]->(n)
    #         WHERE r.end_date IS NULL
    #         RETURN type(r) as rel, n.id as node, n.label as label
    #     """
    #     )

    #     # Fetch Archived Edges
    #     archived = graph.query(
    #         """
    #         MATCH (u:User)-[r]->(n)
    #         WHERE r.end_date IS NOT NULL
    #         RETURN type(r) as rel, n.id as node, r.end_date as ended
    #     """
    #     )

    #     print("🟢 ACTIVE STATE:")
    #     if not active:
    #         print("   (Empty)")
    #     for row in active:
    #         print(f"   (User) --[{row['rel']}]--> ({row['node']}) [{row['label']}]")

    #     print("\n🔴 HISTORY (Archived):")
    #     if not archived:
    #         print("   (None)")
    #     for row in archived:
    #         print(
    #             f"   (User) --[{row['rel']}]--> ({row['node']}) [Ended: {row['ended']}]"
    #         )
    #     print("=" * 60)

    # # ============================================================
    # # 📨 Message 1: Initialization
    # # Setting identity, interest, and specific preference.
    # # ============================================================
    # msg_1 = "Hi, I'm Alex. I really love Python development. Please give me detailed, in-depth answers."
    # print(f"\nUser: {msg_1}")
    # await agent.process_interaction(msg_1)
    # print_graph_snapshot("AFTER MESSAGE 1")
    # # EXPECTED:
    # # Active: INTERESTED_IN -> python, HAS_PREFERENCE -> detailed
    # # History: None

    # # ============================================================
    # # 📨 Message 2: Expansion
    # # Adding a new skill (Accumulative change).
    # # ============================================================
    # msg_2 = "I am also starting to learn Docker for containerization. could you explain to me some details on docker?"
    # print(f"\nUser: {msg_2}")
    # await agent.process_interaction(msg_2)
    # print_graph_snapshot("AFTER MESSAGE 2")
    # # EXPECTED:
    # # Active: ... + LEARNING -> docker
    # # History: None

    # # ============================================================
    # # 📨 Message 3: Preference Shift (Conflict Type 1)
    # # Changing "Detailed" -> "Brief".
    # # The 'detailed' edge should close.
    # # ============================================================
    # msg_3 = (
    #     "Actually, your answers are too long. Keep them brief and concise from now on."
    # )
    # print(f"\nUser: {msg_3}")
    # await agent.process_interaction(msg_3)
    # print_graph_snapshot("AFTER MESSAGE 3")
    # # EXPECTED:
    # # Active: HAS_PREFERENCE -> brief, INTERESTED_IN -> python, LEARNING -> docker
    # # History: HAS_PREFERENCE -> detailed

    # # ============================================================
    # # 📨 Message 4: Sentiment Shift (Conflict Type 2)
    # # Changing "Love Python" -> "Dislike Python".
    # # The 'INTERESTED_IN' Python edge should close.
    # # ============================================================
    # msg_4 = "Also, I'm tired of Python. It's too slow. I fucking hate it now. same with docker, so troublesome"
    # print(f"\nUser: {msg_4}")
    # await agent.process_interaction(msg_4)
    # print_graph_snapshot("AFTER MESSAGE 4")
    # # EXPECTED:
    # # Active: DISLIKES -> python, HAS_PREFERENCE -> brief, LEARNING -> docker
    # # History: HAS_PREFERENCE -> detailed, INTERESTED_IN -> python

    msg_5 = "could you describe what you know about "
    print(f"\nUser: {msg_5}")
    await agent.process_interaction(msg_5)
    # print_graph_snapshot("AFTER MESSAGE 4")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
