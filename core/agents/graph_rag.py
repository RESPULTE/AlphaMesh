from typing import Any, Dict, List, Literal

# Preserving strict imports as requested
from core.services import service_manager
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field

# ============================================================================
# 1. Structured Schema Definitions (Pydantic)
# ============================================================================
# These schemas enforce the strict typology of the graph to prevent hallucinations.

# Define allowed Node Labels to keep the graph clean
AllowedNodeLabels = Literal[
    "User",
    "Concept",
    "Technology",
    "Skill",
    "Project",
    "Organization",
    "Location",
    "Person",
]

# Define allowed Relationship Types to ensure traversability
AllowedRelTypes = Literal[
    "INTERESTED_IN",
    "WORKS_ON",
    "HAS_SKILL",
    "USES_TECHNOLOGY",
    "LOCATED_IN",
    "EMPLOYED_BY",
    "RELATED_TO",
    "MENTIONED",
]


class GraphNode(BaseModel):
    """Represents a Node in the Knowledge Graph."""

    id: str = Field(
        description="Unique identifier for the node (usually the name in lowercase)."
    )
    label: AllowedNodeLabels = Field(description="The primary label of the node.")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Attributes of the node (e.g., proficiency, status).",
    )


class GraphRelationship(BaseModel):
    """Represents a directed Relationship between two nodes."""

    source_node_id: str = Field(description="The ID of the source node.")
    target_node_id: str = Field(description="The ID of the target node.")
    type: AllowedRelTypes = Field(description="The type of relationship.")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Attributes of the relationship (e.g., strength, date).",
    )


class KnowledgeGraphUpdate(BaseModel):
    """The structured output container for the extraction process."""

    nodes: List[GraphNode] = Field(
        description="List of entities identified in the conversation."
    )
    relationships: List[GraphRelationship] = Field(
        description="List of relationships identified. Always link the 'User' if the info pertains to them."
    )


# ============================================================================
# 2. Knowledge Extractor (Structured)
# ============================================================================


class KnowledgeExtractor:
    """
    Uses LLM with Structured Output to extract precise graph updates.
    """

    def __init__(self, llm):
        # We enforce the schema using the bind_tools or with_structured_output paradigm
        self.llm = llm.with_structured_output(KnowledgeGraphUpdate)

    async def extract_knowledge(self, chat_history: List[Any]) -> KnowledgeGraphUpdate:
        """
        Extracts entities and relationships based on the conversation history.
        Uses the last few messages to maintain context resolution.
        """

        system_prompt = """You are a top-tier Knowledge Graph Engineer.
Your goal is to extract structured knowledge from a conversation to build a user profile and domain graph.

RULES:
1. **Focus on the User**: Identify who the user is, what they work on, what they know, and what they want.
2. **Ignore Small Talk**: Do not extract entities for greetings, polite filler, or generic statements.
3. **Resolution**: If the user says "I work on *it*", look at the assistant's previous message to resolve "it".
4. **Consistency**: Use consistent naming (e.g., "Python" not "python" or "Python 3").
5. **Privacy**: Do NOT extract the raw message content. Only extract the facts.

When extracting the User, always use the ID 'user_core' (or the specific user name if known) and Label 'User'.
"""

        # We format the history into a string for the extraction context
        formatted_history = "\n".join(
            [
                f"{msg.type.upper()}: {msg.content}" for msg in chat_history[-4:]
            ]  # Look at last 4 messages
        )

        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    f"Analyze this interaction and extract knowledge:\n\n{formatted_history}",
                ),
            ]
        )

        try:
            # The chain automatically returns a KnowledgeGraphUpdate object
            chain = extraction_prompt | self.llm
            result: KnowledgeGraphUpdate = await chain.ainvoke({})
            return result
        except Exception as e:
            print(f"Extraction Error: {e}")
            return KnowledgeGraphUpdate(nodes=[], relationships=[])


# ============================================================================
# 3. Graph Manager
# ============================================================================


class GraphManager:
    """Manages Neo4j operations using optimized Cypher queries."""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._initialize_schema()

    def _initialize_schema(self):
        """Create constraints to ensure data integrity."""
        # Note: In production, specific label constraints are better than generic ones.
        queries = [
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX entity_id IF NOT EXISTS FOR (n:Entity) ON (n.id)",
        ]
        for q in queries:
            try:
                self.graph.query(q)
            except Exception as e:
                # Ignore if constraint already exists
                pass

    def upsert_knowledge(self, knowledge: KnowledgeGraphUpdate):
        """
        Batch updates the graph using UNWIND for performance.
        Handles both Nodes and Relationships.
        """
        if not knowledge.nodes and not knowledge.relationships:
            return

        # 1. Upsert Nodes
        # We convert the Pydantic models to dicts
        nodes_data = [
            {"id": n.id, "label": n.label, "props": n.properties}
            for n in knowledge.nodes
        ]

        # Dynamic Cypher usually requires APOC for dynamic labels,
        # but to stay standard, we can handle the User distinct from generic concepts if needed.
        # Here is a generic approach dealing with dynamic labels is tricky in pure Cypher params.
        # We will iterate node types to handle labels safely.

        for node in nodes_data:
            # Sanitize label to prevent injection (though Pydantic validates this via Literal)
            label = node["label"]
            query = f"""
            MERGE (n:{label} {{id: $id}})
            ON CREATE SET n += $props, n.created_at = timestamp()
            ON MATCH SET n += $props, n.updated_at = timestamp()
            """
            self.graph.query(query, {"id": node["id"], "props": node["props"]})

        # 2. Upsert Relationships
        # We assume nodes exist (created above) or will be created loosely.
        for rel in knowledge.relationships:
            rel_type = rel.type
            query = f"""
            MATCH (s {{id: $source_id}})
            MATCH (t {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            ON CREATE SET r += $props, r.created_at = timestamp()
            ON MATCH SET r += $props, r.updated_at = timestamp()
            """
            try:
                self.graph.query(
                    query,
                    {
                        "source_id": rel.source_node_id,
                        "target_id": rel.target_node_id,
                        "props": rel.properties,
                    },
                )
            except Exception as e:
                print(f"Error creating relationship {rel_type}: {e}")

    def get_user_context(self, user_id: str) -> str:
        """
        Retrieves the immediate neighborhood of the User node.
        """
        query = """
        MATCH (u:User {id: $user_id})-[r]->(n)
        RETURN type(r) as relationship, n.id as entity, labels(n) as labels
        LIMIT 50
        """
        result = self.graph.query(query, {"user_id": user_id})

        if not result:
            return "No prior context known about the user."

        context_str = "Known User Context:\n"
        for row in result:
            # Clean up labels list
            lbl = (
                [l for l in row["labels"] if l != "Entity"][0]
                if row["labels"]
                else "Entity"
            )
            context_str += f"- User {row['relationship']} {row['entity']} ({lbl})\n"

        return context_str


# ============================================================================
# 4. Main Evolving Agent
# ============================================================================


class EvolvingGraphRAGAgent:
    """
    Main controller.
    1. Fetches Graph Context.
    2. Chats with User (using History + Context).
    3. Extracts new info (Structured).
    4. Updates Graph (No raw history stored).
    """

    def __init__(self, graph: Neo4jGraph, llm, user_id: str = "user_core"):
        self.user_id = user_id

        # Services
        self.graph = graph
        self.llm = llm

        # Components
        self.graph_manager = GraphManager(self.graph)
        self.extractor = KnowledgeExtractor(self.llm)

        # In-memory history (NOT stored in graph)
        self.chat_history: List[Any] = []

    async def process_interaction(self, user_input: str) -> str:
        """Full pipeline execution."""

        # 1. Retrieve Dynamic Context
        user_context = self.graph_manager.get_user_context(self.user_id)

        # 2. Update memory temporarily for the response generation
        temp_history = self.chat_history + [HumanMessage(content=user_input)]

        # 3. Generate Response
        response_content = await self._generate_response(
            user_input, user_context, temp_history
        )

        # 4. Update memory officially
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response_content))

        # 5. Extract and Evolve Graph (Background Task ideally)
        # We pass the full history so the LLM can resolve references
        knowledge_update = await self.extractor.extract_knowledge(self.chat_history)

        # 6. Inject the 'User' anchor if missing from extraction
        # (Ensures attributes are linked to the specific ID of this session)
        self._enforce_user_anchor(knowledge_update)

        print(
            f"DEBUG: Extracted {len(knowledge_update.nodes)} nodes, {len(knowledge_update.relationships)} rels."
        )

        # 7. Persist to Graph
        self.graph_manager.upsert_knowledge(knowledge_update)

        return response_content

    async def _generate_response(
        self, user_input: str, context: str, history: List[Any]
    ) -> str:
        """
        Generates the actual chat response using context.
        """
        system_msg = f"""You are a helpful AI assistant.
        
CONTEXT FROM KNOWLEDGE GRAPH:
{context}

INSTRUCTIONS:
- Use the context to personalize the conversation.
- If the user asks about something in the context, refer to it.
- Do not explicitly mention "I found this in the database". act naturally.
"""
        # We only keep the last 10 messages for the chat generation to fit context window
        recent_history = history[-10:] if len(history) > 10 else history

        messages = [SystemMessage(content=system_msg)] + recent_history

        response = await self.llm.ainvoke(messages)
        return response.content

    def _enforce_user_anchor(self, knowledge: KnowledgeGraphUpdate):
        """
        Helper to ensure that if the LLM extracted 'User' generic node,
        it is mapped to our session's actual user_id.
        """
        for node in knowledge.nodes:
            if node.label == "User":
                node.id = self.user_id  # Overwrite generic ID with actual session ID

        for rel in knowledge.relationships:
            # If the extraction logic used a generic name for user, align it
            # This logic depends on the LLM outputting a consistent ID for 'self'
            # Simpler approach: If the prompt instructs to use 'user_core', it aligns naturally.
            if rel.source_node_id.lower() in ["user", "me", "i"]:
                rel.source_node_id = self.user_id
            if rel.target_node_id.lower() in ["user", "me", "i"]:
                rel.target_node_id = self.user_id


# ============================================================================
# Example Execution
# ============================================================================


async def main():
    # 1. Setup
    # Assumes service_manager is configured in your environment
    graph = service_manager.get_graph()
    llm = service_manager.get_agent()

    agent = EvolvingGraphRAGAgent(graph, llm, user_id="user_12345")

    # 2. Simulate Chat
    inputs = [
        "Hi, I'm John. I work as a Python Backend Engineer.",
        "I am currently looking into Neo4j for a project.",
        "Does the system know what my tech stack is?",
        "I also enjoy hiking in my free time.",
    ]

    for inp in inputs:
        print(f"\nUser: {inp}")
        response = await agent.process_interaction(inp)
        print(f"Assistant: {response}")

    # 3. Verify Graph State (Direct Query)
    print("\n--- Graph Verification ---")
    result = graph.query("MATCH (n)-[r]->(m) RETURN n.id, type(r), m.id")
    for r in result:
        print(f"{r['n.id']} --[{r['type(r)']}]--> {r['m.id']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
