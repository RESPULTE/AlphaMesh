import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List

# Preserved existing imports
from core.services import service_manager
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

# Pydantic for strict schema validation
from pydantic import BaseModel, Field

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. GRAPH SCHEMA (Strict Typing)
# ============================================================================


class NodeLabel(str, Enum):
    """Allowed Node Labels to prevent hallucinated types."""

    USER = "User"
    TOPIC = "Topic"
    SKILL = "Skill"
    LOCATION = "Location"
    PERSON = "Person"
    ORGANIZATION = "Organization"
    EVENT = "Event"
    PREFERENCE = "Preference"


class RelationType(str, Enum):
    """Allowed Relationship Types to enforce graph consistency."""

    INTERESTED_IN = "INTERESTED_IN"
    HAS_SKILL = "HAS_SKILL"
    LIVES_IN = "LIVES_IN"
    KNOWS = "KNOWS"
    WORKS_AT = "WORKS_AT"
    ATTENDED = "ATTENDED"
    DISLIKES = "DISLIKES"
    MENTIONED = "MENTIONED"  # Fallback for weak connections


class GraphNode(BaseModel):
    """Represents a single node in the extraction."""

    id: str = Field(
        description="Unique identifier/name for the node (e.g., 'Python', 'New York')."
    )
    label: NodeLabel = Field(description="The strict type of the node.")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Attributes like sentiment, confidence, or specific details.",
    )


class GraphRelationship(BaseModel):
    """Represents a directed relationship between two nodes."""

    source_id: str = Field(description="The ID of the source node.")
    target_id: str = Field(description="The ID of the target node.")
    type: RelationType = Field(description="The strict relationship type.")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context attributes (e.g., 'since_year', 'context').",
    )


class KnowledgeGraphExtraction(BaseModel):
    """The complete structured output from the LLM."""

    nodes: List[GraphNode]
    relationships: List[GraphRelationship]


# ============================================================================
# 2. NEO4J SERVICE (Low-Level Operations)
# ============================================================================


class Neo4jService:
    """Handles raw DB interactions with atomic merges and indexing."""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Ensures performance indexes exist."""
        commands = [
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Topic) ON (n.id)",
            "CREATE INDEX skill_id_index IF NOT EXISTS FOR (n:Skill) ON (n.id)",
        ]
        for cmd in commands:
            try:
                self.graph.query(cmd)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")

    async def persist_data(self, data: KnowledgeGraphExtraction):
        """
        Atomically merges nodes and relationships.
        Uses UNWIND for batch processing efficiency.
        """
        if not data.nodes and not data.relationships:
            return

        # 1. Merge Nodes
        # We handle dynamic labels by grouping nodes by label first or using a generic approach with APOC
        # simplified here: We assume ID uniqueness across labels or use a Generic Label + Specific Label

        node_query = """
        UNWIND $nodes AS n
        MERGE (e:Entity {id: n.id})
        ON CREATE SET e.created_at = timestamp()
        SET e.updated_at = timestamp(),
            e += n.properties
        WITH e, n
        CALL apoc.create.addLabels(e, [n.label]) YIELD node
        RETURN count(node)
        """

        # Prepare node data dicts
        node_params = [
            {"id": n.id, "label": n.label.value, "properties": n.properties}
            for n in data.nodes
        ]

        try:
            self.graph.query(node_query, {"nodes": node_params})
        except Exception as e:
            logger.error(f"Error merging nodes: {e}")

        # 2. Merge Relationships
        rel_query = """
        UNWIND $rels AS r
        MATCH (s:Entity {id: r.source_id})
        MATCH (t:Entity {id: r.target_id})
        MERGE (s)-[rel:RELATIONSHIP {type: r.type}]->(t)
        ON CREATE SET rel.created_at = timestamp()
        SET rel += r.properties, rel.updated_at = timestamp()
        WITH s, t, r, rel
        CALL apoc.refactor.setType(rel, r.type) YIELD output
        RETURN count(output)
        """
        # Note: Dynamic relationship types in pure Cypher require APOC or literal injection.
        # Safe Approach: Iterate or use APOC. Here we use APOC `setType` or standard Cypher injection if strictly validated.

        # Since we use Enums, we can trust the types. However, MERGE doesn't accept dynamic types easily.
        # We will do a loop for reliability in this specific implementation, or use a specific query per type.
        # For production speed, we group by relationship type.

        for rel in data.relationships:
            query = f"""
            MATCH (s:Entity {{id: $source_id}})
            MATCH (t:Entity {{id: $target_id}})
            MERGE (s)-[r:{rel.type.value}]->(t)
            ON CREATE SET r.created_at = timestamp()
            SET r += $properties, r.updated_at = timestamp()
            """
            try:
                self.graph.query(
                    query,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "properties": rel.properties,
                    },
                )
            except Exception as e:
                logger.error(f"Error merging relationship {rel.type}: {e}")


# ============================================================================
# 3. KNOWLEDGE MEMORY (The "Write" Path)
# ============================================================================


class KnowledgeGraphMemory:
    """Handles extracting insights and persisting them to Neo4j."""

    def __init__(self, llm, neo4j_service: Neo4jService):
        self.llm = llm
        self.neo4j_service = neo4j_service

        # Configure the structured LLM
        self.structured_llm = self.llm.with_structured_output(KnowledgeGraphExtraction)

        self.prompt = ChatPromptTemplate.from_template(
            """
        You are a Knowledge Graph Architect building a digital twin of the USER.
        
        Your goal is to extract structured knowledge from the chat history.
        
        RULES:
        1. **User-Centric**: Focus primarily on the 'User' node. Connect facts to them.
           - If user says "I love Python", create (User)-[:INTERESTED_IN]->(Topic:Python).
           - Do NOT create isolated facts like (Python)-[:IS_A]->(Language) unless relevant to the user's context.
        2. **Strict Schema**: You can ONLY use the allowed NodeLabels and RelationTypes provided.
        3. **Resolution**: Resolve "I", "me", "my" to the node ID "User_Main".
        4. **Deduplication**: If a fact implies a stronger version of an existing one, output it to update properties.
        
        Existing Chat Context:
        {history}
        
        Extract the nodes and relationships now.
        """
        )

    async def extract_and_save(self, history: List[Any]):
        """Runs in background to evolve graph."""
        try:
            # Format history for prompt
            history_text = "\n".join(
                [f"{msg.type}: {msg.content}" for msg in history[-4:]]
            )  # Last 4 messages

            chain = self.prompt | self.structured_llm
            result: KnowledgeGraphExtraction = await chain.ainvoke(
                {"history": history_text}
            )

            # Ensure "User_Main" is always of type User if present
            for node in result.nodes:
                if node.id == "User_Main":
                    node.label = NodeLabel.USER

            logger.info(
                f"Extracted {len(result.nodes)} nodes and {len(result.relationships)} relations."
            )
            await self.neo4j_service.persist_data(result)

        except Exception as e:
            logger.error(f"Extraction failed: {e}")


# ============================================================================
# 4. GRAPH RETRIEVER (The "Read" Path)
# ============================================================================


class GraphRAGRetriever:
    """Retrieves relevant context by traversing the graph."""

    def __init__(self, graph: Neo4jGraph, user_id: str = "User_Main"):
        self.graph = graph
        self.user_id = user_id

    async def get_context(self, user_query: str) -> str:
        """
        Performs a hybrid retrieval:
        1. Identifies entities in the query.
        2. Fetches the User's direct profile.
        3. Traverses 1-hop from query entities.
        """

        # 1. Always get direct user profile (Strongest Context)
        user_profile_query = """
        MATCH (u:User {id: $user_id})-[r]->(n)
        RETURN n.id as node, type(r) as rel, n.properties as props
        LIMIT 15
        """
        user_data = self.graph.query(user_profile_query, {"user_id": self.user_id})

        # 2. Keyword/Entity Search (Simple implementation)
        # In a full prod system, use vector search here.
        # For now, we use a simple case-insensitive substring match on the query.
        # "Tell me about Python" -> matches node "Python"

        entity_search_query = """
        MATCH (n:Entity)
        WHERE toLower($query) CONTAINS toLower(n.id) AND n.id <> $user_id
        MATCH (n)-[r]-(m)
        RETURN n.id as source, type(r) as rel, m.id as target
        LIMIT 10
        """
        rel_data = self.graph.query(
            entity_search_query, {"query": user_query, "user_id": self.user_id}
        )

        # Format Context
        context_lines = ["**User Profile:**"]
        for row in user_data:
            context_lines.append(f"- User {row['rel']} {row['node']} ({row['props']})")

        if rel_data:
            context_lines.append("\n**Relevant Topic Graph:**")
            for row in rel_data:
                context_lines.append(
                    f"- {row['source']} --[{row['rel']}]--> {row['target']}"
                )

        return "\n".join(context_lines)


# ============================================================================
# 5. MAIN AGENT (The Orchestrator)
# ============================================================================


class DigitalTwinAgent:
    """
    Main controller.
    1. Retrieve Context
    2. Generate Answer
    3. Trigger Evolution (Background)
    """

    def __init__(self):
        self.llm = service_manager.get_agent()  # Getting LLM
        self.graph = service_manager.get_graph()  # Getting Neo4jGraph

        self.neo4j_service = Neo4jService(self.graph)
        self.memory = KnowledgeGraphMemory(self.llm, self.neo4j_service)
        self.retriever = GraphRAGRetriever(self.graph)

        self.chat_history: List[Any] = []

        # Answer Generation Prompt
        self.response_prompt = ChatPromptTemplate.from_template(
            """
        You are a helpful AI assistant. You have access to a 'Digital Twin' graph of the user.
        
        Graph Context:
        {context}
        
        Chat History:
        {chat_history}
        
        User: {input}
        
        Answer the user naturally. If the context shows they know something, acknowledge it.
        """
        )

    async def chat(self, user_message: str) -> str:
        # 1. Retrieve
        context = await self.retriever.get_context(user_message)

        # 2. Generate
        # Format history for the generation model
        history_msgs = [f"{m.type}: {m.content}" for m in self.chat_history[-5:]]

        chain = self.response_prompt | self.llm
        response = await chain.ainvoke(
            {
                "context": context,
                "chat_history": "\n".join(history_msgs),
                "input": user_message,
            }
        )

        ai_message = response.content

        # 3. Update State
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=ai_message))

        # 4. Evolve Graph (Background Task)
        # In a web server (FastAPI), utilize BackgroundTasks. Here we await for simplicity.
        await self.memory.extract_and_save(self.chat_history)

        return ai_message


# ============================================================================
# EXECUTION
# ============================================================================


async def main():
    print("Initializing Digital Twin System...")
    agent = DigitalTwinAgent()

    # Simulate a user session
    inputs = [
        "Hi, I am a data scientist living in Malaysia.",
        "I specialize in Python and LangChain.",
        "What do you know about my tech stack?",
        "I'm also learning Rust now, it's difficult but cool.",
        "Where do I live again?",
    ]

    for inp in inputs:
        print(f"\nUser: {inp}")
        response = await agent.chat(inp)
        print(f"Agent: {response}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
