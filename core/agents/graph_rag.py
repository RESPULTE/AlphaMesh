"""
Evolving Graph RAG System with Neo4j and LangChain
Continuously builds and expands knowledge graph from user interactions
"""

from datetime import datetime
from typing import Any, Dict, List

from core.config import settings
from core.services import service_manager
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from pydantic import BaseModel, Field

# ============================================================================
# Configuration
# ============================================================================


class GraphRAGConfig(BaseModel):
    """Configuration for Graph RAG system"""

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    openai_api_key: str = Field(default="")
    model_name: str = Field(default="gpt-4")


# ============================================================================
# Entity and Relation Extraction
# ============================================================================

EXTRACTION_PROMPT = """You are an expert at extracting entities and relationships from conversational text.

Extract the following from the conversation:
1. Entities: People, places, organizations, concepts, topics, preferences, facts
2. Relationships: How entities relate to each other
3. User attributes: Any information about the user (preferences, background, goals, etc.)

Format your response as JSON:
{{
    "entities": [
        {{"name": "entity_name", "type": "entity_type", "properties": {{"key": "value"}}}}
    ],
    "relationships": [
        {{"source": "entity1", "target": "entity2", "type": "relationship_type", "properties": {{"key": "value"}}}}
    ],
    "user_attributes": [
        {{"attribute": "attribute_name", "value": "attribute_value", "context": "context"}}
    ]
}}

Conversation to analyze:
User: {user_message}
Assistant: {assistant_message}

Current timestamp: {timestamp}
"""


class KnowledgeExtractor:
    """Extracts structured knowledge from conversations"""

    def __init__(self, llm):
        self.llm = llm
        self.extraction_prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)

    async def extract_knowledge(
        self, user_message: str, assistant_message: str
    ) -> Dict[str, Any]:
        """Extract entities and relationships from conversation"""

        prompt = self.extraction_prompt.format(
            user_message=user_message,
            assistant_message=assistant_message,
            timestamp=datetime.now().isoformat(),
        )

        response = await self.llm.ainvoke(prompt)

        # Parse JSON response
        import json

        try:
            raw_text: str = response.content
            knowledge = json.loads(
                raw_text.removeprefix("```json\n").removesuffix("\n```")
            )
            print(json.dumps(knowledge, indent=2))

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            knowledge = {"entities": [], "relationships": [], "user_attributes": []}

        return knowledge


# ============================================================================
# Graph Manager
# ============================================================================


class GraphManager:
    """Manages the Neo4j knowledge graph"""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._initialize_constraints()

    def _initialize_constraints(self):
        """Create uniqueness constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                self.graph.query(constraint)
            except Exception as e:
                print(f"Constraint already exists or error: {e}")

    def add_or_update_entity(
        self, name: str, entity_type: str, properties: Dict[str, Any]
    ):
        """Add or update an entity in the graph"""

        set_create = ["e.type = $entity_type", "e.created_at = timestamp()"]
        set_match = ["e.updated_at = timestamp()"]

        if properties:
            prop_assignments = [f"e.{k} = ${k}" for k in properties]
            set_create += prop_assignments
            set_match += prop_assignments

        create_str = ", ".join(set_create)
        match_str = ", ".join(set_match)

        query = f"""
        MERGE (e:Entity {{name: $name}})
        ON CREATE SET {create_str}
        ON MATCH SET {match_str}
        RETURN e
        """

        params = {"name": name, "entity_type": entity_type, **properties}
        self.graph.query(query, params)

    def add_relationship(
        self, source: str, target: str, rel_type: str, properties: Dict[str, Any] = None
    ):
        """Add a relationship between entities"""

        properties = properties or {}
        properties["created_at"] = datetime.now().isoformat()

        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])

        query = f"""
        MATCH (s:Entity {{name: $source}})
        MATCH (t:Entity {{name: $target}})
        MERGE (s)-[r:{rel_type} {{{props_str}}}]->(t)
        RETURN r
        """

        params = {"source": source, "target": target, **properties}

        try:
            self.graph.query(query, params)
        except Exception as e:
            print(f"Error creating relationship: {e}")

    def add_user_attribute(
        self, user_id: str, attribute: str, value: str, context: str
    ):
        """Add or update user attributes"""

        query = """
        MERGE (u:User {id: $user_id})
        ON CREATE SET u.created_at = timestamp()
        SET u.updated_at = timestamp()
        MERGE (u)-[r:HAS_ATTRIBUTE]->(a:Attribute {name: $attribute})
        ON CREATE SET a.value = $value, a.context = $context, a.created_at = timestamp()
        ON MATCH SET a.value = $value, a.context = $context, a.updated_at = timestamp()
        RETURN u, a
        """

        self.graph.query(
            query,
            {
                "user_id": user_id,
                "attribute": attribute,
                "value": value,
                "context": context,
            },
        )

    def store_conversation(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        conversation_id: str,
    ):
        """Store conversation in graph"""

        query = """
        MERGE (u:User {id: $user_id})
        MERGE (c:Conversation {id: $conversation_id})
        ON CREATE SET c.created_at = timestamp()
        CREATE (m:Message {
            user_message: $user_message,
            assistant_message: $assistant_message,
            timestamp: timestamp()
        })
        MERGE (u)-[:HAD_CONVERSATION]->(c)
        MERGE (c)-[:CONTAINS]->(m)
        RETURN m
        """

        self.graph.query(
            query,
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
            },
        )

    def get_user_context(self, user_id: str) -> str:
        """Retrieve user context from graph"""

        query = """
        MATCH (u:User {id: $user_id})-[:HAS_ATTRIBUTE]->(a:Attribute)
        RETURN a.name AS attribute, a.value AS value, a.context AS context
        ORDER BY a.updated_at DESC
        LIMIT 20
        """

        results = self.graph.query(query, {"user_id": user_id})

        if not results:
            return "No user context available."

        context_parts = []
        for record in results:
            context_parts.append(
                f"- {record['attribute']}: {record['value']} (Context: {record['context']})"
            )

        return "User Context:\n" + "\n".join(context_parts)

    def query_related_entities(self, entity_name: str, depth: int = 2) -> str:
        """Query entities related to a given entity"""

        query = f"""
        MATCH path = (e:Entity {{name: $entity_name}})-[*1..{depth}]-(related:Entity)
        RETURN related.name AS name, related.type AS type, 
               [r IN relationships(path) | type(r)] AS relationships
        LIMIT 20
        """

        results = self.graph.query(query, {"entity_name": entity_name})

        if not results:
            return f"No related entities found for {entity_name}."

        context_parts = [f"Entities related to {entity_name}:"]
        for record in results:
            context_parts.append(
                f"- {record['name']} ({record['type']}) via {' -> '.join(record['relationships'])}"
            )

        return "\n".join(context_parts)


# ============================================================================
# Main Graph RAG Agent
# ============================================================================


class EvolvingGraphRAGAgent:
    """Main agent that evolves understanding through graph interactions"""

    def __init__(self, config: GraphRAGConfig, user_id: str = "default_user"):
        self.config = config
        self.user_id = user_id
        self.conversation_id = f"conv_{datetime.now().timestamp()}"

        # Initialize components
        self.llm = service_manager.get_agent()

        self.graph = Neo4jGraph(
            url=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
        )

        self.graph_manager = GraphManager(self.graph)
        self.extractor = KnowledgeExtractor(self.llm)

        # Conversation history
        self.chat_history: List[Any] = []

    async def process_interaction(self, user_message: str) -> str:
        """Process user interaction and evolve the graph"""

        # 1. Get user context from graph
        user_context = self.graph_manager.get_user_context(self.user_id)

        # 2. Generate response with context
        response = await self._generate_response(user_message, user_context)

        # 3. Extract knowledge from interaction
        knowledge = await self.extractor.extract_knowledge(user_message, response)

        # 4. Update graph with extracted knowledge
        await self._update_graph(knowledge, user_message, response)

        # 5. Store conversation
        self.graph_manager.store_conversation(
            self.user_id, user_message, response, self.conversation_id
        )

        # 6. Update chat history
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=response))

        return response

    async def _generate_response(self, user_message: str, user_context: str) -> str:
        """Generate contextualized response"""

        system_prompt = f"""You are a helpful AI assistant with access to a knowledge graph.
        
{user_context}

Use this context to provide personalized and contextual responses. 
If the user asks about topics related to their context, incorporate that information naturally."""

        messages = [
            ("system", system_prompt),
            *[
                (msg.type, msg.content) for msg in self.chat_history[-10:]
            ],  # Last 10 messages
            ("human", user_message),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def _update_graph(
        self, knowledge: Dict[str, Any], user_message: str, assistant_message: str
    ):
        """Update graph with extracted knowledge"""

        # Add entities
        for entity in knowledge.get("entities", []):
            self.graph_manager.add_or_update_entity(
                name=entity["name"],
                entity_type=entity["type"],
                properties=entity.get("properties", {}),
            )

        # Add relationships
        for rel in knowledge.get("relationships", []):
            self.graph_manager.add_relationship(
                source=rel["source"],
                target=rel["target"],
                rel_type=rel["type"],
                properties=rel.get("properties", {}),
            )

        # Add user attributes
        for attr in knowledge.get("user_attributes", []):
            self.graph_manager.add_user_attribute(
                user_id=self.user_id,
                attribute=attr["attribute"],
                value=attr["value"],
                context=attr.get("context", ""),
            )

    def query_graph(self, query: str) -> str:
        """Query the graph using Cypher"""

        qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm, graph=self.graph, verbose=True
        )

        try:
            result = qa_chain.invoke({"query": query})
            return result["result"]
        except Exception as e:
            return f"Error querying graph: {e}"

    def get_user_profile(self) -> Dict[str, Any]:
        """Get complete user profile from graph"""

        query = """
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)-[:HAS_ATTRIBUTE]->(a:Attribute)
        OPTIONAL MATCH (u)-[:HAD_CONVERSATION]->(c:Conversation)
        RETURN u, 
               collect(DISTINCT {attribute: a.name, value: a.value}) AS attributes,
               count(DISTINCT c) AS conversation_count
        """

        result = self.graph.query(query, {"user_id": self.user_id})

        if result:
            return result[0]
        return {}


# ============================================================================
# Example Usage
# ============================================================================


async def main():
    """Example usage of the Evolving Graph RAG system"""

    # Configuration
    config = GraphRAGConfig(
        neo4j_uri=settings.NEO4J_URL,
        neo4j_username=settings.NEO4J_USERNAME,
        neo4j_password=settings.NEO4J_PASSWORD,
        openai_api_key=settings.GOOGLE_API_KEY,
        model_name=settings.LLM_MODEL,
    )

    # Initialize agent
    agent = EvolvingGraphRAGAgent(config, user_id="user_123")

    # Simulate conversation
    conversations = [
        "Hi! I'm a software engineer interested in machine learning.",
        "I'm currently working on a project involving natural language processing.",
        "Can you tell me about graph databases?",
        "What do you remember about my interests?",
    ]

    for user_msg in conversations:
        print(f"\n{'='*60}")
        print(f"User: {user_msg}")
        response = await agent.process_interaction(user_msg)
        print(f"Assistant: {response}")

    # Query the graph
    print(f"\n{'='*60}")
    print("User Profile:")
    profile = agent.get_user_profile()
    print(profile)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
