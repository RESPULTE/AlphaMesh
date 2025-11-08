# components/rag_handler.py
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_neo4j import GraphCypherQAChain
from langchain_community.vectorstores import Chroma

from . import prompts

logger = logging.getLogger(__name__)

# Pydantic models for structured output
class VectorDoc(BaseModel):
    text: str = Field(description="The text content of the document to be added to the vector store.")
    entities: List[str] = Field(description="A list of key entities present in the text.")

class VectorUpdates(BaseModel):
    add_documents: List[VectorDoc] = Field(default_factory=list, description="List of new documents to add to the vector store.")
    delete_by_entity: List[str] = Field(default_factory=list, description="List of entity names. All documents associated with these entities will be deleted.")

class MemoryModification(BaseModel):
    graph_updates: List[str] = Field(default_factory=list, description="A list of Cypher statements to execute.")
    vector_updates: VectorUpdates = Field(default_factory=VectorUpdates, description="Instructions for updating the vector store.")


class UserProfileRAG:
    """
    Handles synchronized Retrieval and Storage for a user's profile across
    a knowledge graph (long-term) and a vector store (short-term).
    """
    def __init__(self, user_id: str, llm: BaseChatModel, graph, vector_store: Chroma):
        if not all([user_id, llm, graph, vector_store]):
            raise ValueError("All arguments are required.")
            
        self.user_id = user_id
        self.llm = llm
        self.graph = graph
        self.vector_store = vector_store
        self.memory_modification_parser = JsonOutputParser(pydantic_object=MemoryModification)

    def _delete_vector_docs_by_entity(self, entity: str):
        """Finds and deletes documents in Chroma by a metadata entity tag."""
        try:
            # This logic assumes ChromaDB and its specific 'where' filter capabilities.
            # The '$contains' operator works on list-type metadata fields.
            docs_to_delete = self.vector_store.get(
                where={"$and": [
                    {"user_id": self.user_id}, 
                    {"entities": {"$in": entity}}
                ]}
            )
            
            if ids_to_delete := docs_to_delete.get("ids"):
                self.vector_store.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} vector documents for user {self.user_id} related to entity: '{entity}'")
            else:
                logger.info(f"No vector documents found to delete for entity: '{entity}'")
        except Exception as e:
            logger.error(f"Error deleting vector docs for entity '{entity}' for user {self.user_id}: {e}")

    def _consolidate_memories(self, conversation_turn: str):
        """
        Uses an LLM to generate and execute a plan for updating both the graph
        and vector store memories based on the latest conversation.
        """
        # CORRECTED: Simplified PromptTemplate instantiation. It now correctly infers
        # input_variables and properly injects the partial 'format_instructions'.
        prompt = PromptTemplate(
            template=prompts.MEMORY_MODIFICATION_TEMPLATE,
            partial_variables={"format_instructions": self.memory_modification_parser.get_format_instructions()}
        )
        
        modification_chain = prompt | self.llm | self.memory_modification_parser

        try:
            modification_plan = modification_chain.invoke({
                "user_id": self.user_id, 
                "conversation_turn": conversation_turn
            })

            if not modification_plan:
                logger.info("Memory consolidation resulted in no changes.")
                return
            
            # 1. Execute Graph Updates
            if graph_queries := modification_plan.get('graph_updates'):
                for query in graph_queries:
                    logger.info(f"Executing Cypher: {query}")
                    self.graph.query(query)
                logger.info(f"Updated knowledge graph for user {self.user_id}.")

            # 2. Execute Vector Store Updates
            if vector_plan := modification_plan.get('vector_updates'):
                # 2a. Deletions (must happen before additions)
                if entities_to_delete := vector_plan.get('delete_by_entity'):
                    self._delete_vector_docs_by_entity(entities_to_delete)
                
                # 2b. Additions
                if docs_to_add_data := vector_plan.get('add_documents'):
                    docs_to_add = [
                        Document(
                            page_content=doc['text'],
                            metadata={"user_id": self.user_id, "entities": ", ".join(doc['entities'])}
                        ) for doc in docs_to_add_data
                    ]
                    if docs_to_add:
                        self.vector_store.add_documents(docs_to_add)
                        logger.info(f"Added {len(docs_to_add)} new documents to vector store for user {self.user_id}.")

        except Exception as e:
            logger.error(f"Failed to consolidate memories for user {self.user_id}: {e}", exc_info=True)

    # ... (the rest of the UserProfileRAG class and generate_augmented_response function are unchanged)
    def update_memories_from_turn(self, user_input: str, llm_output: str):
        """
        Public method to trigger memory consolidation after a conversation turn.
        """
        conversation_turn = f"User: \"{user_input}\"\nAssistant: \"{llm_output}\""
        self._consolidate_memories(conversation_turn)

    def _retrieve_short_term_context(self, user_input: str) -> str:
        """Retrieves relevant recent conversation topics from the vector store."""
        try:
            docs = self.vector_store.similarity_search(user_input, k=3, filter={"user_id": self.user_id})
            return "\n".join([doc.page_content for doc in docs]) if docs else "No recent context available."
        except Exception as e:
            logger.warning(f"Could not retrieve from vector store for user {self.user_id}: {e}")
            return "No recent context available."

    def _retrieve_long_term_context(self) -> str:
        """Retrieves summarized user facts from the knowledge graph."""
        try:
            graph_query = f"Summarize all known information about the user with id '{self.user_id}', including their name, interests, profession, and goals."
            cypher_chain = GraphCypherQAChain.from_llm(graph=self.graph, llm=self.llm, validate_cypher=True, allow_dangerous_requests=True)
            result = cypher_chain.invoke({"query": graph_query})
            return result.get('result', 'No long-term profile available.')
        except Exception as e:
            logger.warning(f"Could not retrieve from graph for user {self.user_id}: {e}")
            return "No long-term profile available."

    def get_context_for_prompt(self, user_input: str) -> dict:
        """
        Retrieves all necessary context for the LLM. This is a read-only operation.
        """
        short_term_context = self._retrieve_short_term_context(user_input)
        logger.info(f"Retrieved short-term context: {short_term_context}")

        long_term_context = self._retrieve_long_term_context()
        logger.info(f"Retrieved long-term context: {long_term_context}")

        return {
            "short_term_context": short_term_context,
            "long_term_context": long_term_context
        }


def generate_augmented_response(
    llm: BaseChatModel,
    short_term_context: str,
    long_term_context: str,
    user_input: str
) -> str:
    # This function remains unchanged
    augmented_prompt = PromptTemplate.from_template(prompts.AUGMENTED_RESPONSE_TEMPLATE)
    generation_chain = augmented_prompt | llm
    response = generation_chain.invoke({
        "short_term_context": short_term_context,
        "long_term_context": long_term_context,
        "user_input": user_input
    })
    return response.content