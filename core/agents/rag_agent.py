"""
hybrid_rag_subgraph.py

A reusable LangGraph subgraph module implementing Hybrid RAG with:
- Query Enhancement (Rewriting)
- Retrieval Validation (Relevance Grading)
- Answer Generation & Validation (Hallucination Check)
- Storage Decision (Conditional Upsert)

References:
- Retrieval: https://docs.langchain.com/oss/python/langchain/retrieval
- RAG Patterns: https://docs.langchain.com/oss/python/langchain/rag
- Subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs#subgraphs
- Agentic RAG: https://docs.langchain.com/oss/python/langgraph/agentic-rag
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Literal

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START

# Configure Logging
logger = logging.getLogger(__name__)

# --- Configuration Models (Pydantic v2) ---


class HybridRAGConfig(BaseModel):
    """Configuration for the Hybrid RAG Subgraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_retrieval_attempts: int = Field(
        default=3, description="Max retries for query refinement."
    )
    min_relevance_score: float = Field(
        default=0.5, description="Threshold for document relevance."
    )
    auto_upsert: bool = Field(
        default=False,
        description="If True, automatically upserts valid new docs to vector_store.",
    )
    human_review_required: bool = Field(
        default=False, description="If True, skips upsert and flags for review."
    )
    domain_allowlist: List[str] = Field(
        default_factory=list, description="Allowed domains for storage."
    )


# --- State Definition ---


class HybridRAGState(TypedDict):
    """
    Represents the state of the RAG flow.
    Ref: https://docs.langchain.com/oss/python/langgraph/concepts/low_level#state
    """

    query: str
    original_query: str
    documents: List[Document]
    generation: str
    retrieval_attempts: int
    is_relevant: bool
    is_grounded: bool
    upsert_status: str  # 'skipped', 'stored', 'pending_review', 'failed'
    error: Optional[str]


# --- Internal Helper Models for Structured Output ---


class GradeDocuments(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(
        description="Documents are relevant to the query, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination check."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswerQuality(BaseModel):
    """Binary score for answer usefulness."""

    binary_score: str = Field(description="Answer addresses the query, 'yes' or 'no'")


# --- The Subgraph Builder Class ---


class HybridRAGBuilder:
    """
    Builder class to encapsulate logic and dependencies for the subgraph.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        vector_store: Optional[VectorStore] = None,
        config: Optional[HybridRAGConfig] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.vector_store = vector_store
        self.config = config or HybridRAGConfig()

        # -- Prompts --
        # Ref: https://docs.langchain.com/oss/python/langchain/rag#query-transformations
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant that rewrites queries to be more effective for retrieval. \n"
            "Original query: {query} \n"
            "Output only the rewritten query, nothing else."
        )

        # Ref: https://docs.langchain.com/oss/python/langgraph/agentic-rag#retrieval-grader
        self.retrieval_grader_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grader assessing relevance of a retrieved document to a user question. \n"
                    "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n"
                    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.",
                ),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {query}",
                ),
            ]
        )

        self.rag_prompt = ChatPromptTemplate.from_template(
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \n"
            "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
            "Question: {query} \n"
            "Context: {context} \n"
            "Answer:"
        )

        # Ref: https://docs.langchain.com/oss/python/langgraph/agentic-rag#hallucination-grader
        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n"
                    "Give a binary score 'yes' or 'no'. 'yes' means the answer is fully supported by the facts.",
                ),
                (
                    "human",
                    "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
                ),
            ]
        )

        self.answer_grader_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grader assessing whether an answer addresses / resolves a question. \n"
                    "Give a binary score 'yes' or 'no'. 'yes' means the answer resolves the question.",
                ),
                ("human", "User question: {query} \n\n LLM generation: {generation}"),
            ]
        )

    # --- Nodes ---

    def rewrite_query(self, state: HybridRAGState) -> Dict[str, Any]:
        """
        Node: Rewrites the query to improve retrieval.
        """
        logger.info("---NODE: REWRITE QUERY---")
        query = state["query"]

        # Simple chain for rewriting
        chain = self.rewrite_prompt | self.llm
        better_query = chain.invoke({"query": query})

        return {
            "query": better_query.content,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    def retrieve(self, state: HybridRAGState) -> Dict[str, Any]:
        """
        Node: Retrieves documents using the injected retriever.
        Ref: https://docs.langchain.com/oss/python/langchain/retrieval
        """
        logger.info("---NODE: RETRIEVE---")
        query = state["query"]
        try:
            documents = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {"error": str(e), "documents": []}

        return {"documents": documents}

    def grade_documents(self, state: HybridRAGState) -> Dict[str, Any]:
        """
        Node: Filters retrieved documents for relevance.
        Ref: https://docs.langchain.com/oss/python/langgraph/agentic-rag#retrieval-grader
        """
        logger.info("---NODE: GRADE DOCUMENTS---")
        query = state["query"]
        documents = state["documents"]

        # Use structured output for reliable grading
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        grader_chain = self.retrieval_grader_prompt | structured_llm_grader

        filtered_docs = []
        for doc in documents:
            score = grader_chain.invoke({"query": query, "document": doc.page_content})
            if score.binary_score.lower() == "yes":
                filtered_docs.append(doc)

        is_relevant = len(filtered_docs) > 0
        return {"documents": filtered_docs, "is_relevant": is_relevant}

    def generate(self, state: HybridRAGState) -> Dict[str, Any]:
        """
        Node: Generates the answer using retrieved context.
        """
        logger.info("---NODE: GENERATE---")
        query = state["query"]
        documents = state["documents"]

        # Format context
        context = "\n\n".join([d.page_content for d in documents])

        chain = self.rag_prompt | self.llm | StrOutputParser()
        generation = chain.invoke({"context": context, "query": query})

        return {"generation": generation}

    def validate_answer(self, state: HybridRAGState) -> Dict[str, Any]:
        """
        Node: Checks for hallucinations and answer quality.
        Ref: https://docs.langchain.com/oss/python/langgraph/agentic-rag#hallucination-grader
        """
        logger.info("---NODE: VALIDATE ANSWER---")
        query = state["query"]
        documents = state["documents"]
        generation = state["generation"]

        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        hallucination_chain = self.hallucination_prompt | structured_llm_grader

        # Check 1: Hallucination
        grade_hallucination = hallucination_chain.invoke(
            {"documents": documents, "generation": generation}
        )

        if grade_hallucination.binary_score.lower() == "yes":
            # Check 2: Answer Quality
            structured_llm_qa = self.llm.with_structured_output(GradeAnswerQuality)
            answer_chain = self.answer_grader_prompt | structured_llm_qa
            grade_answer = answer_chain.invoke(
                {"query": query, "generation": generation}
            )

            if grade_answer.binary_score.lower() == "yes":
                return {"is_grounded": True}

        return {"is_grounded": False}

    def decide_and_store(self, state: HybridRAGState) -> Dict[str, Any]:
        """
        Node: Decides whether to store documents in the vector store.
        Logic: Checks config flags, domain allowlists, and deduplication (naive).
        """
        logger.info("---NODE: DECIDE AND STORE---")

        if not self.vector_store or not self.config.auto_upsert:
            return {"upsert_status": "skipped"}

        if self.config.human_review_required:
            return {"upsert_status": "pending_review"}

        docs_to_store = []
        for doc in state["documents"]:
            # Security/Safety: Domain Allowlist Check
            source = doc.metadata.get("source", "")
            if self.config.domain_allowlist and source:
                if not any(domain in source for domain in self.config.domain_allowlist):
                    continue  # Skip unauthorized domains

            # Heuristic: Content Length & Quality (Simple example)
            if len(doc.page_content) < 50:
                continue

            # Add ID if missing to help with idempotency
            if "id" not in doc.metadata:
                doc.metadata["id"] = str(uuid.uuid4())

            docs_to_store.append(doc)

        if docs_to_store:
            try:
                # Ref: https://docs.langchain.com/oss/python/langchain/retrieval/vector_stores
                self.vector_store.add_documents(docs_to_store)
                logger.info(f"Upserted {len(docs_to_store)} documents.")
                return {"upsert_status": "stored"}
            except Exception as e:
                logger.error(f"Upsert failed: {e}")
                return {"upsert_status": "failed"}

        return {"upsert_status": "skipped_criteria"}

    # --- Conditional Edges ---

    def route_retrieval(
        self, state: HybridRAGState
    ) -> Literal["rewrite_query", "generate"]:
        """
        Edge: Decides whether to regenerate query or proceed to generation.
        """
        if state["is_relevant"]:
            return "generate"

        if state["retrieval_attempts"] >= self.config.max_retrieval_attempts:
            # Fallback: try to generate with what we have or just end (here we try generate)
            logger.warning(
                "Max retrieval attempts reached. Proceeding with best effort."
            )
            return "generate"

        return "rewrite_query"

    def route_validation(
        self, state: HybridRAGState
    ) -> Literal["rewrite_query", "decide_and_store"]:
        """
        Edge: If answer is hallucinated/bad, retry retrieval (or generation).
        """
        if state["is_grounded"]:
            return "decide_and_store"

        if state["retrieval_attempts"] >= self.config.max_retrieval_attempts:
            return "decide_and_store"  # Give up and return what we have

        return "rewrite_query"  # Loop back to try finding better context

    # --- Graph Construction ---

    def build(self) -> StateGraph:
        workflow = StateGraph(HybridRAGState)

        # Add Nodes
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("validate_answer", self.validate_answer)
        workflow.add_node("decide_and_store", self.decide_and_store)

        # Add Edges
        # Entry point logic: could start at rewrite or retrieve.
        # We start at retrieve for the first pass using original query.
        workflow.add_edge(START, "retrieve")

        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            self.route_retrieval,
            {"rewrite_query": "rewrite_query", "generate": "generate"},
        )

        workflow.add_edge("rewrite_query", "retrieve")

        workflow.add_edge("generate", "validate_answer")

        workflow.add_conditional_edges(
            "validate_answer",
            self.route_validation,
            {"rewrite_query": "rewrite_query", "decide_and_store": "decide_and_store"},
        )

        workflow.add_edge("decide_and_store", END)

        return workflow.compile()


# --- Public Factory Function ---


def create_hybrid_rag_subgraph(
    *,
    name: str = "hybrid_rag",
    retriever: BaseRetriever,
    llm: BaseChatModel,
    vector_store: Optional[VectorStore] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Creates a compiled LangGraph subgraph for Hybrid RAG.

    Args:
        name: Name of the subgraph (unused in logic, useful for debugging context).
        retriever: The LangChain Retriever instance.
        llm: The ChatModel instance (must support structured output for best results).
        vector_store: Optional VectorStore for upserting new knowledge.
        config: Dictionary matching HybridRAGConfig fields.

    Returns:
        A StateGraph runnable.
    """
    rag_config = HybridRAGConfig(**(config or {}))
    builder = HybridRAGBuilder(
        retriever=retriever, llm=llm, vector_store=vector_store, config=rag_config
    )
    return builder.build()


if __name__ == "__main__":
    from core.services import service_manager

    app = create_hybrid_rag_subgraph(
        retriever=service_manager.get_vector_store_retriever(),
        llm=service_manager.get_agent(),
        vector_store=service_manager.get_vector_store(),
    )

    initial_state: HybridRAGState = {
        "query": "What is the capital of France?",
        "original_query": "What is the capital of France?",
        "documents": [],
        "generation": "",
        "retrieval_attempts": 0,
        "is_relevant": False,
        "is_grounded": False,
        "upsert_status": "",
        "error": None,
    }

    for chunk in app.stream(initial_state):
        for node, update in chunk.items():
            print(f"\n--- Update from node: {node} ---")
            print(update)
