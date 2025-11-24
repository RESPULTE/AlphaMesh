import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# New Import for Semantic Chunking
from langchain_experimental.text_splitter import SemanticChunker

# Configure Logging
logger = logging.getLogger(__name__)

# --- Configuration Models ---


class VectorStoreConfig(BaseModel):
    """Configuration for the Financial Vector Store Manager."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    min_relevance_score: float = Field(
        default=0.5, description="Threshold for document relevance grading."
    )
    domain_allowlist: List[str] = Field(
        default_factory=list, description="Allowed domains for storage security."
    )


# --- Data Models for Structured Output (Metadata) ---


class FinancialArticleMetadata(BaseModel):
    """Schema for extracting structured financial data from text."""

    ticker: str = Field(
        description="The primary stock ticker symbol mentioned (e.g., AAPL, TSLA). If none, use 'GENERAL'."
    )
    company_name: str = Field(description="The full name of the company.")
    category: Literal[
        "Earnings", "M&A", "Macro", "Product Launch", "Executive Change", "Other"
    ] = Field(description="The broad category of the news.")
    event_type: str = Field(
        description="Specific event description (e.g., 'Revenue Beat', 'Guidance Cut', 'CEO Resignation')."
    )
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="The overall market sentiment of the article regarding the ticker."
    )
    key_figures: Optional[str] = Field(
        description="Key financial numbers mentioned (e.g., '$10B revenue', 'EPS $1.20')."
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(
        description="Documents are relevant to the query, 'yes' or 'no'"
    )


# --- Centralized Prompts ---

MANAGER_PROMPTS = {
    "retrieval_grader": ChatPromptTemplate.from_messages(
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
    ),
    "metadata_extractor": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial analyst. Extract structured metadata from the following news article. "
                "Focus on the primary entity and the market impact.",
            ),
            ("human", "Article Content:\n{document_content}"),
        ]
    ),
    "summarizer": ChatPromptTemplate.from_template(
        "You are an expert financial summarizer. Summarize the following document to preserve key facts, "
        "numbers, and semantic meaning for future retrieval.\n"
        "Focus on: Earnings results, Guidance, and Strategic moves.\n\n"
        "Original Document:\n{document_content}\n\n"
        "Summary:"
    ),
}

# --- The Manager Class ---


class FinancialVectorStoreManager:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        embeddings: Embeddings,
        vector_store: VectorStore,
        config: Optional[VectorStoreConfig] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.config = config or VectorStoreConfig()

        # --- Components ---

        # 1. Grader
        self._grader = MANAGER_PROMPTS[
            "retrieval_grader"
        ] | self.llm.with_structured_output(GradeDocuments)

        # 2. Semantic Chunker
        self._semantic_chunker = SemanticChunker(
            embeddings=self.embeddings, breakpoint_threshold_type="percentile"
        )

        # 3. Ingestion Processing Chain (Parallel Execution)
        # This replaces the separate calls in the previous version
        self._ingestion_chain = RunnableParallel(
            {
                "fin_meta": MANAGER_PROMPTS["metadata_extractor"]
                | self.llm.with_structured_output(FinancialArticleMetadata),
                "summary": MANAGER_PROMPTS["summarizer"] | self.llm | StrOutputParser(),
                "raw_text": RunnablePassthrough(),
            }
        )

    # --- Pipeline Step: Ingestion ---

    def _article_exists(self, url: str) -> bool:
        """Checks if an article with this URL already exists in the vector store."""
        try:
            # Note: Syntax depends on specific VectorStore (Chroma, Pinecone, etc.)
            # We search for 1 document with this specific source URL.
            # If your VectorStore supports .get(where=...), use that instead for speed.
            results = self.vector_store.similarity_search(
                "check_existence", k=1, filter={"url": url}
            )
            return len(results) > 0
        except Exception:
            # If filter fails or DB is empty, assume it doesn't exist
            return False

    def ingest_article(self, raw_text: str, source_metadata: Dict[str, Any]) -> bool:
        """
        Ingests an article with deduplication, parallel processing, and semantic chunking.
        """
        if not raw_text:
            return False

        # 1. Deduplication Check
        url = source_metadata.get("url")
        if url and self._article_exists(url):
            logger.info(f"Skipping duplicate article: {url}")
            return False

        try:
            logger.info("Processing article (Metadata + Summary)...")

            # 2. Execute Parallel Chain (Metadata & Summary)
            # Input: {"document_content": raw_text} -> Output: Dict with keys 'fin_meta', 'summary', 'raw_text'
            result = self._ingestion_chain.invoke({"document_content": raw_text})

            fin_meta: FinancialArticleMetadata = result["fin_meta"]
            summary_text: str = result["summary"]

            # 3. Semantic Chunking
            logger.info("Performing semantic chunking...")
            chunks = self._semantic_chunker.create_documents([raw_text])

            # 4. Metadata Enrichment
            enriched_documents = []
            base_metadata = {
                **source_metadata,
                **fin_meta.model_dump(),
                "summary": summary_text,
                "ingest_timestamp": datetime.now().isoformat(),
            }

            for i, chunk in enumerate(chunks):
                chunk_meta = base_metadata.copy()
                chunk_meta["chunk_index"] = i
                chunk_meta["chunk_id"] = self._generate_content_hash(chunk.page_content)

                chunk.metadata = chunk_meta
                enriched_documents.append(chunk)

            # 5. Store
            logger.info(
                f"Storing {len(enriched_documents)} chunks for {fin_meta.ticker}..."
            )
            self.vector_store.add_documents(enriched_documents)

            return True

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            return False

    # --- Pipeline Step: Retrieval & Grading ---

    def retrieve(
        self, query: str, filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieves documents. Optionally applies metadata filters (e.g., ticker='AAPL').
        """
        # Note: Depending on the specific VectorStore implementation,
        # you might need to pass filter_dict to the retriever differently.
        # Here we assume the retriever is pre-configured or we use the vector_store directly for filtering.

        raw_docs = self._retrieve_documents(query, filter_dict)
        return self.grade_documents(query, raw_docs)

    def _retrieve_documents(
        self, query: str, filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        try:
            # If the retriever supports dynamic filtering (like Chroma asRetriever search_kwargs)
            if filter_dict and hasattr(self.retriever, "search_kwargs"):
                # This is a hacky way to inject filters dynamically;
                # in production, use self.vector_store.similarity_search(query, filter=filter_dict)
                documents = self.vector_store.similarity_search(
                    query, k=5, filter=filter_dict
                )
            else:
                documents = self.retriever.invoke(query)

            logger.info(f"Retrieved {len(documents)} documents.")
            return documents
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def grade_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Grades relevance. Returns only relevant documents.
        """
        if not documents or not query:
            return []

        batch_inputs = [
            {"query": query, "document": doc.page_content} for doc in documents
        ]

        try:
            scores = self._grader.batch(batch_inputs)
        except Exception as e:
            logger.error(f"Batch grading failed: {e}")
            return []

        filtered_docs = []
        for doc, score in zip(documents, scores):
            if score.binary_score.lower() == "yes":
                filtered_docs.append(doc)

        logger.info(f"Graded {len(documents)} docs. {len(filtered_docs)} relevant.")
        return filtered_docs

    # --- Helpers ---

    def _generate_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    from core.services import service_manager

    manager = service_manager.get_vector_store_manager()
    # 2. Ingest
    article_text = "Apple Inc. (AAPL) reported Q4 revenue of $89.5B..."
    source_meta = {
        "url": "https://finance.yahoo.com/...",
        "source": "Yahoo Finance",
        "publish_time": "2023-11-02",
    }

    manager.ingest_article(article_text, source_meta)

    # 3. Retrieve with Filter
    # "What was the sentiment on Apple's earnings?"
    results = manager.retrieve("earnings sentiment", filter_dict={"ticker": "AAPL"})
    print(f"Retrieved {len(results)} relevant documents for AAPL earnings sentiment.")
    for doc in results:
        print(doc.page_content)
