import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel, ConfigDict, Field

# Internal Imports
from core.services import service_manager

logger = logging.getLogger(__name__)

# --- Configuration Models ---


class VectorStoreConfig(BaseModel):
    """Configuration for the Financial Vector Store Manager."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    min_relevance_score: float = Field(
        default=0.5, description="Threshold for document relevance grading."
    )
    chunk_size_tokens: int = Field(
        default=500, description="Target size for chunks if not using semantic."
    )


# --- Metadata Models ---


class FinancialArticleMetadata(BaseModel):
    """Schema for extracting structured financial metadata for Vector Filtering."""

    ticker: str = Field(
        description="The primary stock ticker (e.g., AAPL). Use 'GENERAL' if none."
    )
    company_name: str = Field(description="The full name of the company.")
    category: Literal["Earnings", "M&A", "Macro", "Product", "Executive", "Other"] = (
        Field(description="Broad category of the content.")
    )
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Market sentiment."
    )


# --- Prompts ---

MANAGER_PROMPTS = {
    "metadata_extractor": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial analyst. Extract metadata for vector storage filtering.",
            ),
            ("human", "Article:\n{document_content}"),
        ]
    ),
    "summarizer": ChatPromptTemplate.from_template(
        "Summarize the following financial text to preserve key figures and strategic shifts.\n\nText:\n{document_content}\n\nSummary:"
    ),
}


class VectorStoreManager:
    """
    Manages the 'Flesh' of the memory: ChromaDB.
    Responsible for:
    1. Cleaning and Chunking text (Semantic Chunking).
    2. Extracting Metadata for pre-filtering (Metadata Enrichment).
    3. Embedding and Writing to Vector Store.
    4. Similarity Search (Retrieval).
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        llm: Optional[BaseChatModel] = None,
        embeddings: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[VectorStoreConfig] = None,
    ):
        # Service Injection with Fallbacks
        self.llm = llm or service_manager.get_agent(temperature=0.0)
        self.embeddings = embeddings or service_manager.get_embedding_func()
        self.vector_store = vector_store or service_manager.get_vector_store()
        self.retriever = retriever or self.vector_store.as_retriever()

        self.config = config or VectorStoreConfig()

        # 1. Semantic Chunker (Intelligent Splitting)
        self._semantic_chunker = SemanticChunker(
            embeddings=self.embeddings, breakpoint_threshold_type="percentile"
        )

        # 2. Ingestion Chain
        self._ingestion_chain = RunnableParallel(
            {
                "fin_meta": MANAGER_PROMPTS["metadata_extractor"]
                | self.llm.with_structured_output(FinancialArticleMetadata),
                "summary": (
                    lambda x: (
                        MANAGER_PROMPTS["summarizer"] | self.llm | StrOutputParser()
                        if x["summarize"]
                        else (lambda _: "")
                    )
                ),
                "raw_text": lambda x: x["document_content"],
            }
        )

    # --- Ingestion Interface ---

    async def ingest_article(
        self,
        raw_text: str,
        source_metadata: Dict[str, Any],
        should_summarize: bool = False,
    ) -> bool:
        """
        Ingests text into the Vector Store asynchronously.
        """
        if not raw_text:
            return False

        # 1. Deduplication (Async check)
        url = source_metadata.get("url")
        if url and await self._article_exists(url):
            logger.info(f"Vector Store: Skipping duplicate {url}")
            return False

        try:
            # 2. LLM Processing (Metadata + Summary)
            processed = await self._ingestion_chain.ainvoke(
                {"document_content": raw_text, "summarize": should_summarize}
            )

            meta_obj: FinancialArticleMetadata = processed["fin_meta"]
            summary_text: str = processed["summary"]

            # 3. Create Documents (Semantic Chunking)
            # Run in thread to avoid blocking loop during embedding calls inside chunker
            chunks = await asyncio.to_thread(
                self._semantic_chunker.create_documents, [raw_text]
            )

            # 4. Enrich Metadata
            enriched_docs = []
            base_meta = {
                **source_metadata,
                **meta_obj.model_dump(),  # Flatten extracted metadata
                "ingest_timestamp": datetime.now().isoformat(),
                "has_summary": should_summarize,
            }

            # Add summary to the first chunk or all chunks?
            # Strategy: Store summary in a separate 'Summary Document' or just metadata.
            # Here, we add it to metadata of all chunks for retrieval context.
            if summary_text:
                base_meta["summary_context"] = summary_text

            for i, chunk in enumerate(chunks):
                chunk.metadata = base_meta.copy()
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_id"] = self._generate_hash(chunk.page_content)
                enriched_docs.append(chunk)

            # 5. Write to Store
            await self.vector_store.aadd_documents(enriched_docs)
            logger.info(f"Vector Ingest Success: {len(enriched_docs)} chunks stored.")
            return True

        except Exception as e:
            logger.error(f"Vector Ingest Failed: {e}")
            return False

    # --- Retrieval Interface ---

    def retrieve(
        self, query: str, filter_dict: Optional[Dict] = None, k: int = 5
    ) -> List[Document]:
        """
        Standard Vector Retrieval.
        Supports metadata filtering (e.g., ticker='AAPL').
        """
        try:
            if filter_dict:
                # Direct vector store call allows filtering
                docs = self.vector_store.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                # Retriever abstraction
                docs = self.vector_store.similarity_search(query, k=k)

            return docs
        except Exception as e:
            logger.error(f"Vector Retrieval Failed: {e}")
            return []

    # --- Helpers ---

    async def _article_exists(self, url: str) -> bool:
        """Checks existence by URL to prevent duplicates."""
        try:
            # Chroma specific filtering
            results = await self.vector_store.asimilarity_search(
                "check", k=1, filter={"url": url}
            )
            return len(results) > 0
        except Exception:
            return False

    def _generate_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
