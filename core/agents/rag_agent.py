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
from langchain_core.runnables import (
    RunnableBranch,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStore

# New Import for Semantic Chunking
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel, ConfigDict, Field

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


class VectorStoreManager:
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

        # 2. Semantic Chunker
        self._semantic_chunker = SemanticChunker(
            embeddings=self.embeddings, breakpoint_threshold_type="percentile"
        )

        # 3. Ingestion Processing Chain (Parallel Execution)
        summary_chain = RunnableBranch(
            # If should_summarize is True
            (
                lambda x: x["summarize"] is True,
                MANAGER_PROMPTS["summarizer"] | self.llm | StrOutputParser(),
            ),
            # Default branch (skip summary)
            RunnablePassthrough() | (lambda x: ""),
        )

        self._ingestion_chain = RunnableParallel(
            {
                "fin_meta": MANAGER_PROMPTS["metadata_extractor"]
                | self.llm.with_structured_output(FinancialArticleMetadata),
                "summary": summary_chain,
                "raw_text": RunnablePassthrough(),
            }
        )

    # --- Pipeline Step: Ingestion ---

    async def _article_exists(self, url: str) -> bool:
        """
        Checks if an article with this URL already exists in the vector store.
        Uses Async search.
        """
        try:
            # Uses asimilarity_search (Async)
            results = await self.vector_store.asimilarity_search(
                "check_existence", k=1, filter={"url": url}
            )
            return len(results) > 0
        except Exception:
            return False

    async def ingest_article(
        self,
        raw_text: str,
        source_metadata: Dict[str, Any],
        should_summarize: bool = False,
    ) -> bool:
        """
        Ingests an article ASYNCHRONOUSLY.
        """
        if not raw_text:
            return False

        # 1. Deduplication Check (Async)
        url = source_metadata.get("url")
        if url:
            exists = await self._article_exists(url)
            if exists:
                logger.info(f"Skipping duplicate article: {url}")
                return False

        try:
            # 2. Execute Parallel Chain (Metadata & Summary) - ASYNC INVOKE
            # This allows the LLM calls to happen without blocking the loop
            result = await self._ingestion_chain.ainvoke(
                {"document_content": raw_text, "summarize": should_summarize}
            )

            fin_meta: FinancialArticleMetadata = result["fin_meta"]
            summary_text: str = result["summary"]

            # 3. Semantic Chunking
            # Note: SemanticChunker.create_documents is usually synchronous but calls Embeddings API.
            # We wrap it in to_thread to prevent it from blocking the event loop while waiting for embeddings.
            chunks = await asyncio.to_thread(
                self._semantic_chunker.create_documents, [raw_text]
            )

            # 4. Metadata Enrichment
            base_metadata = {
                **source_metadata,
                **fin_meta.model_dump(),
                "ingest_timestamp": datetime.now().isoformat(),
            }
            if should_summarize:
                base_metadata["summary"] = summary_text

            enriched_documents = []
            for i, chunk in enumerate(chunks):
                chunk_meta = base_metadata.copy()
                chunk_meta["chunk_index"] = i
                chunk_meta["chunk_id"] = self._generate_content_hash(chunk.page_content)

                chunk.metadata = chunk_meta
                enriched_documents.append(chunk)

            # 5. Store (Async Add)
            # Uses aadd_documents
            await self.vector_store.aadd_documents(enriched_documents)

            return True

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            return False

    # --- Pipeline Step: Retrieval & Grading ---

    def retrieve(
        self, query: str, filter_dict: Optional[Dict] = None, k: int = 10
    ) -> List[Document]:
        """
        Retrieves documents synchronously.
        (Kept sync as requested, or can be upgraded to async if needed).
        """
        return self._retrieve_documents(query, filter_dict, k=k)

    def _retrieve_documents(
        self, query: str, filter_dict: Optional[Dict] = None, k: int = 10
    ) -> List[Document]:
        try:
            if filter_dict and hasattr(self.retriever, "search_kwargs"):
                documents = self.vector_store.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                documents = self.retriever.invoke(query)

            logger.info(f"Retrieved {len(documents)} documents.")
            return documents
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    # --- Helpers ---

    def _generate_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # Example usage requires an async loop now
    from core.services import service_manager

    async def main():
        manager = service_manager.get_vector_store_manager()
        article_text = "Apple Inc. (AAPL) reported Q4 revenue of $89.5B..."
        source_meta = {
            "url": "https://finance.yahoo.com/...",
            "source": "Yahoo Finance",
            "publish_time": "2023-11-02",
        }

        # Notice the await
        await manager.ingest_article(article_text, source_meta, should_summarize=True)

        results = manager.retrieve("earnings sentiment", filter_dict={"ticker": "AAPL"})
        print(f"Retrieved {len(results)} relevant documents.")

    asyncio.run(main())
