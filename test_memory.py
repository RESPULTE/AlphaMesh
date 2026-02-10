import asyncio
import logging
import pytest
from lightrag.lightrag import LightRAG
from core.memory import FinancialMemory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_financial_memory_flow():
    """
    Integration test for the full memory pipeline.
    Requires valid .env with Neo4j, Qdrant, and Gemini credentials.
    """
    memory = FinancialMemory()
    await memory.initialize()
    
    user_id = "test_user_001"
    
    try:
        # 1. Ingest a document (Domain Knowledge)
        logger.info("Testing document ingestion...")
        doc_res = await memory.ingest_document(
            user_id=user_id,
            text="The S&P 500 is a stock market index tracking the stock performance of 500 of the largest companies listed on stock exchanges in the United States.",
            metadata={"source": "investopedia", "topic": "stock market"}
        )
        assert doc_res.status == "completed"

        # 2. Ingest a conversation (Personal Context)
        logger.info("Testing conversation ingestion...")
        conv_res = await memory.ingest_conversation(
            user_id=user_id,
            messages=[
                {"role": "user", "content": "I want to start investing in the S&P 500 index fund."},
                {"role": "assistant", "content": "That's a diversified choice! How much are you planning to invest?"},
                {"role": "user", "content": "I have about $5000 in my savings for this."}
            ],
            date_range=("2026-02-10T14:00:00", "2026-02-10T14:05:00")
        )
        assert conv_res.status == "completed"
        
        # Wait for user ingestion to finish
        logger.info(f"Waiting for user processing (track_id: {conv_res.user_track_id})...")
        for i in range(30):
            status = await memory.get_ingestion_status(conv_res.user_track_id, f"user_{user_id}")
            if status.status == "processed":
                logger.info("User processing completed.")
                break
            if status.status == "failed":
                raise RuntimeError(f"User processing failed: {status.message}")
            if i % 5 == 0:
                logger.info(f"Still waiting... ({status.status})")
            await asyncio.sleep(2)
        else:
            raise TimeoutError("User processing timed out.")

        # 3. Query the memory
        logger.info("Testing query...")
        query_res = await memory.query(user_id, "What are my thoughts on S&P 500 and how much do I want to invest?")
        
        logger.info(f"Global Context: {query_res.global_context[:100]}...")
        logger.info(f"User Context: {query_res.user_context[:100]}...")
        logger.info(f"Merged Context: {query_res.merged_context[:200]}...")
        
        assert "S&P 500" in query_res.merged_context
        assert "5000" in query_res.merged_context

        # 4. Cleanup
        # logger.info("Testing data deletion...")
        # await memory.delete_user_data(user_id)
        
    finally:
        await memory.close()

if __name__ == "__main__":
    asyncio.run(test_financial_memory_flow())
