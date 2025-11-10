from langchain_neo4j import Neo4jGraph
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

_embedding_instance: GoogleGenerativeAIEmbeddings = None
_llm_instance: ChatGoogleGenerativeAI = None
_graph_instance: Neo4jGraph = None
_vector_instance: Chroma = None


def get_llm() -> ChatGoogleGenerativeAI | None:
    global _llm_instance
    if _llm_instance:
        return _llm_instance
    try:
        _llm_instance = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,
        )
        return _llm_instance
    except Exception as e:
        logger.error(f"Failed to initialize Google LLM: {e}")
        return None


def get_graph() -> Neo4jGraph | None:
    global _graph_instance
    if _graph_instance:
        return _graph_instance
    try:
        _graph_instance = Neo4jGraph(
            url=settings.NEO4J_URL,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
        )
        return _graph_instance
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j Graph: {e}")
        return None


def get_embedding_func():
    global _embedding_instance
    if _embedding_instance:
        return _embedding_instance

    try:
        return GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY
        )
    except Exception as e:
        logger.error(f"Failed to initialize google embedding function: {e}")
        return None


def get_vector_store() -> Chroma | None:
    global _vector_instance
    if _vector_instance:
        return _vector_instance
    try:
        _vector_instance = Chroma(
            collection_name=settings.CHROMA_NAME,
            embedding_function=get_embedding_func(),
            persist_directory=settings.CHROMA_PATH,
        )
        return _vector_instance
    except Exception as e:
        logger.error(f"Failed to initialize Chroma Vector Store: {e}")
        return None
