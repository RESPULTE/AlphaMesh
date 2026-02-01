import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    GOOGLE_API_KEY = os.getenv("LLM_BINDING_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    NEO4J_URL = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db_user")
    CHROMA_NAME = os.getenv("CHROMA_NAME")

    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


settings = Settings()
