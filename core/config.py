from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(dotenv_path="./AlphaMesh/core/.env")


class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    NEO4J_URL = os.getenv("NEO4J_URL")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db_user")
    CHROMA_NAME = os.getenv("CHROMA_NAME")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")


settings = Settings()
