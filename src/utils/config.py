import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Local model names
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # File paths
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
    PDF_PATH = os.getenv("PDF_PATH", "./data/Owners_Manual.pdf")

    # Chunk settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 950))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

    # Retriever settings
    K = int(os.getenv("K", 5))  # number of retrieved chunks
