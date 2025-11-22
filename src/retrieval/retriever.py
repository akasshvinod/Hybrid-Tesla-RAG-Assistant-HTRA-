from __future__ import annotations
import time
from typing import List, Optional, Dict, Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

# ---------------------------------------------------------
# 1. Load embedding model for retrieval: Must match ingestion!
# ---------------------------------------------------------
def load_embedding_model():
    print(f">>> RETRIEVAL EMBEDDING MODEL: {Config.EMBED_MODEL}")
    return OllamaEmbeddings(model=Config.EMBED_MODEL)

# ---------------------------------------------------------
# 2. Load vector DB (persistent) with correct embedding function
# ---------------------------------------------------------
def load_vector_db() -> Chroma:
    """
    Load existing ChromaDB vector store for retrieval,
    using explicit embedding_function for consistency.
    """
    logger.info("Loading ChromaDB for retrieval...")
    try:
        vectordb = Chroma(
            collection_name="tesla_manual_rag",
            embedding_function=load_embedding_model(),
            persist_directory=Config.CHROMA_PATH,
        )
    except Exception as e:
        logger.error(f"Could not load ChromaDB vector store: {e}")
        raise
    return vectordb

# ---------------------------------------------------------
# 3. Build metadata filter dynamically
# ---------------------------------------------------------
def build_metadata_filter(
    chapter: Optional[str] = None,
    heading: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Returns a dict for Chroma metadata filter, or None.
    """
    flt = {}
    if chapter:
        flt["chapter"] = chapter
    if heading:
        flt["heading"] = heading
    return flt or None

# ---------------------------------------------------------
# 4. Hybrid metadata-aware retrieval: Filter + Similarity
# ---------------------------------------------------------
def retrieve_docs(
    query: str,
    k: Optional[int] = None,
    chapter: Optional[str] = None,
    heading: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hybrid retrieval using metadata filter (if set) + similarity search.
    Returns: dict with docs, latency, metadata_filter.
    """
    vectordb = load_vector_db()
    metadata_filter = build_metadata_filter(chapter, heading)
    logger.info(f"Retrieval filter: {metadata_filter}")

    k = k if k is not None else Config.K
    start = time.perf_counter()
    docs = vectordb.similarity_search(query=query, k=k, filter=metadata_filter)
    latency = round((time.perf_counter() - start) * 1000, 2) #seconds to milliseconds with rounds to two decimal places
    logger.info(f"Retrieved {len(docs)} docs in {latency} ms")

    return {
        "docs": docs,
        "latency_ms": latency,
        "metadata_filter": metadata_filter,
    }

# ---------------------------------------------------------
# 5. Grounded/Safe retrieval: Fallbacks and logging
# ---------------------------------------------------------
def safe_retrieve(
    query: str,
    chapter: Optional[str] = None,
    heading: Optional[str] = None,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs hybrid retrieval and adds `no_answer` when no docs are found.
    """
    result = retrieve_docs(query, k=k, chapter=chapter, heading=heading)
    if not result["docs"]:
        logger.warning(f"No relevant docs found for query: '{query}'")
        result["no_answer"] = True
    else:
        result["no_answer"] = False
    return result

# ---------------------------------------------------------
# 6. CLI Quick Test (single-query demo)
# ---------------------------------------------------------
def main():
    """
    Test using:
      python -m src.retrieval.retriever
    """
    query = "How do I charge my Model 3?"
    chapter = "Charging"

    result = safe_retrieve(query, chapter=chapter)

    print(f"\n=== RETRIEVAL RESULT ===")
    print(f"Query: {query}")
    print(f"Filter: {result['metadata_filter']}")
    print(f"Latency: {result['latency_ms']} ms\n")

    for i, doc in enumerate(result["docs"][:3], start=1):
        print(f"--- Doc {i} ---")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Section: {doc.metadata.get('heading')}")
        print(doc.page_content[:300])
        print("-------------------------\n")

if __name__ == "__main__":
    main()
