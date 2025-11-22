from __future__ import annotations
from typing import List, Tuple
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.utils.logger import get_logger
from src.utils.config import Config
from src.utils.custom_types import Chunk

logger = get_logger(__name__)

# ---------------------------------------------------------
# 1. Load embedding model (with debug print)
# ---------------------------------------------------------
def load_embedding_model() -> OllamaEmbeddings:
    """
    Loads OllamaEmbeddings for the specified model,
    logs debug info about the model name.
    """
    logger.info(f"Loading embedding model: {Config.EMBED_MODEL}")
    print(f">>> INGESTION EMBEDDING MODEL: {Config.EMBED_MODEL}")
    embedder = OllamaEmbeddings(model=Config.EMBED_MODEL)
    # Try to get vector dimension (if supported)
    try:
        print(f">>> Embedding dimension: {getattr(embedder, 'dimensions', 'Unknown')}")
    except Exception:
        pass
    return embedder

# ---------------------------------------------------------
# 2. Initialize Chroma (persistent, with embedding model print)
# ---------------------------------------------------------
def init_chroma() -> Chroma:
    """
    Initialize a persistent ChromaDB collection.
    Passes the embedding model explicitly, logs info.
    """
    logger.info(f"Initializing ChromaDB at: {Config.CHROMA_PATH}")
    print(f">>> CHROMA INIT EMBEDDING MODEL: {Config.EMBED_MODEL}")
    return Chroma(
        collection_name="tesla_manual_rag",
        embedding_function=load_embedding_model(),
        persist_directory=Config.CHROMA_PATH,
    )

# ---------------------------------------------------------
# 3. Metadata flattener for ChromaDB ingestion (Pydantic v2+)
# ---------------------------------------------------------
def flatten_chunk_metadata(chunk: Chunk) -> dict:
    """
    Expands chunk.metadata, merges any SectionMeta fields, and outputs flat dict
    with ONLY primitives (str/int/float/bool/None) for ChromaDB.
    """
    meta = dict(chunk.metadata)
    if chunk.section:
        # Use Pydantic v2+ model_dump instead of dict
        meta.update(chunk.section.model_dump(exclude_none=True))
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, type(None)))}

# ---------------------------------------------------------
# 4. Prepare ChromaDB ingestion inputs
# ---------------------------------------------------------
def prepare_chroma_inputs(chunks: List[Chunk]) -> Tuple[List[str], List[str], List[dict]]:
    """
    Converts List[Chunk] into tuple of (ids, texts, flat_metadatas) for ChromaDB.
    """
    ids, texts, metadatas = [], [], []
    for ch in chunks:
        ids.append(ch.id)
        texts.append(ch.text)
        metadatas.append(flatten_chunk_metadata(ch))
    return ids, texts, metadatas

# ---------------------------------------------------------
# 5. INGESTION PIPELINE
# ---------------------------------------------------------
def ingest_chunks(chunks: List[Chunk]):
    """
    Full ingestion pipeline:
      - validate chunks
      - embed + store them in Chroma
      - database is auto-persisted on disk
    """
    logger.info(f"Starting ingestion of {len(chunks)} chunks...")
    vectordb = init_chroma()
    ids, texts, metadatas = prepare_chroma_inputs(chunks)
    for i in tqdm(range(len(texts)), desc="Ingesting chunks"):
        vectordb.add_texts(
            ids=[ids[i]],
            texts=[texts[i]],
            metadatas=[metadatas[i]]
        )
    logger.info("Ingestion complete. Vector store saved to disk.")

# ---------------------------------------------------------
# 6. CLI for full ingestion
# ---------------------------------------------------------
def main():
    """
    CLI entrypoint:
      python -m src.ingestion.ingest
    Loads, cleans, chunks and ingests the entire manual.
    """
    from src.parsing.pdf_reader import read_pdf_pages
    from src.parsing.text_cleaner import clean_page_text
    from src.parsing.heading_extractor import extract_section_metadata
    from src.ingestion.chunker import run_chunk_pipeline

    logger.info("Loading full PDF for ingestion...")
    pages = [pg for pg in read_pdf_pages()]
    for pg in pages:
        pg.text = clean_page_text(pg.text, pg.page_no)
    logger.info(f"Total cleaned pages: {len(pages)}")

    sections = extract_section_metadata(pages)
    logger.info(f"Sections found: {len(sections)}")

    chunks = run_chunk_pipeline(pages, sections)
    logger.info(f"Total final chunks: {len(chunks)}")

    print(f"\n>>> INGESTION EMBEDDING MODEL (main): {Config.EMBED_MODEL}\n")
    ingest_chunks(chunks)

if __name__ == "__main__":
    main()
