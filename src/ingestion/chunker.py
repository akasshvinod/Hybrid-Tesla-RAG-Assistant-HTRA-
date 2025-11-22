from __future__ import annotations
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger
from src.utils.custom_types import Page, SectionMeta, Chunk
from src.utils.config import Config

logger = get_logger(__name__)

# ---------------------------------------------------------
# 1. Merge all valid page texts for chunking
# ---------------------------------------------------------
def merge_pages_into_text(pages: List[Page]) -> str:
    """
    Merge non-empty Page objects into a single text block
    for chunking.
    """
    return "\n\n".join(pg.text.strip() for pg in pages if pg.text.strip())

# ---------------------------------------------------------
# 2. Fast lookup: page number -> section metadata
# ---------------------------------------------------------
def build_section_lookup(sections: List[SectionMeta]) -> Dict[int, SectionMeta]:
    """
    Maps each page_no to its SectionMeta (first found section per page).
    """
    lookup = {}
    for sec in sections:
        if sec.page_start not in lookup:
            lookup[sec.page_start] = sec
    return lookup

# ---------------------------------------------------------
# 3. Chunk text with high-quality semantic split
# ---------------------------------------------------------
def chunk_text(text: str) -> List[str]:
    """
    Split the manual text into semantic chunks for RAG.
    Uses LangChain RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.info(f"Chunks produced: {len(chunks)}")
    return chunks

# ---------------------------------------------------------
# 4. Heuristic matching: chunk text → original page number
# ---------------------------------------------------------
def estimate_page_number(chunk: str, pages: List[Page], window: int = 20) -> int:
    """
    Estimate original page number using the first window characters.
    Looks for inclusion in page texts (robust for manuals).
    """
    first_part = chunk[:window].strip().lower()
    for p in pages:
        if first_part and first_part in p.text.lower():
            return p.page_no
    return 1  # fallback (first page)

# ---------------------------------------------------------
# 5. Build Chunk objects w/ metadata
# ---------------------------------------------------------
def create_chunk_objects(
    text_chunks: List[str],
    pages: List[Page],
    section_lookup: Dict[int, SectionMeta],
    min_length: int = 40
) -> List[Chunk]:
    """
    Attaches metadata and builds Chunk objects.
    Ignores very short/noise chunks.
    """
    final_chunks = []
    for idx, text in enumerate(text_chunks):
        if len(text.strip()) < min_length:
            continue
        page_num = estimate_page_number(text, pages)
        section_meta = section_lookup.get(page_num)
        chunk = Chunk(
            id=f"chunk-{idx+1}",
            text=text,
            section=section_meta,
            metadata={
                "page": page_num,
                "source": "Tesla Model 3 Owner's Manual"
            }
        )
        final_chunks.append(chunk)
    logger.info(f"Final Chunk objects: {len(final_chunks)}")
    return final_chunks

# ---------------------------------------------------------
# 6. Metadata flattening utility for ChromaDB compatibility
# ---------------------------------------------------------
def flatten_chunk_metadata(chunk: Chunk) -> dict:
    """
    Expands chunk.metadata and merges any SectionMeta fields flat.
    Returns a dict ONLY with primitives (str/int/float/bool/None).
    """
    meta = dict(chunk.metadata)  # Start with basic metadata
    if chunk.section:
        section_fields = chunk.section.dict(exclude_none=True)
        meta.update(section_fields)
    # Remove any values that are non-primitive
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, type(None)))}

# ---------------------------------------------------------
# 7. Full pipeline: merge → chunk → annotate → produce
# ---------------------------------------------------------
def run_chunk_pipeline(pages: List[Page], sections: List[SectionMeta]) -> List[Chunk]:
    """
    End-to-end: merges pages, chunks text, links back
    to section metadata, returns full Chunk objects.
    """
    logger.info("Running chunking pipeline...")
    text = merge_pages_into_text(pages)
    text_chunks = chunk_text(text)
    section_lookup = build_section_lookup(sections)
    final_chunks = create_chunk_objects(text_chunks, pages, section_lookup)
    return final_chunks

# ---------------------------------------------------------
# Example: Prepare for ChromaDB ingestion
# ---------------------------------------------------------
def prepare_chromadb_records(chunks: List[Chunk]) -> List[dict]:
    """
    Prepares records for ChromaDB ingestion with flat metadata.
    Each record is a dictionary with keys: id, text, embedding, metadata.
    """
    records = []
    for chunk in chunks:
        record = {
            "id": chunk.id,
            "text": chunk.text,
            "embedding": chunk.embedding,
            "metadata": flatten_chunk_metadata(chunk)
        }
        records.append(record)
    return records

# ---------------------------------------------------------
# CLI/test: python -m src.ingestion.chunker
# ---------------------------------------------------------
def main():
    from src.parsing.pdf_reader import read_first_n_pages
    from src.parsing.heading_extractor import extract_section_metadata

    pages = read_first_n_pages(6)
    sections = extract_section_metadata(pages)
    chunks = run_chunk_pipeline(pages, sections)

    print("\n===== CHUNK SAMPLES =====\n")
    for c in chunks[:5]:
        print(f"ID:         {c.id}")
        print(f"Page:       {c.metadata['page']}")
        print(f"Section:    {c.section.heading if c.section else 'None'}")
        print(f"FlatMetadata: {flatten_chunk_metadata(c)}")
        print(f"Text: {c.text[:250].strip()} ...")
        print("-----------------------------------------\n")

if __name__ == "__main__":
    main()
