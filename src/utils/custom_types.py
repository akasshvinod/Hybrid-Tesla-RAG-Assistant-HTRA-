from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# ------------------------------
# PAGE MODEL (raw extracted page)
# ------------------------------
class Page(BaseModel):
    """
    One page extracted from the PDF.
    Only contains raw text and page geometry.
    Used in parsing and cleaning.
    """
    page_no: int = Field(..., description="1-based page number from PDF document.")
    text: str = Field(..., description="Raw extracted text of the page.")
    width: Optional[float] = Field(None, description="Page width in points (optional, for layout parsing).")
    height: Optional[float] = Field(None, description="Page height in points (optional, for layout parsing).")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


# ------------------------------
# SECTION METADATA
# ------------------------------
class SectionMeta(BaseModel):
    """
    Metadata giving semantic context to a chunk.
    Provided by heading detection.
    Used for smarter retrieval/routing in RAG.
    """
    chapter: Optional[str] = Field(None, description="Main chapter name (e.g., 'Charging', 'Safety').")
    heading: Optional[str] = Field(None, description="Main section heading in the manual (H1/H2).")
    subheading: Optional[str] = Field(None, description="Subheading (H3, etc.).")
    page_start: Optional[int] = Field(None, description="First page of the section.")
    page_end: Optional[int] = Field(None, description="Last page of the section.")
    keywords: List[str] = Field(default_factory=list, description="Suggested keywords derived from the section text.")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


# ------------------------------
# CHUNK MODEL (vectorization unit)
# ------------------------------
class Chunk(BaseModel):
    """
    One processed chunk of text, ready for embedding and retrieval.
    Encapsulates the chunk, any embedding, rich semantic metadata,
    and any additional attributes for logging or routing.
    """
    id: Optional[str] = Field(None, description="Unique ID for chunk (assigned by pipeline or ChromaDB).")
    text: str = Field(..., description="Chunked content, usually from manual after parsing and cleaning.")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for the chunk (None until computed).")
    section: Optional[SectionMeta] = Field(None, description="Linked SectionMeta for semantic context.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Flat metadata dict (ready for ChromaDB ingestion).")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def to_chroma_doc(self) -> Dict[str, Any]:
        """
        Converts the Chunk into a ChromaDB-compatible dict
        by merging all metadata and SectionMeta flat.
        """
        meta = self.metadata.copy()
        if self.section:
            meta.update(self.section.dict(exclude_none=True))
        return {
            "id": self.id,
            "text": self.text,
            "metadata": meta
        }


# ------------------------------
# Example Usage (Optional)
# ------------------------------
if __name__ == "__main__":
    # Example Page
    pg = Page(page_no=1, text="Charging Instructions Overview", width=595, height=842)
    
    # Example SectionMeta
    sm = SectionMeta(chapter="Charging", heading="Charging Overview", subheading=None,
                     page_start=1, page_end=3, keywords=["charging", "connector", "battery"])

    # Example Chunk
    ch = Chunk(id="chunk-001", text="How to use the charging connector...", section=sm)
    print(ch.to_chroma_doc())
