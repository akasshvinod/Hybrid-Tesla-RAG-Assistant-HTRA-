from __future__ import annotations
import fitz  # PyMuPDF
from typing import Generator, List, Optional
from src.utils.logger import get_logger
from src.utils.config import Config
from src.utils.custom_types import Page

logger = get_logger(__name__)

def open_pdf(path: Optional[str] = None) -> fitz.Document:
    """
    Opens the PDF file using PyMuPDF.
    Returns an open fitz.Document object.

    Args:
        path (str | None): Path to PDF. If None, uses Config.PDF_PATH.
    """
    pdf_path = path or Config.PDF_PATH
    logger.info(f"Opening PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        return doc
    except Exception as e:
        logger.error(f"Unable to open PDF file at '{pdf_path}': {e}")
        raise

def extract_page_text(page: fitz.Page) -> str:
    """
    Extracts structured text from a single PDF page using block coordinates.

    Args:
        page (fitz.Page): The PDF page object.

    Returns:
        str: Merged text content, in top-down order.
    """
    blocks = page.get_text("blocks")  # [x0, y0, x1, y1, text, ...]
    # Sort by vertical position, then left-right for layout faithfulness
    
    blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))
    
    text_parts = [b[4].strip() for b in blocks_sorted if b[4] and b[4].strip()]
    return "\n".join(text_parts)

def read_pdf_pages(path: Optional[str] = None) -> Generator[Page, None, None]:
    """
    Yields Page objects (as defined in src.utils.types) for every PDF page.

    Args:
        path (str | None): Optional override path to PDF.
    Yields:
        Page: Page object containing extracted text and metadata.
    """
    doc = open_pdf(path)
    logger.info(f"Total pages detected: {doc.page_count}")

    total, failed = 0, 0
    try:
        for idx, page in enumerate(doc, start=1):
            try:
                raw_text = extract_page_text(page)
                pg = Page(
                    page_no=idx,
                    text=raw_text,
                    width=page.rect.width,
                    height=page.rect.height,
                )
                yield pg
                total += 1
            except Exception as e:
                logger.error(f"Failed reading page {idx}: {e}")
                yield Page(page_no=idx, text="")
                failed += 1
    finally:
        doc.close()
        logger.info(f"Finished reading PDF: {total} pages yielded, {failed} page(s) failed.")

def read_first_n_pages(n: int = 5, path: Optional[str] = None) -> List[Page]:
    """
    Utility to quickly scan the first N pages, for debugging or notebook demos.

    Args:
        n (int): Number of pages to return.
        path (str | None): Optional path override.

    Returns:
        List[Page]: First N Page objects.
    """
    pages = []
    for p in read_pdf_pages(path):
        pages.append(p)
        if len(pages) >= n:
            break
    logger.info(f"Collected {len(pages)} pages for preview.")
    return pages

def main():
    logger.info("Running PDF Reader Smoke Test...")

    pages = read_first_n_pages(3)
    logger.info(f"Extracted {len(pages)} pages")

    for p in pages:
        print(f"\n--- PAGE {p.page_no} ---")
        print(p.text[:800])  # print only first 800 chars
        print("\n=========================\n")

if __name__ == "__main__":
    main()