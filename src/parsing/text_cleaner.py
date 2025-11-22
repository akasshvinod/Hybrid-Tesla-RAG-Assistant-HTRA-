from __future__ import annotations
import re
from typing import Optional
from src.utils.logger import get_logger
from src.parsing.pdf_reader import read_first_n_pages

logger = get_logger(__name__)

# ----------------------------------------
# PAGE STRUCTURE CLEANING UTILS
# ----------------------------------------

# Compile page number patterns for performance
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^page\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^\d+\s*/\s*\d+\s*$"),
    re.compile(r"^page\s*\|\s*\d+$", re.IGNORECASE),  # e.g., "Page | 13"
]

def remove_page_numbers(text: str) -> str:
    """
    Removes standalone page numbers and variants (typically at header/footer).
    """
    cleaned_lines = []
    for line in text.splitlines():
        l = line.strip().lower()
        if any(pat.match(l) for pat in PAGE_NUMBER_PATTERNS):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def dehyphenate(text: str) -> str:
    """
    Removes hyphenation at line breaks (e.g., "exam-\nple" → "example").
    Only removes hyphens if the next line starts with a letter.
    """
    # Handles hyphenated line breaks between words.
    return re.sub(r"(?<=\w)-\n(?=\w)", "", text)

def remove_headers_footers(text: str) -> str:
    """
    Removes likely headers and footers using simple title heuristics.

    Logic:
    - Removes first line if it's a short, uppercase, or likely header.
    - Removes last line if it's a likely footer (matches page number/title).
    """
    lines = text.splitlines()
    if len(lines) <= 2:
        return text

    # Header candidate
    first = lines[0].strip()
    if (len(first.split()) <= 5
        and (first.isupper() or len(first) < 25 or first.startswith("TESLA"))):
        lines = lines[1:]

    # Footer candidate
    last = lines[-1].strip().lower()
    if (last.startswith("page")
        or re.match(r"^\d+$", last)
        or re.match(r"^copyright", last)
        or re.match(r"^tesla", last)):
        lines = lines[:-1]

    return "\n".join(lines)

def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace:
    - Removes multiple blank lines
    - Removes trailing spaces on each line
    - Reduces excessive newlines
    """
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    return text.strip()

# ----------------------------------------
# MAIN CLEAN PIPELINE
# ----------------------------------------
def clean_page_text(text: str, page_no: Optional[int] = None, debug: bool = False) -> str:
    """
    Run all cleaning steps for a single page.
    Args:
        text (str): Raw page text
        page_no (int): For logging (optional)
        debug (bool): If True, prints before/after samples
    Returns:
        str: Cleaned page text
    """
    if not text.strip():
        return ""

    if page_no is not None:
        logger.info(f"Cleaning page {page_no}")

    t = text
    t_orig = t
    t = remove_page_numbers(t)
    t = dehyphenate(t)
    t = remove_headers_footers(t)
    t = normalize_whitespace(t)

    if debug:
        print(f"\n----- [DEBUG] Page {page_no} Original -----\n{t_orig[:500]}")
        print(f"\n----- [DEBUG] Page {page_no} Cleaned -----\n{t[:500]}")

    return t
def main():
    pages = read_first_n_pages(3)

    for p in pages:
        print(f"\n===== CLEANED PAGE {p.page_no} =====")
        cleaned = clean_page_text(p.text, p.page_no)
        print(cleaned[:800])
        print("\n=============================\n")

if __name__ == "__main__":
    main()