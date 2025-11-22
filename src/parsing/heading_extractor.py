from __future__ import annotations
from typing import List, Set, Optional
import re
from src.utils.logger import get_logger
from src.utils.custom_types import Page, SectionMeta
from src.parsing.pdf_reader import read_first_n_pages

logger = get_logger(__name__)

# -------------------------------------------------------------
# Known chapters explicitly from Tesla manuals (strong signals)
# -------------------------------------------------------------
KNOWN_CHAPTERS: Set[str] = {
    "Overview",
    "Driving",
    "Charging",
    "Autopilot",
    "Safety",
    "Interior",
    "Exterior",
    "Controls",
    "Maintenance",
    "Specifications",
    "Troubleshooting",
    "Emergency",
    "Warning",
}

# TOC-specific patterns (very strong signals)
TOC_HEADING_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z ]+?)\s*\.{3,}\s*\d+$"
)


# -------------------------------------------------------------
# 1) Robust + Hybrid heading extraction
# -------------------------------------------------------------
def extract_heading_candidates(text: str) -> List[str]:
    """
    Extract candidate headings from cleaned page text.

    Detection layers:
      1. TOC-style headings (Title.......122)
      2. ALL CAPS lines
      3. Title Case lines
      4. Short multi-word clean lines
    """

    candidates = set()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for ln in lines:

        # --- 1. TOC-style "Autopilot Features......105" ---
        m = TOC_HEADING_PATTERN.match(ln)
        if m:
            candidates.add(m.group(1).strip())
            continue

        # --- 2. ALL CAPS headings ---
        if ln.isupper() and 3 <= len(ln) <= 60:
            candidates.add(ln.title().strip())
            continue

        # --- 3. Title Case (H1/H2 style) ---
        if re.match(r"^([A-Z][a-z]+(\s+[A-Z][a-z]+){1,5})$", ln):
            candidates.add(ln.strip())
            continue

        # --- 4. Clean 2–6 word candidates ---
        if 2 <= len(ln.split()) <= 6 and re.fullmatch(r"[A-Za-z ]+", ln):
            candidates.add(ln.strip())

    return sorted(candidates)


# -------------------------------------------------------------
# 2) Stronger chapter detection
# -------------------------------------------------------------
def detect_chapter(text: str) -> Optional[str]:
    """
    A stronger chapter detection algorithm:
      - Gives priority to exact chapter matches
      - Uses keyword frequency to boost confidence
      - Detects TOC page as 'Overview' explicitly
    """

    # TOC detection → treat entire page as 'Overview'
    if "..." in text or TOC_HEADING_PATTERN.search(text):
        return "Overview"

    chapter_scores = {}

    for chapter in KNOWN_CHAPTERS:
        matches = re.findall(rf"\b{re.escape(chapter)}\b", text, re.IGNORECASE)
        if matches:
            chapter_scores[chapter] = len(matches)

    if not chapter_scores:
        return None

    # Pick chapter with highest match frequency
    return max(chapter_scores, key=chapter_scores.get)


# -------------------------------------------------------------
# 3) Extract sections from pages + assign early metadata
# -------------------------------------------------------------
def extract_section_metadata(pages: List[Page]) -> List[SectionMeta]:
    """
    Returns SectionMeta list with early metadata:
      - heading
      - detected chapter
      - page_start/page_end (filled later)
    """

    sections: List[SectionMeta] = []

    for pg in pages:
        headings = extract_heading_candidates(pg.text)
        chapter = detect_chapter(pg.text)

        for h in headings:
            # Filtering out garbage like copyright lines
            if "©" in h or "Inc" in h or "Tesla" in h:
                continue

            sm = SectionMeta(
                chapter=chapter,
                heading=h,
                page_start=pg.page_no,
                page_end=pg.page_no,
                keywords=[]
            )
            sections.append(sm)

    logger.info(
        f"Extracted {len(sections)} high-quality section candidates "
        f"from {len(pages)} pages."
    )
    return merge_section_spans(sections)


# -------------------------------------------------------------
# 4) Merge adjacent headings into multi-page section spans
# -------------------------------------------------------------
def merge_section_spans(sections: List[SectionMeta]) -> List[SectionMeta]:
    """
    Example:
       Page 1: 'Charging'
       Page 2: 'Charging'
       Page 3: 'Charging Overview'

    → merge contiguous blocks into:
       Charging: pages 1–2
       Charging Overview: page 3
    """

    if not sections:
        return []

    merged = []
    prev = sections[0]

    for sec in sections[1:]:
        # Same heading → extend page_end
        if sec.heading == prev.heading:
            prev.page_end = sec.page_end
            continue

        merged.append(prev)
        prev = sec

    merged.append(prev)
    return merged


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    pages = read_first_n_pages(8)
    sections = extract_section_metadata(pages)

    print("\n===== EXTRACTED SECTION METADATA =====\n")
    for sec in sections:
        print(
            f"Page {sec.page_start:03d}-{sec.page_end:03d} | "
            f"Chapter={sec.chapter or 'Unknown':12} | "
            f"Heading={sec.heading}"
        )


if __name__ == "__main__":
    main()
