from __future__ import annotations

from pathlib import Path
from typing import Any


def _join_blocks(blocks: list[tuple[float, float, float, float, str]]) -> str:
    ordered = sorted(blocks, key=lambda b: (round(b[1] / 8.0), b[0]))
    return "\n".join(b[4] for b in ordered)


def _column_split(
    blocks: list[tuple[float, float, float, float, str]], page_width: float
) -> str:
    """Split two-column pages on the largest gap in left-edge x coordinates."""
    if len(blocks) < 4 or page_width <= 0:
        return _join_blocks(blocks)
    xs = sorted(b[0] for b in blocks)
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    gap, idx = max(gaps)
    # Ignore small gaps (single-column or just indented paragraphs).
    if gap < page_width * 0.12:
        return _join_blocks(blocks)
    split_x = (xs[idx] + xs[idx + 1]) / 2
    left = [b for b in blocks if (b[0] + b[2]) / 2 < split_x]
    right = [b for b in blocks if (b[0] + b[2]) / 2 >= split_x]
    if len(left) < 2 or len(right) < 2:
        return _join_blocks(blocks)
    return _join_blocks(left) + "\n\n" + _join_blocks(right)


def extract_page_pymupdf(page: object) -> str:
    data = page.get_text("dict")  # type: ignore[attr-defined]
    blocks: list[tuple[float, float, float, float, str]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        x0, y0, x1, y1 = block["bbox"]
        parts: list[str] = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if line_text.strip():
                parts.append(line_text.strip())
        text = "\n".join(parts).strip()
        if text:
            blocks.append((float(x0), float(y0), float(x1), float(y1), text))
    if not blocks:
        return str(page.get_text("text") or "")  # type: ignore[attr-defined]
    width = float(getattr(getattr(page, "rect", None), "width", 0.0) or 0.0)
    return _column_split(blocks, width)


def extract_page_plumber(page: Any) -> str:
    return str(page.extract_text() or "")


def extract_pdf_pages(path: Path, extractor: str) -> list[str]:
    """Return 1-based page texts. pymupdf is default; plumber is the ablation baseline."""
    if extractor == "plumber":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return [extract_page_plumber(p) for p in pdf.pages]
    import pymupdf

    doc = pymupdf.open(path)
    try:
        return [extract_page_pymupdf(doc[i]) for i in range(doc.page_count)]
    finally:
        doc.close()


def infer_title(first_page: str, fallback: str) -> str:
    for raw in first_page.splitlines():
        line = " ".join(raw.split())
        if len(line) < 8 or len(line) > 160:
            continue
        if line.lower().startswith(("abstract", "keywords", "introduction")):
            continue
        return line
    return fallback
