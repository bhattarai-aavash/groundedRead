from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from dsqa.extract import extract_pdf_pages, infer_title


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    context: str = ""
    embed_text: str = field(default="")

    def __post_init__(self) -> None:
        if not self.embed_text:
            self.embed_text = self.text


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_id(source: str, page: int, text: str) -> str:
    return hashlib.sha1(f"{source}:{page}:{text}".encode("utf-8")).hexdigest()[:16]


def chunk_page(
    text: str,
    source: str,
    page: int,
    chunk_chars: int,
    chunk_overlap: int,
    context: str = "",
) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > chunk_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            for i in range(0, len(para), chunk_chars):
                pieces.append(para[i : i + chunk_chars])
            continue
        candidate = para if not buf else f"{buf}\n\n{para}"
        if len(candidate) <= chunk_chars:
            buf = candidate
        else:
            pieces.append(buf)
            buf = para
    if buf:
        pieces.append(buf)

    chunks: list[Chunk] = []
    prev = ""
    for i, piece in enumerate(pieces):
        # Overlap from the pre-overlap piece so nested overlaps don't compound.
        body = piece if i == 0 or chunk_overlap <= 0 else prev[-chunk_overlap:] + piece
        embed_text = f"{context}\n{body}" if context else body
        chunks.append(
            Chunk(
                id=_chunk_id(source, page, body),
                text=body,
                source=source,
                page=page,
                context=context,
                embed_text=embed_text,
            )
        )
        prev = piece
    return chunks


def contextual_prefix(source: str, page: int, title: str) -> str:
    """One-line document context prepended at embed time (Anthropic-style contextual retrieval)."""
    title = title.strip() or source
    return f"This excerpt is from {source} page {page}, paper: {title}."


def ingest_pdf(path: Path, chunk_chars: int, chunk_overlap: int, extractor: str, contextual: bool) -> list[Chunk]:
    chunks: list[Chunk] = []
    empty_pages = 0
    source = path.name
    try:
        pages = extract_pdf_pages(path, extractor)
    except Exception as exc:
        print(f"warning: {source}: failed to read ({exc})")
        return []
    title = infer_title(pages[0] if pages else "", source.rsplit(".", 1)[0].replace("-", " "))
    for i, raw in enumerate(pages, start=1):
        cleaned = clean_text(raw)
        if not cleaned:
            empty_pages += 1
            continue
        prefix = contextual_prefix(source, i, title) if contextual else ""
        chunks.extend(chunk_page(cleaned, source, i, chunk_chars, chunk_overlap, prefix))
    if empty_pages:
        print(
            f"warning: {source}: {empty_pages} page(s) had no text layer "
            "(scanned PDF; OCR needed) — skipped"
        )
    return chunks


def ingest_dir(path: str | Path) -> list[Chunk]:
    from dsqa.config import CONFIG

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"ingest dir not found: {root}")
    all_chunks: list[Chunk] = []
    pdfs = sorted(root.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {root}")
        return all_chunks
    for pdf in pdfs:
        file_chunks = ingest_pdf(
            pdf,
            CONFIG.chunk_chars,
            CONFIG.chunk_overlap,
            CONFIG.extractor,
            CONFIG.contextual,
        )
        print(f"{pdf.name}: {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)
    return all_chunks
