"""Section-aware chunking: pack whole paragraphs up to a size cap, never crossing a section."""

import re

from finsight.schemas import Chunk, ChunkMetadata, Filing, make_chunk_id

MAX_CHARS = 1800  # ~400 tokens: safely under bge-small's 512-token limit
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, cur = [], ""
    for sent in _SENTENCE.split(paragraph):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    # A single sentence longer than max_chars: hard-split as a last resort.
    return [p[i : i + max_chars] for p in pieces for i in range(0, len(p), max_chars)]


def chunk_section(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    chunks, cur = [], ""
    for para in (p for line in text.split("\n") if (p := line.strip())):
        for piece in _split_long(para, max_chars):
            if cur and len(cur) + len(piece) + 1 > max_chars:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n{piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_filing(filing: Filing, sections: dict[str, str]) -> list[Chunk]:
    out: list[Chunk] = []
    for section, body in sections.items():
        for idx, text in enumerate(chunk_section(body)):
            out.append(
                Chunk(
                    chunk_id=make_chunk_id(filing.accession_no, section, idx),
                    chunk_index=idx,
                    text=text,
                    metadata=ChunkMetadata(
                        ticker=filing.ticker,
                        form_type=filing.form_type,
                        fiscal_year=filing.fiscal_year,
                        section=section,
                        accession_no=filing.accession_no,
                    ),
                )
            )
    return out
