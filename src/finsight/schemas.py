"""Data contracts shared by ingestion and retrieval."""

import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel

_NAMESPACE = uuid.UUID("6f1d5b7e-3a52-4c1e-9d0a-5a1f0c2b9e11")


class Filing(BaseModel):
    accession_no: str  # e.g. 0000320193-24-000123; the idempotency key
    ticker: str
    cik: int
    form_type: str
    fiscal_year: int  # derived from period-of-report year (approximation, see edgar.py)
    filing_date: date
    report_date: date | None
    primary_document: str

    @property
    def url(self) -> str:
        nodash = self.accession_no.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{nodash}/{self.primary_document}"
        )


class ChunkMetadata(BaseModel):
    ticker: str
    form_type: str
    fiscal_year: int
    section: str  # e.g. "Item 1A"
    accession_no: str


class Chunk(BaseModel):
    chunk_id: str  # deterministic uuid so re-ingestion overwrites, never duplicates
    chunk_index: int
    text: str
    metadata: ChunkMetadata


def make_chunk_id(accession_no: str, section: str, index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{accession_no}:{section}:{index}"))


@dataclass
class Hit:
    """One retrieved chunk, as returned by a search."""

    chunk_id: str
    text: str
    ticker: str
    form_type: str
    fiscal_year: int
    section: str
    accession_no: str
    score: float = 0.0

    @property
    def source(self) -> str:
        return f"{self.ticker} {self.form_type} FY{self.fiscal_year}, {self.section}"


def hit_from_metadata(chunk_id: str, md: Any, score: float = 0.0) -> Hit:
    return Hit(
        chunk_id=str(chunk_id),
        text=md["text"],
        ticker=md["ticker"],
        form_type=md["form_type"],
        fiscal_year=int(md["fiscal_year"]),  # Pinecone returns numbers as floats
        section=md["section"],
        accession_no=md["accession_no"],
        score=float(score),
    )


@dataclass(frozen=True)
class SearchFilter:
    """Metadata pre-filter: narrows the search space BEFORE ranking."""

    tickers: tuple[str, ...] = ()
    fiscal_year: int | None = None
    form_type: str | None = None
    section: str | None = None

    def as_key(self) -> dict:
        return {
            "tickers": list(self.tickers), "fiscal_year": self.fiscal_year,
            "form_type": self.form_type, "section": self.section,
        }  # fmt: skip
