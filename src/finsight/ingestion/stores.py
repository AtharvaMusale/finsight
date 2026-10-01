"""DuckDB storage: filing registry, chunk records, XBRL facts. Vectors live in Pinecone."""

import duckdb

from finsight.config import Settings
from finsight.schemas import Chunk, Filing

_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession_no VARCHAR PRIMARY KEY,
    ticker VARCHAR, cik BIGINT, form_type VARCHAR, fiscal_year INTEGER,
    filing_date DATE, report_date DATE, url VARCHAR
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR PRIMARY KEY,
    accession_no VARCHAR, section VARCHAR, chunk_index INTEGER, n_chars INTEGER
);
CREATE TABLE IF NOT EXISTS indexed (
    accession_no VARCHAR, store VARCHAR, n_chunks INTEGER, indexed_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (accession_no, store)
);
CREATE TABLE IF NOT EXISTS facts (
    cik BIGINT, ticker VARCHAR, taxonomy VARCHAR, tag VARCHAR, unit VARCHAR,
    period_start DATE, period_end DATE, value DOUBLE,
    fiscal_year INTEGER, fiscal_period VARCHAR, form_type VARCHAR,
    accession_no VARCHAR, filed DATE,
    PRIMARY KEY (cik, taxonomy, tag, unit, period_start, period_end, accession_no)
);
"""


class DuckStore:
    def __init__(self, settings: Settings):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(settings.duckdb_path))
        self.con.execute(_SCHEMA)

    def is_indexed(self, accession_no: str, store: str) -> bool:
        """Has this filing been written to THIS vector store? (A new index starts empty.)"""
        row = self.con.execute(
            "SELECT 1 FROM indexed WHERE accession_no = ? AND store = ?", [accession_no, store]
        ).fetchone()
        return row is not None

    def mark_indexed(self, accession_no: str, store: str, n_chunks: int) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO indexed (accession_no, store, n_chunks) VALUES (?, ?, ?)",
            [accession_no, store, n_chunks],
        )

    def chunk_ids(self, accession_no: str) -> list[str]:
        """Chunk ids from the previous ingest, so a re-chunk can delete stale vectors by id."""
        rows = self.con.execute(
            "SELECT chunk_id FROM chunks WHERE accession_no = ?", [accession_no]
        ).fetchall()
        return [r[0] for r in rows]

    def save_filing(self, filing: Filing, chunks: list[Chunk]) -> None:
        """Written LAST in the pipeline: if a run crashes earlier, the filing is retried."""
        self.con.execute("DELETE FROM chunks WHERE accession_no = ?", [filing.accession_no])
        self.con.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
            [
                [c.chunk_id, filing.accession_no, c.metadata.section, c.chunk_index, len(c.text)]
                for c in chunks
            ],
        )
        self.con.execute(
            "INSERT OR REPLACE INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                filing.accession_no, filing.ticker, filing.cik, filing.form_type,
                filing.fiscal_year, filing.filing_date, filing.report_date, filing.url,
            ],
        )  # fmt: skip

    def save_facts(self, ticker: str, cik: int, companyfacts: dict, min_year: int) -> int:
        """Load us-gaap facts reported in 10-K/10-Q. Numbers later come from here, not the LLM.

        Instant facts (balance sheet) have no start date; we store start = end for them."""
        rows = []
        for taxonomy, tags in companyfacts.get("facts", {}).items():
            if taxonomy != "us-gaap":
                continue
            for tag, body in tags.items():
                for unit, entries in body.get("units", {}).items():
                    for e in entries:
                        if e.get("form") not in ("10-K", "10-Q") or e.get("fy", 0) < min_year:
                            continue
                        rows.append([
                            cik, ticker, taxonomy, tag, unit,
                            e.get("start") or e["end"], e["end"], e["val"],
                            e.get("fy"), e.get("fp"), e.get("form"), e["accn"], e.get("filed"),
                        ])  # fmt: skip
        if rows:
            self.con.executemany(
                "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
        return len(rows)
