"""Ingestion entrypoint: EDGAR -> parse -> chunk -> hosted embeddings -> Pinecone + DuckDB.

Run from the finsight/ folder:  uv run python -m finsight.ingestion.pipeline
"""

import argparse
import logging
from datetime import date

from finsight.config import Settings, get_settings
from finsight.ingestion.chunker import chunk_filing
from finsight.ingestion.edgar import (
    EdgarClient,
    filings_from_submissions,
    fiscal_years_from_companyfacts,
)
from finsight.ingestion.embedder import Embedder, estimate_tokens
from finsight.ingestion.parser import html_to_text, split_sections
from finsight.ingestion.pinecone_store import PineconeStore
from finsight.ingestion.stores import DuckStore

log = logging.getLogger("finsight.ingest")


def run(
    settings: Settings, tickers: list[str], force: bool = False, estimate: bool = False
) -> None:
    """Ingest filings. With estimate=True nothing is embedded, uploaded or saved: it only
    parses the cached filings and reports how many hosted-embedding tokens a real run needs."""
    edgar = EdgarClient(settings)
    duck = DuckStore(settings)
    store = PineconeStore(settings)
    embedder = Embedder(settings)
    if not estimate:
        store.ensure_ready()
    min_year = date.today().year - settings.years
    n_chunks_total, est_tokens = 0, 0

    try:
        for ticker in tickers:
            cik = edgar.ticker_to_cik(ticker)
            subs = edgar.get_submissions(cik)
            filings = filings_from_submissions(ticker, cik, subs, settings.forms, min_year)
            log.info("%s (CIK %s): %d filings in scope", ticker, cik, len(filings))

            # Fetched once: gives accurate fiscal years now and the numbers loaded below.
            facts = edgar.get_companyfacts(cik)
            fy_by_accn = fiscal_years_from_companyfacts(facts)

            for filing in filings:
                filing = filing.model_copy(
                    update={"fiscal_year": fy_by_accn.get(filing.accession_no, filing.fiscal_year)}
                )
                if not force and duck.is_indexed(filing.accession_no, store.name):
                    log.info("skip %s (already in %s)", filing.accession_no, store.name)
                    continue
                text = html_to_text(edgar.get_document(filing))
                sections = split_sections(text, filing.form_type)
                chunks = chunk_filing(filing, sections)
                if not chunks:
                    log.warning("no sections parsed for %s %s", ticker, filing.accession_no)
                    continue
                texts = [c.text for c in chunks]
                if estimate:
                    n_chunks_total += len(chunks)
                    est_tokens += sum(estimate_tokens(x) for x in texts) * 2  # dense + sparse
                    continue
                store.replace_filing(
                    duck.chunk_ids(filing.accession_no),
                    chunks,
                    embedder.embed_dense(texts),
                    embedder.embed_sparse(texts),
                )
                duck.save_filing(filing, chunks)
                duck.mark_indexed(filing.accession_no, store.name, len(chunks))  # last: done
                log.info(
                    "ingested %s %s FY%s: %d chunks in %s",
                    ticker, filing.form_type, filing.fiscal_year, len(chunks), list(sections),
                )  # fmt: skip

            if not estimate:
                n = duck.save_facts(ticker, cik, facts, min_year)
                log.info("%s: %d XBRL facts loaded", ticker, n)
    finally:
        edgar.close()

    if estimate:
        used, limit = embedder.budget.used(), settings.embed_token_budget
        log.info(
            "ESTIMATE: %d chunks to embed, ~%s tokens (dense + sparse). Budget this month: "
            "%s used of %s. %s",
            n_chunks_total, f"{est_tokens:,}", f"{used:,}", f"{limit:,}",
            "Fits." if used + est_tokens <= limit else "WOULD EXCEED the budget.",
        )  # fmt: skip


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest SEC filings into Pinecone + DuckDB")
    ap.add_argument("--tickers", nargs="+", help="override configured tickers")
    ap.add_argument("--force", action="store_true", help="re-ingest already-ingested filings")
    ap.add_argument(
        "--estimate", action="store_true", help="count chunks and tokens only; embed/upload nothing"
    )
    args = ap.parse_args()
    settings = get_settings()
    run(settings, args.tickers or settings.tickers, force=args.force, estimate=args.estimate)


if __name__ == "__main__":
    main()
