"""Ask a question from the terminal.

Run from the finsight/ folder:
  uv run python -m finsight.retrieval.cli "What supply chain risks does NVIDIA cite?" --ticker NVDA
"""

import argparse
import logging

from finsight.config import get_settings
from finsight.ingestion.embedder import Embedder
from finsight.ingestion.pinecone_store import PineconeStore
from finsight.retrieval.answer import answer_question, make_llm
from finsight.retrieval.cache import JsonCache
from finsight.retrieval.rerank import make_reranker
from finsight.retrieval.search import HybridSearcher, build_filter


def main() -> None:
    for noisy in ("httpx", "pinecone", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description="Ask a question about ingested SEC filings")
    ap.add_argument("question")
    ap.add_argument("--ticker", nargs="+", help="restrict to these tickers")
    ap.add_argument("--year", type=int, help="fiscal year")
    ap.add_argument("--form", choices=["10-K", "10-Q"])
    ap.add_argument("--section", help='e.g. "Item 1A" (risk factors) or "Item 7" (MD&A)')
    ap.add_argument("--retrieve-only", action="store_true", help="skip the LLM; show top chunks")
    args = ap.parse_args()

    s = get_settings()
    cache = JsonCache(s.cache_dir)
    searcher = HybridSearcher(s, Embedder(s), PineconeStore(s), cache=cache)
    flt = build_filter(args.ticker, args.year, args.form, args.section)

    candidates = searcher.search(args.question, flt)
    hits = make_reranker(s).rerank(args.question, candidates, s.final_k)

    if args.retrieve_only:
        for i, h in enumerate(hits, 1):
            print(f"\n[C{i}] {h.source}  (score {h.score:.3f})\n{h.text[:500]}")
        return

    ans = answer_question(args.question, hits, make_llm(s, cache))
    print(f"\n{ans.text}\n\nSources:")
    for label, h in ans.sources.items():
        print(f"  [{label}] {h.source}  (accession {h.accession_no})")
    if ans.invalid_citations:
        print(f"\nWARNING: answer cited labels that don't exist: {ans.invalid_citations}")
    print(f"\n{ans.disclaimer}")


if __name__ == "__main__":
    main()
