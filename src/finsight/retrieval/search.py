"""Hybrid retrieval on Pinecone: embed the question (dense + sparse), search once, cache."""

from dataclasses import asdict

import duckdb

from finsight.config import Settings
from finsight.ingestion.embedder import Embedder
from finsight.ingestion.pinecone_store import PineconeStore
from finsight.observability import annotate, traceable
from finsight.retrieval.cache import JsonCache
from finsight.schemas import Hit, SearchFilter
from finsight.tracing import span


def build_filter(
    tickers: list[str] | None = None,
    fiscal_year: int | None = None,
    form_type: str | None = None,
    section: str | None = None,
) -> SearchFilter | None:
    if not (tickers or fiscal_year is not None or form_type or section):
        return None
    return SearchFilter(tuple(t.upper() for t in tickers or []), fiscal_year, form_type, section)


def corpus_version(settings: Settings) -> str:
    """Changes whenever filings are (re)indexed, so cached search results can't go stale."""
    try:
        con = duckdb.connect(str(settings.duckdb_path), read_only=True)
        try:
            row = con.execute(
                "SELECT count(*), coalesce(sum(n_chunks), 0), max(indexed_at) FROM indexed "
                "WHERE store = 'pinecone'"
            ).fetchone()
        finally:
            con.close()
        return str(row)
    except duckdb.Error:
        return "unknown"


def _as_documents(out) -> dict:
    """LangSmith shows a retriever's passages as documents when its output has this shape."""
    hits = out.get("output") if isinstance(out, dict) else out
    return {
        "documents": [
            {
                "page_content": h.text,
                "metadata": {"source": h.source, "score": h.score, "chunk_id": h.chunk_id},
            }
            for h in (hits or [])
        ]
    }


class HybridSearcher:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: PineconeStore,
        cache: JsonCache | None = None,
        version: str | None = None,
    ):
        self._s, self._embedder, self._store, self._cache = settings, embedder, store, cache
        self._version = version if version is not None else corpus_version(settings)
        self._query_vecs: dict[str, tuple] = {}  # the same question is re-searched in sweeps

    def search(
        self,
        question: str,
        flt: SearchFilter | None = None,
        limit: int | None = None,
        alpha: float | None = None,
    ) -> list[Hit]:
        """alpha overrides the configured dense/sparse balance for this call (used by sweeps)."""
        s = self._s
        limit = limit or s.retrieve_k
        alpha = s.hybrid_alpha if alpha is None else alpha
        key = JsonCache.make_key(
            "search", question, flt.as_key() if flt else None, limit, alpha,
            s.sparse_score_scale, s.dense_model, s.sparse_model,
            s.pinecone_index, s.pinecone_namespace, self._version,
        )  # fmt: skip
        with span("search", limit=limit, alpha=alpha, filtered=flt is not None) as a:
            if self._cache and (cached := self._cache.get(key)) is not None:
                a.update(cached=True, hits=len(cached))
                return [Hit(**h) for h in cached]
            hits = self._search_live(question, flt, limit, alpha)
            a.update(cached=False, hits=len(hits))
            if self._cache:
                self._cache.set(key, [asdict(h) for h in hits])
            return hits

    @traceable(run_type="retriever", name="pinecone.search", process_outputs=_as_documents)
    def _search_live(self, question, flt, limit, alpha) -> list[Hit]:
        annotate(filtered=flt is not None, limit=limit, alpha=alpha)
        s = self._s
        self._store.connect()
        if question not in self._query_vecs:
            self._query_vecs[question] = (
                self._embedder.embed_dense([question], kind="query")[0],
                self._embedder.embed_sparse([question], kind="query")[0],
            )
        dense, sparse = self._query_vecs[question]
        return self._store.search(
            dense, sparse, flt, limit, alpha, sparse_scale=s.sparse_score_scale
        )
