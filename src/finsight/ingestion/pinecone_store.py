"""Pinecone vector store: one serverless index holding dense + sparse values per chunk.

A single index with the dotproduct metric is what Pinecone requires for hybrid (sparse-dense)
records. Each chunk's metadata carries ticker / form_type / fiscal_year / section / accession_no
so searches can pre-filter, plus the chunk text itself so a hit needs no second lookup.
"""

import logging
from collections.abc import Iterator
from typing import Any

from finsight.config import Settings
from finsight.ingestion.embedder import SparseVec
from finsight.schemas import Chunk, Hit, SearchFilter, hit_from_metadata

log = logging.getLogger("finsight.pinecone")
UPSERT_BATCH = 100
DELETE_BATCH = 1000


def chunk_metadata(c: Chunk) -> dict[str, Any]:
    return {**c.metadata.model_dump(), "chunk_index": c.chunk_index, "text": c.text}


def to_record(c: Chunk, dense: list[float], sparse: SparseVec) -> dict[str, Any]:
    rec: dict[str, Any] = {"id": c.chunk_id, "values": dense, "metadata": chunk_metadata(c)}
    if sparse.indices:  # Pinecone rejects an empty sparse vector; dense-only is still valid
        rec["sparse_values"] = {"indices": sparse.indices, "values": sparse.values}
    return rec


def to_pinecone_filter(f: SearchFilter | None) -> dict[str, Any] | None:
    """Translate our neutral filter into Pinecone's metadata filter (fields are AND-ed)."""
    if f is None:
        return None
    out: dict[str, Any] = {}
    if f.tickers:
        out["ticker"] = {"$in": [t.upper() for t in f.tickers]}
    if f.fiscal_year is not None:
        out["fiscal_year"] = {"$eq": f.fiscal_year}
    if f.form_type:
        out["form_type"] = {"$eq": f.form_type}
    if f.section:
        out["section"] = {"$eq": f.section}
    return out or None


class PineconeStore:
    name = "pinecone"

    def __init__(self, settings: Settings, client: Any = None, index: Any = None):
        self._s = settings
        self._client = client  # injectable for tests
        self._index = index

    def _pc(self) -> Any:
        if self._client is None:
            if self._s.pinecone_api_key is None:
                raise RuntimeError("PINECONE_API_KEY is not set (add it to your local .env)")
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self._s.pinecone_api_key.get_secret_value())
        return self._client

    @property
    def index(self) -> Any:
        if self._index is None:
            raise RuntimeError("call ensure_ready() first")
        return self._index

    def connect(self) -> None:
        """Attach to the existing index; never creates one (searching must not spend anything)."""
        if self._index is not None:
            return
        s, pc = self._s, self._pc()
        if not pc.has_index(s.pinecone_index):
            raise RuntimeError(
                f"Pinecone index {s.pinecone_index!r} does not exist; run the ingestion pipeline"
            )
        self._attach(pc.describe_index(s.pinecone_index).host)

    def _attach(self, host: str) -> None:
        s = self._s
        self._index = self._pc().Index(host=host)
        dim = self._index.describe_index_stats().dimension
        if dim != s.dense_dim:
            raise RuntimeError(
                f"index {s.pinecone_index!r} has dimension {dim} but FINSIGHT_DENSE_DIM is "
                f"{s.dense_dim}; use a different FINSIGHT_PINECONE_INDEX or fix the setting"
            )

    def ensure_ready(self) -> None:
        """Create the index if missing (never modifies or deletes an existing one)."""
        s, pc = self._s, self._pc()
        if not pc.has_index(s.pinecone_index):
            from pinecone import ServerlessSpec

            log.info(
                "creating Pinecone index %r (%s dims, dotproduct)", s.pinecone_index, s.dense_dim
            )
            pc.create_index(
                name=s.pinecone_index,
                dimension=s.dense_dim,
                metric="dotproduct",  # required for sparse-dense records
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
                timeout=300,
            )
        self._attach(pc.describe_index(s.pinecone_index).host)

    def count(self) -> int:
        """Vectors in our namespace. Pinecone's counts lag writes by a few seconds."""
        ns = self.index.describe_index_stats().namespaces or {}
        info = ns.get(self._s.pinecone_namespace)
        if info is None:
            return 0
        return int(info["vector_count"] if isinstance(info, dict) else info.vector_count)

    def replace_filing(
        self,
        old_ids: list[str],
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVec],
    ) -> None:
        """Idempotent write for one filing: drop its previous chunks by id, then upsert.

        Ids come from DuckDB (the previous ingest), so a re-chunk that yields fewer chunks
        cannot leave stale vectors behind. Deleting ids that don't exist is harmless, but a
        namespace that doesn't exist yet (first ever write) answers 404: nothing to delete."""
        from pinecone import NotFoundError

        ns = self._s.pinecone_namespace
        for i in range(0, len(old_ids), DELETE_BATCH):
            try:
                self.index.delete(ids=old_ids[i : i + DELETE_BATCH], namespace=ns)
            except NotFoundError:
                break  # namespace is created by the upsert below
        records = [to_record(c, d, sp) for c, d, sp in zip(chunks, dense, sparse, strict=True)]
        self.index.upsert(
            vectors=records, namespace=ns, batch_size=UPSERT_BATCH, show_progress=False
        )

    def search(
        self,
        dense: list[float],
        sparse: SparseVec,
        flt: SearchFilter | None,
        limit: int,
        alpha: float,
        sparse_scale: float = 1.0,
    ) -> list[Hit]:
        """Hybrid query in ONE request. Pinecone has no server-side fusion, so we balance the two
        signals client-side by scaling the query vectors: alpha for dense, (1 - alpha) for sparse.
        Sparse scores are unbounded and far larger than dense ones, so `sparse_scale` puts them on
        a comparable scale first (see Settings.sparse_score_scale)."""
        kwargs: dict[str, Any] = {
            "top_k": limit,
            "vector": [v * alpha for v in dense],
            "filter": to_pinecone_filter(flt),
            "include_metadata": True,
            "namespace": self._s.pinecone_namespace,
        }
        if sparse.indices:
            kwargs["sparse_vector"] = {
                "indices": sparse.indices,
                "values": [v * (1 - alpha) * sparse_scale for v in sparse.values],
            }
        resp = self.index.query(**kwargs)
        return [hit_from_metadata(m.id, m.metadata, m.score) for m in resp.matches]

    def scan(self) -> Iterator[dict[str, Any]]:
        """Yield every stored chunk's metadata (incl. text). Used to validate eval labels against
        what is actually in the index. Cost: a handful of list/fetch calls (tiny read units)."""
        ns = self._s.pinecone_namespace
        for page in self.index.list(namespace=ns, limit=100):
            ids = [v["id"] if isinstance(v, dict) else v.id for v in page.vectors]
            if not ids:
                continue
            for vec in self.index.fetch(ids=ids, namespace=ns).vectors.values():
                md = dict(vec.metadata)
                md["fiscal_year"] = int(md["fiscal_year"])  # Pinecone returns floats
                yield md
