from datetime import date
from types import SimpleNamespace

import pytest

from finsight.config import Settings
from finsight.ingestion.embedder import BudgetExceeded, Embedder, SparseVec, TokenBudget
from finsight.ingestion.pinecone_store import PineconeStore, to_record
from finsight.ingestion.stores import DuckStore
from finsight.schemas import Chunk, ChunkMetadata, Filing, make_chunk_id


def _settings(tmp_path, **kw) -> Settings:
    return Settings(
        _env_file=None, sec_user_agent="app me@example.com", data_dir=tmp_path,
        pinecone_api_key="test-key", **kw,
    )  # fmt: skip


def _chunk(i: int, text: str = "some filing text") -> Chunk:
    meta = ChunkMetadata(
        ticker="NVDA", form_type="10-K", fiscal_year=2025, section="Item 1A", accession_no="acc1"
    )
    return Chunk(
        chunk_id=make_chunk_id("acc1", "Item 1A", i), chunk_index=i, text=text, metadata=meta
    )


# ---------- token budget ----------


def test_budget_tracks_usage_and_blocks_overspend(tmp_path):
    b = TokenBudget(tmp_path / "u.json", limit=1000)
    assert b.used() == 0
    b.record(900)
    assert b.used() == 900
    b.check(100)  # exactly at the limit is allowed
    with pytest.raises(BudgetExceeded, match="EMBED_TOKEN_BUDGET"):
        b.check(101)


def test_budget_resets_each_month(tmp_path, monkeypatch):
    b = TokenBudget(tmp_path / "u.json", limit=1000)
    b.record(900)
    monkeypatch.setattr(b, "_month", lambda: "2099-01")
    assert b.used() == 0  # a new month starts from zero
    b.record(5)
    assert b.used() == 5


def test_budget_survives_a_corrupt_ledger(tmp_path):
    (tmp_path / "u.json").write_text("{not json")
    assert TokenBudget(tmp_path / "u.json", 10).used() == 0


# ---------- hosted embeddings ----------


class FakeInference:
    def __init__(self, fail_first: int = 0, error: Exception | None = None):
        self.calls: list[dict] = []
        self.fail_first, self.error = fail_first, error or ConnectionError("blip")

    def embed(self, *, model, inputs, parameters):
        self.calls.append({"model": model, "inputs": list(inputs), "parameters": parameters})
        if len(self.calls) <= self.fail_first:
            raise self.error
        sparse = "sparse" in model
        data = [
            SimpleNamespace(sparse_indices=[1, 2], sparse_values=[0.5, 0.25])
            if sparse
            else SimpleNamespace(values=[0.1, 0.2, 0.3])
            for _ in inputs
        ]
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=10 * len(inputs)))


def _embedder(tmp_path, inference, **kw):
    s = _settings(tmp_path, **kw)
    return Embedder(s, client=SimpleNamespace(inference=inference)), s


def test_embedding_is_batched_and_tokens_are_recorded(tmp_path):
    inf = FakeInference()
    emb, _ = _embedder(tmp_path, inf, embed_batch=2)
    vecs = emb.embed_dense(["a", "b", "c", "d", "e"])
    assert len(vecs) == 5 and [len(c["inputs"]) for c in inf.calls] == [2, 2, 1]
    assert emb.budget.used() == 50  # taken from the API's usage report, not the estimate


def test_dense_params_pass_dimension_and_input_type(tmp_path):
    inf = FakeInference()
    emb, s = _embedder(tmp_path, inf)
    emb.embed_dense(["q"], kind="query")
    call = inf.calls[0]
    assert call["model"] == s.dense_model
    assert call["parameters"] == {"input_type": "query", "truncate": "END", "dimension": 1024}


def test_sparse_passages_get_long_sequences_but_queries_do_not(tmp_path):
    inf = FakeInference()
    emb, _ = _embedder(tmp_path, inf)
    out = emb.embed_sparse(["doc"], kind="passage")
    emb.embed_sparse(["q"], kind="query")
    assert out == [SparseVec([1, 2], [0.5, 0.25])]
    assert inf.calls[0]["parameters"]["max_tokens_per_sequence"] == 2048
    assert "max_tokens_per_sequence" not in inf.calls[1]["parameters"]


def test_over_budget_batch_is_never_sent(tmp_path):
    inf = FakeInference()
    emb, _ = _embedder(tmp_path, inf, embed_token_budget=5)
    with pytest.raises(BudgetExceeded):
        emb.embed_dense(["x" * 400])  # ~100 estimated tokens
    assert inf.calls == []


def test_transient_errors_retry_then_succeed(tmp_path, monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)  # no real waiting in tests
    inf = FakeInference(fail_first=1)
    emb, _ = _embedder(tmp_path, inf)
    assert len(emb.embed_dense(["a"])) == 1 and len(inf.calls) == 2


def test_retries_are_capped_at_three_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    inf = FakeInference(fail_first=99)
    emb, _ = _embedder(tmp_path, inf)
    with pytest.raises(ConnectionError):
        emb.embed_dense(["a"])
    assert len(inf.calls) == 3


def test_bad_input_errors_are_not_retried(tmp_path):
    inf = FakeInference(fail_first=99, error=ValueError("bad input"))
    emb, _ = _embedder(tmp_path, inf)
    with pytest.raises(ValueError):
        emb.embed_dense(["a"])
    assert len(inf.calls) == 1


def test_missing_key_fails_fast_without_network(tmp_path):
    s = Settings(_env_file=None, sec_user_agent="app me@example.com", data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        Embedder(s).embed_dense(["a"])


# ---------- Pinecone store ----------


class FakeIndex:
    def __init__(self, dim=1024, namespaces=None):
        self.deleted: list[list[str]] = []
        self.upserts: list[dict] = []
        self._stats = SimpleNamespace(dimension=dim, namespaces=namespaces or {})

    def delete(self, *, ids, namespace):
        self.deleted.append(list(ids))

    def upsert(self, *, vectors, namespace, batch_size, show_progress):
        self.upserts.append({"vectors": vectors, "namespace": namespace, "batch": batch_size})

    def describe_index_stats(self):
        return self._stats


class FakeClient:
    def __init__(self, exists: bool, index: FakeIndex):
        self.exists, self.index, self.created = exists, index, []

    def has_index(self, name):
        return self.exists

    def create_index(self, **kw):
        self.created.append(kw)

    def describe_index(self, name):
        return SimpleNamespace(host="h.svc.pinecone.io")

    def Index(self, host):  # noqa: N802  (mirrors the SDK's method name)
        return self.index


def test_record_carries_metadata_text_and_both_vectors():
    rec = to_record(_chunk(0, "hello"), [0.1, 0.2], SparseVec([3], [0.9]))
    assert rec["id"] == _chunk(0).chunk_id and rec["values"] == [0.1, 0.2]
    assert rec["sparse_values"] == {"indices": [3], "values": [0.9]}
    assert rec["metadata"] == {
        "ticker": "NVDA", "form_type": "10-K", "fiscal_year": 2025, "section": "Item 1A",
        "accession_no": "acc1", "chunk_index": 0, "text": "hello",
    }  # fmt: skip


def test_empty_sparse_vector_is_omitted():
    assert "sparse_values" not in to_record(_chunk(0), [0.1], SparseVec([], []))


def test_ensure_ready_creates_a_dotproduct_index_when_missing(tmp_path):
    client = FakeClient(exists=False, index=FakeIndex())
    PineconeStore(_settings(tmp_path), client=client).ensure_ready()
    (kw,) = client.created
    assert kw["name"] == "finsight" and kw["dimension"] == 1024 and kw["metric"] == "dotproduct"


def test_ensure_ready_never_recreates_an_existing_index(tmp_path):
    client = FakeClient(exists=True, index=FakeIndex())
    PineconeStore(_settings(tmp_path), client=client).ensure_ready()
    assert client.created == []


def test_ensure_ready_rejects_a_dimension_mismatch(tmp_path):
    client = FakeClient(exists=True, index=FakeIndex(dim=384))
    with pytest.raises(RuntimeError, match="dimension 384"):
        PineconeStore(_settings(tmp_path), client=client).ensure_ready()


def test_replace_filing_deletes_old_ids_then_upserts(tmp_path):
    idx = FakeIndex()
    store = PineconeStore(_settings(tmp_path), index=idx)
    old = [f"old{i}" for i in range(2500)]
    chunks = [_chunk(0), _chunk(1)]
    store.replace_filing(old, chunks, [[0.1], [0.2]], [SparseVec([1], [1.0])] * 2)
    assert [len(b) for b in idx.deleted] == [1000, 1000, 500]
    (up,) = idx.upserts
    assert [v["id"] for v in up["vectors"]] == [c.chunk_id for c in chunks]
    assert up["namespace"] == "filings"


def test_first_ingest_has_nothing_to_delete(tmp_path):
    idx = FakeIndex()
    PineconeStore(_settings(tmp_path), index=idx).replace_filing(
        [], [_chunk(0)], [[0.1]], [SparseVec([1], [1.0])]
    )
    assert idx.deleted == [] and len(idx.upserts) == 1


def test_count_reads_our_namespace_only(tmp_path):
    ns = {"filings": SimpleNamespace(vector_count=42), "other": SimpleNamespace(vector_count=7)}
    store = PineconeStore(_settings(tmp_path), index=FakeIndex(namespaces=ns))
    assert store.count() == 42
    assert PineconeStore(_settings(tmp_path), index=FakeIndex()).count() == 0


# ---------- DuckDB per-store marker ----------


def test_indexed_marker_is_per_store_and_idempotent(tmp_path):
    duck = DuckStore(_settings(tmp_path))
    filing = Filing(
        accession_no="acc1", ticker="NVDA", cik=1, form_type="10-K", fiscal_year=2025,
        filing_date=date(2025, 2, 26), report_date=date(2025, 1, 26), primary_document="d.htm",
    )  # fmt: skip
    chunks = [_chunk(0), _chunk(1)]
    assert not duck.is_indexed("acc1", "pinecone")
    duck.save_filing(filing, chunks)
    assert not duck.is_indexed("acc1", "pinecone")  # a filing row alone is not "indexed"
    duck.mark_indexed("acc1", "pinecone", 2)
    duck.mark_indexed("acc1", "pinecone", 2)  # re-marking must not raise or duplicate
    assert duck.is_indexed("acc1", "pinecone") and not duck.is_indexed("acc1", "other-store")
    assert sorted(duck.chunk_ids("acc1")) == sorted(c.chunk_id for c in chunks)


def test_first_ever_write_survives_a_missing_namespace(tmp_path):
    """Regression: Pinecone answers 404 'Namespace not found' when deleting from a namespace
    that has never been written to. That must not stop the very first ingest."""
    from pinecone import NotFoundError

    class EmptyIndex(FakeIndex):
        def delete(self, *, ids, namespace):
            raise NotFoundError("Namespace not found")

    idx = EmptyIndex()
    PineconeStore(_settings(tmp_path), index=idx).replace_filing(
        ["stale-from-another-store"], [_chunk(0)], [[0.1]], [SparseVec([1], [1.0])]
    )
    assert len(idx.upserts) == 1  # the write still happened
