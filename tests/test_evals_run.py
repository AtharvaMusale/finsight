from types import SimpleNamespace

from finsight.config import Settings
from finsight.evals.run import evaluate
from finsight.ingestion.embedder import SparseVec
from finsight.ingestion.pinecone_store import PineconeStore
from finsight.retrieval.search import HybridSearcher
from finsight.schemas import Hit


def _settings(tmp_path, **kw) -> Settings:
    return Settings(_env_file=None, sec_user_agent="app me@example.com", data_dir=tmp_path, **kw)


class PagedIndex:
    """list() yields pages of ids; fetch() returns their metadata (numbers come back as floats)."""

    def __init__(self, pages):
        self.pages, self.fetched = pages, []

    def list(self, *, namespace, limit):
        for ids in self.pages:
            yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in ids])

    def fetch(self, *, ids, namespace):
        self.fetched.append(list(ids))
        md = {"ticker": "NVDA", "fiscal_year": 2025.0, "text": "t", "section": "Item 1A"}
        return SimpleNamespace(vectors={i: SimpleNamespace(metadata=dict(md, id=i)) for i in ids})


def test_scan_pages_through_every_id_and_fixes_the_year_type(tmp_path):
    idx = PagedIndex([["a", "b"], ["c"], []])  # an empty page must be tolerated
    rows = list(PineconeStore(_settings(tmp_path), index=idx).scan())
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["fiscal_year"] == 2025 and isinstance(rows[0]["fiscal_year"], int)
    assert idx.fetched == [["a", "b"], ["c"]]


def _hit(n: int, text: str) -> Hit:
    return Hit(f"id{n}", text, "NVDA", "10-K", 2025, "Item 1A", "acc")


class ScriptedStore:
    def __init__(self):
        self.connected = 0

    def connect(self):
        self.connected += 1

    def search(self, dense, sparse, flt, limit, alpha, sparse_scale=1.0):
        # the relevant chunk is buried at rank 3 of the pool
        return [_hit(1, "noise"), _hit(2, "noise"), _hit(3, "export control")][:limit]


class Emb:
    def embed_dense(self, texts, kind="passage"):
        return [[0.1]]

    def embed_sparse(self, texts, kind="passage"):
        return [SparseVec([1], [1.0])]


def _searcher(s) -> HybridSearcher:
    return HybridSearcher(s, Emb(), ScriptedStore(), version="v")


ROWS = [
    {
        "id": "q1", "type": "single", "question": "export controls?", "smoke": True,
        "expect_abstain": False, "filters": {"tickers": ["NVDA"], "year": 2025, "form": "10-K"},
        "targets": [{"ticker": "NVDA", "pattern": "export control"}],
    },
    {
        "id": "u1", "type": "unanswerable", "question": "tesla?", "smoke": False,
        "expect_abstain": True, "filters": {}, "targets": [],
    },
]  # fmt: skip


def test_evaluate_measures_pool_topk_and_skips_unanswerable_retrieval(tmp_path):
    s = _settings(tmp_path, final_k=2, retrieve_k=3)
    results, summary = evaluate(ROWS, s, _searcher(s))
    assert summary["n_questions"] == 2 and summary["n_answerable"] == 1
    r = summary["retrieval"]
    assert r["pool@3"]["hit"] == 1.0  # found somewhere in the candidate pool...
    assert r["top@2"]["hit"] == 0.0 and r["final@2"]["hit"] == 0.0  # ...but not in the top 2
    assert r["pool@3"]["mrr"] == round(1 / 3, 3)
    assert "final" not in results[1]  # abstain questions have no retrieval targets
    assert summary["by_type"] == {"single": r["final@2"]}


def test_evaluate_answer_checks_score_abstention_and_citations(tmp_path):
    s = _settings(tmp_path, final_k=3, retrieve_k=3)
    llm = lambda msgs: "I could not find that in the provided filings."  # noqa: E731
    _, summary = evaluate(ROWS, s, _searcher(s), llm=llm)
    a = summary["answers"]
    assert a["abstain_correct"] == 1.0  # the unanswerable question was correctly declined
    assert a["false_abstain"] == 1.0  # ...but declining the answerable one is a miss
