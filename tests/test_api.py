import time

import pytest
from fastapi.testclient import TestClient

from finsight.api import app as api
from finsight.config import Settings
from finsight.orchestrator.graph import Deps
from finsight.retrieval.rerank import NoRerank
from test_graph import FakeLLM, FakeSearcher


def _deps(llm) -> Deps:
    return Deps(
        settings=Settings(_env_file=None, sec_user_agent="t t@example.com"),
        searcher=FakeSearcher(),
        reranker=NoRerank(),
        llm=llm,
        facts_db=None,
    )


@pytest.fixture
def client_for(monkeypatch):
    def make(llm, timeout=30.0):
        monkeypatch.setattr(api, "_build_deps", lambda: _deps(llm))
        real = Settings(_env_file=None, sec_user_agent="t t@example.com", request_timeout_s=timeout)
        monkeypatch.setattr(api, "get_settings", lambda: real)
        return TestClient(api.app)  # used as a context manager so lifespan builds the graph

    return make


def test_health_and_trace_header(client_for):
    with client_for(FakeLLM()) as c:
        r = c.get("/health")
    assert r.json() == {"status": "ok"} and len(r.headers["X-Trace-Id"]) == 32


def test_ask_returns_cited_answer_with_matching_trace_id(client_for):
    with client_for(FakeLLM(answer="Apple cites it [C1].")) as c:
        r = c.post("/ask", json={"question": "Does Apple mention tariffs in fiscal 2025?"})
    body = r.json()
    assert r.status_code == 200
    assert body["trace_id"] == r.headers["X-Trace-Id"]
    assert body["sources"]["C1"]["accession_no"] == "acc-AAPL"
    assert body["route"] == "text" and body["invalid_citations"] == []
    assert "Not financial advice" in body["disclaimer"]


def test_input_validation(client_for):
    with client_for(FakeLLM()) as c:
        assert c.post("/ask", json={"question": "hi"}).status_code == 422  # too short
        assert c.post("/ask", json={"question": "x" * 501}).status_code == 422  # too long
        assert c.post("/ask", json={}).status_code == 422


def test_slow_request_times_out_with_504(client_for):
    class Slow(FakeLLM):
        def __call__(self, messages):
            time.sleep(1.0)
            return super().__call__(messages)

    with client_for(Slow(), timeout=0.05) as c:
        r = c.post("/ask", json={"question": "Does Apple mention tariffs in fiscal 2025?"})
    assert r.status_code == 504 and r.headers["X-Trace-Id"] in r.json()["detail"]


def test_internal_errors_do_not_leak_details(client_for):
    class Boom(FakeLLM):
        def __call__(self, messages):
            raise RuntimeError("secret internal detail")

    with client_for(Boom()) as c:
        r = c.post("/ask", json={"question": "Does Apple mention tariffs in fiscal 2025?"})
    assert r.status_code == 502 and "secret" not in r.text
