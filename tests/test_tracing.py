"""Tracing: span trees, privacy rules, cross-agent trees, the API endpoint and the CLI."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from finsight import tracing
from finsight.api import app as api
from finsight.config import Settings
from finsight.orchestrator.graph import ask
from finsight.retrieval.answer import make_llm
from finsight.retrieval.cache import JsonCache
from finsight.trace_cli import main as cli_main
from finsight.trace_cli import render
from test_a2a import HOSTS, Mesh, _deps, _settings, agent, analyst_app, with_remote_agents, workers
from test_graph import FakeLLM, _graph


@pytest.fixture
def traces(tmp_path):
    d = tmp_path / "traces"
    tracing.configure("test", d)
    yield d
    tracing.tracer.disable()  # other tests must see tracing off


def spans_of(d, trace_id=None):
    return tracing.read_spans(d, trace_id)


# ---------- the tracer ----------


def test_tracing_is_off_until_configured(tmp_path):
    tracing.tracer.disable()
    with tracing.span("x", n=1) as a:
        a["more"] = 2
    assert not list(tmp_path.glob("**/*.jsonl"))


def test_nested_spans_form_a_tree_with_timings(traces):
    with tracing.adopt("t-1", None), tracing.span("outer"), tracing.span("inner", items=3) as a:
        a["late"] = True
    by_name = {s["name"]: s for s in spans_of(traces, "t-1")}
    assert by_name["inner"]["parent"] == by_name["outer"]["span"]
    assert by_name["outer"]["parent"] is None
    assert by_name["inner"]["attrs"] == {"items": 3, "late": True}
    assert by_name["outer"]["ms"] >= by_name["inner"]["ms"] >= 0


def test_errors_record_the_type_but_never_the_message(traces):
    with tracing.adopt("t-err", None), pytest.raises(RuntimeError), tracing.span("boom"):
        raise RuntimeError("secret-token-abc /Users/me/.env")
    (s,) = spans_of(traces, "t-err")
    assert s["ok"] is False and s["error"] == "RuntimeError"
    assert "secret-token-abc" not in json.dumps(s)


def test_attributes_are_bounded_scalars(traces):
    with tracing.adopt("t-attr", None), tracing.span("a", text="x" * 5000, obj={"k": "v"}, n=1.5):
        pass
    (s,) = spans_of(traces, "t-attr")
    assert len(s["attrs"]["text"]) == 200 and isinstance(s["attrs"]["obj"], str)
    assert s["attrs"]["n"] == 1.5


def test_a_write_failure_never_breaks_the_request(tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    tracing.configure("test", blocker)  # mkdir under a file fails with OSError
    try:
        with tracing.span("still-works") as a:
            a["ok"] = 1
    finally:
        tracing.tracer.disable()


def test_reader_skips_corrupt_lines(traces):
    with tracing.adopt("t-ok", None), tracing.span("good"):
        pass
    (path,) = traces.glob("traces-*.jsonl")
    path.write_text(path.read_text() + "{not json\n[1,2]\n")
    assert [s["name"] for s in spans_of(traces, "t-ok")] == ["good"]


# ---------- the graph ----------


def test_a_graph_run_is_a_span_tree_without_any_text(traces):
    llm = FakeLLM(answer="AAPL flagged that tariffs may affect its supply chain. [C1]")
    graph, _ = _graph(llm)
    question = "Does Apple mention tariffs in fiscal 2025?"
    out = ask(graph, question, trace_id="trace-graph-1")
    spans = spans_of(traces, "trace-graph-1")
    names = [s["name"] for s in spans]
    for expected in ("ask", "node.plan", "node.retrieve", "node.compose", "node.verify",
                     "node.finalize"):  # fmt: skip
        assert expected in names
    root = next(s for s in spans if s["name"] == "ask")
    assert root["parent"] is None and root["attrs"]["route"] == "text"
    assert root["attrs"]["verified"] is True
    node = {s["name"]: s for s in spans}
    assert node["node.retrieve"]["parent"] == root["span"]
    assert node["node.retrieve"]["attrs"]["hits"] == len(out["hits"])
    assert node["node.verify"]["attrs"]["issues"] == 0
    blob = (next(traces.glob("traces-*.jsonl"))).read_text()
    assert "tariffs" not in blob and "Does Apple" not in blob  # no question, no chunk text


def test_a_revision_shows_up_as_extra_spans(traces):
    bad = "Apple tariff costs were $999 billion [C1]."
    good = "AAPL flagged that tariffs may affect its supply chain. [C1]"
    graph, _ = _graph(FakeLLM(answer=[bad, good]))
    ask(graph, "Does Apple mention tariffs in fiscal 2025?", trace_id="trace-rev")
    names = [s["name"] for s in spans_of(traces, "trace-rev")]
    assert names.count("node.verify") == 2 and names.count("node.revise") == 1


# ---------- LLM calls ----------


def _fake_completion(monkeypatch, content="hello", fail=False):
    import litellm

    def completion(**kw):
        if fail:
            raise RuntimeError("provider said: key FAKE-PROVIDER-SECRET is invalid")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
        )

    monkeypatch.setattr(litellm, "completion", completion)


def test_llm_spans_record_model_tokens_and_cache_hits_but_not_the_prompt(
    traces, tmp_path, monkeypatch
):
    _fake_completion(monkeypatch)
    s = Settings(_env_file=None, sec_user_agent="t t@example.com", llm_model="openai/fake")
    llm = make_llm(s, JsonCache(tmp_path / "cache"))
    with tracing.adopt("t-llm", None):
        llm([{"role": "user", "content": "CONFIDENTIAL-PROMPT-TEXT"}])
        llm([{"role": "user", "content": "CONFIDENTIAL-PROMPT-TEXT"}])  # served from cache
    live, cached = spans_of(traces, "t-llm")
    assert live["attrs"]["cached"] is False and live["attrs"]["prompt_tokens"] == 11
    assert live["attrs"]["completion_tokens"] == 3 and live["attrs"]["model"] == "openai/fake"
    assert cached["attrs"]["cached"] is True
    assert "CONFIDENTIAL" not in next(traces.glob("traces-*.jsonl")).read_text()


def test_a_failing_llm_call_records_the_error_type_only(traces, monkeypatch):
    _fake_completion(monkeypatch, fail=True)
    s = Settings(_env_file=None, sec_user_agent="t t@example.com", llm_model="openai/fake")
    with tracing.adopt("t-llm-err", None), pytest.raises(RuntimeError):
        make_llm(s)([{"role": "user", "content": "hi"}])
    (span,) = spans_of(traces, "t-llm-err")
    assert span["ok"] is False and span["error"] == "RuntimeError"
    assert "FAKE-PROVIDER-SECRET" not in next(traces.glob("traces-*.jsonl")).read_text()


# ---------- across agents ----------


def test_one_question_is_one_tree_across_the_analyst_and_its_workers(traces):
    worker_mesh, _ = workers()
    llm = FakeLLM(answer="AAPL flagged that tariffs may affect its supply chain. [C1]")
    local = _deps(llm=llm, a2a_retrieval_url=HOSTS["retrieval"], a2a_verifier_url=HOSTS["verifier"])
    make = lambda: with_remote_agents(  # noqa: E731
        local, {"retrieval": worker_mesh, "verifier": worker_mesh}
    )
    analyst = Mesh({"analyst": analyst_app(_settings(), make, HOSTS["analyst"])})
    agent("analyst", analyst).call({"question": "Does Apple mention tariffs?"}, "trace-mesh")

    spans = spans_of(traces, "trace-mesh")
    by_id = {s["span"]: s for s in spans}
    handles = [s for s in spans if s["name"] == "a2a.handle"]
    assert sorted(s["attrs"]["skill"] for s in handles) == [
        "ask", "search_filings", "verify_answer",
    ]  # fmt: skip
    for h in handles:
        if h["attrs"]["skill"] != "ask":  # worker spans hang under the analyst's a2a.call span
            assert by_id[h["parent"]]["name"] == "a2a.call"
    roots = [s for s in spans if s["parent"] not in by_id]
    assert [r["name"] for r in roots] == ["a2a.call"]  # exactly one tree


# ---------- API + CLI ----------


@pytest.fixture
def client(tmp_path, monkeypatch):
    real = Settings(_env_file=None, sec_user_agent="t t@example.com", data_dir=tmp_path)
    monkeypatch.setattr(api, "get_settings", lambda: real)
    monkeypatch.setattr(api, "_build_deps", lambda: _deps(FakeLLM(answer="Apple cites it [C1].")))
    with TestClient(api.app) as c:
        yield c
    tracing.tracer.disable()


def test_the_api_serves_the_spans_of_a_request_by_trace_id(client):
    r = client.post("/ask", json={"question": "Does Apple mention tariffs in fiscal 2025?"})
    trace_id = r.headers["X-Trace-Id"]
    body = client.get(f"/traces/{trace_id}").json()
    assert body["trace_id"] == trace_id
    assert {"ask", "node.plan", "node.compose"} <= {s["name"] for s in body["spans"]}


def test_trace_endpoint_rejects_bad_and_unknown_ids(client):
    assert client.get("/traces/" + "x" * 65).status_code == 422
    assert client.get("/traces/unknown-id").status_code == 404


def test_cli_renders_a_timeline_and_totals(traces, capsys):
    graph, _ = _graph(FakeLLM(answer="AAPL flagged that tariffs may affect its supply chain. [C1]"))
    ask(graph, "Does Apple mention tariffs in fiscal 2025?", trace_id="trace-cli")
    text = render(spans_of(traces, "trace-cli"))
    assert text.splitlines()[0].startswith("trace trace-cli")
    assert "node.plan" in text and "  node.verify" in text  # children are indented
    assert "llm calls" in text and "errors 0" in text
    assert cli_main(["--last", "--dir", str(traces)]) == 0
    assert "trace trace-cli" in capsys.readouterr().out


def test_cli_handles_a_missing_folder_and_missing_trace(tmp_path, capsys):
    assert cli_main(["--last", "--dir", str(tmp_path / "nope")]) == 1
    (tmp_path / "t").mkdir()
    assert cli_main(["abc", "--dir", str(tmp_path / "t")]) == 1
    assert "no spans found" in capsys.readouterr().out
