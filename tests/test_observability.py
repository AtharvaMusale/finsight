"""LangSmith export: off by default, never leaks the key, and nests what it should.

Every test runs with the network blocked and a mock LangSmith client, so nothing is sent.
"""

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests.adapters
from langsmith import Client, tracing_context

from finsight import observability as obs
from finsight.config import Settings
from finsight.orchestrator.graph import ask
from finsight.retrieval.answer import make_llm
from finsight.retrieval.cache import JsonCache
from finsight.retrieval.search import _as_documents
from finsight.schemas import Hit
from finsight.tracing import adopt
from test_graph import FakeLLM, _graph, facts_db  # noqa: F401  (fixture reuse)

ENV_VARS = (
    "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT",
    "LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS",
)  # fmt: skip
FAKE_KEY = "ls-FAKE-KEY-for-tests-only"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)


@pytest.fixture
def clean_env(monkeypatch):
    # setenv-then-delenv registers the variables so configure_langsmith's writes are undone
    for v in ENV_VARS:
        monkeypatch.setenv(v, "x")
        monkeypatch.delenv(v)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, sec_user_agent="t t@example.com", **kw)


@pytest.fixture
def client():
    c = MagicMock(spec=Client)
    c.tracing_queue = None
    c.info = MagicMock()
    return c


def runs(client, method):
    return [c.kwargs for c in client.method_calls if c[0] == method]


# ---------- configuration ----------


def test_off_by_default_and_changes_nothing(clean_env):
    assert obs.configure_langsmith(settings()) is False
    assert not any(v in os.environ for v in ENV_VARS)


def test_turned_on_without_a_key_stays_off_and_says_so(clean_env, caplog):
    with caplog.at_level(logging.WARNING, logger="finsight.observability"):
        assert obs.configure_langsmith(settings(LANGSMITH_TRACING=True)) is False
    assert "LANGSMITH_API_KEY is not set" in caplog.text
    assert "LANGSMITH_TRACING" not in os.environ


def test_turned_on_exports_the_standard_variables_and_never_logs_the_key(clean_env, caplog):
    s = settings(
        LANGSMITH_TRACING=True, LANGSMITH_API_KEY=FAKE_KEY, LANGSMITH_PROJECT="demo",
        LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com", LANGSMITH_HIDE_INPUTS=True,
    )  # fmt: skip
    with caplog.at_level(logging.DEBUG, logger="finsight.observability"):
        assert obs.configure_langsmith(s) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == FAKE_KEY
    assert os.environ["LANGSMITH_PROJECT"] == "demo"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert "LANGSMITH_HIDE_OUTPUTS" not in os.environ
    assert FAKE_KEY not in caplog.text and "inputs hidden" in caplog.text
    assert FAKE_KEY not in repr(s)  # SecretStr keeps it out of reprs too


def test_settings_read_the_standard_names_from_an_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=" + FAKE_KEY + "\nLANGSMITH_PROJECT=from-file\n"
    )
    s = Settings(_env_file=env, sec_user_agent="t t@example.com")
    assert s.langsmith_tracing is True and s.langsmith_project == "from-file"
    assert s.langsmith_api_key.get_secret_value() == FAKE_KEY
    assert Settings(_env_file=None, sec_user_agent="t t@example.com").langsmith_tracing is False


# ---------- what reaches LangSmith ----------


def test_a_question_is_one_root_run_with_node_children_tied_to_our_trace_id(client):
    graph, _ = _graph(FakeLLM(answer="AAPL flagged that tariffs may affect its supply chain. [C1]"))
    with tracing_context(enabled=True, client=client, project_name="test"):
        out = ask(graph, "Does Apple mention tariffs in fiscal 2025?", trace_id="trace-ls-1")
    assert out["answer"].text  # tracing must not change the answer
    created = runs(client, "create_run")
    root = [r for r in created if r["name"] == "finsight.ask"]
    assert len(root) == 1 and not root[0].get("parent_run_id")
    nodes = {r["name"] for r in created if r.get("parent_run_id")}
    assert {"plan", "retrieve", "compose", "verify", "finalize"} <= nodes
    for r in created:
        assert r["extra"]["metadata"]["finsight_trace_id"] == "trace-ls-1"


def test_nothing_is_sent_when_langsmith_is_off(client):
    graph, _ = _graph(FakeLLM(answer="AAPL flagged that tariffs may affect its supply chain. [C1]"))
    with tracing_context(enabled=False, client=client):
        out = ask(graph, "Does Apple mention tariffs in fiscal 2025?", trace_id="trace-ls-off")
    assert out["answer"].text
    assert client.method_calls == []


def _fake_completion(monkeypatch):
    import litellm

    def completion(**kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
        )

    monkeypatch.setattr(litellm, "completion", completion)


def test_llm_runs_report_model_usage_cache_flag_and_our_trace_id(client, tmp_path, monkeypatch):
    _fake_completion(monkeypatch)
    s = settings(llm_model="anthropic/claude-haiku-4-5", ANTHROPIC_API_KEY="fake")
    llm = make_llm(s, JsonCache(tmp_path / "cache"))
    with tracing_context(enabled=True, client=client), adopt("trace-ls-2", None):
        assert llm([{"role": "user", "content": "hi"}]) == "hello"
        assert llm([{"role": "user", "content": "hi"}]) == "hello"  # second call: cache hit
    created = [r for r in runs(client, "create_run") if r["name"] == "llm"]
    assert len(created) == 2 and all(r["run_type"] == "llm" for r in created)
    meta = created[0]["extra"]["metadata"]
    assert meta["ls_provider"] == "anthropic" and meta["ls_model_name"] == "claude-haiku-4-5"
    ended = [r["extra"]["metadata"] for r in runs(client, "update_run") if r.get("extra")]
    live = [m for m in ended if m.get("cached") is False]
    cached = [m for m in ended if m.get("cached") is True]
    assert live and live[0]["usage_metadata"] == {
        "input_tokens": 11, "output_tokens": 3, "total_tokens": 14,
    }  # fmt: skip
    assert cached and "usage_metadata" not in cached[0]
    assert all(m["finsight_trace_id"] == "trace-ls-2" for m in live + cached)


def test_retriever_passages_are_shaped_as_documents():
    hit = Hit("c1", "Tariffs may raise costs.", "AAPL", "10-K", 2025, "Item 1A", "a1", 0.9)
    expected = {"documents": [{
        "page_content": "Tariffs may raise costs.",
        "metadata": {"source": "AAPL 10-K FY2025, Item 1A", "score": 0.9, "chunk_id": "c1"},
    }]}  # fmt: skip
    assert _as_documents([hit]) == expected
    assert _as_documents({"output": [hit]}) == expected  # either shape the SDK may pass
    assert _as_documents(None) == {"documents": []}


def test_sql_tool_runs_send_the_query_but_never_the_database_handle(client, facts_db):  # noqa: F811
    sql = "SELECT ticker, value FROM v_facts WHERE ticker='NVDA'"
    with tracing_context(enabled=True, client=client), adopt("trace-ls-3", None):
        res = facts_db.run(sql)
    assert res.rows
    (run,) = [r for r in runs(client, "create_run") if r["name"] == "sql"]
    assert run["run_type"] == "tool"
    assert set(run["inputs"]) == {"sql"} and run["inputs"]["sql"] == sql
    # annotate() runs inside the function, so the trace ID arrives with the end-of-run update
    ended = [r["extra"]["metadata"] for r in runs(client, "update_run") if r.get("extra")]
    assert any(m.get("finsight_trace_id") == "trace-ls-3" for m in ended)


def test_helpers_are_harmless_outside_any_run():
    obs.annotate(cached=True)  # no open run: must not raise
    obs.set_llm_usage(1, 2)
    obs.set_llm_usage(None, None)

    @obs.traceable(run_type="chain", name="x")
    def add(a, b):
        return a + b

    assert add(1, 2) == 3  # tracing is off by default: the function just runs
