"""The web UI and the read-only endpoints it uses (/status, /traces). Localhost only."""

import json

import pytest
from fastapi.testclient import TestClient

from finsight.api import app as api
from finsight.config import Settings
from test_api import _deps
from test_graph import FakeLLM

SECRET = "sk-ant-THIS-MUST-NEVER-APPEAR-IN-A-RESPONSE"


@pytest.fixture
def client_with(monkeypatch):
    def make(**settings_kw):
        s = Settings(_env_file=None, sec_user_agent="t t@example.com", **settings_kw)
        monkeypatch.setattr(api, "_build_deps", lambda: _deps(FakeLLM()))
        monkeypatch.setattr(api, "get_settings", lambda: s)
        return TestClient(api.app)

    return make


def _span(trace, span, parent, name, ts, **attrs):
    return {
        "trace": trace, "span": span, "parent": parent, "name": name, "service": "api",
        "ts": ts, "ms": 5.0, "ok": True, "error": None, "attrs": attrs,
    }  # fmt: skip


# ---------- serving the page ----------


def test_root_redirects_to_the_ui_and_the_page_is_served(client_with):
    with client_with() as c:
        r = c.get("/", follow_redirects=False)
        assert r.status_code == 307 and r.headers["location"] == "/ui/"
        page = c.get("/ui/")
        assert page.status_code == 200 and "FinSight" in page.text
        for asset in ("/ui/ui.js", "/ui/ui.css"):
            assert c.get(asset).status_code == 200
        assert c.get("/health").json() == {"status": "ok"}  # the mount never shadows the API


def test_sample_data_is_a_complete_trace_with_a_cited_answer(client_with):
    with client_with() as c:
        sample = c.get("/ui/sample.json").json()
    resp, spans = sample["response"], sample["trace"]["spans"]
    assert resp["trace_id"] == sample["trace"]["trace_id"]
    assert "[C4]" in resp["answer"] and "C4" in resp["sources"]
    roots = [s for s in spans if s["parent"] is None]
    assert len(roots) == 1 and roots[0]["name"] == "ask"
    ids = {s["span"] for s in spans}
    assert all(s["parent"] in ids for s in spans if s["parent"])


def test_the_page_never_writes_server_text_as_html():
    # The answer comes from a model reading untrusted filings. It must only ever reach the page
    # as text, so the script must not use any HTML-injecting API.
    js = (api.STATIC_DIR / "ui.js").read_text()
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert banned not in js


# ---------- /status ----------


def test_status_reports_presence_but_never_a_key_value(client_with):
    with client_with(ANTHROPIC_API_KEY=SECRET, PINECONE_API_KEY=SECRET + "-2") as c:
        r = c.get("/status")
    body = r.json()
    assert body["keys"] == {"anthropic": True, "pinecone": True}
    assert body["mode"] == "in-process" and all(not a["configured"] for a in body["agents"])
    assert SECRET not in r.text


def test_status_with_no_keys_and_a_missing_database(client_with, tmp_path):
    with client_with(data_dir=tmp_path) as c:
        body = c.get("/status").json()
    assert body["keys"] == {"anthropic": False, "pinecone": False} and body["facts_db"] is False


def test_status_marks_a_configured_but_unreachable_agent_as_down(client_with):
    # port 9 (discard) refuses connections on localhost, so the probe fails fast
    with client_with(a2a_facts_url="http://127.0.0.1:9") as c:
        body = c.get("/status").json()
    by = {a["name"]: a for a in body["agents"]}
    assert body["mode"] == "a2a"
    assert by["facts"]["configured"] is True and by["facts"]["up"] is False
    assert by["retrieval"]["configured"] is False and by["retrieval"]["up"] is None


# ---------- /traces ----------


def test_trace_list_is_newest_first_and_only_has_summary_fields(client_with, tmp_path):
    d = tmp_path / "traces"
    d.mkdir()
    rows = [
        _span("old", "a1", None, "ask", 100.0, route="text", revisions=0, verified=True),
        _span("new", "b1", None, "ask", 200.0, route="both", revisions=1, verified=False),
        _span("new", "b2", "b1", "node.plan", 200.1, route="both"),
    ]  # fmt: skip
    lines = [json.dumps(r) for r in rows] + ["not json"]  # a bad line must be skipped
    (d / "traces-2026-01-01.jsonl").write_text("\n".join(lines) + "\n")
    with client_with(data_dir=tmp_path) as c:
        body = c.get("/traces").json()
    assert body["tracing"] is True
    assert [t["trace_id"] for t in body["traces"]] == ["new", "old"]  # children are not listed
    top = body["traces"][0]
    assert top["route"] == "both" and top["verified"] is False and top["revisions"] == 1
    expected = {"trace_id", "ts", "ms", "ok", "route", "revisions", "verified", "question_chars"}
    assert set(top) == expected


def test_trace_list_when_tracing_is_off_or_empty(client_with, tmp_path):
    with client_with(data_dir=tmp_path, tracing=False) as c:
        assert c.get("/traces").json() == {"tracing": False, "traces": []}
    with client_with(data_dir=tmp_path) as c:
        assert c.get("/traces").json() == {"tracing": True, "traces": []}
