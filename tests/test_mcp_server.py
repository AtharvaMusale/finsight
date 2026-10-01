"""The MCP server, exercised through the real protocol (in-memory client), with fakes behind it."""

import json
import time

import anyio
import pytest
from mcp import Client

from finsight.config import Settings
from finsight.mcp_server.server import MAX_RESULTS, build_server
from finsight.orchestrator.graph import Deps
from finsight.retrieval.rerank import NoRerank
from finsight.tools.calculator import OPERATIONS
from test_graph import FakeLLM, FakeSearcher, facts_db  # noqa: F401  (fixture reuse)

REVENUE_SQL = (
    "SELECT ticker, metric, period_type, period_end, value, accession_no FROM v_facts "
    "WHERE ticker='NVDA' AND metric='Revenue' AND period_type='annual' ORDER BY period_end"
)


def _deps(llm=None, facts=None, timeout=30.0) -> Deps:
    return Deps(
        settings=Settings(
            _env_file=None, sec_user_agent="t t@example.com", request_timeout_s=timeout
        ),
        searcher=FakeSearcher(),
        reranker=NoRerank(),
        llm=llm or FakeLLM(),
        facts_db=facts,
    )


def call(server, tool: str, args: dict):
    """(is_error, payload): payload is parsed JSON on success, the error text on failure."""

    async def go():
        async with Client(server) as c:
            r = await c.call_tool(tool, args)
            text = "".join(getattr(b, "text", "") for b in r.content)
            return r.is_error, (text if r.is_error else json.loads(text))

    return anyio.run(go)


def test_nothing_is_built_until_a_tool_is_used():
    built = []
    build_server(lambda: built.append(1) or _deps())
    assert built == []  # startup stays fast and credential-free


def test_exposes_four_read_only_tools():
    async def go():
        async with Client(build_server(_deps)) as c:
            return (await c.list_tools()).tools

    tools = anyio.run(go)
    assert {t.name for t in tools} == {
        "search_filings", "query_financials", "calculate", "ask_finsight",
    }  # fmt: skip
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


def test_calculate_operations_match_the_calculator():
    async def go():
        async with Client(build_server(_deps)) as c:
            return {t.name: t for t in (await c.list_tools()).tools}["calculate"]

    schema = anyio.run(go).input_schema
    assert set(schema["properties"]["op"]["enum"]) == set(OPERATIONS)


# ---------- calculate ----------


def test_calculate_returns_value_and_formula():
    err, out = call(
        build_server(_deps), "calculate", {"op": "pct_change", "inputs": {"old": 60, "new": 130}}
    )
    assert not err and out["value"] == pytest.approx(116.6667) and "|60.0|" in out["formula"]


@pytest.mark.parametrize(
    "args, expect",
    [
        ({"op": "ratio", "inputs": {"numerator": 1, "denominator": 0}}, "zero"),
        ({"op": "ratio", "inputs": {"numerator": 1}}, "needs exactly"),
        ({"op": "__import__('os')", "inputs": {}}, "op"),  # not an allowed operation
    ],
)
def test_calculate_rejects_bad_input_with_a_safe_message(args, expect):
    err, text = call(build_server(_deps), "calculate", args)
    assert err and expect in text


# ---------- search_filings ----------


def test_search_passes_filters_and_labels_results_as_untrusted_data():
    deps = _deps()
    err, out = call(
        build_server(lambda: deps),
        "search_filings",
        {"query": "tariffs and trade policy", "tickers": ["aapl"], "fiscal_year": 2025,
         "form_type": "10-K", "section": "Item 1A", "limit": 2},
    )  # fmt: skip
    assert not err
    assert deps.searcher.filters == [(("AAPL",), 2025)]  # tickers are normalised
    assert [r["label"] for r in out["results"]] == ["C1", "C2"]  # limit applied
    assert out["results"][0]["accession_no"] == "acc-AAPL"
    assert "untrusted" in out["notice"]


@pytest.mark.parametrize(
    "args, expect",
    [
        ({"query": "tariffs", "tickers": ["TSLA"]}, "unknown ticker"),
        ({"query": "tariffs", "section": "Item 1A; DROP TABLE"}, "section must look like"),
        ({"query": "tariffs", "limit": MAX_RESULTS + 1}, "less than or equal"),
        ({"query": "hi"}, "at least 3"),
        ({"query": "x" * 501}, "at most 500"),
        ({"query": "tariffs", "form_type": "8-K"}, "10-K"),
    ],
)
def test_search_validates_inputs(args, expect):
    err, text = call(build_server(_deps), "search_filings", args)
    assert err and expect in text


# ---------- query_financials ----------


def test_query_financials_returns_rows_and_computed_growth(facts_db):  # noqa: F811
    llm = FakeLLM(sql=REVENUE_SQL)
    err, out = call(
        build_server(lambda: _deps(llm, facts_db)),
        "query_financials",
        {"question": "How did NVIDIA's revenue change in fiscal 2025?"},
    )
    assert not err
    assert out["rows"][1][4] == 130.0
    (calc,) = out["calculations"]
    assert calc["unit"] == "percent" and calc["value"] == pytest.approx(116.6667)


def test_query_financials_blocks_writes_and_reports_a_safe_error(facts_db):  # noqa: F811
    llm = FakeLLM(sql="DROP TABLE facts")
    err, text = call(
        build_server(lambda: _deps(llm, facts_db)), "query_financials", {"question": "drop it all"}
    )
    assert err and "could not build a valid query" in text
    assert facts_db.run("SELECT count(*) FROM v_facts").rows[0][0] > 0  # data untouched


def test_query_financials_without_a_facts_database_says_so():
    err, text = call(build_server(_deps), "query_financials", {"question": "NVIDIA revenue?"})
    assert err and "not available" in text


# ---------- ask_finsight ----------


def test_ask_finsight_returns_the_same_payload_as_the_http_api():
    llm = FakeLLM(answer="Apple cites it [C1].")
    err, out = call(
        build_server(lambda: _deps(llm)),
        "ask_finsight",
        {"question": "Does Apple mention tariffs in fiscal 2025?"},
    )
    assert not err
    assert out["sources"]["C1"]["accession_no"] == "acc-AAPL"
    assert "Not financial advice" in out["disclaimer"]
    assert out["verification"]["ok"] is True and out["verification"]["critic_ran"] is True
    assert len(out["trace_id"]) == 32


def test_the_graph_is_built_once_and_reused():
    llm = FakeLLM(answer="Apple cites it [C1].")
    built = []
    server = build_server(lambda: built.append(1) or _deps(llm))

    async def go():
        async with Client(server) as c:
            for _ in range(2):
                await c.call_tool(
                    "ask_finsight", {"question": "Does Apple mention tariffs in 2025?"}
                )

    anyio.run(go)
    assert built == [1]


# ---------- failure handling ----------


def test_unexpected_errors_do_not_leak_details_but_carry_a_trace_id():
    class Boom(FakeLLM):
        def __call__(self, messages):
            raise RuntimeError("secret internal detail /Users/me/.env")

    err, text = call(
        build_server(lambda: _deps(Boom())),
        "ask_finsight",
        {"question": "Does Apple mention tariffs?"},
    )
    assert err and "secret" not in text and ".env" not in text and "trace" in text


def test_slow_calls_time_out():
    class Slow(FakeLLM):
        def __call__(self, messages):
            time.sleep(1.0)
            return super().__call__(messages)

    err, text = call(
        build_server(lambda: _deps(Slow(), timeout=0.05)),
        "ask_finsight",
        {"question": "Does Apple mention tariffs in 2025?"},
    )
    assert err and "exceeded" in text and "trace" in text
