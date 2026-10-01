"""A2A agents over in-process ASGI: no ports, no network, fakes behind them."""

import logging
import time

import anyio
import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from finsight.agents.common import (
    AgentError,
    RemoteAgent,
    RemoteAgentError,
    SkillExecutor,
    build_agent_app,
    make_card,
)
from finsight.agents.remote import (
    RemoteSearcher,
    make_facts_agent,
    make_verifier,
    with_remote_agents,
)
from finsight.agents.services import analyst_app, facts_app, retrieval_app, verifier_app
from finsight.config import Settings
from finsight.orchestrator.graph import Deps, ask, build_graph
from finsight.retrieval.rerank import NoRerank
from finsight.schemas import SearchFilter
from finsight.tools.sql_tool import SqlToolError
from test_graph import FakeLLM, FakeSearcher, facts_db  # noqa: F401  (fixture reuse)
from test_mcp_server import REVENUE_SQL

HOSTS = {
    "retrieval": "http://retrieval", "facts": "http://facts", "verifier": "http://verifier",
    "analyst": "http://analyst", "rogue": "http://rogue",
}  # fmt: skip


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, sec_user_agent="t t@example.com", a2a_timeout_s=10, **kw)


def _deps(llm=None, facts=None, searcher=None, **kw) -> Deps:
    return Deps(
        settings=_settings(**kw),
        searcher=searcher or FakeSearcher(),
        reranker=NoRerank(),
        llm=llm or FakeLLM(),
        facts_db=facts,
    )


class Mesh(httpx.AsyncBaseTransport):
    """Routes each request to the in-process ASGI app registered for its host name."""

    def __init__(self, apps: dict):
        self._t = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._t[request.url.host].handle_async_request(request)


def agent(name: str, mesh: Mesh, **kw) -> RemoteAgent:
    return RemoteAgent(
        HOSTS[name], timeout_s=kw.pop("timeout_s", 10), max_bytes=kw.pop("max_bytes", 512_000),
        transport=mesh, **kw,
    )  # fmt: skip


def workers(llm=None, facts=None, searcher=None, **kw):
    """(mesh, searcher) with retrieval, facts and verifier agents registered."""
    s = _settings(**kw)
    searcher = searcher or FakeSearcher()
    make = lambda: _deps(llm, facts, searcher, **kw)  # noqa: E731
    return Mesh({
        "retrieval": retrieval_app(s, make, HOSTS["retrieval"]),
        "facts": facts_app(s, make, HOSTS["facts"]),
        "verifier": verifier_app(s, make, HOSTS["verifier"]),
    }), searcher  # fmt: skip


def get(mesh: Mesh, host: str, path: str) -> httpx.Response:
    async def go():
        async with httpx.AsyncClient(transport=mesh, base_url=HOSTS[host]) as c:
            return await c.get(path)

    return anyio.run(go)


# ---------- discovery ----------


@pytest.mark.parametrize(
    "name, skill", [("retrieval", "search_filings"), ("facts", "query_financials"),
                    ("verifier", "verify_answer")],
)  # fmt: skip
def test_each_agent_publishes_a_card_with_its_skill_and_a_health_route(name, skill):
    mesh, _ = workers()
    card = get(mesh, name, "/.well-known/agent-card.json").json()
    assert card["name"] == f"finsight-{name}"
    assert [s["id"] for s in card["skills"]] == [skill]
    assert card["supportedInterfaces"][0]["url"] == f"{HOSTS[name]}/rpc"
    assert get(mesh, name, "/health").json()["agent"] == f"finsight-{name}"


# ---------- retrieval ----------


def test_remote_search_returns_typed_hits_and_applies_the_filter():
    mesh, searcher = workers()
    remote = RemoteSearcher(agent("retrieval", mesh))
    hits = remote.search("tariffs", SearchFilter(("aapl",), 2025, "10-K", "Item 1A"), limit=2)
    assert searcher.filters == [(("AAPL",), 2025)]  # tickers normalised on the agent
    assert hits and isinstance(hits[0].fiscal_year, int)  # ints survive the wire (not 2025.0)
    assert hits[0].accession_no == "acc-AAPL"


@pytest.mark.parametrize(
    "args, expect",
    [
        ({"query": "hi"}, "invalid arguments"),
        ({"query": "tariffs", "tickers": ["TSLA"]}, "unknown ticker"),
        ({"query": "tariffs", "section": "Item 1A; DROP TABLE"}, "section"),
        ({"query": "tariffs", "limit": 999}, "limit"),
        ({"query": "tariffs", "form_type": "8-K"}, "form_type"),
    ],
)
def test_agent_rejects_bad_arguments_with_a_safe_message(args, expect):
    mesh, _ = workers()
    with pytest.raises(RemoteAgentError, match=expect):
        agent("retrieval", mesh).call(args, "t1")


def test_internal_errors_carry_a_trace_id_but_no_details():
    class Boom(FakeSearcher):
        def search(self, *a, **k):
            raise RuntimeError("secret internal detail /Users/me/.env")

    mesh, _ = workers(searcher=Boom())
    with pytest.raises(RemoteAgentError) as e:
        agent("retrieval", mesh).call({"query": "tariffs please"}, "trace-abc")
    assert "secret" not in str(e.value) and ".env" not in str(e.value)
    assert "trace-abc" in str(e.value)


def test_slow_agent_times_out_on_the_client():
    class Slow(FakeSearcher):
        def search(self, *a, **k):
            time.sleep(1.0)
            return super().search(*a, **k)

    mesh, _ = workers(searcher=Slow())
    with pytest.raises(RemoteAgentError, match="timed out"):
        agent("retrieval", mesh, timeout_s=0.1).call({"query": "tariffs please"}, "t1")


def test_oversized_requests_are_refused_before_sending():
    mesh, _ = workers()
    with pytest.raises(RemoteAgentError, match="too large"):
        agent("retrieval", mesh, max_bytes=100).call({"query": "x" * 300}, "t1")


def test_unreachable_agent_is_reported_not_raised_raw():
    class Down(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("refused")

    a = RemoteAgent("http://nowhere", timeout_s=2, max_bytes=10_000, transport=Down())
    with pytest.raises(RemoteAgentError, match="unavailable"):
        a.call({"query": "tariffs"}, "t1")


# ---------- untrusted responses ----------


class _AnyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


def _rogue(handler):
    """An agent whose handler we control, to test how the CALLER treats what comes back."""
    card = make_card(
        name="rogue", description="x", skill_id="x", skill_name="x", skill_description="x",
        examples=[], base_url=HOSTS["rogue"],
    )  # fmt: skip
    ex = SkillExecutor(handler, _AnyArgs, skill="x", timeout_s=5, max_bytes=50_000)
    mesh = Mesh({"rogue": build_agent_app(card, ex, 50_000)})
    return RemoteAgent(HOSTS["rogue"], timeout_s=5, max_bytes=50_000, transport=mesh)


def test_malformed_hits_from_a_remote_agent_are_rejected():
    rogue = _rogue(lambda a, t: {"hits": [{"chunk_id": 123, "text": "x"}]})
    with pytest.raises(RemoteAgentError, match="malformed hits"):
        RemoteSearcher(rogue).search("tariffs please")


def test_remote_error_text_is_truncated_and_single_line():
    def fail(args, trace):
        raise AgentError("line one\n" + "A" * 1000)

    with pytest.raises(RemoteAgentError) as e:
        _rogue(fail).call({"question": "anything at all"}, "t1")
    assert "\n" not in str(e.value) and len(str(e.value)) < 350


# ---------- facts ----------


def test_remote_facts_returns_rows_and_the_orchestrator_derives_growth_locally(facts_db):  # noqa: F811
    llm = FakeLLM(route="facts", sql=REVENUE_SQL, answer="Revenue rose [F1][F2][K1].")
    mesh, _ = workers(llm=llm, facts=facts_db)
    deps = _deps(llm=FakeLLM(route="facts", answer="Revenue rose [F1][F2][K1]."))
    deps.facts_agent = make_facts_agent(agent("facts", mesh))
    out = ask(build_graph(deps), "How did NVIDIA's revenue change in fiscal 2025?")
    assert out["sql"].rows[1][4] == 130.0
    assert out["derived"] and out["derived"][0].result.value == pytest.approx(116.6667)
    assert out["answer"].calc_sources  # K1 exists because the orchestrator computed it


def test_facts_agent_blocks_writes_and_the_graph_degrades_gracefully(facts_db):  # noqa: F811
    mesh, _ = workers(llm=FakeLLM(sql="DROP TABLE facts"), facts=facts_db)
    with pytest.raises(SqlToolError, match="could not build a valid query"):
        make_facts_agent(agent("facts", mesh))("drop it all", "t1")
    deps = _deps(llm=FakeLLM(route="facts", answer="Tariffs matter [C1]."))
    deps.facts_agent = make_facts_agent(agent("facts", mesh))
    out = ask(build_graph(deps), "What was NVIDIA revenue in fiscal 2025?")
    assert out["errors"] and "valid query" in out["errors"][0]  # recorded, not crashed
    assert facts_db.run("SELECT count(*) FROM v_facts").rows[0][0] > 0  # data untouched


# ---------- verifier ----------


def test_remote_verifier_catches_a_hallucinated_number():
    mesh, _ = workers()
    verify = make_verifier(agent("verifier", mesh))
    hits = FakeSearcher().search("q", SearchFilter(("AAPL",), 2025))
    v = verify("Apple tariff costs were $999 billion [C1].", hits, None, [], "t1")
    assert [i.kind for i in v.issues] == ["unsupported_number"]


def test_remote_verifier_passes_a_faithful_answer():
    mesh, _ = workers()
    hits = FakeSearcher().search("q", SearchFilter(("AAPL",), 2025))
    v = make_verifier(agent("verifier", mesh))(
        "AAPL flagged that tariffs and trade policy may affect its supply chain. [C1]",
        hits, None, [], "t1",
    )  # fmt: skip
    assert v.ok and v.critic_ran and v.claims_checked == 1


def test_verifier_outage_fails_closed_with_an_honest_caveat():
    class Down(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("refused")

    llm = FakeLLM(answer="Apple cites it [C1].")
    deps = _deps(llm=llm)
    deps.verifier = make_verifier(
        RemoteAgent("http://verifier", timeout_s=2, max_bytes=50_000, transport=Down())
    )
    out = ask(build_graph(deps), "Does Apple mention tariffs in fiscal 2025?")
    ans = out["answer"]
    assert "could not be independently verified" in ans.text  # never passed off as verified
    assert ans.verification.issues[0].kind == "verifier_unavailable"
    assert ans.revisions == 0 and llm.calls.count("compose") == 1  # no pointless rewrite


# ---------- wiring ----------


def test_only_configured_capabilities_are_delegated():
    deps = with_remote_agents(_deps(a2a_verifier_url="http://verifier"))
    assert deps.verifier is not None
    assert deps.facts_agent is None and isinstance(deps.searcher, FakeSearcher)
    untouched = _deps()
    assert with_remote_agents(untouched) is untouched  # nothing configured: same object


def test_analyst_delegates_retrieval_and_verification_and_one_trace_id_follows(caplog):
    caplog.set_level(logging.INFO, logger="finsight.a2a")
    worker_mesh, _ = workers()
    llm = FakeLLM(answer="AAPL flagged that tariffs may affect its supply chain. [C1]")
    local = _deps(llm=llm, a2a_retrieval_url=HOSTS["retrieval"], a2a_verifier_url=HOSTS["verifier"])
    make = lambda: with_remote_agents(  # noqa: E731
        local, {"retrieval": worker_mesh, "verifier": worker_mesh}
    )
    # the workers are registered in one mesh, routed by host
    analyst = Mesh({"analyst": analyst_app(_settings(), make, "http://analyst")})
    trace = "trace-e2e-0001"
    result = agent("analyst", analyst).call({"question": "Does Apple mention tariffs?"}, trace)

    assert result["trace_id"] == trace
    assert result["sources"]["C1"]["accession_no"] == "acc-AAPL"
    assert result["verification"]["ok"] is True and result["verification"]["critic_ran"] is True
    handled = [r.getMessage() for r in caplog.records if "handled in" in r.getMessage()]
    assert len(handled) == 3  # analyst + retrieval + verifier
    assert all(f"trace={trace}" in m for m in handled)  # the SAME id crossed every hop
