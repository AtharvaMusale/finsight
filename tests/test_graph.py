import json
import re

import duckdb
import pytest

from finsight.config import Settings
from finsight.ingestion.stores import _SCHEMA
from finsight.orchestrator.compose import NO_EVIDENCE, build_messages, parse_citations
from finsight.orchestrator.derive import derive_calcs
from finsight.orchestrator.graph import Deps, _per_query_k, ask, build_graph
from finsight.retrieval.rerank import NoRerank
from finsight.schemas import Hit
from finsight.tools.sql_tool import FactsDB, SqlResult

COLS = ["ticker", "metric", "period_type", "period_end", "value", "accession_no"]


def _hit(ticker: str, n: int = 1, fy: int = 2025) -> Hit:
    text = f"{ticker} excerpt {n}: tariffs and trade policy may affect our supply chain."
    return Hit(
        chunk_id=f"{ticker}{n}", text=text, ticker=ticker, form_type="10-K",
        fiscal_year=fy, section="Item 1A", accession_no=f"acc-{ticker}",
    )  # fmt: skip


class FakeSearcher:
    """Returns chunks for whichever tickers the metadata filter asks for."""

    def __init__(self):
        self.filters: list[tuple] = []

    def search(self, question, flt, limit=None):
        tickers, year = (list(flt.tickers), flt.fiscal_year) if flt else ([], None)
        self.filters.append((tuple(tickers), year))
        return [_hit(t, n, year or 2025) for t in (tickers or ["AAPL"]) for n in (1, 2, 3)]


class FakeLLM:
    """Stands in for every model call, dispatching on the system prompt."""

    def __init__(self, route="text", sql="SELECT 1", answer="Answer [C1].", critic="auto"):
        self.route, self.sql, self.critic = route, sql, critic
        self.answers = [answer] if isinstance(answer, str) else list(answer)
        self.calls: list[str] = []
        self.compose_messages: list[list[dict]] = []

    def _critic_reply(self, prompt: str) -> str:
        if self.critic != "auto":
            return self.critic
        # A faithful critic: quote the start of the first evidence item of every claim.
        results = []
        for m in re.finditer(
            r'<claim i="(\d+)">.*?<evidence id="\w+">\n(.*?)\n</evidence>', prompt, re.S
        ):
            results.append({"i": int(m.group(1)), "verdict": "supported", "quote": m.group(2)[:30]})
        return json.dumps({"results": results})

    def __call__(self, messages):
        system = messages[0]["content"]
        if system.startswith("Classify"):
            self.calls.append("route")
            return self.route
        if system.startswith("You write DuckDB SQL"):
            self.calls.append("sql")
            return f"```sql\n{self.sql}\n```"
        if system.startswith("You check whether claims"):
            self.calls.append("critic")
            return self._critic_reply(messages[1]["content"])
        self.calls.append("compose")
        self.last_prompt = messages[1]["content"]
        self.compose_messages.append(messages)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


@pytest.fixture
def facts_db(tmp_path):
    path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA)
    con.execute(
        "INSERT INTO filings VALUES ('a1','NVDA',1,'10-K',2025,'2025-02-26','2025-01-26','u')"
    )
    con.executemany(
        "INSERT INTO facts VALUES (1,'NVDA','us-gaap','Revenues','USD',?,?,?,2025,'FY','10-K','a1',"
        "'2025-02-26')",
        [("2023-01-30", "2024-01-28", 60.0), ("2024-01-29", "2025-01-26", 130.0)],
    )
    con.close()
    return FactsDB(path)


def _graph(llm, facts_db=None, **settings):
    searcher = FakeSearcher()
    deps = Deps(
        settings=Settings(_env_file=None, sec_user_agent="t t@example.com", **settings),
        searcher=searcher,
        reranker=NoRerank(),
        llm=llm,
        facts_db=facts_db,
    )
    return build_graph(deps), searcher


def test_cross_company_question_retrieves_per_company():
    llm = FakeLLM(answer="Both mention it [C1][C4].")
    graph, searcher = _graph(llm)
    out = ask(graph, "Which of Apple and Microsoft mention tariffs in fiscal 2025?")
    assert searcher.filters == [(("AAPL",), 2025), (("MSFT",), 2025)]
    assert {h.ticker for h in out["hits"]} == {"AAPL", "MSFT"}  # both sides got slots
    # Budget allows 4 per company (6 + 2 over 2 sub-queries); the fake index has only 3 each.
    assert sorted(h.ticker for h in out["hits"]) == ["AAPL"] * 3 + ["MSFT"] * 3
    tickers = {out["answer"].sources[k].ticker for k in ("C1", "C4")}
    assert tickers == {"AAPL", "MSFT"}
    assert "route" not in llm.calls  # no metric word, so no routing model call


def test_facts_route_uses_sql_and_derives_growth(facts_db):
    sql = (
        "SELECT ticker, metric, period_type, period_end, value, accession_no FROM v_facts "
        "WHERE ticker='NVDA' AND metric='Revenue' AND period_type='annual' ORDER BY period_end"
    )
    llm = FakeLLM(route="facts", sql=sql, answer="Revenue rose [F1][F2][K1].")
    graph, searcher = _graph(llm, facts_db)
    out = ask(graph, "How did NVIDIA's revenue change in fiscal 2025?")
    assert searcher.filters == []  # pure facts route never touched the filings index
    assert out["sql"].rows[1][4] == 130.0
    assert "value=130" in llm.last_prompt and '<calc id="K1"' in llm.last_prompt
    ans = out["answer"]
    assert ans.fact_sources["F2"]["value"] == 130.0
    assert ans.calc_sources["K1"].result.value == pytest.approx(116.6667)  # (130-60)/60*100
    assert not ans.invalid_citations


def test_facts_route_falls_back_to_filing_text_when_sql_finds_nothing(facts_db):
    empty_sql = "SELECT ticker, value FROM v_facts WHERE ticker='ZZZ'"
    graph, searcher = _graph(FakeLLM(route="facts", sql=empty_sql), facts_db)
    out = ask(graph, "What was Nvidia revenue?")
    assert out["sql"].rows == []
    assert searcher.filters  # fell back to retrieval
    assert out["answer"].sources


def test_both_route_with_no_evidence_terminates():
    """Regression guard for the retrieve <-> facts loop: must end with the abstention text."""
    llm = FakeLLM(route="both", sql="SELECT ticker FROM v_facts WHERE ticker='ZZZ'")
    graph, searcher = _graph(llm, facts_db=None)
    searcher.search = lambda q, f, limit=None: []
    out = ask(graph, "What was the revenue and why?")
    assert out["answer"].text == NO_EVIDENCE
    assert "compose" not in llm.calls  # no evidence, so no synthesis call either


def test_sql_rejection_is_recorded_and_does_not_crash(facts_db):
    graph, _ = _graph(FakeLLM(route="facts", sql="DROP VIEW v_facts"), facts_db)
    out = ask(graph, "What was Nvidia revenue?")
    assert any("gave up" in e for e in out["errors"])
    assert out["answer"].text  # fell back to text and still produced an answer


def test_trace_id_is_propagated():
    graph, _ = _graph(FakeLLM())
    assert ask(graph, "q", trace_id="abc123")["trace_id"] == "abc123"


@pytest.mark.parametrize("n, expected", [(1, 6), (2, 4), (3, 2), (9, 1)])
def test_chunk_budget_split(n, expected):
    assert _per_query_k(6, n) == expected


# ---------- derive + compose ----------


def _rows():
    return [
        ["AAPL", "Revenue", "annual", "2024-09-28", 100.0, "a"],
        ["AAPL", "Revenue", "annual", "2025-09-27", 110.0, "b"],
        ["AAPL", "GrossProfit", "annual", "2025-09-27", 44.0, "b"],
    ]


def test_derive_growth_and_margin_reference_their_source_rows():
    d = {x.label: x for x in derive_calcs(COLS, _rows())}
    growth = d["AAPL Revenue % change, 2024-09-28 to 2025-09-27"]
    assert growth.result.value == 10.0 and growth.from_rows == (1, 2)
    margin = d["AAPL GrossProfit margin (% of Revenue), period ending 2025-09-27"]
    assert margin.result.value == 40.0 and margin.from_rows == (3, 2)


def test_derive_needs_the_right_columns_and_skips_zero_base():
    assert derive_calcs(["ticker", "value"], [["AAPL", 1.0]]) == []
    rows = [["A", "Revenue", "annual", "2024-01-01", 0.0, "x"],
            ["A", "Revenue", "annual", "2025-01-01", 5.0, "y"]]  # fmt: skip
    assert derive_calcs(COLS, rows) == []  # undefined growth is skipped, not raised


def test_parse_citations_maps_all_three_kinds_and_flags_invented():
    sql = SqlResult("q", COLS, _rows())
    derived = derive_calcs(COLS, _rows())
    hits = [_hit("AAPL")]
    ans = parse_citations("a [C1] b [F3] c [K1] d [C7] e [F9] f [K99]", hits, sql, derived)
    assert list(ans.sources) == ["C1"] and list(ans.fact_sources) == ["F3"]
    assert list(ans.calc_sources) == ["K1"]
    assert ans.invalid_citations == ["C7", "F9", "K99"]


def test_prompt_formats_numbers_exactly_and_marks_evidence_kinds():
    sql = SqlResult("q", COLS, _rows())
    msgs = build_messages("q?", [_hit("AAPL")], sql, derive_calcs(COLS, _rows()))
    user = msgs[1]["content"]
    assert '<chunk id="C1"' in user and '<fact id="F2">' in user
    assert "value=110" in user and "Never calculate" in msgs[0]["content"]


# ---------- verify / revise cycle ----------

Q = "Does Apple mention tariffs in fiscal 2025?"
BAD = "Apple says tariffs cost it $4.2 billion last year [C1]."  # number appears in no evidence
GOOD = "Apple says tariffs may hurt its results [C1]."


def test_a_clean_draft_is_verified_once_and_not_revised():
    llm = FakeLLM(answer=GOOD)
    graph, _ = _graph(llm)
    ans = ask(graph, Q)["answer"]
    assert llm.calls == ["compose", "critic"] and ans.revisions == 0
    assert ans.verification.ok and ans.verification.critic_ran and "Caveat" not in ans.text


def test_a_hallucinated_number_triggers_one_revision_with_specific_feedback():
    llm = FakeLLM(answer=[BAD, GOOD])
    graph, _ = _graph(llm)
    ans = ask(graph, Q)["answer"]
    assert llm.calls == ["compose", "critic", "compose", "critic"]
    assert ans.revisions == 1 and ans.text.startswith("Apple says tariffs may hurt")
    feedback = llm.compose_messages[1][-1]["content"]  # what the reviser was told
    assert "unsupported_number" in feedback and "$4.2 billion" in feedback
    assert llm.compose_messages[1][-2] == {"role": "assistant", "content": BAD}  # it sees its draft
    assert ans.verification.ok and "Caveat" not in ans.text


def test_the_revision_cap_is_hard_and_a_caveat_is_added_when_problems_remain():
    llm = FakeLLM(answer=BAD)  # the model never fixes it
    graph, _ = _graph(llm, max_revisions=1)
    ans = ask(graph, Q)["answer"]
    assert llm.calls.count("compose") == 2 and ans.revisions == 1  # original + exactly one retry
    assert not ans.verification.ok
    assert "Caveat: 1 statement" in ans.text and "could not be verified" in ans.text


def test_zero_revisions_means_verify_and_caveat_only():
    llm = FakeLLM(answer=BAD)
    graph, _ = _graph(llm, max_revisions=0)
    ans = ask(graph, Q)["answer"]
    assert llm.calls.count("compose") == 1 and ans.revisions == 0 and "Caveat" in ans.text


def test_a_critic_that_cannot_verify_fails_closed():
    llm = FakeLLM(answer=GOOD, critic="not json")
    graph, _ = _graph(llm, max_revisions=0)
    ans = ask(graph, Q)["answer"]
    assert [i.kind for i in ans.verification.issues] == ["unsupported_claim"]
    assert "Caveat" in ans.text


def test_verification_can_be_switched_off():
    llm = FakeLLM(answer=BAD)
    graph, _ = _graph(llm, verify=False)
    ans = ask(graph, Q)["answer"]
    assert llm.calls == ["compose"] and ans.verification is None and "Caveat" not in ans.text


def test_no_evidence_skips_composition_and_verification_entirely():
    llm = FakeLLM()
    graph, searcher = _graph(llm)
    searcher.search = lambda q, f, limit=None, alpha=None: []
    ans = ask(graph, Q)["answer"]
    assert ans.text == NO_EVIDENCE and llm.calls == []
