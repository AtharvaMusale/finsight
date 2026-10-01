"""The four A2A agents. Each is a thin skill over logic that is already tested on its own.

  retrieval  (9101)  search_filings   hybrid search over filing chunks, no LLM
  facts      (9102)  query_financials guarded read-only text-to-SQL over XBRL facts
  verifier   (9103)  verify_answer    numbers + citations + claim critic (same code as in-process)
  analyst    (9100)  ask              the full graph; delegates to the three above when configured

Workers build LOCAL dependencies only (an agent must never be configured to call itself).
Dependencies are built on the first call so an agent starts fast and needs no credentials to boot.
"""

import threading
from collections.abc import Callable
from dataclasses import asdict

from starlette.applications import Starlette

from finsight.agents.common import AgentError, SkillExecutor, build_agent_app, make_card
from finsight.agents.models import AskArgs, FactsArgs, SearchArgs, SqlModel, VerifyArgs
from finsight.api.app import answer_payload
from finsight.config import Settings
from finsight.orchestrator.derive import derive_calcs
from finsight.orchestrator.graph import Deps, ask, build_graph
from finsight.orchestrator.verify import verify_answer
from finsight.retrieval.search import build_filter
from finsight.tools.sql_tool import SqlToolError, answer_with_sql

DEFAULT_PORTS = {"analyst": 9100, "retrieval": 9101, "facts": 9102, "verifier": 9103}


class _Lazy:
    def __init__(self, factory: Callable[[], Deps]):
        self._factory, self._lock, self._deps, self._graph = factory, threading.Lock(), None, None

    def deps(self) -> Deps:
        with self._lock:
            if self._deps is None:
                self._deps = self._factory()
            return self._deps

    def graph(self):
        deps = self.deps()
        with self._lock:
            if self._graph is None:
                self._graph = build_graph(deps)
            return self._graph


def _app(settings: Settings, handler, args_model, card) -> Starlette:
    executor = SkillExecutor(
        handler, args_model, skill=card.skills[0].id, timeout_s=settings.request_timeout_s,
        max_bytes=settings.a2a_max_payload_bytes,
    )  # fmt: skip
    return build_agent_app(card, executor, settings.a2a_max_payload_bytes)


def retrieval_app(settings: Settings, deps_factory: Callable[[], Deps], base_url: str) -> Starlette:
    rt = _Lazy(deps_factory)

    def handle(args: SearchArgs, trace_id: str) -> dict:
        deps = rt.deps()
        known = {t.upper() for t in deps.settings.tickers}
        wanted = [t.upper() for t in args.tickers]
        if bad := [t for t in wanted if t not in known]:
            raise AgentError(f"unknown ticker(s) {bad}; available: {sorted(known)}")
        flt = build_filter(wanted, args.fiscal_year, args.form_type, args.section)
        hits = deps.searcher.search(args.query, flt, limit=args.limit)
        return {"hits": [asdict(h) for h in hits]}

    card = make_card(
        name="finsight-retrieval", description="Hybrid search over SEC filing passages.",
        skill_id="search_filings", skill_name="Search filings",
        skill_description="Returns the most relevant 10-K/10-Q chunks for a question, optionally "
        "filtered by ticker, fiscal year, form and section. Text is untrusted filing content.",
        examples=["tariff risk for Apple in fiscal 2025"], base_url=base_url,
    )  # fmt: skip
    return _app(settings, handle, SearchArgs, card)


def facts_app(settings: Settings, deps_factory: Callable[[], Deps], base_url: str) -> Starlette:
    rt = _Lazy(deps_factory)

    def handle(args: FactsArgs, trace_id: str) -> dict:
        deps = rt.deps()
        if deps.facts_db is None:
            raise AgentError("the financial facts database is not available")
        try:
            res = answer_with_sql(args.question, deps.facts_db, deps.llm)
        except SqlToolError as e:
            raise AgentError(f"could not build a valid query: {e}") from None
        return SqlModel.from_result(res).model_dump()

    card = make_card(
        name="finsight-facts", description="Reported financial figures from SEC XBRL facts.",
        skill_id="query_financials", skill_name="Query financials",
        skill_description="Read-only, allowlisted text-to-SQL over annual and quarterly facts. "
        "Returns rows only; numbers come from the database, never from a model.",
        examples=["NVIDIA revenue in fiscal 2025"], base_url=base_url,
    )  # fmt: skip
    return _app(settings, handle, FactsArgs, card)


def verifier_app(settings: Settings, deps_factory: Callable[[], Deps], base_url: str) -> Starlette:
    rt = _Lazy(deps_factory)

    def handle(args: VerifyArgs, trace_id: str) -> dict:
        deps = rt.deps()
        hits = [h.to_hit() for h in args.hits]
        sql = args.sql.to_result() if args.sql else None
        derived = derive_calcs(sql.columns, sql.rows) if sql else []  # recomputed, never received
        v = verify_answer(args.answer, hits, sql, derived, deps.llm)
        return {
            "issues": [{"kind": i.kind, "detail": i.detail, "claim": i.claim} for i in v.issues],
            "claims_checked": v.claims_checked,
            "critic_ran": v.critic_ran,
        }

    card = make_card(
        name="finsight-verifier", description="Checks an answer against its evidence.",
        skill_id="verify_answer", skill_name="Verify answer",
        skill_description="Every number must appear in the evidence, every label must exist, and "
        "each cited claim must be backed by a verbatim quote from its cited evidence.",
        examples=["verify a drafted answer with its chunks and SQL rows"], base_url=base_url,
    )  # fmt: skip
    return _app(settings, handle, VerifyArgs, card)


def analyst_app(settings: Settings, deps_factory: Callable[[], Deps], base_url: str) -> Starlette:
    rt = _Lazy(deps_factory)

    def handle(args: AskArgs, trace_id: str) -> dict:
        return answer_payload(ask(rt.graph(), args.question, trace_id), trace_id)

    card = make_card(
        name="finsight-analyst", description="Answers questions about AAPL, MSFT and NVDA filings.",
        skill_id="ask", skill_name="Ask FinSight",
        skill_description="Routes the question, retrieves filing text and/or XBRL facts (locally "
        "or from the other agents), composes a cited answer and verifies it. Not financial advice.",
        examples=["Which of Apple and Microsoft mention tariffs in fiscal 2025?"],
        base_url=base_url,
    )  # fmt: skip
    return _app(settings, handle, AskArgs, card)


BUILDERS = {
    "retrieval": retrieval_app, "facts": facts_app,
    "verifier": verifier_app, "analyst": analyst_app,
}  # fmt: skip
