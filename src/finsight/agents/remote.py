"""Adapters that let the orchestrator graph use remote A2A agents through its existing seams.

  RemoteSearcher        looks like HybridSearcher.search (the local reranker still runs)
  make_facts_agent      (question, trace_id) -> SqlResult, failures become SqlToolError so the
                        graph's existing "facts unavailable, fall back to filings" path applies
  make_verifier         (answer, hits, sql, derived, trace_id) -> Verification; it raises on any
                        failure and the graph fails closed (caveat, no pointless rewrite)

Numbers stay local: the facts agent returns SQL rows only. Growth and margins are recomputed from
those rows by the orchestrator, never taken from a remote "calculation".
"""

import dataclasses
from dataclasses import asdict

import httpx
from pydantic import ValidationError

from finsight.agents.common import RemoteAgent, RemoteAgentError
from finsight.agents.models import HitsResult, SqlModel, VerificationResult
from finsight.orchestrator.derive import Derived
from finsight.orchestrator.graph import Deps
from finsight.orchestrator.verify import Issue, Verification
from finsight.schemas import Hit, SearchFilter
from finsight.tools.sql_tool import SqlResult, SqlToolError
from finsight.tracing import trace_id_var


class RemoteSearcher:
    def __init__(self, agent: RemoteAgent):
        self._agent = agent

    def search(
        self,
        question: str,
        flt: SearchFilter | None = None,
        limit: int | None = None,
        alpha: float
        | None = None,  # the retrieval agent owns its alpha; a per-call override is ignored
    ) -> list[Hit]:
        args: dict = {"query": question}
        if flt:
            args["tickers"] = list(flt.tickers)
            for key in ("fiscal_year", "form_type", "section"):
                if (value := getattr(flt, key)) is not None:
                    args[key] = value
        if limit:
            args["limit"] = limit
        result = self._agent.call(args, trace_id_var.get())
        try:
            return [h.to_hit() for h in HitsResult.model_validate(result).hits]
        except ValidationError:
            raise RemoteAgentError("retrieval agent returned malformed hits") from None


def make_facts_agent(agent: RemoteAgent):
    def call(question: str, trace_id: str) -> SqlResult:
        try:
            result = agent.call({"question": question}, trace_id)
            return SqlModel.model_validate(result).to_result()
        except RemoteAgentError as e:
            raise SqlToolError(str(e)) from None
        except ValidationError:
            raise SqlToolError("facts agent returned malformed rows") from None

    return call


def make_verifier(agent: RemoteAgent):
    def call(
        answer: str,
        hits: list[Hit],
        sql: SqlResult | None,
        derived: list[Derived],  # recomputed by the verifier from the same rows; not sent
        trace_id: str,
    ) -> Verification:
        args = {
            "answer": answer,
            "hits": [asdict(h) for h in hits],
            "sql": SqlModel.from_result(sql).model_dump() if sql else None,
        }
        result = agent.call(args, trace_id)
        try:
            parsed = VerificationResult.model_validate(result)
        except ValidationError:
            raise RemoteAgentError("verifier returned a malformed verdict") from None
        return Verification(
            issues=[Issue(i.kind, i.detail, i.claim) for i in parsed.issues],
            claims_checked=parsed.claims_checked,
            critic_ran=parsed.critic_ran,
        )

    return call


def with_remote_agents(
    deps: Deps, transports: dict[str, httpx.AsyncBaseTransport] | None = None
) -> Deps:
    """Swap in A2A clients for every capability that has a URL configured; others stay local."""
    s, transports = deps.settings, transports or {}

    def agent(url: str, name: str) -> RemoteAgent:
        return RemoteAgent(
            url, timeout_s=s.a2a_timeout_s, max_bytes=s.a2a_max_payload_bytes,
            transport=transports.get(name),
        )  # fmt: skip

    changes: dict = {}
    if s.a2a_retrieval_url:
        changes["searcher"] = RemoteSearcher(agent(s.a2a_retrieval_url, "retrieval"))
    if s.a2a_facts_url:
        changes["facts_agent"] = make_facts_agent(agent(s.a2a_facts_url, "facts"))
    if s.a2a_verifier_url:
        changes["verifier"] = make_verifier(agent(s.a2a_verifier_url, "verifier"))
    return dataclasses.replace(deps, **changes) if changes else deps
