"""LangGraph orchestrator.

    plan --text--> retrieve --both--> facts --> compose
      |                |                 ^
      |                +----text-------> compose
      +----facts-----> facts --(no rows, no chunks)--> retrieve --> compose

    compose --> verify --issues & revisions < cap--> revise --> verify   (bounded cycle)
                  |--ok, or cap reached--> finalize (appends a caveat if issues remain) --> END

Nodes are thin wrappers over components that are tested on their own (router, search, rerank, SQL
tool, calculator, verifier). The graph adds shared state, branching, the "facts found nothing, fall
back to filing text" path, and the verify/revise cycle, whose only loop is capped by max_revisions.
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from finsight.config import Settings
from finsight.observability import langsmith_config
from finsight.orchestrator.compose import NO_EVIDENCE, GraphAnswer, build_messages, parse_citations
from finsight.orchestrator.derive import Derived, derive_calcs
from finsight.orchestrator.router import Plan, plan_question
from finsight.orchestrator.verify import (
    Issue,
    Verification,
    caveat,
    feedback_message,
    verify_answer,
)
from finsight.retrieval.rerank import NoRerank, PineconeReranker
from finsight.retrieval.search import HybridSearcher, build_filter
from finsight.schemas import Hit
from finsight.tools.sql_tool import FactsDB, SqlResult, SqlToolError, answer_with_sql
from finsight.tracing import span, trace_id_var

log = logging.getLogger("finsight.graph")
LLM = Callable[[list[dict]], str]
Reranker = NoRerank | PineconeReranker


@dataclass
class Deps:
    settings: Settings
    searcher: HybridSearcher
    reranker: Reranker
    llm: LLM  # cheap model: routing + text-to-SQL
    facts_db: FactsDB | None
    synth_llm: LLM | None = None  # stronger model for the final answer only
    # Optional A2A delegation (step 7). When set, these replace the in-process path.
    # facts_agent(question, trace_id) -> SqlResult; raises SqlToolError on any failure.
    facts_agent: Callable[[str, str], SqlResult] | None = None
    # verifier(answer, hits, sql, derived, trace_id) -> Verification; may raise (fails closed).
    verifier: (
        Callable[[str, list[Hit], SqlResult | None, list[Derived], str], Verification] | None
    ) = None  # noqa: E501


class State(TypedDict, total=False):
    question: str
    trace_id: str
    plan: Plan
    hits: list[Hit]
    sql: SqlResult | None
    derived: list[Derived]
    errors: list[str]
    draft: str
    verification: Verification
    revisions: int
    answer: GraphAnswer


def _per_query_k(final_k: int, n_sub: int) -> int:
    """Split the chunk budget across sub-queries so every company/year gets retrieval slots."""
    budget = final_k if n_sub == 1 else final_k + 2
    return max(1, budget // n_sub)


def retrieve_for_plan(
    plan: Plan, searcher: HybridSearcher, reranker: Reranker, final_k: int
) -> list[Hit]:
    """Search + rerank once per sub-query (each with its own metadata filter), then merge.

    Shared by the graph's retrieve node and the eval harness, so evals measure the real thing."""
    per_k = _per_query_k(final_k, len(plan.sub_queries))
    merged: dict[str, Hit] = {}
    for sub in plan.sub_queries:
        flt = build_filter(list(sub.tickers), sub.fiscal_year, sub.form_type, sub.section)
        candidates = searcher.search(sub.question, flt)
        for h in reranker.rerank(sub.question, candidates, per_k):
            merged.setdefault(h.chunk_id, h)
    return list(merged.values())


def _node_attrs(name: str, out: dict) -> dict:
    """Counts and flags only: a span never carries prompts, filing text or numbers."""
    if name == "plan":
        plan = out["plan"]
        return {"route": plan.route, "subqueries": len(plan.sub_queries)}
    if name == "retrieve":
        return {"hits": len(out["hits"])}
    if name == "facts":
        sql = out.get("sql")
        return {"rows": len(sql.rows) if sql else 0, "failed": sql is None}
    if name == "compose":
        return {"draft_chars": len(out.get("draft", "")), "no_evidence": "answer" in out}
    if name == "verify":
        v = out["verification"]
        return {
            "issues": len(v.issues), "kinds": ",".join(sorted({i.kind for i in v.issues})),
            "critic_ran": v.critic_ran, "claims": v.claims_checked,
        }  # fmt: skip
    if name == "revise":
        return {"revisions": out["revisions"]}
    if name == "finalize":
        v = out["answer"].verification
        return {"caveat": bool(v and v.issues)}
    return {}


def _traced(name: str, fn):
    def wrapper(state: State) -> dict:
        with span(f"node.{name}") as a:
            out = fn(state)
            a.update(_node_attrs(name, out))
            return out

    return wrapper


def build_graph(deps: Deps):
    s = deps.settings

    def plan_node(state: State) -> dict:
        plan = plan_question(state["question"], deps.llm)
        log.info(
            "trace=%s route=%s subqueries=%d", state["trace_id"], plan.route, len(plan.sub_queries)
        )
        return {"plan": plan, "errors": []}

    def retrieve_node(state: State) -> dict:
        hits = retrieve_for_plan(state["plan"], deps.searcher, deps.reranker, s.final_k)
        return {"hits": hits}

    def facts_node(state: State) -> dict:
        if deps.facts_db is None and deps.facts_agent is None:
            return {"sql": None, "derived": [], "errors": [*state["errors"], "facts DB missing"]}
        try:
            if deps.facts_agent is not None:
                sql = deps.facts_agent(state["question"], state["trace_id"])
            else:
                sql = answer_with_sql(state["question"], deps.facts_db, deps.llm)
        except SqlToolError as e:
            log.warning("trace=%s sql failed: %s", state["trace_id"], e)
            return {"sql": None, "derived": [], "errors": [*state["errors"], str(e)]}
        return {"sql": sql, "derived": derive_calcs(sql.columns, sql.rows)}

    def compose_node(state: State) -> dict:
        hits = state.get("hits", [])
        sql = state.get("sql")
        derived = state.get("derived", [])
        if not hits and not (sql and sql.rows):
            return {"answer": GraphAnswer(text=NO_EVIDENCE), "draft": ""}
        messages = build_messages(state["question"], hits, sql, derived)
        return {"draft": (deps.synth_llm or deps.llm)(messages), "revisions": 0}

    def verify_node(state: State) -> dict:
        hits, sql, derived = state.get("hits", []), state.get("sql"), state.get("derived", [])
        if deps.verifier is not None:
            try:
                v = deps.verifier(state["draft"], hits, sql, derived, state["trace_id"])
            except Exception:  # fail closed: never pass an answer because the verifier was down
                log.exception("trace=%s remote verifier failed", state["trace_id"])
                v = Verification(
                    issues=[Issue("verifier_unavailable", "verification service unavailable")]
                )
        else:
            # The critic is the cheap model; the answer itself may come from a stronger one.
            v = verify_answer(state["draft"], hits, sql, derived, deps.llm)
        log.info(
            "trace=%s verify: %d issues, %d revisions so far",
            state["trace_id"], len(v.issues), state["revisions"],
        )  # fmt: skip
        return {"verification": v}

    def revise_node(state: State) -> dict:
        messages = build_messages(
            state["question"], state.get("hits", []), state.get("sql"), state.get("derived", [])
        )
        messages += [
            {"role": "assistant", "content": state["draft"]},
            {"role": "user", "content": feedback_message(state["verification"].issues)},
        ]
        return {
            "draft": (deps.synth_llm or deps.llm)(messages),
            "revisions": state["revisions"] + 1,
        }

    def finalize_node(state: State) -> dict:
        ans = parse_citations(
            state["draft"], state.get("hits", []), state.get("sql"), state.get("derived", [])
        )
        v = state.get("verification")
        ans.verification, ans.revisions = v, state.get("revisions", 0)
        if v and v.issues:  # never hand back unverified text without saying so
            ans.text = f"{ans.text}\n\n{caveat(v.issues)}"
        return {"answer": ans}

    def after_plan(state: State) -> str:
        return "facts" if state["plan"].route == "facts" else "retrieve"

    def after_retrieve(state: State) -> str:
        return "facts" if state["plan"].route == "both" else "compose"

    def after_facts(state: State) -> str:
        # Only the pure-facts route falls back; on "both", retrieval already ran (this guard is
        # what keeps the graph loop-free: retrieve -> facts -> retrieve would never end).
        sql = state.get("sql")
        empty = not (sql and sql.rows) and not state.get("hits")
        return "retrieve" if empty and state["plan"].route == "facts" else "compose"

    def after_compose(state: State) -> str:
        if "answer" in state:  # no evidence: already answered, nothing to verify
            return END
        return "verify" if s.verify else "finalize"

    def after_verify(state: State) -> str:
        # The revise <-> verify cycle is bounded by max_revisions: this guard is what ends it.
        issues = state["verification"].issues
        fixable = [i for i in issues if i.kind != "verifier_unavailable"]  # a rewrite can't fix it
        if fixable and state["revisions"] < s.max_revisions:
            return "revise"
        return "finalize"

    g = StateGraph(State)
    for name, fn in [
        ("plan", plan_node), ("retrieve", retrieve_node), ("facts", facts_node),
        ("compose", compose_node), ("verify", verify_node), ("revise", revise_node),
        ("finalize", finalize_node),
    ]:  # fmt: skip
        g.add_node(name, _traced(name, fn))
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, ["retrieve", "facts"])
    g.add_conditional_edges("retrieve", after_retrieve, ["facts", "compose"])
    g.add_conditional_edges("facts", after_facts, ["retrieve", "compose"])
    g.add_conditional_edges("compose", after_compose, ["verify", "finalize", END])
    g.add_conditional_edges("verify", after_verify, ["revise", "finalize"])
    g.add_edge("revise", "verify")
    g.add_edge("finalize", END)
    return g.compile()


def ask(graph, question: str, trace_id: str | None = None) -> State:
    """Run one question through the graph. Returns the final state (answer, plan, sql, ...)."""
    trace_id = trace_id or uuid.uuid4().hex
    token = trace_id_var.set(trace_id)
    try:
        with span("ask", question_chars=len(question)) as a:
            state = graph.invoke(
                {"question": question, "trace_id": trace_id}, config=langsmith_config(trace_id)
            )
            ans = state.get("answer")
            a.update(
                route=state["plan"].route,
                revisions=getattr(ans, "revisions", 0),
                verified=bool(ans and ans.verification and ans.verification.ok),
            )
            return state
    finally:
        trace_id_var.reset(token)
