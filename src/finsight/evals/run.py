"""Run the golden eval against the Pinecone index.

Run from the finsight/ folder:
  uv run python -m finsight.evals.run --validate                      # check labels vs the index
  uv run python -m finsight.evals.run --name baseline                 # retrieval metrics
  uv run python -m finsight.evals.run --name alpha --alpha 0.3 0.5 0.7   # compare alpha values
  uv run python -m finsight.evals.run --name rr --rerank pinecone     # hosted rerank (budgeted)
  uv run python -m finsight.evals.run --name ans --llm                # + Claude answer checks
  uv run python -m finsight.evals.run --smoke                         # small CI-sized subset

Retrieval runs use the golden set's hand-written filters (the router that infers them arrives
in step 4), so they measure the search itself, not the planning.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from finsight.api.app import _build_deps
from finsight.config import Settings, get_settings
from finsight.evals.golden import load_corpus, load_golden, validate
from finsight.evals.metrics import mean, score_answer, score_retrieval
from finsight.ingestion.embedder import Embedder
from finsight.ingestion.pinecone_store import PineconeStore
from finsight.observability import configure_langsmith
from finsight.orchestrator.graph import ask, build_graph, retrieve_for_plan
from finsight.orchestrator.router import plan_question
from finsight.orchestrator.verify import verify_answer
from finsight.retrieval.answer import answer_question, make_llm
from finsight.retrieval.cache import JsonCache
from finsight.retrieval.rerank import make_reranker
from finsight.retrieval.search import HybridSearcher, build_filter

RESULTS_DIR = Path("evals/results")


def _agg(rows: list[dict], key: str) -> dict:
    scores = [r[key] for r in rows if key in r]
    return {
        "hit": round(mean([s["hit"] for s in scores]), 3),
        "coverage": round(mean([s["coverage"] for s in scores]), 3),
        "mrr": round(mean([s["rr"] for s in scores]), 3),
    }


def evaluate(
    rows: list[dict], settings: Settings, searcher: HybridSearcher, llm=None, use_router=False
):
    """One full pass over the golden set with the given settings (alpha, rerank, k)."""
    reranker = make_reranker(settings)
    k = settings.final_k
    results = []
    for r in rows:
        if use_router:
            # Step 4: filters are inferred from the question; the golden filters are ignored.
            final = retrieve_for_plan(plan_question(r["question"]), searcher, reranker, k)
            candidates = top = final  # the router has no separate "pool" stage
        else:
            f = r["filters"]
            flt = build_filter(f.get("tickers"), f.get("year"), f.get("form"), f.get("section"))
            candidates = searcher.search(r["question"], flt, alpha=settings.hybrid_alpha)
            top = candidates[:k]
            final = reranker.rerank(r["question"], candidates, k)
        row: dict = {"id": r["id"], "type": r["type"], "question": r["question"]}
        if r["targets"]:
            for name, hits in [("pool", candidates), ("top", top), ("final", final)]:
                sc = score_retrieval(r["targets"], hits)
                row[name] = {"hit": sc.hit, "coverage": sc.coverage, "rr": sc.reciprocal_rank}
        if llm:
            ans = answer_question(r["question"], final, llm)
            row["answer"] = ans.text
            row["answer_checks"] = score_answer(
                r["targets"], r["expect_abstain"], ans.text,
                list(ans.sources.values()), ans.invalid_citations,
            )  # fmt: skip
        results.append(row)
        print(
            f"  {r['id']:4} "
            + (f"hit={row['final']['hit']!s:5}" if "final" in row else "(no targets)")
        )

    answerable = [x for x in results if "final" in x]
    summary: dict = {
        "n_questions": len(results),
        "n_answerable": len(answerable),
        "retrieval": {
            f"pool@{settings.retrieve_k}": _agg(answerable, "pool"),
            f"top@{k}": _agg(answerable, "top"),
            f"final@{k}": _agg(answerable, "final"),
        },
        "by_type": {
            t: _agg([x for x in answerable if x["type"] == t], "final")
            for t in sorted({x["type"] for x in answerable})
        },
    }
    if llm:
        checks = defaultdict(list)
        for x in results:
            for name, val in x["answer_checks"].items():
                checks[name].append(bool(val))
        summary["answers"] = {name: round(mean(v), 3) for name, v in checks.items()}
    return results, summary


def evaluate_e2e(rows: list[dict], settings: Settings):
    """Every question through the REAL graph (router, SQL route, verifier), like /ask does.

    When verification is off, the final draft is still verified after the fact, so both arms of a
    before/after comparison are measured with the same yardstick."""
    deps = _build_deps(settings)
    graph = build_graph(deps)
    results = []
    for r in rows:
        st = ask(graph, r["question"], trace_id=r["id"])
        ans, hits = st["answer"], st.get("hits", [])
        v = ans.verification
        if v is None and st.get("draft"):
            v = verify_answer(st["draft"], hits, st.get("sql"), st.get("derived", []), deps.llm)
        row: dict = {
            "id": r["id"], "type": r["type"], "question": r["question"],
            "route": st["plan"].route, "answer": ans.text, "revisions": ans.revisions,
            "issues": [i.kind for i in v.issues] if v else [],
        }  # fmt: skip
        if r["targets"]:
            sc = score_retrieval(r["targets"], hits)
            row["final"] = {"hit": sc.hit, "coverage": sc.coverage, "rr": sc.reciprocal_rank}
        row["answer_checks"] = score_answer(
            r["targets"], r["expect_abstain"], ans.text,
            list(ans.sources.values()), ans.invalid_citations,
        )  # fmt: skip
        results.append(row)
        print(
            f"  {r['id']:4} route={row['route']:5} issues={row['issues']} revisions={ans.revisions}"
        )

    checks = defaultdict(list)
    for x in results:
        for name, val in x["answer_checks"].items():
            checks[name].append(bool(val))
    kinds: dict[str, int] = defaultdict(int)
    for x in results:
        for k in x["issues"]:
            kinds[k] += 1
    summary = {
        "n_questions": len(results),
        "answers": {name: round(mean(v), 3) for name, v in checks.items()},
        "verification": {
            "answers_with_issues": round(mean([bool(x["issues"]) for x in results]), 3),
            "issue_kinds": dict(kinds),
            "answers_revised": round(mean([x["revisions"] > 0 for x in results]), 3),
            "routes": {k: sum(x["route"] == k for x in results) for k in ("text", "facts", "both")},
        },
    }
    return results, summary


def _print_table(runs: list[dict], k: int) -> None:
    print(
        f"\n{'config':28} {'pool':>6} {'hit@' + str(k):>7} {'cover':>7} {'mrr':>6}   by type (hit)"
    )
    for run in runs:
        cfg, sm = run["config"], run["summary"]
        final = sm["retrieval"][f"final@{k}"]
        pool = next(v for name, v in sm["retrieval"].items() if name.startswith("pool"))
        types = " ".join(f"{t[:5]}={v['hit']:.2f}" for t, v in sm["by_type"].items())
        label = f"alpha={cfg['alpha']} rerank={cfg['rerank']}" + (
            " +router" if cfg.get("router") else ""
        )
        print(
            f"{label:28} {pool['hit']:>6.2f} {final['hit']:>7.2f} "
            f"{final['coverage']:>7.2f} {final['mrr']:>6.2f}   {types}"
        )


def main() -> None:
    for noisy in ("httpx", "pinecone", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="only questions flagged smoke")
    ap.add_argument("--llm", action="store_true", help="also generate answers and check them")
    ap.add_argument("--name", default="run", help="results file name")
    ap.add_argument("--alpha", type=float, nargs="+", help="one or more hybrid alpha values")
    ap.add_argument("--rerank", choices=["none", "pinecone"], help="override the rerank setting")
    ap.add_argument("--e2e", action="store_true", help="run the full graph (as /ask does)")
    ap.add_argument(
        "--verify", choices=["on", "off"], help="override the verify setting (use with --e2e)"
    )
    ap.add_argument(
        "--router",
        action="store_true",
        help="step 4: infer filters from the question and split it into sub-queries",
    )
    args = ap.parse_args()

    s = get_settings()
    configure_langsmith(s)  # no-op unless LANGSMITH_TRACING is on in .env
    store = PineconeStore(s)
    rows = load_golden(smoke_only=args.smoke)

    if args.validate:
        store.connect()
        corpus = load_corpus(store)
        problems = validate(load_golden(), corpus)
        print(f"{len(corpus)} chunks in index; {len(load_golden())} questions")
        print("\n".join(problems) if problems else "All targets match at least one chunk.")
        raise SystemExit(1 if problems else 0)

    if args.e2e:
        cfg = s.model_copy(
            update={
                "hybrid_alpha": (args.alpha or [s.hybrid_alpha])[0],
                "verify": (args.verify or ("on" if s.verify else "off")) == "on",
            }
        )
        print(f"\n== e2e verify={cfg.verify} model={cfg.synthesis_model or cfg.llm_model}")
        results, summary = evaluate_e2e(rows, cfg)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{args.name}.json"
        cfgd = {"verify": cfg.verify, "max_revisions": cfg.max_revisions, "alpha": cfg.hybrid_alpha,
                "llm_model": cfg.llm_model, "synthesis_model": cfg.synthesis_model}  # fmt: skip
        out.write_text(
            json.dumps({"config": cfgd, "summary": summary, "questions": results}, indent=2)
        )
        print("\n" + json.dumps(summary, indent=2) + f"\n\nSaved {out}")
        raise SystemExit(0)

    alphas = args.alpha or [s.hybrid_alpha]
    if args.llm and len(alphas) > 1:
        raise SystemExit("--llm answers one configuration; pass a single --alpha")

    store.connect()
    cache = JsonCache(s.cache_dir)
    searcher = HybridSearcher(
        s, Embedder(s), store, cache=cache
    )  # shared: embeds each question once
    llm = make_llm(s, cache, s.synthesis_model) if args.llm else None  # same model as /ask

    runs = []
    for alpha in alphas:
        cfg = s.model_copy(update={"hybrid_alpha": alpha, "rerank": args.rerank or s.rerank})
        print(f"\n== alpha={alpha} rerank={cfg.rerank}")
        results, summary = evaluate(rows, cfg, searcher, llm, use_router=args.router)
        runs.append({
            "config": {
                "alpha": alpha, "rerank": cfg.rerank, "retrieve_k": cfg.retrieve_k,
                "final_k": cfg.final_k, "dense_model": cfg.dense_model,
                "sparse_model": cfg.sparse_model, "llm_model": cfg.llm_model if llm else None,
                "smoke": args.smoke, "router": args.router,
            },
            "summary": summary,
            "questions": results,
        })  # fmt: skip

    _print_table(runs, s.final_k)
    for run in runs:
        if "answers" in run["summary"]:
            print("\nanswer checks:", json.dumps(run["summary"]["answers"], indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
