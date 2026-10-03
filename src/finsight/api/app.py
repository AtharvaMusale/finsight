"""HTTP API.

Run from the finsight/ folder:
  uv run uvicorn finsight.api.app:app --host 127.0.0.1 --port 8000
  curl -s localhost:8000/ask -H 'content-type: application/json' \
       -d '{"question": "Which of Apple and Microsoft mention tariffs in fiscal 2025?"}'

Every response carries an X-Trace-Id header (also in the JSON body and the logs) so one request
can be followed through the graph. The whole request has a hard timeout.
"""

import asyncio
import logging
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from finsight.config import Settings, get_settings
from finsight.ingestion.embedder import Embedder
from finsight.ingestion.pinecone_store import PineconeStore
from finsight.observability import configure_langsmith
from finsight.orchestrator.graph import Deps, ask, build_graph
from finsight.retrieval.answer import make_llm
from finsight.retrieval.cache import JsonCache
from finsight.retrieval.rerank import make_reranker
from finsight.retrieval.search import HybridSearcher
from finsight.tools.sql_tool import FactsDB
from finsight.tracing import configure as configure_tracing
from finsight.tracing import read_spans

log = logging.getLogger("finsight.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _build_deps(settings: Settings | None = None, remote: bool = True) -> Deps:
    """Local dependencies; with remote=True, capabilities that have an A2A URL are delegated."""
    s = settings or get_settings()
    cache = JsonCache(s.cache_dir)
    facts_db = FactsDB(s.duckdb_path, timeout_s=s.sql_timeout_s) if s.duckdb_path.exists() else None
    deps = Deps(
        settings=s,
        searcher=HybridSearcher(s, Embedder(s), PineconeStore(s), cache=cache),
        reranker=make_reranker(s),
        llm=make_llm(s, cache),
        synth_llm=make_llm(s, cache, s.synthesis_model) if s.synthesis_model else None,
        facts_db=facts_db,
    )
    if remote:
        from finsight.agents.remote import with_remote_agents  # lazy: agents import this module

        deps = with_remote_agents(deps)
    return deps


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_tracing("api", s.trace_dir, s.tracing)
    configure_langsmith(s)  # before the graph is built: the SDK caches its first env read
    app.state.graph = build_graph(_build_deps())
    yield


app = FastAPI(title="FinSight", lifespan=lifespan)


@app.middleware("http")
async def trace_id_header(request: Request, call_next):
    request.state.trace_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Trace-Id"] = request.state.trace_id
    return response


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/ui/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    """What this deployment is set up to do. Booleans and names only, never a key or its value."""
    s = get_settings()
    urls = {
        "retrieval": s.a2a_retrieval_url,
        "facts": s.a2a_facts_url,
        "verifier": s.a2a_verifier_url,
    }

    async def probe(client: httpx.AsyncClient, url: str | None) -> bool | None:
        if not url:
            return None  # not configured: this capability runs in-process
        try:
            return (await client.get(url.rstrip("/") + "/health")).status_code == 200
        except httpx.HTTPError:
            return False

    # Only the URLs from this deployment's own settings are probed, never a caller-supplied one.
    async with httpx.AsyncClient(timeout=1.0) as client:
        ups = await asyncio.gather(*(probe(client, u) for u in urls.values()))
    agents = [
        {"name": n, "url": u, "configured": bool(u), "up": up}
        for (n, u), up in zip(urls.items(), ups, strict=True)
    ]
    return {
        "model": s.llm_model,
        "verify": s.verify,
        "max_revisions": s.max_revisions,
        "tracing": s.tracing,
        "langsmith": bool(s.langsmith_tracing and s.langsmith_api_key),
        "keys": {
            "anthropic": s.anthropic_api_key is not None,
            "pinecone": s.pinecone_api_key is not None,
        },
        "facts_db": s.duckdb_path.exists(),
        "mode": "a2a" if any(a["configured"] for a in agents) else "in-process",
        "agents": agents,
    }  # fmt: skip


_TRACE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ROOT_ATTRS = ("route", "revisions", "verified", "question_chars")


@app.get("/traces")
def list_traces(limit: int = 15) -> dict:
    """The newest questions this deployment answered: one summary row per trace."""
    s = get_settings()
    if not s.tracing:
        return {"tracing": False, "traces": []}
    spans = read_spans(s.trace_dir) if s.trace_dir.exists() else []
    roots = [r for r in spans if r.get("parent") is None and r.get("name") == "ask"]
    roots.sort(key=lambda r: r.get("ts", 0), reverse=True)
    rows = [
        {
            "trace_id": r["trace"], "ts": r.get("ts"), "ms": r.get("ms"), "ok": r.get("ok"),
            **{k: (r.get("attrs") or {}).get(k) for k in _ROOT_ATTRS},
        }
        for r in roots[: max(1, min(limit, 50))]
    ]  # fmt: skip
    return {"tracing": True, "traces": rows}


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    """Spans this deployment recorded for one trace (counts and timings, never text)."""
    s = get_settings()
    if not s.tracing:
        raise HTTPException(404, "tracing is off")
    if not _TRACE_ID.match(trace_id):
        raise HTTPException(422, "invalid trace id")
    spans = read_spans(s.trace_dir, trace_id) if s.trace_dir.exists() else []
    if not spans:
        raise HTTPException(404, "no spans for that trace id")
    return {"trace_id": trace_id, "spans": sorted(spans, key=lambda r: r.get("ts", 0))}


def answer_payload(state: dict, trace_id: str) -> dict:
    """The JSON shape of one answered question. Shared by the HTTP API and the MCP server."""
    ans, plan = state["answer"], state["plan"]
    v = ans.verification
    return {
        "trace_id": trace_id,
        "answer": ans.text,
        "disclaimer": ans.disclaimer,
        "route": plan.route,
        "sub_queries": [asdict(sq) for sq in plan.sub_queries],
        "sources": {
            k: {"source": h.source, "accession_no": h.accession_no, "chunk_id": h.chunk_id}
            for k, h in ans.sources.items()
        },
        "facts": ans.fact_sources,
        "calculations": {
            k: {
                "label": d.label,
                "value": d.result.value,
                "unit": d.result.unit,
                "formula": d.result.formula,
                "from": [f"F{i}" for i in d.from_rows],
            }
            for k, d in ans.calc_sources.items()
        },  # fmt: skip
        "verification": None
        if v is None
        else {
            "ok": v.ok,
            "claims_checked": v.claims_checked,
            "critic_ran": v.critic_ran,
            "revisions": ans.revisions,
            "issues": [{"kind": i.kind, "detail": i.detail} for i in v.issues],
        },
        "invalid_citations": ans.invalid_citations,
        "warnings": state.get("errors", []),
    }


@app.post("/ask")
async def ask_endpoint(body: AskRequest, request: Request) -> dict:
    trace_id = request.state.trace_id
    timeout = get_settings().request_timeout_s
    loop = asyncio.get_running_loop()
    try:
        # The graph is synchronous (model + DB calls), so it runs in a worker thread.
        state = await asyncio.wait_for(
            loop.run_in_executor(None, ask, request.app.state.graph, body.question, trace_id),
            timeout=timeout,
        )
    except TimeoutError:
        log.warning("trace=%s timed out after %ss", trace_id, timeout)
        raise HTTPException(504, f"request exceeded {timeout:.0f}s (trace {trace_id})") from None
    except Exception:
        # Details stay in the server log; callers only get the trace id to quote.
        log.exception("trace=%s failed", trace_id)
        raise HTTPException(502, f"upstream failure (trace {trace_id})") from None

    return answer_payload(state, trace_id)


# Mounted last so it can never shadow an API route.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
