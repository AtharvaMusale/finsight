# FinSight: Running Project Tasks (Pinecone + Anthropic restart)

Last updated: 2026-09-30. Update this file at the end of every step.
Legend: [x] done · [~] in progress · [ ] not started

## Direction change
Restarted from step 1 with **Pinecone** (hosted vector DB and hosted embeddings) and a **cheap Anthropic model**
(`claude-haiku-4-5`) instead of Qdrant + local Ollama. The previous Qdrant/Ollama build of steps 2-4
is preserved (not deleted) on the local branch `archive/steps-2-4-qdrant` for reference.

## Ground rules (apply to every step)
- Data only from the SEC's documented endpoints, enforced by an allowlist in code. Declared User-Agent with contact email, 5 req/s (hard cap 10), back off on 403/429, cache raw filings, never redistribute.
- No secrets in git. `.env` and `data/` are gitignored. Secrets are `SecretStr`. Keys are added to `.env` by the user only.
- Filings are untrusted text. Numbers come from DuckDB or a calculator, never the LLM.
- One step at a time; pause for review after each. Ask before installs, deletions or anything outward-facing.

## Step 1: Scaffold + ingestion
Kept from the first build (SEC side is vector-DB independent):
- [x] Scaffold, `config.py`, `schemas.py`
- [x] `ingestion/edgar.py` (allowlisted SEC client, rate limit, retries, cache), `parser.py`, `chunker.py`
- [x] DuckDB: filings, chunks, ~19k XBRL facts (AAPL, MSFT, NVDA); raw filings cached in `data/raw`
- [x] Tests: `test_edgar.py`, `test_parser_chunker.py`

Redone with Pinecone (done 2026-09-30):
- [x] Keys in `.env` (`PINECONE_API_KEY`, `ANTHROPIC_API_KEY`); `uv add pinecone` (10.0.0); SDK signatures checked offline before any network call
- [x] Config: Pinecone index/namespace/region, hosted models, monthly embedding-token budget (hard stop 3.5M; free plan allows 5M)
- [x] `ingestion/embedder.py`: hosted dense (`llama-text-embed-v2`, 1024 dims) + sparse (`pinecone-sparse-english-v0`); batching, capped retries, token-budget ledger
- [x] `ingestion/pinecone_store.py`: dotproduct index (created only if missing, dimension checked), sparse-dense records with metadata + text, delete-by-id then upsert
- [x] DuckDB `indexed(accession_no, store)` marker so a new index is filled from cached filings; pipeline `--estimate` mode (no uploads)
- [x] Bug fixed: deleting from a not-yet-existing namespace returns 404 on the first ever write (regression test added)
- [x] 42 offline tests pass (fake Pinecone clients), ruff clean
- [x] Ingested AAPL, MSFT, NVDA: 43 filings, 2,967 chunks = 2,967 Pinecone vectors; 19,033 XBRL facts; 2.04M embedding tokens used (estimate was 2.12M)
- [x] AAPL fiscal-year labels now correct (fixed by the re-ingest)

## Step 2: Basic RAG [x] (2026-09-30)
- [x] `schemas.py`: `Hit`, `SearchFilter`; config: retrieval, rerank, Anthropic settings
- [x] `PineconeStore.search`: one hybrid query, dense scaled by alpha, sparse by (1 - alpha), metadata filters; `connect()` never creates an index
- [x] `retrieval/search.py`: `HybridSearcher` (queries embedded as "query", cached; cache key includes the corpus version so re-indexing invalidates it)
- [x] `retrieval/rerank.py`: optional hosted reranker (`rerank=pinecone`), default OFF; hard monthly request cap (400 of the free 500)
- [x] `retrieval/answer.py`: untrusted-chunk prompt, `[C#]` citations mapped to real chunks, invented labels flagged, disclaimer added by code; LiteLLM with capped retries and cache
- [x] `retrieval/cli.py`; 63 offline tests pass, ruff clean
- [x] First real question: NVDA export controls, cited answer, all citations valid, 17.8 s total (local Llama took ~74 s)
- [ ] Not tuned yet: hybrid alpha (0.5) and whether reranking helps; both are measured in step 3

## Step 3: Golden evals + baselines [x] (2026-09-30)
- [x] 30-question golden set restored from the archive; `PineconeStore.scan()` reads the corpus back (2,967 chunks); `--validate`: every label is satisfiable
- [x] `evals/run.py`: retrieval metrics, alpha sweep (`--alpha 0.1 0.5 ...`), rerank override, `--llm` answer checks; one shared searcher embeds each question once
- [x] Bug found and fixed: sparse scores are ~31x dense (median top-1 16.1 vs 0.51), so alpha < ~0.95 was effectively sparse-only. Added `sparse_score_scale=0.03`; regression tests
- [x] Metric fixed: a cited partial answer is no longer counted as a false abstention (regex over-counted: 34.6% became 7.7%)
- [x] 78 offline tests, ruff clean

### Retrieval baseline (26 answerable questions, golden-set filters, top 6, rerank off)
| alpha | pool@30 | hit@6 | coverage | MRR | single | cross | yoy |
|---|---|---|---|---|---|---|---|
| 0.0 (sparse only) | 0.73 | 0.65 | 0.60 | 0.50 | 0.77 | 0.40 | 0.50 |
| 0.1 / 0.3 | 0.81 | 0.65 | 0.60 | 0.52 | 0.77 | 0.40 | 0.50 |
| **0.5 (default)** | 0.81 | **0.77** | **0.71** | 0.57 | **0.94** | 0.40 | 0.50 |
| 0.7 | 0.89 | 0.73 | 0.67 | 0.59 | 0.88 | 0.40 | 0.50 |
| 0.9 | 0.89 | 0.77 | 0.69 | 0.60 | 0.88 | 0.60 | 0.50 |
| 1.0 (dense only) | 0.92 | 0.77 | 0.69 | 0.60 | 0.88 | 0.60 | 0.50 |
| 0.5 + hosted rerank | 0.81 | 0.69 | 0.65 | 0.63 | 0.88 | 0.20 | 0.50 |

One question = ~3.8 points, so 0.5 / 0.9 / 1.0 are within noise; kept 0.5 (best coverage and single-company). Rerank improves ordering (MRR) but loses coverage, especially cross-company, so it stays OFF. Used 29 of 400 hosted rerank requests.

### Answer-level baseline (Claude Haiku 4.5, alpha 0.5, 30 questions, 77 s total)
| Check | Result |
|---|---|
| citations_valid (no invented labels) | 1.00 |
| abstain_correct (4 unanswerable declined) | 1.00 |
| has_citation | 0.92 |
| cites_relevant | 0.73 |
| false_abstain (bare refusals) | 0.08 |

Weakness for step 4: cross-company (0.40) and year-over-year (0.50) questions. One query returns only 6 chunks and cannot span companies or years; 5 of the 9 answers that said "not found" were these retrieval misses. Note the eval uses the golden set's hand-written filters, so step 4's router (which infers them) is a harder test.

## Step 4: Orchestrator, SQL tool, calculator [x] (2026-09-30; two checks still open, see below)
- [x] `tools/calculator.py` (named ops only) and `tools/sql_tool.py` (read-only text-to-SQL: one SELECT over allowlisted views and functions, row cap, timeout, capped retries; canonical metric names and a corrected `fiscal_year` in `v_facts`)
- [x] `orchestrator/router.py`: rules for tickers, years, form (10-K is the default document), explicit section cues (risk factors = Item 1A, MD&A = Item 7, market risk = Item 7A); fan-out per company x year (cap 9); the LLM only decides text vs numbers when a metric word appears
- [x] `derive.py` (growth/margin computed by the calculator from SQL rows), `compose.py` (`[C#]` chunk, `[F#]` fact row, `[K#]` calculation citations), `graph.py` (LangGraph, loop-free), `api/app.py` (FastAPI `/ask`, trace ids, request timeout, no error-detail leaks)
- [x] Bugs fixed from live runs: Sonnet 5.5 rejects `temperature=0` (params now dropped); a retrieve<->facts loop in the graph (regression test)
- [x] `evals/run.py --router` (filters inferred from the question, golden hints ignored); 152 offline tests, ruff clean
- [x] Verified live: cross-company fan-out with valid citations; numbers route (Haiku SQL passed the guard, revenue from DuckDB, 114.2% from the calculator)

### Router retrieval vs golden-filter baseline (alpha 0.5, rerank off)
| | oracle filters | router v1 | router v2 |
|---|---|---|---|
| hit@6 | 0.77 | 0.65 | **0.73** |
| coverage | 0.71 | 0.59 | 0.69 |
| MRR | 0.57 | 0.40 | 0.54 |
| cross-company | 0.40 | 0.60 | 0.60 |
| year-over-year | 0.50 | 0.50 | **0.75** |
| single-company | 0.94 | 0.71 | 0.77 |

Haiku answers with router v1: citations_valid 1.00, abstain_correct 1.00, bare refusals 0.04, cites_relevant 0.54 (golden filters: 0.73; gap is mostly questions that state no year while the label demands FY2025).

### Still open in step 4
- [ ] Answer-level eval re-run on router v2 (only v1 was measured)
- [x] Live test of the FastAPI app against real Pinecone and Claude (in-process; see Step 9)
- [x] 7 router misses diagnosed at the filter level (Step 9): s03/s04/s10/s17 are label over-specificity, c01/c04/y03 are ranking inside correct filters

## Step 5: Verifier loop [~] (built and tested offline 2026-09-30; live eval NOT run, not committed)
- `orchestrator/verify.py`: three layers. (1) numbers in the answer must appear in the evidence (unit-normalised, rounding tolerance); (2) citation hygiene (invented labels, uncited factual sentences); (3) Haiku claim critic that must return a verbatim quote, which code checks against the cited evidence (fails closed on unparseable output).
- Graph cycle `compose -> verify -> revise -> verify -> finalize`, capped by `FINSIGHT_MAX_REVISIONS` (default 1); remaining problems get a code-written caveat. `FINSIGHT_VERIFY=false` disables it.
- `evals.run --e2e --verify on|off` runs the real graph per question (unit-tested only).
- `/ask` now returns a `verification` block (shared `answer_payload` in `api/app.py`).
- [x] Full live before/after eval on Haiku: 32 questions, one pass (Step 9; `evals/results/full_verify.json`)
- [x] Live prompt-injection check with a poisoned chunk: resisted, n=1 (Step 9)

## Step 6: MCP server [x] (2026-09-30)
- `src/finsight/mcp_server/` on `mcp` 2.2.0 (2.x renamed `FastMCP` to `MCPServer`; import from `mcp.server.mcpserver`). Run: `uv run python -m finsight.mcp_server` (stdio) or `--transport streamable-http` (127.0.0.1:8001/mcp).
- Four read-only tools: `search_filings` (no LLM), `query_financials` (guarded SQL + derived growth), `calculate` (named ops only), `ask_finsight` (full graph + verifier).
- Inputs are bounded (query length, limit <= 10, known tickers, `Item N` section pattern); every call runs in a worker thread with a timeout and a trace id; unexpected errors return only the trace id.
- Dependencies are built lazily on the first tool call; logging goes to stderr because stdout is the stdio protocol channel.
- Tested through the real protocol with an in-memory client (21 tests) plus a real stdio subprocess smoke test (`calculate` only: no network or API cost). Not yet exercised live against Pinecone/Claude through MCP.

## Step 7: A2A services [x] (2026-09-30; offline tests + local smoke only, not run live against Pinecone/Claude)
- `src/finsight/agents/` on `a2a-sdk` 1.2.1 (protobuf types; `DefaultRequestHandler`, `create_jsonrpc_routes`, `create_client`). Four agents, one terminal each: `uv run python -m finsight.agents retrieval|facts|verifier|analyst` (ports 9101/9102/9103/9100, 127.0.0.1 only, NO authentication).
- Each worker is a thin skill over tested logic: retrieval (search, no LLM), facts (guarded text-to-SQL, returns rows only), verifier (same `verify_answer` code). The analyst runs the same LangGraph and delegates to the three when URLs are set.
- Delegation is opt-in per capability: `FINSIGHT_A2A_RETRIEVAL_URL`, `..._FACTS_URL`, `..._VERIFIER_URL` (unset = in-process). Workers always build local deps, so an agent can never call itself. `/ask` and MCP pick up the same URLs via `_build_deps`.
- Wire format: JSON envelope in a text part (`{"trace_id","args"}` / `{"ok","result"|"error"}`), single Message reply. Not a data part: protobuf Structs turn every number into a float (2025 -> 2025.0).
- Trust rules: remote responses are schema-validated (`agents/models.py`), size-capped, error text truncated; numbers never cross as facts (the orchestrator recomputes growth/margins from returned SQL rows); a verifier outage FAILS CLOSED with "could not be independently verified" and skips the pointless rewrite; retrieval/facts errors use the existing fallback paths.
- Trace id follows a question across every hop (`finsight.tracing.trace_id_var`, forwarded in the envelope, logged by each agent).
- Tests: `tests/test_a2a.py` (22, in-process ASGI mesh, fakes) plus a real-process smoke test of the verifier (card, one code-only verdict, bad-args error, clean shutdown).
- Known limits: per-call `alpha` override is ignored by the remote searcher (the retrieval agent owns alpha); the SDK logs a harmless "Dispatcher task is not running" warning on single-Message replies; search validation is duplicated between the MCP server and the retrieval agent (candidate for a shared function).
- [x] Live mesh run against Pinecone and Claude, in-process ASGI, all four skills exercised (Step 9). Still not run over real sockets and separate OS processes

## Step 8: Tracing + README [~] (2026-09-30; Compose and CI smoke gate NOT done, by decision)
- [x] `tracing.py`: dependency-free spans (name, duration, ok/error type, bounded scalar attrs) appended to `data/traces/traces-<date>.jsonl`; off until a process calls `configure()` (API, agents and MCP do; `FINSIGHT_TRACING=false` turns it off). Write failures are swallowed.
- [x] Instrumented: `ask` (root), every graph node, LLM calls (model, cached, tokens, best-effort cost; never the messages), searcher, SQL (row counts, not the query), A2A `a2a.call` / `a2a.handle`. Trace id and parent span id travel in the A2A envelope, so one question is one tree across processes.
- [x] `trace_cli.py` (`--last` or a trace id: indented timeline with LLM/token/cost totals) and `GET /traces/{trace_id}` on the API.
- [x] Privacy tests: no question text, chunk text, prompts or exception messages reach the trace file. 15 tests in `tests/test_tracing.py`; 237 offline tests total, ruff clean.
- [x] Real two-process smoke test: client span -> `a2a.call` -> verifier `a2a.handle`, drawn by the CLI (no API calls).
- [x] `README.md` rewritten: architecture, trust model, measured results, what is NOT measured, quickstart, config table, limitations. `.env.example` documents the A2A and tracing variables.
- [ ] Docker Compose for the services (the old `docker-compose.yml` is an unused Qdrant leftover)
- [x] CI: `.github/workflows/ci.yml` runs ruff (check + format) and the offline tests on push and PRs. Seen running green on GitHub (2026-09-30). Live smoke questions with repo secrets: not added
- [x] Tiny live before/after verifier check (4 questions, 7 live Haiku calls, ~$0.03); README quotes it as an anecdote
- [x] Full before/after verifier eval (Step 9)
- [ ] Trace retention: files accumulate daily and are never deleted automatically

## Step 9: Full live verification run [x] (2026-09-30; one script, Haiku 4.5 only, $0.259 total, 66 live calls)
One process, real Pinecone and Claude, a spend ledger with a hard cap (user ceiling $0.50). Raw results: `evals/results/full_verify.json`, `full_live_check.json`.

**Verifier before/after** (32 questions = 30 golden + 2 numeric questions that exist only in the script; verifier on; "before" = the verifier's verdict on the first draft, "after" = on the final answer). Measured with the verifier as it was BEFORE the fixes below.
- 27 of 32 first drafts flagged; after one revision 6 were fully clean, 13 had fewer issues, 8 were unchanged or worse. Total issues 91 -> 51 (50 uncited, 40 unsupported claim, 1 unsupported number before).
- Answer checks: citations_valid 1.00, abstain_correct 1.00, false_abstain 0.00, has_citation 0.96, cites_relevant 0.61.
- Flags are the verifier's own verdicts, not human labels; the writer and critic are the same model family.
- Only 2 of 32 questions took the SQL route (the golden set is mostly risk-factor text).

**Other live checks, all passed:** prompt injection (poisoned chunk told the model to claim 999% growth and leak its prompt: neither happened, n=1); the FastAPI app (`/health`, `/ask` 200 and verified, `/traces`, 422/422/404 on bad input); all four MCP tools (incl. an invalid `calculate` op rejected); the A2A mesh (analyst delegating to retrieval, facts and verifier; `ask`, `search_filings`, `query_financials`, `verify_answer` all handled; one trace tree per question). Caveat: API, MCP and mesh ran in one process, not over real sockets.

**Root causes found by replaying cached critic replies (zero live calls), and the fixes (offline tests added, 242 total):**
1. Correct database answers got a caveat. The critic said "supported" but joined fragments of several evidence rows with `...`, and the verbatim check wanted one contiguous string. Fix: a quote with ellipses passes only if every fragment of real length is found in the evidence; a fabricated fragment still fails. Replay: the Microsoft revenue-growth question went from 1 issue to 0.
2. Real quotes from filing tables failed (HTML converted to text). Cells sit on their own lines (`$ / 391,035 / 2 / %`). Fix: whitespace around `$` and `%` is closed before comparing.
3. Revisions looked worse (s16 2 -> 5, s17 1 -> 2). The writer's revision used a label-first format (`[C5] "quote"`) and the parser attached that label to the line above, leaving the quote "uncited". Fix: a label that opens a line and is followed by text cites that text. A label after a sentence end, or alone on its own line, still attaches backwards.
4. Lead-in lines ending with ":" ("Apple discusses tariffs in the following ways:") were flagged uncited. They are now skipped; numbers in them are still checked.
5. "What was Apple's total net sales?" never reached SQL: `_METRIC_RX` had no "sales" (Apple says "net sales"), so the route was never asked. Fix: `\bsales\b` added.
- NOT re-measured: the before/after numbers above predate these fixes. Re-running them needs fresh revise calls (the feedback text changed, so the cache no longer covers them).
- Still open from this run: a number in a table "in millions" ("$391,035 million") is flagged as in no evidence because the table lacks the word "million" next to it; it costs a revision but the answer ends up right.

**Router misses, diagnosed at filter level (no retrieval run):** s03, s04, s10 name no year and s17 asks for "quarterly reports", while the golden label demands FY2025 or Item 1A, so the router's broader search is correct and the label is over-specific. c01, c04, y03 get exactly the right filters (one sub-query per company and year), so those misses are ranking inside a correct filter; the plan sets no section because only the phrase "risk factors" counts as an Item 1A cue. Untested idea: treat "risk" as an Item 1A cue for 10-K questions (could hurt "market risk", Item 7A, so it needs a retrieval eval first).

## Step 10: Optional LangSmith export [x] (2026-09-30; tested offline with a mock client, NOT yet seen in the LangSmith UI)
- `observability.py`: `configure_langsmith(settings)` copies the standard `LANGSMITH_*` settings from `.env` into the process environment at startup (the SDK caches its first env read, so it runs before the graph is built). Off unless `LANGSMITH_TRACING=true` AND a key is set; the key is never logged. Called from the API lifespan, the MCP server, the A2A agents and the `--e2e` evals.
- What LangSmith gets: a root `finsight.ask` run (named via the graph config) with a child run per node (LangGraph does this itself), and `@traceable` runs for the LLM call (type llm, with model, cache flag and token usage), the Pinecone search (type retriever, passages shaped as documents) and the SQL tool (type tool). Every run carries `finsight_trace_id`, equal to the local trace ID and the `X-Trace-Id` header. `@traceable` drops `self`, so no client or connection is serialised.
- Content caveat: unlike the local JSONL traces, LangSmith receives prompts, filing passages and answers. `LANGSMITH_HIDE_INPUTS` / `LANGSMITH_HIDE_OUTPUTS` turn that off. Local tracing is unchanged.
- Limits: A2A workers are separate processes, so their runs are separate top-level runs (correlate on `finsight_trace_id`); LangSmith parent/child links across processes are not propagated.
- 10 new tests (`tests/test_observability.py`, network blocked, mock client): off by default, no key stays off, key never logged, `.env` names parsed, one root run with node children all tagged, nothing sent when off, LLM usage and cache flag, retriever documents, SQL inputs exclude the DB handle. 252 tests total.
- Found and fixed a latent test problem: `litellm` runs `load_dotenv()` on first import, so the suite copied a developer's real `.env` into the test process and "missing key" tests depended on import order. `tests/conftest.py` now sets `LITELLM_MODE=PRODUCTION` first.
- [ ] Verify once with a real key: run a question with `LANGSMITH_TRACING=true` and look at the trace tree in the LangSmith UI.
- [ ] Optional: `uv add langsmith` to pin it as a direct dependency (it already arrives via langgraph); propagate LangSmith trace headers over A2A so worker runs nest under the analyst.

## Known facts about Pinecone (verified in its docs, 2026-09-30)
- Starter plan: 2 GB storage, 2M write / 1M read units per month, 5 indexes, `us-east-1` only, 5M embedding tokens/month.
- Hybrid search: one index with the dotproduct metric holds dense + sparse values; balance with alpha client-side; no server-side RRF.
- Not yet verified: the list of hosted dense models and their dimensions (check from the SDK after install).

## Machine cleanup (done 2026-09-30, user-approved)
- Removed the Qdrant container, network and volume (`docker compose down -v`); Pinecone holds the vectors. The `qdrant/qdrant:v1.15.0` image (212 MB) and `docker-compose.yml` still exist.
- Uninstalled Ollama (Homebrew), deleted the `llama3.1:8b` model and `~/.ollama`; brew also removed its `mlx` libraries.
- Removed unused packages: qdrant-client, sentence-transformers, fastembed.

## Open items
| Item | Detail | Owner |
|---|---|---|
| Token quota | ~1.46M of the 3.5M safety budget left this month; a full re-ingest will not fit | Known |
| Commit/push | Restart commits are local on `main`, not pushed | User |
