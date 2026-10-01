"""Optional LangSmith export. Off unless you turn it on; the local JSONL tracing is unaffected.

Turn it on by putting these in `.env` (you add the key yourself; it is never logged):
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=...
  LANGSMITH_PROJECT=finsight          # optional
  LANGSMITH_ENDPOINT=...              # optional (for example the EU endpoint)
  LANGSMITH_HIDE_INPUTS=true          # optional: send no inputs (prompts, questions, filing text)
  LANGSMITH_HIDE_OUTPUTS=true         # optional: send no outputs (answers, retrieved passages)

What you get in LangSmith: one `finsight.ask` run per question with a child run per graph node
(LangGraph does this itself), and inside them the LLM calls (with token usage), the Pinecone
search (as a retriever, with its passages) and the SQL tool. Every run carries the same
`finsight_trace_id` as the local trace files and the API's `X-Trace-Id` header, so one question
can be followed across both. Worker agents run as separate processes, so their runs are separate
top-level runs; filter on `finsight_trace_id` to line them up.

Unlike the local trace files, LangSmith receives CONTENT (prompts, filing passages, answers)
unless you set the hide flags above. It is opt-in for that reason.

`langsmith` ships as a dependency of langgraph. If it were ever missing, every helper here is a
no-op so nothing else breaks.
"""

import logging
import os
from collections.abc import Callable

from finsight.config import Settings
from finsight.tracing import trace_id_var

log = logging.getLogger("finsight.observability")

try:
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:  # pragma: no cover - langsmith always arrives with langgraph
    _traceable = None

    def get_current_run_tree():  # type: ignore[misc]
        return None


def traceable(**kwargs) -> Callable:
    """`langsmith.traceable(**kwargs)` as a decorator, or a no-op if langsmith is unavailable.

    Tracing is decided per call: with LangSmith off, a decorated function just runs."""

    def decorate(fn: Callable) -> Callable:
        return _traceable(**kwargs)(fn) if _traceable else fn

    return decorate


def configure_langsmith(s: Settings) -> bool:
    """Export the LangSmith settings to the process environment, once, at startup.

    The LangSmith SDK reads its configuration from environment variables and caches the first
    read, so this must run before the first traced call. Returns True only when export is on."""
    if not s.langsmith_tracing:
        return False
    if _traceable is None:  # pragma: no cover
        log.warning(
            "LANGSMITH_TRACING is on but the langsmith package is missing; export stays off"
        )
        return False
    if s.langsmith_api_key is None:
        log.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is not set; export stays off")
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    if s.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = s.langsmith_endpoint
    for flag, on in (
        ("LANGSMITH_HIDE_INPUTS", s.langsmith_hide_inputs),
        ("LANGSMITH_HIDE_OUTPUTS", s.langsmith_hide_outputs),
    ):
        if on:
            os.environ[flag] = "true"
    log.info(
        "LangSmith export on (project=%s, inputs %s, outputs %s)",
        s.langsmith_project,
        "hidden" if s.langsmith_hide_inputs else "sent",
        "hidden" if s.langsmith_hide_outputs else "sent",
    )  # the key is never logged
    return True


def langsmith_config(trace_id: str) -> dict:
    """RunnableConfig for `graph.invoke`: names the root run and ties it to our trace ID."""
    return {
        "run_name": "finsight.ask",
        "tags": ["finsight"],
        "metadata": {"finsight_trace_id": trace_id},
    }


def annotate(**metadata: object) -> None:
    """Attach our trace ID (and any extras) to the LangSmith run that is currently open."""
    try:
        run = get_current_run_tree()
        if run is not None:
            run.add_metadata({"finsight_trace_id": trace_id_var.get(), **metadata})
    except Exception:  # observability must never break a request
        log.debug("could not annotate the LangSmith run", exc_info=True)


def set_llm_usage(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """Report token counts on the open LLM run so LangSmith can show usage and cost."""
    try:
        run = get_current_run_tree()
        if run is not None and prompt_tokens is not None and completion_tokens is not None:
            run.set(
                usage_metadata={
                    "input_tokens": int(prompt_tokens),
                    "output_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens) + int(completion_tokens),
                }
            )
    except Exception:
        log.debug("could not set LangSmith usage", exc_info=True)
