"""Lightweight tracing: one span per step, appended to a JSONL file, no dependency or backend.

A trace is every span that shares a trace id. `ask()` sets the id for a graph run; A2A calls carry
it (and the calling span's id) in the request envelope, so a question that crosses the analyst,
retrieval, facts and verifier processes is one tree. Read it back with:
  uv run python -m finsight.trace_cli --last          # newest trace as a timeline
  uv run python -m finsight.trace_cli <trace_id>

Rules this module enforces:
  * Tracing never breaks a request: write failures are logged once and swallowed.
  * Spans hold counts, sizes, model names and error TYPES. Never prompts, filing text, SQL
    results, keys, or exception messages (messages can contain secrets).
  * Off until a process calls `configure()` (servers do; unit tests do not), so importing the
    library has no side effects.
"""

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from pathlib import Path

log = logging.getLogger("finsight.tracing")

trace_id_var: ContextVar[str] = ContextVar("finsight_trace_id", default="-")
_span_var: ContextVar[str | None] = ContextVar("finsight_span_id", default=None)

_MAX_STR = 200


def _clean(value: object) -> object:
    if isinstance(value, bool | int | float) or value is None:
        return value
    return str(value)[:_MAX_STR]


class _Tracer:
    def __init__(self) -> None:
        self.enabled = False
        self.service = "finsight"
        self.directory: Path | None = None
        self._lock = threading.Lock()
        self._warned = False

    def configure(self, service: str, directory: Path | None, enabled: bool = True) -> None:
        self.service, self.directory = service, directory
        self.enabled = bool(enabled and directory is not None)

    def disable(self) -> None:
        self.enabled, self.directory = False, None

    def _write(self, record: dict) -> None:
        if self.directory is None:
            return
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f"traces-{date.today().isoformat()}.jsonl"
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except OSError:
            if not self._warned:  # once: a full disk must not flood the log either
                self._warned = True
                log.warning("could not write trace spans; tracing continues without them")

    @contextmanager
    def span(self, name: str, **attrs: object) -> Iterator[dict]:
        """Time a block. Yields a dict; add attributes (counts, flags) before the block ends."""
        bag: dict = dict(attrs)
        if not self.enabled:
            yield bag
            return
        span_id, parent = uuid.uuid4().hex[:16], _span_var.get()
        token = _span_var.set(span_id)
        start, t0, error = time.time(), time.perf_counter(), None
        try:
            yield bag
        except BaseException as e:
            error = type(e).__name__  # the type only: messages can carry secrets
            raise
        finally:
            _span_var.reset(token)
            self._write(
                {
                    "trace": trace_id_var.get(), "span": span_id, "parent": parent,
                    "name": name, "service": self.service, "ts": round(start, 6),
                    "ms": round((time.perf_counter() - t0) * 1000, 2),
                    "ok": error is None, "error": error,
                    "attrs": {k: _clean(v) for k, v in bag.items()},
                }
            )  # fmt: skip


tracer = _Tracer()
span = tracer.span


def configure(service: str, directory: Path | None, enabled: bool = True) -> None:
    tracer.configure(service, directory, enabled)


def current_span_id() -> str | None:
    return _span_var.get()


@contextmanager
def adopt(trace_id: str, parent_span_id: str | None) -> Iterator[None]:
    """Continue a trace started in another process (an A2A request's envelope)."""
    t_token, s_token = trace_id_var.set(trace_id), _span_var.set(parent_span_id)
    try:
        yield
    finally:
        _span_var.reset(s_token)
        trace_id_var.reset(t_token)


# ---------------------------------------------------------------- reading


def read_spans(directory: Path, trace_id: str | None = None, files: int = 3) -> list[dict]:
    """Spans from the newest `files` daily files, optionally one trace. Bad lines are skipped."""
    out: list[dict] = []
    for path in sorted(directory.glob("traces-*.jsonl"))[-files:]:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and (trace_id is None or rec.get("trace") == trace_id):
                out.append(rec)
    return out


def latest_trace_id(directory: Path) -> str | None:
    """The trace whose root span started last (roots: spans with no parent)."""
    roots = [r for r in read_spans(directory) if r.get("parent") is None and "ts" in r]
    return max(roots, key=lambda r: r["ts"])["trace"] if roots else None
