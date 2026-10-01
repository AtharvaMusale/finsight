"""Print one trace as a timeline tree.

Run from the finsight/ folder:
  uv run python -m finsight.trace_cli --last
  uv run python -m finsight.trace_cli <trace_id>
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from finsight.config import get_settings
from finsight.tracing import latest_trace_id, read_spans


def _attrs(attrs: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in attrs.items() if v not in (None, ""))


def render(spans: list[dict]) -> str:
    if not spans:
        return "no spans found"
    by_id = {s["span"]: s for s in spans}
    kids: dict[str | None, list[dict]] = defaultdict(list)
    for s in spans:
        # A span whose parent was never recorded (another process, older file) is shown as a root.
        kids[s["parent"] if s.get("parent") in by_id else None].append(s)
    for group in kids.values():
        group.sort(key=lambda s: s.get("ts", 0))

    lines: list[str] = []

    def walk(span: dict, depth: int) -> None:
        mark = "" if span.get("ok", True) else f"  !! {span.get('error')}"
        name = f"{'  ' * depth}{span['name']}"
        lines.append(
            f"{name:<34}{span['ms']:>9.1f} ms  [{span['service']}]  {_attrs(span['attrs'])}{mark}"
        )
        for child in kids.get(span["span"], []):
            walk(child, depth + 1)

    for root in kids[None]:
        walk(root, 0)

    start = min(s["ts"] for s in spans)
    end = max(s["ts"] + s["ms"] / 1000 for s in spans)
    llm = [s for s in spans if s["name"] == "llm"]
    live = [s for s in llm if not s["attrs"].get("cached")]
    tokens_in = sum(s["attrs"].get("prompt_tokens") or 0 for s in live)
    tokens_out = sum(s["attrs"].get("completion_tokens") or 0 for s in live)
    cost = sum(s["attrs"].get("cost_usd") or 0 for s in live)
    errors = sum(1 for s in spans if not s.get("ok", True))
    header = (
        f"trace {spans[0]['trace']}   wall {((end - start) * 1000):.0f} ms   spans {len(spans)}"
    )
    footer = (
        f"llm calls {len(llm)} ({len(llm) - len(live)} cached)   tokens in/out "
        f"{tokens_in}/{tokens_out}   est. cost ${cost:.5f}   errors {errors}"
    )
    return "\n".join([header, "-" * 100, *lines, "-" * 100, footer])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="finsight-trace")
    ap.add_argument("trace_id", nargs="?")
    ap.add_argument("--last", action="store_true", help="the most recent trace")
    ap.add_argument("--dir", type=Path, help="trace folder (default: <data_dir>/traces)")
    args = ap.parse_args(argv)

    directory = args.dir or get_settings().trace_dir
    if not directory.exists():
        print(f"no traces yet in {directory} (servers write them while tracing is on)")
        return 1
    trace_id = args.trace_id or (latest_trace_id(directory) if args.last else None)
    if not trace_id:
        print("give a trace id, or --last", file=sys.stderr)
        return 2
    spans = read_spans(directory, trace_id)
    print(render(spans))
    return 0 if spans else 1


if __name__ == "__main__":
    raise SystemExit(main())
