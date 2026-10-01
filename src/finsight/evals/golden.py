"""Load the golden set and validate its labels against the real corpus."""

import json
import re
from pathlib import Path

GOLDEN_PATH = Path("evals/golden.jsonl")


def load_golden(path: Path = GOLDEN_PATH, smoke_only: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["smoke"]] if smoke_only else rows


def load_corpus(store) -> list[dict]:
    """Every chunk currently in the vector index (metadata + text)."""
    return list(store.scan())


def count_matching(corpus: list[dict], target: dict) -> int:
    rx = re.compile(target["pattern"], re.IGNORECASE)
    n = 0
    for p in corpus:
        if p["ticker"] != target["ticker"] or not rx.search(p["text"]):
            continue
        if all(p[f] == target[f] for f in ("form_type", "fiscal_year", "section") if f in target):
            n += 1
    return n


def validate(rows: list[dict], corpus: list[dict]) -> list[str]:
    """Return problems: a target that matches no chunk can never be satisfied (bad label)."""
    problems = []
    for r in rows:
        if r["expect_abstain"] and r["targets"]:
            problems.append(f"{r['id']}: abstain question should have no targets")
        if not r["expect_abstain"] and not r["targets"]:
            problems.append(f"{r['id']}: answerable question has no targets")
        for t in r["targets"]:
            if count_matching(corpus, t) == 0:
                problems.append(f"{r['id']}: target matches NO chunk in corpus: {t}")
    return problems
