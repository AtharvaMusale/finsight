"""Pure scoring functions (no I/O), so they are easy to unit test.

A golden question has `targets`: (ticker, form_type, fiscal_year, section, pattern). A retrieved
chunk is RELEVANT to a target when its metadata matches every field the target sets and its text
matches the regex. This gives objective retrieval scoring without an LLM judge.
"""

import re
from dataclasses import dataclass

from finsight.schemas import Hit

_ABSTAIN = re.compile(
    r"could(n't| not) (find|locate)|not (found|contain|mentioned|available|provided)|"
    r"no (relevant|information)|do(es)? not (contain|include|provide|mention)|"
    r"cannot (answer|determine)|unable to",
    re.IGNORECASE,
)


def matches(hit: Hit, target: dict) -> bool:
    if hit.ticker != target["ticker"]:
        return False
    for field in ("form_type", "fiscal_year", "section"):
        if field in target and getattr(hit, field) != target[field]:
            return False
    return (
        bool(re.search(target["pattern"], hit.text, re.IGNORECASE)) if "pattern" in target else True
    )


@dataclass
class RetrievalScore:
    hit: bool  # at least one target covered
    coverage: float  # fraction of targets covered
    first_rank: int | None  # 1-based rank of the first relevant chunk (None if none)

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_rank if self.first_rank else 0.0


def score_retrieval(targets: list[dict], hits: list[Hit]) -> RetrievalScore:
    covered = [any(matches(h, t) for h in hits) for t in targets]
    first = next((i for i, h in enumerate(hits, 1) if any(matches(h, t) for t in targets)), None)
    return RetrievalScore(hit=any(covered), coverage=sum(covered) / len(targets), first_rank=first)


def is_abstention(answer_text: str) -> bool:
    return bool(_ABSTAIN.search(answer_text))


def score_answer(
    targets: list[dict],
    expect_abstain: bool,
    answer_text: str,
    cited: list[Hit],
    invalid_citations: list[str],
) -> dict:
    """Answer-level checks that need no judge model. Faithfulness proper comes in step 5."""
    abstained = is_abstention(answer_text)
    if expect_abstain:
        return {"abstain_correct": abstained}
    # A refusal has no evidence behind it. An answer that cites chunks but says "the excerpts do
    # not include X" is a partial answer, not an abstention (regex alone over-counted these).
    return {
        "false_abstain": abstained and not cited,
        "has_citation": bool(cited),
        "citations_valid": not invalid_citations,
        "cites_relevant": any(matches(h, t) for h in cited for t in targets),
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
