"""Query planning: infer filters from the question and split it into sub-queries.

Why this exists: one hybrid search returns ~6 chunks, so "compare Apple and Microsoft" or "FY2024
vs FY2025" can only ever surface one side. The plan turns such a question into one sub-query per
(company, year), each with its own metadata pre-filter, so every side gets retrieval slots.

Entities (tickers, years, form type) come from deterministic rules: cheap, testable and they
cannot hallucinate. The LLM is used for one small judgement only: is a metric-flavoured question
asking for a NUMBER (route to the SQL tool) or for an explanation (route to the filings)?
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

LLM = Callable[[list[dict]], str]
Route = Literal["text", "facts", "both"]

MAX_SUBQUERIES = 9  # e.g. 3 companies x 3 years; bounds cost for any question

_TICKER_ALIASES = {
    "AAPL": ("apple", "aapl"),
    "MSFT": ("microsoft", "msft"),
    "NVDA": ("nvidia", "nvda"),
}
_TICKER_RX = {
    t: re.compile(r"\b(?:" + "|".join(names) + r")\b", re.IGNORECASE)
    for t, names in _TICKER_ALIASES.items()
}
# Digit lookarounds instead of \b so "FY2019" matches but "12345" does not.
_YEAR_RX = re.compile(r"(?<!\d)(20[12]\d)(?!\d)")
_QUARTERLY_RX = re.compile(r"\b(10-?q|quarterly|quarter)\b", re.IGNORECASE)
# Explicit cues only. "Risk factors" is Item 1A in both forms; MD&A and market risk move between
# 10-K (Item 7 / 7A) and 10-Q (Item 2 / 3). No cue -> no section filter (a wrong one hurts more).
_SECTION_CUES = [
    (re.compile(r"\brisk factors?\b", re.IGNORECASE), {"10-K": "Item 1A", "10-Q": "Item 1A"}),
    (re.compile(r"\bmarket risk\b", re.IGNORECASE), {"10-K": "Item 7A", "10-Q": "Item 3"}),
    (
        re.compile(r"\bmd&a\b|\bmanagement['\u2019]s discussion\b", re.IGNORECASE),
        {"10-K": "Item 7", "10-Q": "Item 2"},
    ),
]
# Only a GATE: a metric word makes the LLM classifier worth calling; it does not decide the route.
_METRIC_RX = re.compile(
    r"revenue|\bsales\b|net income|profit|margin|operating income|earnings per share|\beps\b|"
    r"total assets|liabilit|equity|cash flow|capex|capital expenditure|"
    r"r&d|research and development|repurchase|\bdebt\b|\bcash\b",
    re.IGNORECASE,
)

ROUTE_PROMPT = """Classify the question. Answer with exactly one word.
facts: it asks for a specific financial number (a reported amount, a ratio, a growth rate).
text: it asks for explanation, drivers, risks, descriptions or comparisons of language.
both: it needs a number AND an explanation.
Examples:
"What was Apple's net income in fiscal 2024?" -> facts
"What drove NVIDIA's Data Center revenue growth?" -> text
"How much did Microsoft's revenue grow and why?" -> both"""


@dataclass(frozen=True)
class SubQuery:
    question: str
    tickers: tuple[str, ...] = ()
    fiscal_year: int | None = None
    form_type: str | None = None
    section: str | None = None


@dataclass
class Plan:
    question: str
    route: Route
    tickers: list[str]
    years: list[int]
    form_type: str
    section: str | None
    sub_queries: list[SubQuery] = field(default_factory=list)
    truncated: bool = False  # True if the sub-query cap dropped combinations


def extract_tickers(question: str) -> list[str]:
    return [t for t, rx in _TICKER_RX.items() if rx.search(question)]


def extract_years(question: str) -> list[int]:
    return sorted({int(y) for y in _YEAR_RX.findall(question)})


def extract_form(question: str) -> str:
    """The annual report (10-K) is the default document; quarterly questions say so explicitly.
    Without this default, near-duplicate 10-Q passages crowd the top results."""
    return "10-Q" if _QUARTERLY_RX.search(question) else "10-K"


def extract_section(question: str, form: str) -> str | None:
    for rx, by_form in _SECTION_CUES:
        if rx.search(question):
            return by_form[form]
    return None


def mentions_metric(question: str) -> bool:
    return bool(_METRIC_RX.search(question))


def classify_route(question: str, llm: LLM | None) -> Route:
    """Cheap-model judgement, called only for metric-flavoured questions. Fails safe to 'text'."""
    if llm is None or not mentions_metric(question):
        return "text"
    reply = llm(
        [{"role": "system", "content": ROUTE_PROMPT}, {"role": "user", "content": question}]
    )
    word = re.search(r"\b(facts|text|both)\b", reply.lower())
    return word.group(1) if word else "text"  # type: ignore[return-value]


def plan_question(question: str, llm: LLM | None = None) -> Plan:
    tickers = extract_tickers(question)
    years = extract_years(question)
    form = extract_form(question)
    section = extract_section(question, form)

    # One sub-query per (company, year). No company named -> a single unfiltered search.
    combos = list(product(tickers or [None], years or [None]))
    truncated = len(combos) > MAX_SUBQUERIES
    subs = [
        SubQuery(question, (t,) if t else (), y, form, section) for t, y in combos[:MAX_SUBQUERIES]
    ]
    return Plan(
        question=question,
        route=classify_route(question, llm),
        tickers=tickers,
        years=years,
        form_type=form,
        section=section,
        sub_queries=subs,
        truncated=truncated,
    )
