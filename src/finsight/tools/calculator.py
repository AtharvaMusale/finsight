"""Deterministic calculator: every derived number in an answer comes from here, not the LLM.

Only named operations are exposed (no eval, no free-form expressions), so the model can choose
WHICH calculation to run and on WHICH inputs, but cannot change the arithmetic.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass


class CalculatorError(ValueError):
    """Bad operation name, bad inputs, or an undefined result (e.g. division by zero)."""


@dataclass(frozen=True)
class CalcResult:
    op: str
    inputs: dict[str, float]
    value: float
    unit: str  # "ratio", "percent" or "percent_per_year"
    formula: str  # shown to the user so the arithmetic is auditable


def _num(name: str, x: object) -> float:
    if isinstance(x, bool) or not isinstance(x, int | float):
        raise CalculatorError(f"{name} must be a number, got {x!r}")
    if not math.isfinite(x):
        raise CalculatorError(f"{name} must be finite, got {x!r}")
    return float(x)


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise CalculatorError("denominator is zero")
    return numerator / denominator


def pct_change(old: float, new: float) -> float:
    """Percent change from old to new. abs(old) keeps the sign right for negative bases."""
    if old == 0:
        raise CalculatorError("old value is zero; percent change is undefined")
    return (new - old) / abs(old) * 100


def margin(part: float, whole: float) -> float:
    """part as a percent of whole, e.g. margin(gross_profit, revenue)."""
    return ratio(part, whole) * 100


def cagr(begin: float, end: float, years: float) -> float:
    """Compound annual growth rate, in percent per year."""
    if years <= 0:
        raise CalculatorError("years must be positive")
    if begin <= 0 or end <= 0:
        raise CalculatorError("CAGR needs positive begin and end values")
    return ((end / begin) ** (1 / years) - 1) * 100


# op name -> (function, argument names, unit, formula template)
_OPS: dict[str, tuple[Callable[..., float], tuple[str, ...], str, str]] = {
    "ratio": (ratio, ("numerator", "denominator"), "ratio", "{numerator} / {denominator}"),
    "pct_change": (pct_change, ("old", "new"), "percent", "({new} - {old}) / |{old}| * 100"),
    "margin": (margin, ("part", "whole"), "percent", "{part} / {whole} * 100"),
    "cagr": (
        cagr, ("begin", "end", "years"), "percent_per_year",
        "(({end} / {begin}) ^ (1 / {years}) - 1) * 100",
    ),
}  # fmt: skip

OPERATIONS = tuple(_OPS)


def compute(op: str, **args: float) -> CalcResult:
    if op not in _OPS:
        raise CalculatorError(f"unknown operation {op!r}; allowed: {', '.join(OPERATIONS)}")
    fn, names, unit, template = _OPS[op]
    if set(args) != set(names):
        raise CalculatorError(f"{op} needs exactly these inputs: {', '.join(names)}")
    inputs = {n: _num(n, args[n]) for n in names}
    value = fn(**inputs)
    return CalcResult(
        op=op,
        inputs=inputs,
        value=round(value, 4),
        unit=unit,
        formula=template.format(**inputs),
    )
