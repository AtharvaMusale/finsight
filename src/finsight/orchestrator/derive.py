"""Deterministic derived numbers: growth and margins computed from SQL rows by the calculator.

The answer LLM is never asked to do arithmetic. Anything like "revenue grew 6.4%" is computed here
from the exact figures the SQL tool returned, then handed to the LLM as a citable fact.
"""

from collections import defaultdict
from dataclasses import dataclass

from finsight.tools.calculator import CalcResult, CalculatorError, compute

_MARGIN_PARTS = ("GrossProfit", "OperatingIncome", "NetIncome")
_REQUIRED = {"ticker", "metric", "period_end", "value"}


@dataclass(frozen=True)
class Derived:
    label: str  # human-readable description of what was computed
    result: CalcResult
    from_rows: tuple[int, ...]  # 1-based indexes of the SQL rows used (become [F#] labels)


def derive_calcs(columns: list[str], rows: list[list]) -> list[Derived]:
    if not set(columns) >= _REQUIRED:
        return []
    recs = [dict(zip(columns, r, strict=True)) | {"_i": i} for i, r in enumerate(rows, 1)]
    out: list[Derived] = []

    # Growth between consecutive full-year (or balance-sheet) points of the same series.
    series: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("period_type", "annual") in ("annual", "instant"):
            series[(r["ticker"], r["metric"], r.get("unit"))].append(r)
    for (ticker, metric, _unit), pts in series.items():
        pts.sort(key=lambda r: r["period_end"])
        for a, b in zip(pts, pts[1:], strict=False):
            try:
                res = compute("pct_change", old=a["value"], new=b["value"])
            except CalculatorError:
                continue
            label = f"{ticker} {metric} % change, {a['period_end']} to {b['period_end']}"
            out.append(Derived(label, res, (a["_i"], b["_i"])))

    # Margins: a profit line as a percent of revenue for the same company and period.
    by_period: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        by_period[(r["ticker"], r["period_end"], r.get("period_type"))][r["metric"]] = r
    for (ticker, end, _ptype), metrics in by_period.items():
        rev = metrics.get("Revenue")
        for part in _MARGIN_PARTS:
            if rev and part in metrics:
                try:
                    res = compute("margin", part=metrics[part]["value"], whole=rev["value"])
                except CalculatorError:
                    continue
                label = f"{ticker} {part} margin (% of Revenue), period ending {end}"
                out.append(Derived(label, res, (metrics[part]["_i"], rev["_i"])))
    return out
