"""Read-only text-to-SQL over an allowlisted schema.

The LLM writes SQL; nothing it writes is trusted. Defence in depth:
  1. AST check: DuckDB's own parser (json_serialize_sql) must see exactly ONE plain SELECT that
     touches only the allowlisted views and calls only allowlisted functions.
  2. Engine limits: read-only connection, external file access off, config locked, row cap,
     and a timeout that interrupts the running query.
  3. The model only sees `v_facts` / `v_filings`, never the raw `facts` table.
"""

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb

from finsight.observability import annotate, traceable
from finsight.tracing import span

LLM = Callable[[list[dict]], str]

ALLOWED_TABLES = frozenset({"v_facts", "v_filings"})
ALLOWED_FUNCTIONS = frozenset({
    "count", "count_star", "sum", "avg", "min", "max", "median", "stddev", "any_value",
    "arg_max", "arg_min", "first", "last", "round", "abs", "ceil", "floor", "coalesce", "nullif",
    "greatest", "least", "lag", "lead", "row_number", "rank", "dense_rank", "lower", "upper",
    "date_part", "year", "month", "date_diff", "strftime",
})  # fmt: skip
MAX_SQL_CHARS = 2000
MAX_ROWS = 200

# Canonical metric name -> XBRL us-gaap tags. Companies tag the same idea differently (NVDA uses
# `Revenues`, AAPL and MSFT use `RevenueFromContractWithCustomerExcludingAssessedTax`), so the
# view maps them to one name and the LLM never has to know the tags.
METRIC_TAGS: dict[str, tuple[str, ...]] = {
    "Revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    "NetIncome": ("NetIncomeLoss",),
    "GrossProfit": ("GrossProfit",),
    "OperatingIncome": ("OperatingIncomeLoss",),
    "RnDExpense": ("ResearchAndDevelopmentExpense",),
    "DilutedEPS": ("EarningsPerShareDiluted",),
    "TotalAssets": ("Assets",),
    "TotalLiabilities": ("Liabilities",),
    "StockholdersEquity": ("StockholdersEquity",),
    "Cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "LongTermDebt": ("LongTermDebt",),
    "OperatingCashFlow": ("NetCashProvidedByUsedInOperatingActivities",),
    "CapEx": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "ShareRepurchases": ("PaymentsForRepurchaseOfCommonStock",),
}


def _metric_case() -> str:
    """Built only from constants above (no user input), so string formatting is safe here."""
    whens = " ".join(
        f"WHEN tag = '{tag}' THEN '{name}'" for name, tags in METRIC_TAGS.items() for tag in tags
    )
    return f"CASE {whens} ELSE tag END"


_VIEWS = f"""
CREATE OR REPLACE TEMP VIEW v_facts AS
WITH canon AS (
    SELECT ticker, {_metric_case()} AS metric, unit, period_start, period_end, value,
           form_type, accession_no, filed
    FROM facts
), ranked AS (
    -- The same period is repeated in later filings (comparatives, restatements): keep the latest.
    SELECT *, row_number() OVER (
        PARTITION BY ticker, metric, unit, period_start, period_end
        ORDER BY filed DESC, accession_no DESC
    ) AS rn
    FROM canon
), typed AS (
    SELECT *, CASE
        WHEN period_start = period_end THEN 'instant'
        WHEN date_diff('day', period_start, period_end) BETWEEN 350 AND 380 THEN 'annual'
        WHEN date_diff('day', period_start, period_end) BETWEEN 80 AND 100 THEN 'quarter'
        ELSE 'other' END AS period_type
    FROM ranked WHERE rn = 1
)
SELECT t.ticker, t.metric, t.unit, t.period_start, t.period_end, t.period_type,
       CASE WHEN t.period_type IN ('annual', 'instant') THEN k.fiscal_year END AS fiscal_year,
       t.value, t.form_type, t.accession_no, t.filed
FROM typed t
LEFT JOIN (
    SELECT ticker, report_date, min(fiscal_year) AS fiscal_year
    FROM filings WHERE form_type = '10-K' GROUP BY ticker, report_date
) k ON k.ticker = t.ticker AND k.report_date = t.period_end;

CREATE OR REPLACE TEMP VIEW v_filings AS
SELECT ticker, form_type, fiscal_year, filing_date, report_date, accession_no FROM filings;
"""

SCHEMA_PROMPT = """You write DuckDB SQL over two read-only views.

v_facts: one row per reported XBRL number (latest filed value wins).
  ticker VARCHAR          -- 'AAPL', 'MSFT', 'NVDA'
  metric VARCHAR          -- Revenue, NetIncome, GrossProfit, OperatingIncome, RnDExpense,
                          -- DilutedEPS, TotalAssets, TotalLiabilities, StockholdersEquity, Cash,
                          -- LongTermDebt, OperatingCashFlow, CapEx, ShareRepurchases
  unit VARCHAR            -- 'USD' (full dollars, not millions), 'USD/shares', 'shares'
  period_start, period_end DATE
  period_type VARCHAR     -- 'annual' (full year), 'quarter' (3 months), 'instant' (balance
                          -- sheet date), 'other' (e.g. 6/9-month year-to-date)
  fiscal_year INTEGER     -- company fiscal year, only for annual/instant rows of a year-end
                          -- covered by a 10-K we hold; NULL otherwise
  value DOUBLE
  form_type VARCHAR       -- '10-K' or '10-Q'
  accession_no VARCHAR    -- the source filing
  filed DATE

v_filings: ticker, form_type, fiscal_year, filing_date, report_date, accession_no.

Rules:
- Output exactly one SELECT (a WITH clause is fine) inside a ```sql block. No other text.
- Use only these two views. Always filter unit = 'USD' for dollar amounts.
- Full-year figures: period_type = 'annual'. Balance sheet items: period_type = 'instant'.
- ALWAYS select ticker, metric, period_end, value AND accession_no so each number is citable.
- Use fiscal_year to pick a year; never guess dates. Add ORDER BY and keep results small."""


class SqlToolError(RuntimeError):
    """The SQL was rejected, failed, or timed out. The message is safe to show the model."""


@dataclass
class SqlResult:
    sql: str
    columns: list[str]
    rows: list[list]
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


def _walk(node, visit) -> None:
    if isinstance(node, dict):
        visit(node)
        for v in node.values():
            _walk(v, visit)
    elif isinstance(node, list):
        for v in node:
            _walk(v, visit)


def check_ast(ast: dict) -> None:
    """Reject anything except one SELECT over allowlisted views and functions."""
    if ast.get("error"):
        raise SqlToolError(f"not a valid single SELECT: {ast.get('error_message', 'parse error')}")
    if len(ast.get("statements", [])) != 1:
        raise SqlToolError("exactly one SELECT statement is allowed")

    ctes: set[str] = set()
    _walk(ast, lambda n: ctes.update(e["key"] for e in n.get("cte_map", {}).get("map", [])))

    problems: list[str] = []

    def visit(n: dict) -> None:
        kind = n.get("type")
        if kind == "TABLE_FUNCTION":
            problems.append("table functions are not allowed")
        elif kind == "BASE_TABLE":
            name = n.get("table_name", "")
            qualified = n.get("schema_name") or n.get("catalog_name")
            if qualified or (name not in ALLOWED_TABLES and name not in ctes):
                problems.append(f"table {name!r} is not allowed (use {sorted(ALLOWED_TABLES)})")
        elif n.get("class") == "FUNCTION" and not n.get("is_operator"):
            fname = str(n.get("function_name", "")).lower()
            if fname not in ALLOWED_FUNCTIONS:
                problems.append(f"function {fname!r} is not allowed")

    _walk(ast, visit)
    if problems:
        raise SqlToolError("; ".join(dict.fromkeys(problems)))


def _jsonable(v):
    return v.isoformat() if isinstance(v, date) else v


class FactsDB:
    """Guarded, read-only access to the XBRL facts in DuckDB."""

    def __init__(self, db_path: Path, timeout_s: float = 5.0, max_rows: int = MAX_ROWS):
        self._timeout_s = timeout_s
        self._max_rows = max_rows
        self._root = duckdb.connect(str(db_path), read_only=True)
        self._root.execute("SET enable_external_access=false")
        self._root.execute("SET lock_configuration=true")

    def _cursor(self) -> duckdb.DuckDBPyConnection:
        # Temp views live per connection, so each call gets its own cursor plus the views.
        cur = self._root.cursor()
        cur.execute(_VIEWS)
        return cur

    @traceable(run_type="tool", name="sql")
    def run(self, sql: str) -> SqlResult:
        annotate()
        with span("sql", sql_chars=len(sql)) as a:  # the query text itself is not recorded
            res = self._run(sql)
            a.update(rows=len(res.rows), truncated=res.truncated)
            return res

    def _run(self, sql: str) -> SqlResult:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise SqlToolError("empty query")
        if len(sql) > MAX_SQL_CHARS:
            raise SqlToolError(f"query longer than {MAX_SQL_CHARS} characters")

        cur = self._cursor()
        try:
            ast = json.loads(cur.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
            check_ast(ast)
            timer = threading.Timer(self._timeout_s, cur.interrupt)
            timer.start()
            try:
                # Fetch one extra row so we can tell the caller the result was cut off.
                cur.execute(f"SELECT * FROM ({sql}) LIMIT {self._max_rows + 1}")
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            finally:
                timer.cancel()
        except SqlToolError:
            raise
        except duckdb.InterruptException as e:
            raise SqlToolError(f"query exceeded {self._timeout_s}s") from e
        except duckdb.Error as e:
            raise SqlToolError(f"SQL error: {str(e).splitlines()[0]}") from e
        finally:
            cur.close()

        truncated = len(rows) > self._max_rows
        rows = rows[: self._max_rows]
        return SqlResult(
            sql=sql,
            columns=cols,
            rows=[[_jsonable(v) for v in r] for r in rows],
            truncated=truncated,
        )


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def answer_with_sql(question: str, db: FactsDB, llm: LLM, max_attempts: int = 2) -> SqlResult:
    """Text-to-SQL with a hard attempt cap. On failure the error goes back to the model once."""
    messages = [
        {"role": "system", "content": SCHEMA_PROMPT},
        {"role": "user", "content": question},
    ]
    last_error = "no attempt made"
    for _ in range(max_attempts):
        raw = llm(messages)
        try:
            return db.run(extract_sql(raw))
        except SqlToolError as e:
            last_error = str(e)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That query failed: {e}. Return a corrected query."},
            ]
    raise SqlToolError(f"gave up after {max_attempts} attempts: {last_error}")
