import duckdb
import pytest

from finsight.ingestion.stores import _SCHEMA
from finsight.tools.calculator import CalculatorError, cagr, compute, margin, pct_change, ratio
from finsight.tools.sql_tool import FactsDB, SqlToolError, answer_with_sql, extract_sql

# ---------- calculator ----------


def test_calculator_basics():
    assert pct_change(100, 110) == pytest.approx(10.0)
    assert pct_change(-100, -50) == pytest.approx(50.0)  # abs(old) keeps the sign meaningful
    assert margin(40, 200) == pytest.approx(20.0)
    assert ratio(1, 4) == 0.25
    assert cagr(100, 121, 2) == pytest.approx(10.0)


def test_compute_returns_auditable_result():
    r = compute("pct_change", old=100.0, new=125.0)
    assert r.value == 25.0 and r.unit == "percent"
    assert r.formula == "(125.0 - 100.0) / |100.0| * 100"


@pytest.mark.parametrize(
    "op, args",
    [
        ("pct_change", {"old": 0, "new": 5}),  # division by zero
        ("ratio", {"numerator": 1, "denominator": 0}),
        ("cagr", {"begin": -1, "end": 5, "years": 2}),
        ("cagr", {"begin": 1, "end": 5, "years": 0}),
        ("pct_change", {"old": 1}),  # missing input
        ("pct_change", {"old": 1, "new": 2, "extra": 3}),  # unexpected input
        ("pct_change", {"old": "1", "new": 2}),  # strings are not numbers
        ("pct_change", {"old": True, "new": 2}),  # bools are not numbers
        ("pct_change", {"old": float("nan"), "new": 2}),
        ("eval", {"old": 1, "new": 2}),  # unknown operation
    ],
)
def test_compute_rejects_bad_input(op, args):
    with pytest.raises(CalculatorError):
        compute(op, **args)


# ---------- SQL tool ----------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA)
    con.execute(
        "INSERT INTO filings VALUES ('a1','NVDA',1,'10-K',2025,'2025-02-26','2025-01-26','u'),"
        "('a0','NVDA',1,'10-K',2024,'2024-02-21','2024-01-28','u')"
    )
    facts = [
        # (tag, start, end, value, accession, filed)
        ("Revenues", "2024-01-29", "2025-01-26", 130.0, "a1", "2025-02-26"),
        ("Revenues", "2023-01-30", "2024-01-28", 60.0, "a1", "2025-02-26"),
        ("Revenues", "2023-01-30", "2024-01-28", 61.0, "a0", "2024-02-21"),  # older duplicate
        ("Assets", "2025-01-26", "2025-01-26", 111.0, "a1", "2025-02-26"),  # instant
    ]
    con.executemany(
        "INSERT INTO facts VALUES (1,'NVDA','us-gaap',?,'USD',?,?,?,2025,'FY','10-K',?,?)", facts
    )
    con.close()
    return FactsDB(path, timeout_s=5)


def test_view_maps_metric_and_keeps_latest_filed(db):
    res = db.run(
        "SELECT fiscal_year, value FROM v_facts "
        "WHERE ticker='NVDA' AND metric='Revenue' AND period_type='annual' ORDER BY fiscal_year"
    )
    # FY2024 appears twice in facts (a0=61, a1=60): the later filing wins.
    assert res.rows == [[2024, 60.0], [2025, 130.0]]


def test_view_types_instant_and_annual(db):
    res = db.run("SELECT metric, period_type, fiscal_year FROM v_facts WHERE metric='TotalAssets'")
    assert res.rows == [["TotalAssets", "instant", 2025]]


def test_dates_are_json_friendly_and_trailing_semicolon_ok(db):
    res = db.run("SELECT period_end FROM v_facts WHERE metric='TotalAssets';")
    assert res.rows == [["2025-01-26"]]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM facts",
        "DROP VIEW v_facts",
        "INSERT INTO facts SELECT * FROM facts",
        "SELECT 1; SELECT 2",  # two statements
        "SELECT * FROM facts",  # raw table is not allowlisted
        "SELECT * FROM main.v_facts",  # qualified names are refused
        "SELECT * FROM read_csv('/etc/hosts')",  # table function
        "SELECT * FROM '/etc/hosts'",  # file as table
        "SELECT getenv('HOME')",  # scalar function not allowlisted
        "SELECT * FROM v_facts, (SELECT * FROM facts) x",  # sneaky sub-select on a raw table
        "COPY v_facts TO '/tmp/x.csv'",
        "ATTACH 'x.db'",
        "PRAGMA database_list",
        "SET enable_external_access=true",
        "",
        "SELECT " + "1," * 1200 + "1",  # too long
    ],
)
def test_guard_rejects_unsafe_sql(db, sql):
    with pytest.raises(SqlToolError):
        db.run(sql)


def test_cte_names_are_allowed(db):
    res = db.run(
        "WITH r AS (SELECT value FROM v_facts WHERE metric='Revenue') SELECT count(*) FROM r"
    )
    assert res.rows == [[2]]


def test_row_cap_and_truncation_flag(tmp_path):
    path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA)
    con.executemany(
        "INSERT INTO facts VALUES "
        "(1,'X','us-gaap','Assets','USD',?,?,1,2025,'FY','10-K','a','2025-01-01')",
        [(f"2020-01-{d:02d}", f"2020-01-{d:02d}") for d in range(1, 11)],
    )
    con.close()
    res = FactsDB(path, max_rows=3).run("SELECT * FROM v_facts")
    assert len(res.rows) == 3 and res.truncated


def test_sql_errors_become_tool_errors(db):
    with pytest.raises(SqlToolError):
        db.run("SELECT nope FROM v_facts")


def test_extract_sql_handles_fences_and_plain_text():
    assert extract_sql("Here:\n```sql\nSELECT 1\n```\nbye") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


def test_answer_with_sql_retries_once_with_the_error(db):
    seen = []

    def fake_llm(messages):
        seen.append(messages)
        bad = "```sql\nSELECT * FROM facts\n```"
        good = "```sql\nSELECT value FROM v_facts WHERE metric='TotalAssets'\n```"
        return bad if len(seen) == 1 else good

    res = answer_with_sql("total assets?", db, fake_llm)
    assert res.rows == [[111.0]]
    assert "not allowed" in seen[1][-1]["content"]  # the rejection reason was fed back


def test_answer_with_sql_gives_up_at_the_cap(db):
    calls = []

    def always_bad(messages):
        calls.append(1)
        return "DROP VIEW v_facts"

    with pytest.raises(SqlToolError, match="gave up after 2"):
        answer_with_sql("q", db, always_bad, max_attempts=2)
    assert len(calls) == 2
