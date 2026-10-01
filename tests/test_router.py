import pytest

from finsight.evals.golden import load_golden
from finsight.orchestrator.router import (
    MAX_SUBQUERIES,
    SubQuery,
    classify_route,
    extract_form,
    extract_section,
    extract_tickers,
    extract_years,
    mentions_metric,
    plan_question,
)


def test_extracts_tickers_by_company_name_and_symbol():
    assert extract_tickers("How do Apple and NVIDIA differ?") == ["AAPL", "NVDA"]
    assert extract_tickers("msft capex") == ["MSFT"]
    assert extract_tickers("Tesla production") == []
    assert extract_tickers("pineapple prices") == []  # word boundary, not substring


def test_extracts_years():
    assert extract_years("between fiscal 2024 and fiscal 2025") == [2024, 2025]
    assert extract_years("FY2019 risk factors") == [2019]
    assert extract_years("we sold 12345 units") == []


def test_form_inference():
    assert extract_form("In its quarterly reports, ...") == "10-Q"
    assert extract_form("What did the latest 10-Q say?") == "10-Q"
    assert extract_form("its fiscal 2024 annual report") == "10-K"
    # The annual report is the default document: without it near-duplicate 10-Q chunks crowd
    # the top results (seen on "How does NVIDIA describe its competition?").
    assert extract_form("How does NVIDIA describe competition?") == "10-K"


def test_section_cues_are_explicit_only_and_follow_the_form():
    assert extract_section("What do its risk factors say about tariffs?", "10-K") == "Item 1A"
    assert extract_section("Compare the risk factor language", "10-Q") == "Item 1A"
    assert extract_section("What market risk does it disclose?", "10-K") == "Item 7A"
    assert extract_section("What market risk does it disclose?", "10-Q") == "Item 3"
    assert extract_section("How did the MD&A describe Activision?", "10-K") == "Item 7"
    assert (
        extract_section("Per management\u2019s discussion, what drove growth?", "10-Q") == "Item 2"
    )
    assert extract_section("What drove NVIDIA's data center growth?", "10-K") is None  # no cue


def test_plan_carries_the_section_into_every_subquery():
    p = plan_question("How did Microsoft's risk factors change from fiscal 2024 to fiscal 2025?")
    assert [s.section for s in p.sub_queries] == ["Item 1A", "Item 1A"] and p.section == "Item 1A"


def test_cross_company_question_fans_out_per_company():
    p = plan_question("Compare Apple and Microsoft buybacks in fiscal 2025")
    assert p.sub_queries == [
        SubQuery(p.question, ("AAPL",), 2025, "10-K", None),
        SubQuery(p.question, ("MSFT",), 2025, "10-K", None),
    ]


def test_year_over_year_question_fans_out_per_year():
    p = plan_question("How did Microsoft's OpenAI risk change from fiscal 2024 to fiscal 2025?")
    assert [(s.tickers, s.fiscal_year) for s in p.sub_queries] == [
        (("MSFT",), 2024),
        (("MSFT",), 2025),
    ]


def test_no_entities_gives_one_unfiltered_subquery():
    p = plan_question("What will happen to the market?")
    assert p.sub_queries == [SubQuery(p.question, (), None, "10-K", None)]  # 10-K is the default


def test_subquery_cap_is_enforced_and_flagged():
    q = "Apple Microsoft NVIDIA in 2021 2022 2023 2024 2025"  # 3 x 5 = 15 combinations
    p = plan_question(q)
    assert len(p.sub_queries) == MAX_SUBQUERIES and p.truncated


# ---------- route classification (LLM is faked) ----------


def _llm(reply, calls=None):
    def fake(messages):
        if calls is not None:
            calls.append(messages)
        return reply

    return fake


def test_non_metric_question_never_calls_the_llm():
    calls: list = []
    assert classify_route("What risks does NVIDIA cite?", _llm("facts", calls)) == "text"
    assert calls == []  # the gate saved a model call


@pytest.mark.parametrize(
    "reply, expected",
    [("facts", "facts"), ("Both.", "both"), ("text", "text"), ("no idea", "text"), ("", "text")],
)
def test_route_parsing_fails_safe_to_text(reply, expected):
    assert classify_route("What was Apple's net income?", _llm(reply)) == expected


def test_no_llm_means_text_route():
    assert classify_route("What was Apple's net income?", None) == "text"


# ---------- against the real golden set ----------


def test_router_matches_hand_written_golden_filters():
    """Every filter a human wrote for the golden set must be recoverable from the question."""
    for row in load_golden():
        gold = row["filters"]
        plan = plan_question(row["question"])
        if "tickers" in gold:
            assert plan.tickers == gold["tickers"], row["id"]
        # Some golden filters carry a year the question never states (e.g. s01). A router cannot
        # know that, so only require years the question actually mentions.
        if "year" in gold and str(gold["year"]) in row["question"]:
            assert gold["year"] in plan.years, row["id"]
        if "form" in gold:
            assert plan.form_type == gold["form"], row["id"]


def test_net_sales_counts_as_a_metric_so_the_route_is_decided_by_the_model():
    # Apple reports "net sales", not "revenue"; without this the question never reached SQL.
    assert mentions_metric("What was Apple's total net sales in fiscal 2024?")
    assert mentions_metric("How large were Microsoft's sales in fiscal 2024?")
    assert not mentions_metric("What risks does NVIDIA cite around export controls?")
