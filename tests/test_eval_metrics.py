from finsight.evals.golden import count_matching, validate
from finsight.evals.metrics import is_abstention, matches, score_answer, score_retrieval
from finsight.schemas import Hit


def _hit(text, ticker="NVDA", form="10-K", fy=2025, section="Item 1A") -> Hit:
    return Hit(
        chunk_id="x", text=text, ticker=ticker, form_type=form,
        fiscal_year=fy, section=section, accession_no="acc",
    )  # fmt: skip


T_EXPORT = {
    "ticker": "NVDA", "form_type": "10-K", "fiscal_year": 2025,
    "section": "Item 1A", "pattern": "export control",
}  # fmt: skip


def test_matches_requires_every_field():
    assert matches(_hit("New Export Controls apply"), T_EXPORT)  # case-insensitive text
    assert not matches(_hit("nothing relevant"), T_EXPORT)  # pattern missing
    assert not matches(_hit("export control", ticker="AAPL"), T_EXPORT)
    assert not matches(_hit("export control", fy=2024), T_EXPORT)
    assert not matches(_hit("export control", section="Item 7"), T_EXPORT)


def test_target_without_optional_fields_is_looser():
    loose = {"ticker": "NVDA", "pattern": "export control"}
    assert matches(_hit("export control", fy=2019, section="Item 7", form="10-Q"), loose)


def test_retrieval_score_rank_and_coverage():
    t_aapl = {"ticker": "AAPL", "pattern": "tariff"}
    hits = [_hit("irrelevant"), _hit("export control"), _hit("tariff", ticker="AAPL")]
    sc = score_retrieval([T_EXPORT, t_aapl], hits)
    assert sc.hit and sc.coverage == 1.0
    assert sc.first_rank == 2 and sc.reciprocal_rank == 0.5


def test_retrieval_score_miss():
    sc = score_retrieval([T_EXPORT], [_hit("irrelevant")])
    assert not sc.hit and sc.coverage == 0.0 and sc.first_rank is None
    assert sc.reciprocal_rank == 0.0


def test_abstention_detection():
    assert is_abstention("I could not find this in the provided filings.")
    assert is_abstention("The excerpts do not contain that information.")
    assert not is_abstention("NVIDIA cites export controls [C1].")


def test_answer_checks():
    good = _hit("export control")
    out = score_answer([T_EXPORT], False, "Risk exists [C1].", [good], [])
    assert out == {
        "false_abstain": False, "has_citation": True,
        "citations_valid": True, "cites_relevant": True,
    }  # fmt: skip
    bad = score_answer([T_EXPORT], False, "Risk exists [C9].", [], ["C9"])
    assert not bad["citations_valid"] and not bad["has_citation"]
    assert score_answer([], True, "I could not find that.", [], []) == {"abstain_correct": True}
    assert score_answer([], True, "It will be $500 [C1].", [], []) == {"abstain_correct": False}


def test_validate_flags_unsatisfiable_targets():
    corpus = [{"ticker": "NVDA", "form_type": "10-K", "fiscal_year": 2025,
               "section": "Item 1A", "text": "Export controls tightened"}]  # fmt: skip
    assert count_matching(corpus, T_EXPORT) == 1
    rows = [
        {"id": "ok", "expect_abstain": False, "targets": [T_EXPORT]},
        {"id": "bad", "expect_abstain": False, "targets": [{**T_EXPORT, "pattern": "zzz"}]},
        {"id": "empty", "expect_abstain": False, "targets": []},
        {"id": "u", "expect_abstain": True, "targets": [T_EXPORT]},
    ]
    problems = validate(rows, corpus)
    assert len(problems) == 3
    assert any("bad" in p and "NO chunk" in p for p in problems)


def test_partial_answer_with_citations_is_not_a_false_abstention():
    """Regression: 'excerpts do not include X' inside a cited answer is not a refusal."""
    good = _hit("export control")
    partial = "Apple repurchased shares [C1]. The excerpts do not contain Microsoft's program."
    out = score_answer([T_EXPORT], False, partial, [good], [])
    assert out["false_abstain"] is False and out["has_citation"] is True
    refusal = score_answer([T_EXPORT], False, "I could not find this in the filings.", [], [])
    assert refusal["false_abstain"] is True  # a bare refusal still counts
