from datetime import date
from pathlib import Path

from finsight.ingestion.chunker import chunk_filing, chunk_section
from finsight.ingestion.parser import html_to_text, split_sections
from finsight.schemas import Filing, make_chunk_id

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.html"


def _filing() -> Filing:
    return Filing(
        accession_no="0000000000-25-000001", ticker="TEST", cik=1, form_type="10-K",
        fiscal_year=2025, filing_date=date(2025, 1, 1), report_date=date(2024, 12, 31),
        primary_document="x.htm",
    )  # fmt: skip


def test_split_sections_skips_toc_and_keeps_real_body():
    sections = split_sections(html_to_text(FIXTURE.read_text()), "10-K")
    assert set(sections) == {"Item 1A", "Item 7"}  # 1B and 8 are not indexed
    assert "Supply chain disruption" in sections["Item 1A"]  # body, not the TOC line
    assert "Revenue increased" in sections["Item 7"]


def test_bare_page_marker_lines_do_not_fragment_a_section():
    # Microsoft-style: a bare "Item 1A" marker line repeats on every page inside the section.
    text = (
        "Item 1A. Risk Factors\n" + "First risk paragraph. " * 20 + "\n"
        "Item 1A\n"  # page marker, no period
         + "Second risk paragraph. " * 20 + "\nItem 1B. Unresolved Staff Comments\nNone."
    )
    body = split_sections(text, "10-K")["Item 1A"]
    assert "First risk" in body and "Second risk" in body  # one section, not two fragments


def test_chunks_carry_metadata_and_never_cross_sections():
    sections = split_sections(html_to_text(FIXTURE.read_text()), "10-K")
    chunks = chunk_filing(_filing(), sections)
    assert {c.metadata.section for c in chunks} == {"Item 1A", "Item 7"}
    for c in chunks:
        assert c.metadata.ticker == "TEST"
        assert c.metadata.accession_no == "0000000000-25-000001"
        assert c.metadata.fiscal_year == 2025


def test_chunk_ids_are_deterministic():
    assert make_chunk_id("a", "Item 1A", 0) == make_chunk_id("a", "Item 1A", 0)
    assert make_chunk_id("a", "Item 1A", 0) != make_chunk_id("a", "Item 1A", 1)


def test_long_text_is_split_under_the_cap():
    text = "\n".join(f"Sentence number {i} is here. " * 5 for i in range(200))
    chunks = chunk_section(text, max_chars=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
