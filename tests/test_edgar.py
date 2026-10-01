import pytest
from pydantic import ValidationError

from finsight.config import Settings
from finsight.ingestion.edgar import (
    DisallowedURLError,
    RateLimiter,
    assert_allowed,
    filings_from_submissions,
    fiscal_years_from_companyfacts,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://data.sec.gov/submissions/CIK0000320193.json",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
        "https://www.sec.gov/files/company_tickers.json",
    ],
)
def test_documented_endpoints_are_allowed(url):
    assert_allowed(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/submissions/CIK0000320193.json",  # wrong host
        "http://data.sec.gov/submissions/CIK0000320193.json",  # not https
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany",  # HTML site, not an API
        "https://efts.sec.gov/LatestSearch/rest/search-index",  # undocumented full-text search
        "https://finance.yahoo.com/quote/AAPL",  # third party
        "https://data.sec.gov.evil.com/submissions/CIK0000320193.json",  # lookalike host
    ],
)
def test_everything_else_is_refused(url):
    with pytest.raises(DisallowedURLError):
        assert_allowed(url)


def test_rate_limiter_spaces_calls():
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    rl = RateLimiter(5, clock=lambda: now[0], sleep=sleep)  # 5 rps -> 0.2s apart
    for _ in range(3):
        rl.wait()
    assert slept == pytest.approx([0.2, 0.2])


def test_rate_limiter_is_capped_at_sec_limit():
    now = [0.0]
    slept: list[float] = []
    rl = RateLimiter(
        1000,
        clock=lambda: now[0],
        sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)),
    )
    rl.wait()
    rl.wait()
    assert slept == pytest.approx([0.1])  # never faster than 10 rps


def test_settings_require_contact_email(monkeypatch):
    monkeypatch.delenv("FINSIGHT_SEC_USER_AGENT", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # missing
    with pytest.raises(ValidationError):
        Settings(_env_file=None, sec_user_agent="no contact here")


def test_settings_cap_rps():
    s = Settings(_env_file=None, sec_user_agent="app me@example.com", sec_max_rps=50)
    assert s.sec_max_rps == 10


def test_fiscal_year_comes_from_xbrl_majority_vote():
    def fact(accn, fy):
        return {"accn": accn, "fy": fy, "val": 1, "end": "2024-01-01"}

    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [fact("A", 2024), fact("A", 2024), fact("A", 2023), fact("B", 2025)]
                    }
                }
            }
        }
    }
    assert fiscal_years_from_companyfacts(facts) == {"A": 2024, "B": 2025}


def test_filings_from_submissions_filters_form_and_year():
    data = {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000000000-25-000001",
                    "0000000000-25-000002",
                    "0000000000-20-000009",
                ],
                "filingDate": ["2025-02-01", "2025-03-01", "2020-02-01"],
                "reportDate": ["2024-12-31", "2025-01-15", "2019-12-31"],
                "form": ["10-K", "8-K", "10-K"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            }
        }
    }
    out = filings_from_submissions("TEST", 1, data, ["10-K", "10-Q"], min_year=2023)
    assert [f.accession_no for f in out] == ["0000000000-25-000001"]  # 8-K and 2019 dropped
    assert out[0].url.endswith("/1/000000000025000001/a.htm")
