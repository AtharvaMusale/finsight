"""EDGAR client. Talks ONLY to the SEC's documented endpoints, politely.

Guardrails (SEC fair-access policy):
  * every request carries a User-Agent identifying us (required setting)
  * rate limited, hard-capped at 10 req/s
  * host + path allowlist: anything else raises before a request is made
  * 403/429 are not retried aggressively: we stop instead of hammering
  * raw responses are cached on disk, so each document is fetched once
"""

import json
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from finsight.config import SEC_HARD_MAX_RPS, Settings
from finsight.schemas import Filing

ALLOWED_HOSTS = {"data.sec.gov", "www.sec.gov"}
ALLOWED_PATH_PATTERNS = [
    re.compile(r"^/submissions/CIK\d{10}\.json$"),
    re.compile(r"^/api/xbrl/companyfacts/CIK\d{10}\.json$"),
    re.compile(r"^/Archives/edgar/data/\d+/\d{18}/[\w.\-]+$"),
    re.compile(r"^/files/company_tickers\.json$"),
]


class DisallowedURLError(ValueError):
    pass


class RateLimitedError(RuntimeError):
    """SEC answered 403/429. We stop rather than retry in a tight loop."""


def assert_allowed(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise DisallowedURLError(f"host not allowed: {url}")
    if not any(p.match(parsed.path) for p in ALLOWED_PATH_PATTERNS):
        raise DisallowedURLError(f"path not allowed: {url}")


class RateLimiter:
    """Spaces calls at least 1/rps seconds apart (thread-safe)."""

    def __init__(
        self,
        rps: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._interval = 1.0 / min(rps, SEC_HARD_MAX_RPS)
        self._clock, self._sleep = clock, sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = self._next - now
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
            self._next = now + self._interval


def filings_from_submissions(
    ticker: str, cik: int, data: dict, forms: list[str], min_year: int
) -> list[Filing]:
    """Pure function: parse a submissions JSON into Filing rows (easy to unit test).

    Only reads `filings.recent`, which covers ~1000 latest filings: plenty for 3 years.
    fiscal_year here is only the period-of-report year, a first guess (Apple's Dec 2023
    quarter is fiscal 2024). The pipeline overrides it via fiscal_years_from_companyfacts().
    """
    recent = data["filings"]["recent"]
    out: list[Filing] = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        filing_date = date.fromisoformat(recent["filingDate"][i])
        rd = recent["reportDate"][i]
        report_date = date.fromisoformat(rd) if rd else None
        fiscal_year = (report_date or filing_date).year
        if fiscal_year < min_year:
            continue
        out.append(
            Filing(
                accession_no=recent["accessionNumber"][i],
                ticker=ticker,
                cik=cik,
                form_type=form,
                fiscal_year=fiscal_year,
                filing_date=filing_date,
                report_date=report_date,
                primary_document=recent["primaryDocument"][i],
            )
        )
    return out


def fiscal_years_from_companyfacts(companyfacts: dict) -> dict[str, int]:
    """Map accession_no -> the filing's own fiscal year, as declared in XBRL.

    Every fact carries `accn` (the filing that reported it) and `fy` (that filing's fiscal
    year focus). Comparative prior-year values inside a filing carry the same `fy`, so the
    most common `fy` per accession is the filing's true fiscal year.
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    for tags in companyfacts.get("facts", {}).values():
        for body in tags.values():
            for entries in body.get("units", {}).values():
                for e in entries:
                    if e.get("accn") and e.get("fy"):
                        votes[e["accn"]][e["fy"]] += 1
    return {accn: c.most_common(1)[0][0] for accn, c in votes.items()}


class EdgarClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self._settings = settings
        self._limiter = RateLimiter(settings.sec_max_rps)
        self._client = client or httpx.Client(
            headers={"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
        )
        settings.raw_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        assert_allowed(url)
        self._limiter.wait()
        resp = self._client.get(url)
        if resp.status_code in (403, 429):
            # Not retryable: tenacity only retries HTTPStatusError, so this propagates.
            raise RateLimitedError(f"SEC returned {resp.status_code} for {url}; stopping")
        if resp.status_code >= 500:
            resp.raise_for_status()  # transient: retried with backoff
        if resp.status_code >= 400:
            raise RuntimeError(f"SEC returned {resp.status_code} for {url}")  # not retried
        return resp

    def _cached(self, name: str, url: str) -> str:
        path: Path = self._settings.raw_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8")
        text = self._get(url).text
        path.write_text(text, encoding="utf-8")
        return text

    def ticker_to_cik(self, ticker: str) -> int:
        text = self._cached(
            "company_tickers.json", "https://www.sec.gov/files/company_tickers.json"
        )
        for row in json.loads(text).values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"])
        raise KeyError(f"ticker not found in SEC list: {ticker}")

    def get_submissions(self, cik: int) -> dict:
        # Not cached long-term: new filings appear here, so always refetch (one small call).
        return self._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()

    def get_companyfacts(self, cik: int) -> dict:
        return self._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()

    def get_document(self, filing: Filing) -> str:
        return self._cached(f"{filing.accession_no}.html", filing.url)
