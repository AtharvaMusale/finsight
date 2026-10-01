"""Filing HTML -> plain text -> {section: text}.

Filings have a table of contents that repeats every "Item X" heading, so a naive regex
finds each heading twice. We take every heading match, then for each item keep the match
whose span (to the next heading) is longest: that is the real section, not the TOC line.
"""

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Sections we index. Keys are item ids; labels differ between 10-K and 10-Q.
KEEP_10K = {"1", "1A", "7", "7A"}
KEEP_10Q = {"1A", "2", "3"}  # 10-Q: Item 2 = MD&A, Item 3 = market risk, Part II 1A = risk factors

# The trailing punctuation is mandatory: some filings (e.g. Microsoft) repeat a bare
# "Item 1A" marker line on every page, which is not a heading and would fragment sections.
_HEADING = re.compile(r"^\s*item\s+(\d{1,2}[abc]?)\s*[.:\-–—]", re.IGNORECASE | re.MULTILINE)


def html_to_text(html: str) -> str:
    with warnings.catch_warnings():
        # Some SEC documents are XHTML/XML; the HTML parser handles them fine, so the
        # "parsed XML as HTML" notice is noise, not a problem.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Inline-XBRL hidden header holds machine data, not prose.
    for tag in soup.find_all(["ix:header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln)


def split_sections(text: str, form_type: str) -> dict[str, str]:
    keep = KEEP_10K if form_type == "10-K" else KEEP_10Q
    matches = list(_HEADING.finditer(text))
    best: dict[str, str] = {}
    for i, m in enumerate(matches):
        item = m.group(1).upper()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start() : end].strip()
        if item in keep and len(body) > len(best.get(item, "")):
            best[item] = body
    return {f"Item {k}": v for k, v in best.items()}
