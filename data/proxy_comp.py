"""DEF 14A executive compensation fetcher.

Parses the Summary Compensation Table from the most recent annual proxy
statement (DEF 14A) for a given company CIK, returning total reported
compensation for each named executive officer (NEO).

Used by data/insider_feed.py to contextualise Form 4 open-market purchase
sizes — a $500k purchase by a CEO earning $2M/year is far more significant
than the same purchase by a CEO earning $50M/year.

Source: SEC EDGAR submissions API + filing documents (no auth required).
Cache: logs/proxy_comp_cache.json, TTL 90 days (proxy statements are annual).

Public API::

    from data.proxy_comp import get_exec_compensation, match_compensation

    comp_map = get_exec_compensation("0000320193")  # Apple CIK
    # {"TIMOTHY D COOK": 63208000.0, "LUCA MAESTRI": 27200000.0, ...}

    total = match_compensation("COOK TIMOTHY D", comp_map)
    # 63208000.0
"""

import json
import logging
import os
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup, Tag

from config import EMAIL_FROM, LOG_DIR

logger = logging.getLogger(__name__)

_USER_AGENT = f"InvestorBot {EMAIL_FROM or 'contact@example.com'}"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_REQ_DELAY = 0.15  # stay within SEC's 10 req/s limit
_CACHE_TTL_DAYS = 90  # proxy statements are annual; refresh quarterly
_CACHE_PATH = os.path.join(LOG_DIR, "proxy_comp_cache.json")

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"

_COMP_TABLE_RE = re.compile(r"summary\s+compensation\s+table", re.IGNORECASE)
_TOTAL_COL_RE = re.compile(r"^total", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"[\d,]+")
_TITLE_RE = re.compile(
    r"\b(mr|ms|mrs|dr|jr|sr|ii|iii|iv)\b",
    re.IGNORECASE,
)

_last_req: float = 0.0


def _edgar_sleep() -> None:
    global _last_req
    now = time.monotonic()
    gap = _REQ_DELAY - (now - _last_req)
    if gap > 0:
        time.sleep(gap)
    _last_req = time.monotonic()


# ── Cache I/O ─────────────────────────────────────────────────────────────────


def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError as exc:
        logger.warning(f"proxy_comp_cache: write error: {exc}")


def _is_fresh(entry: dict) -> bool:
    try:
        fetched = date.fromisoformat(entry["fetched"])
        return (date.today() - fetched).days < _CACHE_TTL_DAYS
    except (KeyError, ValueError):
        return False


# ── Name normalisation ────────────────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    """Normalise an executive name for fuzzy token matching.

    Strips titles (Mr., Jr., etc.), punctuation, and extra whitespace.
    Returns uppercase space-separated tokens.

    Examples::

        "Timothy D. Cook"  → "TIMOTHY D COOK"
        "COOK TIMOTHY D"   → "COOK TIMOTHY D"
        "Luca Maestri, CFO" → "LUCA MAESTRI CFO"
    """
    name = re.sub(r"[^a-zA-Z\s]", " ", name)
    name = _TITLE_RE.sub("", name)
    return " ".join(name.upper().split())


# ── HTML parsing ──────────────────────────────────────────────────────────────


def _extract_names_and_totals(table: Tag) -> dict[str, float]:
    """Parse a single HTML table, extracting name→total_usd pairs.

    Locates the header row by finding the first cell whose text starts with
    "Total" (case-insensitive).  Data rows following the header are parsed:
    first cell = name, total-column cell = USD amount.

    Returns an empty dict when the table has no recognised "Total" column or
    no parseable data rows.
    """
    rows = table.find_all("tr")
    if not rows:
        return {}

    # Find the header row — the first row (within first 10) containing "Total"
    header_row_idx = -1
    total_col_idx = -1
    for i, row in enumerate(rows[:10]):
        cells = row.find_all(["th", "td"])
        for j, cell in enumerate(cells):
            if _TOTAL_COL_RE.match(cell.get_text(strip=True)):
                header_row_idx = i
                total_col_idx = j
                break
        if total_col_idx >= 0:
            break

    if total_col_idx < 0:
        return {}

    result: dict[str, float] = {}
    for row in rows[header_row_idx + 1 :]:
        cells = row.find_all(["th", "td"])
        if len(cells) <= total_col_idx:
            continue
        name_text = cells[0].get_text(separator=" ", strip=True)
        total_text = cells[total_col_idx].get_text(strip=True)
        if not name_text or not total_text:
            continue
        m = _AMOUNT_RE.search(total_text)
        if not m:
            continue
        try:
            total = float(m.group().replace(",", ""))
        except ValueError:  # pragma: no cover
            continue  # pragma: no cover
        if total <= 0:
            continue
        norm = _normalize_name(name_text)
        if len(norm) > 2:
            result[norm] = total

    return result


def _parse_comp_table(soup: BeautifulSoup) -> dict[str, float]:
    """Locate and parse the Summary Compensation Table in a DEF 14A document.

    Tries every text node matching 'Summary Compensation Table' (handles TOC
    entries that precede the actual table) and returns the first non-empty
    parse result.
    """
    for text_node in soup.find_all(string=_COMP_TABLE_RE):
        parent = text_node.parent
        if parent is None:  # pragma: no cover
            continue  # pragma: no cover
        table = parent.find_next("table")
        if table is None:
            continue
        result = _extract_names_and_totals(table)
        if result:
            return result
    return {}


# ── EDGAR fetch ───────────────────────────────────────────────────────────────


def _find_def14a(cik: str) -> tuple[str, str] | None:
    """Return (accession_no_dashes, primary_doc) for the most recent DEF 14A.

    Returns None if no DEF 14A is found or if the EDGAR request fails.
    """
    try:
        _edgar_sleep()
        resp = requests.get(_SUBMISSIONS_URL.format(cik=cik), headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug(f"Submissions fetch failed for CIK {cik}: {exc}")
        return None

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    for form, accession, doc in zip(forms, accessions, docs, strict=False):
        if form in ("DEF 14A", "DEF14A"):
            return accession.replace("-", ""), doc

    return None


def _fetch_compensation(cik: str) -> dict[str, float]:
    """Download the most recent DEF 14A for *cik* and extract compensation data."""
    filing = _find_def14a(cik)
    if filing is None:
        return {}

    accession, doc = filing
    url = _FILING_URL.format(cik_int=int(cik), accession=accession, doc=doc)
    try:
        _edgar_sleep()
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.debug(f"DEF 14A download failed {url}: {exc}")
        return {}

    try:
        soup = BeautifulSoup(html, "html.parser")
        return _parse_comp_table(soup)
    except Exception as exc:
        logger.debug(f"DEF 14A parse failed for CIK {cik}: {exc}")
        return {}


# ── Public API ─────────────────────────────────────────────────────────────────


def get_exec_compensation(cik: str) -> dict[str, float]:
    """Return total compensation for each NEO in the most recent DEF 14A.

    Result schema::

        {"TIMOTHY D COOK": 63208000.0, "LUCA MAESTRI": 27200000.0, ...}

    Keys are normalised uppercase names (punctuation and titles stripped).
    Values are total reported compensation in USD as listed in the Summary
    Compensation Table.

    Returns an empty dict when no proxy is found or parsing fails.
    Cached for 90 days — proxy statements are annual.  All network errors
    are caught and logged; callers always receive a plain dict.
    """
    cache = _load_cache()
    entry = cache.get(cik, {})
    if _is_fresh(entry):
        return entry.get("data", {})

    data = _fetch_compensation(cik)
    cache[cik] = {"fetched": date.today().isoformat(), "data": data}
    _save_cache(cache)

    if data:
        logger.info(f"proxy_comp CIK {cik}: {len(data)} executives")
    else:
        logger.debug(f"proxy_comp CIK {cik}: no compensation data extracted")

    return data


def match_compensation(reporter: str, comp_map: dict[str, float]) -> float | None:
    """Fuzzy-match a Form 4 reporter name to an exec compensation entry.

    Uses token Jaccard similarity.  Returns the best-matching compensation
    amount when overlap ≥ 0.5 (at least half the tokens match), else None.

    Examples::

        match_compensation("COOK TIMOTHY D", {"TIMOTHY D COOK": 63208000.0})
        # → 63208000.0  (3/3 tokens overlap)

        match_compensation("UNKNOWN PERSON", {"TIMOTHY D COOK": 63208000.0})
        # → None
    """
    if not comp_map:
        return None
    reporter_tokens = set(_normalize_name(reporter).split())
    if not reporter_tokens:
        return None

    best_score = 0.0
    best_total: float | None = None
    for name, total in comp_map.items():
        name_tokens = set(name.split())
        union = reporter_tokens | name_tokens
        if not union:  # pragma: no cover
            continue  # pragma: no cover
        score = len(reporter_tokens & name_tokens) / len(union)
        if score > best_score:
            best_score = score
            best_total = total

    return best_total if best_score >= 0.5 else None
