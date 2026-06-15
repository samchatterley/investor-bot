"""SEC EDGAR REST API client for corporate event detection.

Fetches and classifies three filing types used as trading signals:

  8-K  (Item 2.02 / 7.01) — earnings results and guidance updates.
       Keyword-classified as positive, negative, or neutral.
       Feeds guidance_change_signal.

  SC 13D / SC 13G — activist investor filings (≥5% ownership disclosure).
       Tracks known activist funds.  Feeds activist_13d_signal.

  424B4 / S-3 / S-1 — secondary offering prospectuses.
       New share issuance → supply shock.  Feeds secondary_offering_short.

All use SEC's public REST API (no authentication required).  Rate limited
to ≤10 req/s; _REQ_DELAY is set conservatively at 0.15 s per request.

Cache: logs/edgar_client_cache.json, refreshed daily per symbol.
All functions degrade gracefully — return empty dicts / None on any failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, timedelta
from functools import lru_cache

import requests

from config import EMAIL_FROM, LOG_DIR, STOCK_UNIVERSE, today_et

logger = logging.getLogger(__name__)

_USER_AGENT = f"InvestorBot {EMAIL_FROM or 'contact@example.com'}"
_HEADERS = {"User-Agent": _USER_AGENT}
_REQ_DELAY = 0.15

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_EFTS_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
_FILING_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/"

_CACHE_PATH = os.path.join(LOG_DIR, "edgar_client_cache.json")
_DEFAULT_LOOKBACK_DAYS = 30


# ── Known activist funds ──────────────────────────────────────────────────────

_ACTIVIST_FUNDS: frozenset[str] = frozenset(
    {
        "Elliott Investment Management",
        "Starboard Value",
        "Icahn Enterprises",
        "Carl Icahn",
        "ValueAct Capital",
        "Pershing Square",
        "Third Point",
        "Trian Fund Management",
        "Nelson Peltz",
        "Engaged Capital",
        "Jana Partners",
        "Sachem Head Capital",
        "Corvex Management",
        "Greenlight Capital",
        "Ancora Holdings",
    }
)

# ── Guidance keywords ─────────────────────────────────────────────────────────

_POSITIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        # Guidance / outlook raised (stems catch non-contiguous phrasing like
        # "raised its full-year revenue guidance").
        "raised",
        "raises",
        "raising",
        "raises guidance",
        "raised guidance",
        "raised its guidance",
        "increases guidance",
        "raises full-year",
        "raised full-year",
        "raises fiscal",
        "raises outlook",
        "raised outlook",
        "raised its outlook",
        "raises forecast",
        "raised forecast",
        "positive guidance",
        "upward revision",
        # Beats / exceeds
        "beat",
        "beats",
        "exceeded",
        "exceeds",
        "exceeding",
        "exceeds expectations",
        "exceeded expectations",
        "exceeded estimates",
        "beat estimates",
        "beat expectations",
        "above consensus",
        "above expectations",
        "ahead of expectations",
        "better than expected",
        "above the high end",
        "topped estimates",
        # Records / strength
        "record revenue",
        "record earnings",
        "record quarter",
        "record results",
        "record sales",
        "all-time high",
        "strong",
        "strong demand",
        "strong growth",
        "strong results",
        "strong quarter",
        "robust",
        "robust demand",
        "robust growth",
        "solid quarter",
        "outperform",
        "outperformed",
        # Growth / margins / capital return
        "accelerating",
        "accelerating growth",
        "double-digit growth",
        "margin expansion",
        "raised dividend",
        "increased dividend",
        "share repurchase",
    }
)

_NEGATIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        # Guidance / outlook lowered
        "lowered",
        "lowers",
        "lowering",
        "lowers guidance",
        "lowered guidance",
        "cuts guidance",
        "cut guidance",
        "reduces guidance",
        "reduced guidance",
        "withdrew guidance",
        "suspended guidance",
        "lowered outlook",
        "lowered its outlook",
        "lowered forecast",
        "reduced outlook",
        "negative guidance",
        "downward revision",
        "cautious outlook",
        # Misses / below
        "miss",
        "missed",
        "missed estimates",
        "missed expectations",
        "below consensus",
        "below expectations",
        "below estimates",
        "below the low end",
        "short of expectations",
        "fell short",
        "worse than expected",
        "weaker than expected",
        "shortfall",
        "revenue shortfall",
        # Weakness / decline
        "weak",
        "weaker",
        "weakness",
        "weakening",
        "weak demand",
        "soft demand",
        "softening demand",
        "softening",
        "deteriorating",
        "declining",
        "declined",
        "revenue decline",
        "sales decline",
        "reduced revenue",
        "margin compression",
        "margin contraction",
        # Distress
        "disappointing",
        "disappointing results",
        "headwind",
        "headwinds",
        "challenging environment",
        "net loss",
        "operating loss",
        "widening loss",
        "impairment",
        "goodwill impairment",
        "writedown",
        "write-down",
        "restructuring",
        "layoffs",
        "going concern",
        "material weakness",
        "downgrade",
        "downgraded",
    }
)


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
        logger.warning("edgar_client: cache write error: %s", exc)


def _is_stale(entry: dict) -> bool:
    return entry.get("_date") != today_et().isoformat()


# ── CIK lookup (shared with insider_feed) ────────────────────────────────────


@lru_cache(maxsize=1)
def _get_cik_map() -> dict[str, str]:
    """Fetch and cache SEC ticker → CIK mapping for the process lifetime."""
    try:
        resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in resp.json().values()}
    except Exception as exc:
        logger.warning("edgar_client: CIK map unavailable: %s", exc)
        return {}


# ── SEC submissions helpers ───────────────────────────────────────────────────


def _get_recent_filings(
    cik: str,
    form_types: list[str],
    lookback_days: int,
) -> list[dict]:
    """Return recent filing metadata for a CIK matching any of the form types."""
    try:
        time.sleep(_REQ_DELAY)
        resp = requests.get(_SUBMISSIONS_URL.format(cik=cik), headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("edgar_client: submissions fetch failed CIK %s: %s", cik, exc)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    items = recent.get("items", [""] * len(forms))

    cutoff = date.today() - timedelta(days=lookback_days)
    results: list[dict] = []
    for form, filing_date, accession, doc, item in zip(
        forms, dates, accessions, docs, items, strict=False
    ):
        if form not in form_types:
            continue
        try:
            fd = date.fromisoformat(filing_date)
        except ValueError:
            continue
        if fd < cutoff:
            break
        results.append(
            {
                "form": form,
                "filing_date": filing_date,
                "accession": accession.replace("-", ""),
                "doc": doc,
                "items": str(item),
            }
        )
    return results


def _strip_html(raw: str, max_chars: int) -> str:
    """Strip HTML tags, collapse whitespace, lowercase, THEN truncate.

    Order matters: truncating the raw HTML first (as the original did) keeps only the document
    header/inline-XBRL boilerplate for modern EDGAR filings and discards the narrative body, which
    is why keyword classification saw nothing. Strip first so max_chars counts real text.
    """
    content = re.sub(r"<[^>]+>", " ", raw)
    content = re.sub(r"\s+", " ", content).strip().lower()
    return content[:max_chars]


def _fetch_doc_text(url: str, max_chars: int = 8000) -> str:
    """GET a single filing document and return its stripped text."""
    try:
        time.sleep(_REQ_DELAY)
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return _strip_html(resp.text, max_chars)
    except Exception as exc:
        logger.debug("edgar_client: filing fetch failed %s: %s", url, exc)
        return ""


def _fetch_filing_text(cik: str, accession: str, doc: str, max_chars: int = 8000) -> str:
    """Download the primary document of a filing and return its stripped text."""
    url = _FILING_BASE.format(cik_int=int(cik), accession=accession) + doc
    return _fetch_doc_text(url, max_chars)


# Filing-index entries that are never the press-release exhibit: the filing index pages, the
# FilingSummary, and the XBRL viewer report fragments (R1.htm, R2.htm, ...).
_EXHIBIT_SKIP = re.compile(r"(index|filingsummary|^r\d+\.htm$)", re.I)


def _fetch_8k_exhibit_text(
    cik: str, accession: str, primary_doc: str, max_chars: int = 20000
) -> str:
    """Return the combined text of an 8-K's press-release / commentary exhibits.

    The primary 8-K document is usually just the cover page; the results and guidance narrative
    live in the EX-99.x exhibits (the press release and any CFO commentary). We list the accession's
    documents and combine the HTML exhibits that are not the cover, the filing index, or XBRL-viewer
    fragments. Falls back to the primary document when no exhibits are found or the listing fails.
    """
    base = _FILING_BASE.format(cik_int=int(cik), accession=accession)
    try:
        time.sleep(_REQ_DELAY)
        resp = requests.get(base + "index.json", headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("directory", {}).get("item", [])
    except Exception as exc:
        logger.debug("edgar_client: index fetch failed %s: %s", base, exc)
        return _fetch_doc_text(base + primary_doc, max_chars)

    exhibits = [
        d["name"]
        for d in items
        if d.get("name", "").lower().endswith((".htm", ".html"))
        and d["name"] != primary_doc
        and not _EXHIBIT_SKIP.search(d["name"])
    ]
    if not exhibits:
        return _fetch_doc_text(base + primary_doc, max_chars)

    parts: list[str] = []
    for name in exhibits:
        chunk = _fetch_doc_text(base + name, max_chars)
        if chunk:
            parts.append(chunk)
        if sum(len(p) for p in parts) >= 2 * max_chars:
            break
    combined = " ".join(parts)
    return combined or _fetch_doc_text(base + primary_doc, max_chars)


# ── Keyword classification ────────────────────────────────────────────────────


def _classify_guidance(text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' from distinct word-boundary keyword hits.

    Matching is word-boundary anchored, not raw substring, for two reasons: it lets single
    distinctive stems ("raised", "shortfall") count, and it avoids false positives where a keyword
    is embedded in an unrelated word (e.g. "raised" inside "praised", "lowered" inside "flowered").
    Counts distinct keywords matched, not total occurrences.
    """
    pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", text))
    neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", text))
    if pos_hits > neg_hits:
        return "positive"
    if neg_hits > pos_hits:
        return "negative"
    return "neutral"


# ── 8-K guidance detection ────────────────────────────────────────────────────


def _fetch_8k_guidance(sym: str, cik: str, lookback_days: int) -> dict | None:
    """Fetch recent 8-K filings and classify guidance sentiment.

    Returns the most recent guidance-relevant 8-K or None if absent.
    Schema: {"sentiment": str, "filing_date": str, "items": str}
    """
    filings = _get_recent_filings(cik, ["8-K", "8-K/A"], lookback_days)
    # Filter to items 2.02 (results) and 7.01 (regulation FD / outlook)
    guidance_filings = [f for f in filings if any(item in f["items"] for item in ["2.02", "7.01"])]
    if not guidance_filings:
        return None

    latest = guidance_filings[0]
    # The cover page carries no results/guidance language — read the EX-99.x exhibits.
    text = _fetch_8k_exhibit_text(cik, latest["accession"], latest["doc"])
    if not text:
        return None

    sentiment = _classify_guidance(text)
    logger.info(
        "8-K %s: %s (%s) sentiment=%s",
        sym,
        latest["filing_date"],
        latest["items"],
        sentiment,
    )
    return {
        "sentiment": sentiment,
        "filing_date": latest["filing_date"],
        "items": latest["items"],
        "guidance_positive": sentiment == "positive",
        "guidance_negative": sentiment == "negative",
    }


# ── SC 13D activist detection ─────────────────────────────────────────────────


def _fetch_13d_activist(sym: str, cik: str, lookback_days: int) -> dict | None:
    """Detect SC 13D / 13G filings from known activist funds.

    Returns filing metadata if an activist is detected, else None.
    Schema: {"activist_name": str, "filing_date": str, "known_activist": bool}
    """
    filings = _get_recent_filings(cik, ["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"], lookback_days)
    if not filings:
        return None

    for filing in filings:
        text = _fetch_filing_text(cik, filing["accession"], filing["doc"], max_chars=4000)
        if not text:
            continue
        known = False
        activist_name = ""
        for fund in _ACTIVIST_FUNDS:
            if fund.lower() in text:
                known = True
                activist_name = fund
                break
        if known:
            logger.info("13D %s: activist=%s filed=%s", sym, activist_name, filing["filing_date"])
            return {
                "activist_name": activist_name,
                "filing_date": filing["filing_date"],
                "known_activist": True,
                "form": filing["form"],
            }

    # Return unknown activist if any 13D exists within window (still notable)
    return {
        "activist_name": "",
        "filing_date": filings[0]["filing_date"],
        "known_activist": False,
        "form": filings[0]["form"],
    }


# ── 424B4 / S-3 secondary offering detection ─────────────────────────────────


def _fetch_secondary_offering(sym: str, cik: str, lookback_days: int) -> dict | None:
    """Detect secondary share offerings (424B4, S-3, S-1) within the lookback window.

    Returns filing metadata if an offering is detected, else None.
    Schema: {"form": str, "filing_date": str, "offering_detected": bool}
    """
    filings = _get_recent_filings(
        cik, ["424B4", "424B3", "S-3", "S-3/A", "S-1", "S-1/A"], lookback_days
    )
    if not filings:
        return None

    latest = filings[0]
    logger.info("Secondary offering %s: %s filed=%s", sym, latest["form"], latest["filing_date"])
    return {
        "form": latest["form"],
        "filing_date": latest["filing_date"],
        "offering_detected": True,
    }


# ── 8-K narrative-event detection (material-context enrichment) ────────────────
#
# These flag the material-context categories (ma_event, accounting_concern, regulatory_event) that
# the experiment's detector reads (experiment/material_context.py). They are direction-agnostic
# enrichment flags, not trading signals: the deterministic engine still selects the candidate, the
# flag marks that it carries context worth AI judgement, and the AI decides the implication.
# Detection is by 8-K item code (cheap, from submissions metadata) — high-precision items fire
# directly; broad items (1.01 material agreement, 8.01 other events) are confirmed by focused
# keywords in the exhibit text to suppress unrelated agreements/announcements.

_ACCOUNTING_ITEMS = ("4.02", "4.01")  # non-reliance/restatement; auditor change
_MA_COMPLETION_ITEM = "2.01"  # completion of acquisition or disposition of assets
_MA_AGREEMENT_ITEM = "1.01"  # entry into a material definitive agreement (confirm with keywords)
_REG_DELISTING_ITEM = "3.01"  # notice of delisting / failure to satisfy a listing rule
_REG_OTHER_ITEM = "8.01"  # other events (confirm with keywords)

# Event-specific multi-word phrases only. Bare "merger"/"acquisition" appear in debt-covenant
# boilerplate ("in the event of a merger, disposition or transfer..."), and "securities and exchange
# commission" / bare "fda" appear in nearly every 8-K, so loose terms produce false positives on
# financing filings (verified against live JPM/GILD/F 8-Ks). These require the actual deal/regulatory
# language. Negative tests in test_edgar_client lock in the boilerplate cases that previously fired.
_MA_KEYWORDS = re.compile(
    r"(agreement and plan of merger|definitive merger agreement|merger agreement|"
    r"definitive agreement to acquire|to be acquired by|tender offer|"
    r"business combination agreement)",
    re.I,
)
_REGULATORY_KEYWORDS = re.compile(
    r"(complete response letter|clinical hold|warning letter|consent decree|"
    r"enforcement action|product recall|antitrust|subpoena|department of justice|"
    r"fda\s+(?:has\s+)?(?:approved|cleared|granted|rejected|declined|accepted)|"
    r"(?:approved|cleared|granted|rejected)\s+by\s+the\s+fda|"
    r"phase (?:3|iii) (?:trial|study|results|data))",
    re.I,
)


def _fetch_accounting_concern(sym: str, filings: list[dict]) -> dict | None:
    """Detect an accounting-concern 8-K: item 4.02 (non-reliance/restatement) or 4.01 (auditor
    change). Item codes are unambiguous here, so no text confirmation is needed."""
    for f in filings:
        if any(it in f["items"] for it in _ACCOUNTING_ITEMS):
            logger.info("8-K accounting-concern %s: %s (%s)", sym, f["filing_date"], f["items"])
            return {"detected": True, "filing_date": f["filing_date"], "items": f["items"]}
    return None


def _fetch_ma_event(sym: str, cik: str, filings: list[dict]) -> dict | None:
    """Detect an M&A 8-K: item 2.01 (completed acquisition/disposition), or item 1.01 (material
    definitive agreement) confirmed by M&A keywords in the exhibit text."""
    for f in filings:
        items = f["items"]
        if _MA_COMPLETION_ITEM in items:
            logger.info("8-K M&A %s: %s (%s) completion", sym, f["filing_date"], items)
            return {
                "detected": True,
                "filing_date": f["filing_date"],
                "items": items,
                "trigger": "2.01",
            }
        if _MA_AGREEMENT_ITEM in items:
            text = _fetch_8k_exhibit_text(cik, f["accession"], f["doc"])
            if text and _MA_KEYWORDS.search(text):
                logger.info("8-K M&A %s: %s (%s) agreement+kw", sym, f["filing_date"], items)
                return {
                    "detected": True,
                    "filing_date": f["filing_date"],
                    "items": items,
                    "trigger": "1.01+kw",
                }
    return None


def _fetch_regulatory_event(sym: str, cik: str, filings: list[dict]) -> dict | None:
    """Detect a regulatory-event 8-K: item 3.01 (delisting / listing-rule failure), or item 8.01
    (other events) confirmed by regulatory keywords in the exhibit text."""
    for f in filings:
        items = f["items"]
        if _REG_DELISTING_ITEM in items:
            logger.info("8-K regulatory %s: %s (%s) delisting", sym, f["filing_date"], items)
            return {
                "detected": True,
                "filing_date": f["filing_date"],
                "items": items,
                "trigger": "3.01",
            }
        if _REG_OTHER_ITEM in items:
            text = _fetch_8k_exhibit_text(cik, f["accession"], f["doc"])
            if text and _REGULATORY_KEYWORDS.search(text):
                logger.info("8-K regulatory %s: %s (%s) other+kw", sym, f["filing_date"], items)
                return {
                    "detected": True,
                    "filing_date": f["filing_date"],
                    "items": items,
                    "trigger": "8.01+kw",
                }
    return None


# ── Per-symbol fetch ──────────────────────────────────────────────────────────


def _live_fetch(sym: str, lookback_days: int) -> dict:
    """Fetch all EDGAR signal types for sym. Returns {} on CIK miss or error."""
    cik_map = _get_cik_map()
    cik = cik_map.get(sym.upper())
    if not cik:
        return {}

    result: dict = {}

    guidance = _fetch_8k_guidance(sym, cik, lookback_days)
    if guidance:
        result["guidance"] = guidance

    activist = _fetch_13d_activist(sym, cik, lookback_days)
    if activist:
        result["activist"] = activist

    offering = _fetch_secondary_offering(sym, cik, lookback_days)
    if offering:
        result["secondary_offering"] = offering

    # The three narrative material-context categories all read 8-K filings — fetch once and share.
    filings_8k = _get_recent_filings(cik, ["8-K", "8-K/A"], lookback_days)

    ma = _fetch_ma_event(sym, cik, filings_8k)
    if ma:
        result["ma_event"] = ma

    accounting = _fetch_accounting_concern(sym, filings_8k)
    if accounting:
        result["accounting_concern"] = accounting

    regulatory = _fetch_regulatory_event(sym, cik, filings_8k)
    if regulatory:
        result["regulatory_event"] = regulatory

    return result


# ── Public prefetch ───────────────────────────────────────────────────────────


def prefetch_edgar_data(
    symbols: list[str] | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> int:
    """Warm the same-day EDGAR cache for all symbols.

    Called from the 07:00 ET pre-market prefetch job.  Safe to call multiple
    times — already-cached symbols are skipped.

    Returns:
        Number of symbols newly fetched.
    """
    if symbols is None:
        symbols = list(STOCK_UNIVERSE)

    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if not missing:
        logger.info("edgar prefetch: cache already warm (%d symbols)", len(today_cache))
        return 0

    for sym in missing:
        today_cache[sym] = _live_fetch(sym, lookback_days)

    _save_cache({today: today_cache})

    n_guidance = sum(1 for v in today_cache.values() if v.get("guidance"))
    n_activist = sum(1 for v in today_cache.values() if v.get("activist"))
    n_offering = sum(1 for v in today_cache.values() if v.get("secondary_offering"))
    n_ma = sum(1 for v in today_cache.values() if v.get("ma_event"))
    n_accounting = sum(1 for v in today_cache.values() if v.get("accounting_concern"))
    n_regulatory = sum(1 for v in today_cache.values() if v.get("regulatory_event"))
    logger.info(
        "edgar prefetch: %d symbols — guidance=%d activist=%d offerings=%d "
        "ma=%d accounting=%d regulatory=%d",
        len(missing),
        n_guidance,
        n_activist,
        n_offering,
        n_ma,
        n_accounting,
        n_regulatory,
    )
    return len(missing)


# ── Public getters ────────────────────────────────────────────────────────────


def get_edgar_signals_batch(
    symbols: list[str],
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict[str, dict]:
    """Return EDGAR signals for all symbols from the daily cache in one read.

    Loads the cache once (avoids N file opens for N symbols).  Symbols missing
    from the cache are fetched individually only when the cache entry is absent —
    when the 07:00 prefetch ran, this is a pure in-memory operation.

    Returns {symbol: entry_dict} for every symbol in ``symbols``.  The entry
    dict has optional keys "guidance", "activist", "secondary_offering",
    "ma_event", "accounting_concern", "regulatory_event".
    """
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict] = cache.get(today, {})

    result: dict[str, dict] = {}
    newly_fetched: list[str] = []
    for sym in symbols:
        if sym in today_cache:
            result[sym] = today_cache[sym]
        else:
            entry = _live_fetch(sym, lookback_days)
            result[sym] = entry
            today_cache[sym] = entry
            newly_fetched.append(sym)

    if newly_fetched:
        _save_cache({today: today_cache})

    return result


def _today_entry(sym: str, lookback_days: int) -> dict:
    """Return today's cached entry for sym, fetching live if missing."""
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict] = cache.get(today, {})

    if sym not in today_cache:
        today_cache[sym] = _live_fetch(sym, lookback_days)
        _save_cache({today: today_cache})

    return today_cache.get(sym, {})


def get_guidance_sentiment(
    sym: str,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict | None:
    """Return the most recent 8-K guidance classification for sym.

    Result schema::
        {
            "sentiment":        str,   # "positive" | "negative" | "neutral"
            "filing_date":      str,
            "items":            str,
            "guidance_positive": bool,
            "guidance_negative": bool,
        }
    Returns None when no recent guidance 8-K was found.
    """
    entry = _today_entry(sym, lookback_days)
    return entry.get("guidance")


def get_activist_filing(
    sym: str,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict | None:
    """Return SC 13D/G activist filing data for sym, or None if none found.

    Result schema::
        {
            "activist_name":  str,
            "filing_date":    str,
            "known_activist": bool,
            "form":           str,
        }
    """
    entry = _today_entry(sym, lookback_days)
    return entry.get("activist")


def get_secondary_offering(
    sym: str,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict | None:
    """Return secondary offering filing data for sym, or None if none found.

    Result schema::
        {
            "form":              str,
            "filing_date":       str,
            "offering_detected": bool,
        }
    """
    entry = _today_entry(sym, lookback_days)
    return entry.get("secondary_offering")


def get_ma_event(
    sym: str,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict | None:
    """Return M&A 8-K event data for sym, or None if none found.

    Result schema::
        {"detected": bool, "filing_date": str, "items": str, "trigger": str}
    """
    entry = _today_entry(sym, lookback_days)
    return entry.get("ma_event")


def get_accounting_concern(
    sym: str,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict | None:
    """Return accounting-concern 8-K data for sym, or None if none found.

    Result schema::
        {"detected": bool, "filing_date": str, "items": str}
    """
    entry = _today_entry(sym, lookback_days)
    return entry.get("accounting_concern")


def get_regulatory_event(
    sym: str,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict | None:
    """Return regulatory-event 8-K data for sym, or None if none found.

    Result schema::
        {"detected": bool, "filing_date": str, "items": str, "trigger": str}
    """
    entry = _today_entry(sym, lookback_days)
    return entry.get("regulatory_event")


def get_edgar_signals(
    sym: str,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict:
    """Return all EDGAR signal types for sym in one call.

    Always returns a dict (empty when no events found).
    Keys present only when the corresponding event was detected:
      "guidance"           — 8-K guidance classification
      "activist"           — SC 13D/G activist filing
      "secondary_offering" — 424B4/S-3 secondary offering
      "ma_event"           — 8-K M&A (item 2.01, or 1.01 + keywords)
      "accounting_concern" — 8-K item 4.02/4.01 (restatement / auditor change)
      "regulatory_event"   — 8-K (item 3.01 delisting, or 8.01 + keywords)
    """
    entry = _today_entry(sym, lookback_days)
    return dict(entry)


# ── Text classification (public for testing) ──────────────────────────────────


def classify_guidance_text(text: str) -> str:
    """Public wrapper around the keyword classifier.

    Returns 'positive', 'negative', or 'neutral'.
    Input text is lowercased internally.
    """
    return _classify_guidance(text.lower())
