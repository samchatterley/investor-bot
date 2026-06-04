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
        "raises guidance",
        "increases guidance",
        "raises full-year",
        "raises fiscal",
        "exceeds expectations",
        "beat estimates",
        "above consensus",
        "strong demand",
        "record revenue",
        "raises outlook",
        "raises forecast",
        "positive guidance",
        "upward revision",
        "accelerating growth",
    }
)

_NEGATIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        "lowers guidance",
        "reduces guidance",
        "cuts guidance",
        "below expectations",
        "missed estimates",
        "disappointing results",
        "challenging environment",
        "headwinds",
        "lowered outlook",
        "lowered forecast",
        "negative guidance",
        "downward revision",
        "softening demand",
        "reduced revenue",
        "revenue shortfall",
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


def _fetch_filing_text(cik: str, accession: str, doc: str, max_chars: int = 8000) -> str:
    """Download the primary document of a filing and return the first max_chars chars."""
    url = _FILING_BASE.format(cik_int=int(cik), accession=accession) + doc
    try:
        time.sleep(_REQ_DELAY)
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        content = resp.text[:max_chars]
        # Strip HTML tags for keyword matching
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).lower()
        return content
    except Exception as exc:
        logger.debug("edgar_client: filing fetch failed %s: %s", url, exc)
        return ""


# ── Keyword classification ────────────────────────────────────────────────────


def _classify_guidance(text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' based on keyword matching."""
    pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text)
    neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text)
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
    text = _fetch_filing_text(cik, latest["accession"], latest["doc"])
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
    logger.info(
        "edgar prefetch: %d symbols — guidance=%d activist=%d offerings=%d",
        len(missing),
        n_guidance,
        n_activist,
        n_offering,
    )
    return len(missing)


# ── Public getters ────────────────────────────────────────────────────────────


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
