"""SEC EDGAR Form 4 fetcher — open-market insider purchase cluster detection.

Queries the SEC's public submissions API (no auth required) to identify stocks
where multiple corporate insiders made open-market purchases within the lookback
window.  Only transaction code 'P' (open-market purchase) with AcquiredDisposed
code 'A' is counted — option exercises, RSU vesting, and gifts are excluded.

Cluster definition: ≥2 distinct insiders purchasing within `lookback_days`.
Single large purchase: 1 insider buying shares worth > `large_buy_usd`.

Cache lives at logs/insider_cache.json and is refreshed once per calendar day.
Call prefetch_insider_activity() from the 07:00 ET pre-market prefetch job to
warm all symbols before open_sells runs; get_insider_activity() returns cached
data instantly when the cache is warm, avoiding 15–20 minute live EDGAR fetches
during market hours.
"""

import json
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from functools import lru_cache

import requests

from config import EMAIL_FROM, LOG_DIR, STOCK_UNIVERSE, today_et
from data.proxy_comp import get_exec_compensation, match_compensation

logger = logging.getLogger(__name__)

# SEC requires a descriptive User-Agent including contact info.
_USER_AGENT = f"InvestorBot {EMAIL_FROM or 'contact@example.com'}"
_HEADERS = {"User-Agent": _USER_AGENT}

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"
_REQ_DELAY = 0.15  # 10 req/s SEC limit → use 6-7/s to stay safe
_MAX_WORKERS = 10  # concurrent symbols; rate limiter keeps global req/s in check
_CACHE_PATH = os.path.join(LOG_DIR, "caching", "insider_cache.json")

# Global rate limiter: enforces _REQ_DELAY between all EDGAR requests across threads.
_rate_lock = threading.Lock()
_last_req_time: float = 0.0


def _edgar_sleep() -> None:
    global _last_req_time
    with _rate_lock:
        now = time.monotonic()
        gap = _REQ_DELAY - (now - _last_req_time)
        if gap > 0:
            time.sleep(gap)
        _last_req_time = time.monotonic()


# ── cache I/O ─────────────────────────────────────────────────────────────────


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
    except OSError as e:
        logger.warning(f"insider_cache: write error: {e}")


# ── CIK lookup ─────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_cik_map() -> dict[str, str]:
    """Fetch and cache SEC's master ticker→CIK mapping for the process lifetime."""
    try:
        resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in resp.json().values()}
    except Exception as exc:
        logger.warning(f"SEC ticker→CIK map unavailable: {exc}")
        return {}


def _recent_form4_filings(cik: str, lookback_days: int) -> list[dict]:
    """Return Form 4 filing metadata for the given CIK within the lookback window."""
    try:
        _edgar_sleep()
        resp = requests.get(_SUBMISSIONS_URL.format(cik=cik), headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug(f"Submissions fetch failed for CIK {cik}: {exc}")
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    cutoff = date.today() - timedelta(days=lookback_days)
    results = []
    for form, filing_date, accession, doc in zip(forms, dates, accessions, docs, strict=False):
        if form != "4":
            continue
        try:
            fd = date.fromisoformat(filing_date)
        except ValueError:
            continue
        if fd < cutoff:
            break  # submissions sorted newest-first; stop at cutoff
        results.append(
            {"filing_date": filing_date, "accession": accession.replace("-", ""), "doc": doc}
        )
    return results


def _parse_form4(cik: str, accession: str, doc: str) -> list[dict]:
    """Download and parse a Form 4 XML, returning open-market purchase records only."""
    # EDGAR's primaryDocument now points at the XSL-styled HTML view (e.g.
    # "xslF345X06/ownership.xml"), which is not parseable as XML. The raw ownership XML lives in
    # the accession root under the same filename, so strip the leading "xsl.../" styling path.
    if doc.lower().startswith("xsl") and "/" in doc:
        doc = doc.split("/", 1)[1]
    url = _FILING_URL.format(cik_int=int(cik), accession=accession, doc=doc)
    try:
        _edgar_sleep()
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:
        logger.debug(f"Form 4 parse failed {url}: {exc}")
        return []

    reporter = next((e.text or "" for e in root.iter("reportingOwnerName")), "").strip()

    transactions = []
    for txn in root.iter("nonDerivativeTransaction"):
        code = (txn.findtext(".//transactionCode") or "").strip()
        ad = (txn.findtext(".//transactionAcquiredDisposedCode/value") or "").strip()
        if code != "P" or ad != "A":
            continue
        try:
            shares_text = txn.findtext(".//transactionShares/value") or "0"
            price_text = txn.findtext(".//transactionPricePerShare/value") or "0"
            date_text = (txn.findtext(".//transactionDate/value") or "").strip()
            transactions.append(
                {
                    "reporter": reporter,
                    "shares": float(shares_text),
                    "price": float(price_text),
                    "date": date_text,
                }
            )
        except (ValueError, TypeError):
            continue

    return transactions


# ── Live fetch (one symbol at a time, no cache) ────────────────────────────────


def _fetch_one(
    sym: str,
    cik_map: dict[str, str],
    lookback_days: int,
    large_buy_usd: float,
) -> tuple[str, dict | None]:
    """Fetch insider activity for a single symbol. Returns (symbol, data|None)."""
    cik = cik_map.get(sym.upper())
    if not cik:
        return sym, None

    filings = _recent_form4_filings(cik, lookback_days)
    all_txns: list[dict] = []
    for filing in filings:
        all_txns.extend(_parse_form4(cik, filing["accession"], filing["doc"]))

    if not all_txns:
        return sym, None

    unique_insiders = len({t["reporter"] for t in all_txns})
    total_shares = sum(t["shares"] for t in all_txns)
    max_notional = max((t["shares"] * t["price"] for t in all_txns), default=0.0)

    # Strong cluster: ≥3 distinct insiders in last 5 calendar days
    five_ago = date.today() - timedelta(days=5)
    recent_txns = []
    for t in all_txns:
        try:
            if date.fromisoformat(t.get("date", "")) >= five_ago:
                recent_txns.append(t)
        except ValueError:
            pass
    strong_cluster = len({t["reporter"] for t in recent_txns}) >= 3

    # Compensation ratio: max (purchase notional / annual comp) across all transactions
    comp_map = get_exec_compensation(cik)
    comp_ratio = 0.0
    for t in all_txns:
        comp = match_compensation(t["reporter"], comp_map)
        if comp and comp > 0:
            comp_ratio = max(comp_ratio, (t["shares"] * t["price"]) / comp)

    data: dict = {
        "insider_cluster": unique_insiders >= 2,
        "insider_unique_insiders": unique_insiders,
        "insider_transaction_count": len(all_txns),
        "insider_total_shares": total_shares,
        "insider_large_buy": max_notional >= large_buy_usd,
        "insider_strong_cluster": strong_cluster,
        "insider_comp_ratio": comp_ratio,
    }
    logger.info(
        f"Insider {sym}: {unique_insiders} insiders, {len(all_txns)} txns, "
        f"cluster={data['insider_cluster']}"
    )
    return sym, data


def _live_fetch(
    symbols: list[str],
    lookback_days: int = 10,
    large_buy_usd: float = 100_000.0,
) -> dict[str, dict | None]:
    """Fetch insider activity from EDGAR for each symbol (parallel, rate-limited).

    Returns every symbol in the result: data dict when activity exists, None
    when no open-market purchases were found.  None acts as a "fetched, nothing
    found" sentinel so repeated calls within a day don't re-query EDGAR.
    """
    cik_map = _get_cik_map()
    result: dict[str, dict | None] = {}

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, sym, cik_map, lookback_days, large_buy_usd): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym, data = future.result()
            result[sym] = data

    return result


# ── Public prefetch ────────────────────────────────────────────────────────────


def prefetch_insider_activity(
    symbols: list[str] | None = None,
    lookback_days: int = 10,
    large_buy_usd: float = 100_000.0,
) -> int:
    """Warm the same-day insider activity cache before market open.

    Called from the 07:00 ET pre-market prefetch job.  Safe to call multiple
    times — already-cached symbols are skipped.  Discards stale date keys so
    the cache file stays small.

    Returns:
        Number of symbols newly fetched (0 if cache was already warm).
    """
    if symbols is None:
        symbols = list(STOCK_UNIVERSE)
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict | None] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if not missing:
        logger.info(f"insider prefetch: cache already warm ({len(today_cache)} symbols)")
        return 0

    fresh = _live_fetch(missing, lookback_days, large_buy_usd)
    today_cache.update(fresh)
    _save_cache({today: today_cache})  # discard previous date keys

    found = sum(1 for v in today_cache.values() if v is not None)
    logger.info(f"insider prefetch: fetched {len(missing)} symbols, {found} with activity")
    return len(missing)


# ── Public getter ──────────────────────────────────────────────────────────────


def get_insider_activity(
    symbols: list[str],
    lookback_days: int = 10,
    large_buy_usd: float = 100_000.0,
) -> dict[str, dict]:
    """Return open-market insider purchase summary for each symbol.

    Uses the same-day cache populated by prefetch_insider_activity().  On a
    cache miss (e.g. symbol added after the prefetch ran), falls back to a live
    EDGAR fetch and saves the result before returning.

    Result schema per symbol::

        {
            "insider_cluster":          bool,   # ≥2 distinct insiders bought
            "insider_unique_insiders":  int,
            "insider_transaction_count": int,
            "insider_total_shares":     float,
            "insider_large_buy":        bool,   # single buy > large_buy_usd
            "insider_strong_cluster":   bool,   # ≥3 distinct insiders in last 5 days
            "insider_comp_ratio":       float,  # max(notional / annual_comp) across txns
        }

    Symbols with no Form 4 activity in the window are omitted from the result.
    All network errors are caught and logged; callers always receive a plain dict.
    """
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict | None] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if missing:
        fresh = _live_fetch(missing, lookback_days, large_buy_usd)
        today_cache.update(fresh)
        _save_cache({today: today_cache})

    return {sym: v for sym in symbols if (v := today_cache.get(sym)) is not None}
