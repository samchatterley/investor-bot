"""SEC EDGAR Form 4 fetcher — open-market insider purchase cluster detection.

Queries the SEC's public submissions API (no auth required) to identify stocks
where multiple corporate insiders made open-market purchases within the lookback
window.  Only transaction code 'P' (open-market purchase) with AcquiredDisposed
code 'A' is counted — option exercises, RSU vesting, and gifts are excluded.

Cluster definition: ≥2 distinct insiders purchasing within `lookback_days`.
Single large purchase: 1 insider buying shares worth > `large_buy_usd`.
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from functools import lru_cache

import requests

from config import EMAIL_FROM

logger = logging.getLogger(__name__)

# SEC requires a descriptive User-Agent including contact info.
_USER_AGENT = f"InvestorBot {EMAIL_FROM or 'contact@example.com'}"
_HEADERS = {"User-Agent": _USER_AGENT}

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"
_REQ_DELAY = 0.15  # 10 req/s SEC limit → use 6-7/s to stay safe


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
        time.sleep(_REQ_DELAY)
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
    url = _FILING_URL.format(cik_int=int(cik), accession=accession, doc=doc)
    try:
        time.sleep(_REQ_DELAY)
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


def get_insider_activity(
    symbols: list[str],
    lookback_days: int = 10,
    large_buy_usd: float = 100_000.0,
) -> dict[str, dict]:
    """Return open-market insider purchase summary for each symbol.

    Result schema per symbol::

        {
            "insider_cluster":        bool,   # ≥2 distinct insiders bought
            "insider_unique_insiders": int,
            "insider_transaction_count": int,
            "insider_total_shares":   float,
            "insider_large_buy":      bool,   # single buy > large_buy_usd
        }

    Symbols with no Form 4 activity in the window are omitted from the result.
    All network errors are caught and logged; callers always receive a plain dict.
    """
    cik_map = _get_cik_map()
    result: dict[str, dict] = {}

    for sym in symbols:
        cik = cik_map.get(sym.upper())
        if not cik:
            continue

        filings = _recent_form4_filings(cik, lookback_days)
        all_txns: list[dict] = []
        for filing in filings:
            all_txns.extend(_parse_form4(cik, filing["accession"], filing["doc"]))

        if not all_txns:
            continue

        unique_insiders = len({t["reporter"] for t in all_txns})
        total_shares = sum(t["shares"] for t in all_txns)
        max_notional = max((t["shares"] * t["price"] for t in all_txns), default=0.0)

        result[sym] = {
            "insider_cluster": unique_insiders >= 2,
            "insider_unique_insiders": unique_insiders,
            "insider_transaction_count": len(all_txns),
            "insider_total_shares": total_shares,
            "insider_large_buy": max_notional >= large_buy_usd,
        }
        logger.info(
            f"Insider {sym}: {unique_insiders} insiders, {len(all_txns)} txns, "
            f"cluster={result[sym]['insider_cluster']}"
        )

    return result
