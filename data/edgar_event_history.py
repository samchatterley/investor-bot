"""Historical EDGAR filing-event feed — point-in-time corporate events for backtesting.

The SEC submissions API (data.sec.gov) returns ~1000 recent filings per company (10+ years for many
names) with filing dates, free and point-in-time. This module exposes them as a historical event
archive so the bot's *catalyst* signals — previously live-only with no history to backtest against —
can finally be validated:

  * secondary_offering_short   → 424B* / S-3 / S-1 filings (dilution / supply shock)
  * activist_13d_signal        → SC 13D / SC 13D/A filings
  * insider_buying / _selling  → Form 4 filings (clustering by date; direction needs the XML)
  * accounting_concern_short   → 8-K with item 4.01 (auditor change) / 4.02 (non-reliance)

One request per symbol (reuses edgar_client's CIK map, headers and rate-limit); cached to disk keyed
by calendar day so a backtest re-run is instant. Parsing is a pure function (fully unit-testable);
the single network call is isolated in ``_fetch_submissions`` for mocking.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date

import requests

import data.edgar_client as _ec
from config import LOG_DIR

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "caching", "edgar_event_history.json")

# Convenience form-prefix groups for the catalyst signals above.
OFFERING_FORMS = ("424B", "S-3", "S-1")
ACTIVIST_FORMS = ("SC 13D",)
INSIDER_FORMS = ("4",)
# 8-K item codes that flag a hard governance red flag (auditor change / non-reliance).
ACCOUNTING_ITEMS = ("4.01", "4.02")


def _fetch_submissions(cik: str) -> dict:
    """Raw submissions JSON for a CIK (the one network call — mocked in tests)."""
    import time

    time.sleep(_ec._REQ_DELAY)
    resp = requests.get(_ec._SUBMISSIONS_URL.format(cik=cik), headers=_ec._HEADERS, timeout=20)
    resp.raise_for_status()
    return dict(resp.json())


def _parse_events(submissions: dict, forms: tuple[str, ...] | None) -> list[dict]:
    """Pure: extract ``[{date, form, accession, items}]`` from a submissions JSON, newest-first.

    ``forms`` is a tuple of form-name *prefixes* (e.g. ``("424B", "S-3")``); None returns all.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    items = recent.get("items", [""] * len(form_list))
    out: list[dict] = []
    for form, filing_date, accession, item in zip(
        form_list, dates, accessions, items, strict=False
    ):
        if forms is not None and not any(form.startswith(p) for p in forms):
            continue
        try:
            date.fromisoformat(filing_date)
        except (ValueError, TypeError):
            continue
        out.append(
            {
                "date": filing_date,
                "form": form,
                "accession": str(accession).replace("-", ""),
                "items": str(item or ""),
            }
        )
    return out


def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH) as f:
            return dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def fetch_events(
    symbol: str, forms: tuple[str, ...] | None = None, use_cache: bool = True
) -> list[dict]:
    """Historical filing events for ``symbol`` (newest-first), optionally filtered by form prefix.

    Cached per symbol keyed by calendar day. Returns [] for unknown tickers or on network failure.
    """
    sym = symbol.upper()
    today = date.today().isoformat()
    cache = _load_cache() if use_cache else {}
    entry = cache.get(sym)
    if use_cache and entry is not None and entry.get("day") == today:
        events = entry["events"]
    else:
        cik = _ec._get_cik_map().get(sym)
        if not cik:
            return []
        try:
            subs = _fetch_submissions(cik)
        except Exception as exc:
            logger.debug("edgar_event_history: fetch failed %s: %s", sym, exc)
            return []
        events = _parse_events(subs, None)
        if use_cache:
            cache[sym] = {"day": today, "events": events}
            _save_cache(cache)
    if forms is None:
        return list(events)
    return [e for e in events if any(e["form"].startswith(p) for p in forms)]
