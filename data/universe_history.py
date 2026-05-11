"""Point-in-time universe integrity for backtest walk-forward.

Two free data sources are combined:

1. Hard-coded IPO / listing dates (_FIRST_TRADEABLE dict) — prevents lookahead
   bias from using symbols that weren't yet public on the fold's start date.
   Sources: SEC EDGAR, exchange records, Wikipedia.

2. Wikipedia S&P 500 constituent-change table — optional enrichment that maps
   each date to the set of symbols actually in the S&P 500 at that time.
   Fetched once per process and cached.  Falls back gracefully when unavailable
   (CI, offline, rate-limited).

Usage::

    from data.universe_history import get_universe_for_date
    symbols = get_universe_for_date(date(2018, 1, 2), candidate_list)
"""

from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# First-trading dates for symbols with known IPOs / listings after 2010.
# Symbols absent from this dict are assumed to have been trading since 2005.
# ---------------------------------------------------------------------------
_FIRST_TRADEABLE: dict[str, date] = {
    # 2019
    "LYFT": date(2019, 3, 29),
    "UBER": date(2019, 5, 10),
    # 2020
    "SNOW": date(2020, 9, 16),
    "PLTR": date(2020, 9, 30),
    "DASH": date(2020, 12, 9),
    "ABNB": date(2020, 12, 10),
    # 2021
    "AFRM": date(2021, 1, 13),
    "COIN": date(2021, 4, 14),
    "SOFI": date(2021, 6, 1),
    "HOOD": date(2021, 7, 29),
    "RKLB": date(2021, 8, 25),
    "GTLB": date(2021, 10, 14),
    "MNDY": date(2021, 6, 10),
    "APP": date(2021, 12, 14),
    # 2023
    "PCOR": date(2023, 10, 12),
    # Ticker change: Block Inc renamed SQ → XYZ on 2023-11-01.
    # Data under "XYZ" starts on that date; earlier history lives under "SQ".
    "XYZ": date(2023, 11, 1),
}

# Last tradeable date for delisted / acquired symbols.
# Extend as symbols in the universe are removed.
_LAST_TRADEABLE: dict[str, date] = {}


# ---------------------------------------------------------------------------
# Wikipedia S&P 500 constituent history (optional enrichment)
# ---------------------------------------------------------------------------

_SP500_TABLE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@lru_cache(maxsize=1)
def _fetch_sp500_changes() -> list[dict]:
    """Return a list of {symbol, added, removed} dicts from Wikipedia.

    Falls back to empty list on any error (network, parse, etc.).
    The caller must treat the empty list as 'no S&P 500 filter applied'.
    """
    try:
        tables = pd.read_html(_SP500_TABLE_URL)
        if len(tables) < 2:
            return []
        changes_df = tables[1]

        # Wikipedia table columns vary by year; normalise to known names.
        changes_df.columns = [str(c).strip().lower() for c in changes_df.columns]

        # Find date, added, removed columns
        date_col = next((c for c in changes_df.columns if "date" in c or "year" in c), None)
        added_col = next((c for c in changes_df.columns if "added" in c or "ticker" in c), None)
        removed_col = next((c for c in changes_df.columns if "removed" in c), None)
        if date_col is None or added_col is None:
            logger.warning("universe_history: unexpected Wikipedia table structure")
            return []

        records: list[dict] = []
        for _, row in changes_df.iterrows():
            raw_date = row.get(date_col)
            if pd.isna(raw_date):
                continue
            try:
                change_date = pd.to_datetime(str(raw_date)).date()
            except Exception:
                continue
            added_sym = str(row.get(added_col, "")).strip().upper()
            if added_sym and added_sym != "NAN":
                records.append({"symbol": added_sym, "added": change_date, "removed": None})
            if removed_col:
                removed_sym = str(row.get(removed_col, "")).strip().upper()
                if removed_sym and removed_sym != "NAN":
                    records.append({"symbol": removed_sym, "added": None, "removed": change_date})
        logger.info(f"universe_history: loaded {len(records)} S&P 500 change events from Wikipedia")
        return records
    except Exception as exc:
        logger.warning(f"universe_history: Wikipedia S&P 500 fetch failed ({exc}); skipping filter")
        return []


def _build_sp500_membership(changes: list[dict]) -> dict[str, tuple[date | None, date | None]]:
    """Convert change events to {symbol: (earliest_add, latest_remove)} mapping."""
    membership: dict[str, tuple[date | None, date | None]] = {}
    for rec in changes:
        sym = rec["symbol"]
        current = membership.get(sym, (None, None))
        added = rec.get("added")
        removed = rec.get("removed")
        new_added = min(filter(None, [current[0], added])) if any([current[0], added]) else None
        new_removed = (
            max(filter(None, [current[1], removed])) if any([current[1], removed]) else current[1]
        )
        membership[sym] = (new_added, new_removed)
    return membership


def get_universe_for_date(
    dt: date,
    candidates: list[str],
    apply_sp500_filter: bool = False,
) -> list[str]:
    """Return the subset of candidates that were publicly tradeable on dt.

    Parameters
    ----------
    dt : date
        The point-in-time date to check against.
    candidates : list[str]
        Full candidate symbol list.
    apply_sp500_filter : bool
        If True, additionally restrict to symbols with evidence of S&P 500
        membership on dt (sourced from Wikipedia).  Defaults to False because
        the Wikipedia table only covers additions since ~2000 and would
        incorrectly exclude long-standing members with no recorded add event.

    Returns
    -------
    list[str]
        Filtered symbol list preserving the input order.
    """
    sp500_membership: dict[str, tuple[date | None, date | None]] = {}
    if apply_sp500_filter:
        changes = _fetch_sp500_changes()
        if changes:
            sp500_membership = _build_sp500_membership(changes)

    result: list[str] = []
    for sym in candidates:
        # IPO / listing date gate
        first = _FIRST_TRADEABLE.get(sym)
        if first is not None and dt < first:
            continue
        # Delisted / acquired gate
        last = _LAST_TRADEABLE.get(sym)
        if last is not None and dt > last:
            continue
        # Optional S&P 500 membership gate (only applied when data available)
        if apply_sp500_filter and sp500_membership:
            membership = sp500_membership.get(sym)
            if membership is not None:
                added, removed = membership
                if added is not None and dt < added:
                    continue
                if removed is not None and dt > removed:
                    continue
        result.append(sym)
    return result
