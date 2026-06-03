"""Post-Earnings Announcement Drift (PEAD) candidate detection.

Identifies stocks where the most recent earnings report beat analyst consensus
by at least _MIN_SURPRISE_PCT within the last _PEAD_WINDOW_DAYS.  Stocks that
beat strongly tend to drift upward for 30–60 days as the full market gradually
reprices the new earnings trajectory (classic PEAD / SUE literature).

Data source: yfinance earnings_dates — covers the last ~4 quarters with
EPS Estimate, Reported EPS, and Surprise(%).  No API key required.

Important: this module returns metadata, not a trading signal.  The actual
prefilter signal (``pead``) is evaluated in execution/stock_scanner.py using
these fields together with the snapshot's live technical state.

Cache lives at logs/earnings_cache.json and is refreshed once per calendar day.
Call prefetch_earnings_data() from the 07:00 ET pre-market prefetch job to warm
all symbols before open_sells/open_buys run; get_earnings_surprise() and
get_earnings_miss() return cached data instantly when warm.  Both functions
share a single per-symbol yfinance fetch so the cache is twice as efficient as
the sequential live path that previously fetched each symbol twice.
"""

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf

from config import ETF_SYMBOLS, LOG_DIR, STOCK_UNIVERSE, today_et

logger = logging.getLogger(__name__)
# yfinance logs at ERROR when no earnings exist (normal for ETFs) — suppress that noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_MIN_SURPRISE_PCT = 5.0  # EPS beat threshold (%)
_PEAD_WINDOW_DAYS = 30  # only consider surprises within this many days
_MAX_MISS_PCT = -5.0  # EPS miss threshold — surprise must be at most this negative
_REQ_DELAY = 0.05  # modest delay — yfinance is not rate-limited by key
_CACHE_PATH = os.path.join(LOG_DIR, "earnings_cache.json")


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
        logger.warning(f"earnings_cache: write error: {e}")


# ── Live fetch (computes beat + miss in one pass per symbol) ───────────────────


def _live_fetch_earnings(
    symbols: list[str],
    lookback_days: int = _PEAD_WINDOW_DAYS,
) -> dict[str, dict | None]:
    """Fetch earnings_dates once per symbol and compute both beat and miss results.

    Uses default thresholds (_MIN_SURPRISE_PCT, _MAX_MISS_PCT).  Returns every
    symbol in the result: {"surprise": dict|None, "miss": dict|None} when
    earnings data exists, or None when the symbol is an ETF or has no data.
    None acts as a "fetched, nothing found" sentinel so the symbol is not
    re-queried within the same calendar day.
    """
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    result: dict[str, dict | None] = {}

    for sym in symbols:
        if sym in ETF_SYMBOLS:
            result[sym] = None
            continue
        try:
            time.sleep(_REQ_DELAY)
            df: pd.DataFrame | None = yf.Ticker(sym).earnings_dates
        except Exception as exc:
            logger.debug(f"earnings_dates fetch failed for {sym}: {exc}")
            result[sym] = None
            continue

        if df is None or df.empty:
            result[sym] = None
            continue

        df = df.dropna(subset=["Reported EPS", "Surprise(%)"])
        if df.empty:
            result[sym] = None
            continue

        df = df[df.index >= pd.Timestamp(cutoff)]
        if df.empty:
            result[sym] = None
            continue

        row = df.iloc[0]
        surprise_pct = float(row["Surprise(%)"])
        earnings_ts = row.name
        earnings_date_str = earnings_ts.date().isoformat()
        days_ago = (datetime.now(UTC) - earnings_ts.to_pydatetime()).days

        surprise_data = None
        if surprise_pct >= _MIN_SURPRISE_PCT:
            surprise_data = {
                "earnings_surprise_pct": round(surprise_pct, 2),
                "earnings_date": earnings_date_str,
                "earnings_days_ago": days_ago,
                "pead_candidate": True,
            }
            logger.debug(
                f"PEAD {sym}: +{surprise_pct:.1f}% surprise on {earnings_date_str} ({days_ago}d ago)"
            )

        miss_data = None
        if surprise_pct <= _MAX_MISS_PCT:
            miss_data = {
                "earnings_miss_pct": round(surprise_pct, 2),
                "earnings_miss_date": earnings_date_str,
                "earnings_miss_days_ago": days_ago,
                "earnings_miss_candidate": True,
            }
            logger.debug(
                f"Neg-PEAD {sym}: {surprise_pct:.1f}% miss on {earnings_date_str} ({days_ago}d ago)"
            )

        result[sym] = {"surprise": surprise_data, "miss": miss_data}

    return result


# ── Public prefetch ────────────────────────────────────────────────────────────


def prefetch_earnings_data(
    symbols: list[str] | None = None,
    lookback_days: int = _PEAD_WINDOW_DAYS,
) -> int:
    """Warm the same-day earnings cache before market open.

    Called from the 07:00 ET pre-market prefetch job.  Fetches earnings_dates
    once per symbol and stores both beat and miss results in a single pass.
    Safe to call multiple times — already-cached symbols are skipped.  Discards
    stale date keys so the cache file stays small.

    Returns:
        Number of symbols newly fetched.
    """
    if symbols is None:
        symbols = list(STOCK_UNIVERSE)
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict | None] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if not missing:
        logger.info(f"earnings prefetch: cache already warm ({len(today_cache)} symbols)")
        return 0

    fresh = _live_fetch_earnings(missing, lookback_days)
    today_cache.update(fresh)
    _save_cache({today: today_cache})

    n_surprise = sum(
        1 for v in today_cache.values() if v is not None and v.get("surprise") is not None
    )
    n_miss = sum(1 for v in today_cache.values() if v is not None and v.get("miss") is not None)
    logger.info(
        f"earnings prefetch: fetched {len(missing)} symbols, "
        f"{n_surprise} PEAD candidates, {n_miss} neg-PEAD candidates"
    )
    return len(missing)


# ── Public getters ─────────────────────────────────────────────────────────────


def get_earnings_surprise(
    symbols: list[str],
    lookback_days: int = _PEAD_WINDOW_DAYS,
    min_surprise: float = _MIN_SURPRISE_PCT,
) -> dict[str, dict]:
    """Return PEAD candidate data for each symbol that beat recent estimates.

    Uses the same-day cache populated by prefetch_earnings_data().  On a cache
    miss (e.g. symbol added after the prefetch ran), falls back to a live
    yfinance fetch and saves the result before returning.

    Result schema per symbol::

        {
            "earnings_surprise_pct": float,   # EPS beat percentage (e.g. 6.34)
            "earnings_date":         str,      # ISO date of the surprise
            "earnings_days_ago":     int,      # calendar days since the report
            "pead_candidate":        bool,     # always True when present
        }

    Symbols with no recent qualifying earnings beat are omitted.
    All network errors are caught and logged; callers always receive a plain dict.
    """
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict | None] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if missing:
        fresh = _live_fetch_earnings(missing, lookback_days)
        today_cache.update(fresh)
        _save_cache({today: today_cache})

    result: dict[str, dict] = {}
    for sym in symbols:
        entry = today_cache.get(sym)
        if (
            entry is not None
            and (surprise := entry.get("surprise")) is not None
            and surprise["earnings_surprise_pct"] >= min_surprise
        ):
            result[sym] = surprise

    logger.info(f"PEAD: {len(result)}/{len(symbols)} symbols with qualifying surprise")
    return result


def get_earnings_miss(
    symbols: list[str],
    lookback_days: int = _PEAD_WINDOW_DAYS,
    max_miss: float = _MAX_MISS_PCT,
) -> dict[str, dict]:
    """Return negative-PEAD candidate data for each symbol that missed recent estimates.

    Uses the same-day cache populated by prefetch_earnings_data() or by a prior
    call to get_earnings_surprise() for the same symbols — no re-fetch needed.

    Result schema per symbol::

        {
            "earnings_miss_pct":       float,  # EPS miss percentage (e.g. -8.2)
            "earnings_miss_date":      str,     # ISO date of the miss
            "earnings_miss_days_ago":  int,     # calendar days since the report
            "earnings_miss_candidate": bool,    # always True when present
        }

    Symbols with no recent qualifying miss are omitted.
    All network errors are caught and logged; callers always receive a plain dict.
    """
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict | None] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if missing:
        fresh = _live_fetch_earnings(missing, lookback_days)
        today_cache.update(fresh)
        _save_cache({today: today_cache})

    result: dict[str, dict] = {}
    for sym in symbols:
        entry = today_cache.get(sym)
        if (
            entry is not None
            and (miss := entry.get("miss")) is not None
            and miss["earnings_miss_pct"] <= max_miss
        ):
            result[sym] = miss

    logger.info(f"Neg-PEAD: {len(result)}/{len(symbols)} symbols with qualifying miss")
    return result
