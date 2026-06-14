"""Short interest data fetcher — live signal enrichment only.

Fetches yfinance.info["shortRatio"] (days-to-cover) for each symbol.
A high days-to-cover ratio (≥ 5) indicates heavy short positioning and
elevated squeeze / mean-reversion risk, which reinforces a bearish thesis
when combined with technical breakdown signals.

NOT BACKTESTABLE: yfinance.info provides the current snapshot value only;
no time-series history is available.  This signal is therefore live-only
and is excluded from backtest/engine.py.

Cache lives at logs/short_interest_cache.json and is refreshed once per
calendar day.  Call prefetch_short_interest() from the 07:00 ET pre-market
prefetch job to warm all symbols before open_sells/open_buys run;
get_short_interest() returns cached data instantly when warm.

Threshold: shortRatio >= _MIN_SHORT_RATIO_DAYS (default 5 days).
"""

import json
import logging
import os
import time

import yfinance as yf

from config import ETF_SYMBOLS, LOG_DIR, STOCK_UNIVERSE, today_et

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_MIN_SHORT_RATIO_DAYS = 5.0  # days-to-cover threshold for "high short interest"
_REQ_DELAY = 0.05
_CACHE_PATH = os.path.join(LOG_DIR, "short_interest_cache.json")


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
        logger.warning(f"short_interest_cache: write error: {e}")


# ── Live fetch ─────────────────────────────────────────────────────────────────


def _live_fetch_short_interest(
    symbols: list[str],
    min_short_ratio: float = _MIN_SHORT_RATIO_DAYS,
) -> dict[str, dict | None]:
    """Fetch shortRatio from yfinance.info for each symbol.

    Returns every symbol in the result: data dict when short ratio exceeds the
    threshold, None otherwise.  None acts as a "fetched, not qualifying"
    sentinel so the symbol is not re-queried within the same calendar day.
    """
    result: dict[str, dict | None] = {}

    for sym in symbols:
        if sym in ETF_SYMBOLS:
            result[sym] = None
            continue
        try:
            time.sleep(_REQ_DELAY)
            info = yf.Ticker(sym).info
        except Exception as exc:
            logger.debug(f"short_interest fetch failed for {sym}: {exc}")
            result[sym] = None
            continue

        short_ratio = info.get("shortRatio")
        if short_ratio is None:
            result[sym] = None
            continue

        try:
            ratio = float(short_ratio)
        except (TypeError, ValueError):
            result[sym] = None
            continue

        short_pct_float = info.get("shortPercentOfFloat")
        try:
            pct_float = float(short_pct_float) if short_pct_float is not None else None
        except (TypeError, ValueError):
            pct_float = None

        if ratio < min_short_ratio:
            result[sym] = None
        else:
            result[sym] = {
                "short_ratio": ratio,
                "high_short_interest": True,
                "short_pct_float": pct_float,
            }

    return result


# ── Public prefetch ────────────────────────────────────────────────────────────


def prefetch_short_interest(
    symbols: list[str] | None = None,
    min_short_ratio: float = _MIN_SHORT_RATIO_DAYS,
) -> int:
    """Warm the same-day short interest cache before market open.

    Called from the 07:00 ET pre-market prefetch job.  Safe to call multiple
    times — already-cached symbols are skipped.  Discards stale date keys so
    the cache file stays small.

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
        logger.info(f"short_interest prefetch: cache already warm ({len(today_cache)} symbols)")
        return 0

    fresh = _live_fetch_short_interest(missing, min_short_ratio)
    today_cache.update(fresh)
    _save_cache({today: today_cache})

    found = sum(1 for v in today_cache.values() if v is not None)
    logger.info(f"short_interest prefetch: fetched {len(missing)} symbols, {found} with high SI")
    return len(missing)


# ── Public getter ──────────────────────────────────────────────────────────────


def get_short_interest(
    symbols: list[str],
    min_short_ratio: float = _MIN_SHORT_RATIO_DAYS,
) -> dict[str, dict]:
    """Return short interest data for each symbol with shortRatio >= min_short_ratio.

    Uses the same-day cache populated by prefetch_short_interest().  Falls back
    to a live yfinance fetch on cache miss.

    Result schema per symbol::

        {
            "short_ratio":              float,        # days-to-cover from yfinance.info
            "high_short_interest":      bool,         # always True when present
            "short_pct_float":          float | None, # short interest as fraction of float
        }

    Symbols with no qualifying data (or below threshold) are omitted.
    All network errors are caught and logged; callers always receive a plain dict.
    """
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict | None] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if missing:
        fresh = _live_fetch_short_interest(missing, min_short_ratio)
        today_cache.update(fresh)
        _save_cache({today: today_cache})

    return {sym: v for sym in symbols if (v := today_cache.get(sym)) is not None}
