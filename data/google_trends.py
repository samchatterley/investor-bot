"""Google Trends signal using pytrends.

Detects unusual search volume spikes for a stock's ticker or company name.
A spike (current week > 150% of 12-week average) in a positive market context
supplements the bullish signal.

Requires the `pytrends` package. Degrades gracefully to False when pytrends is
not installed or the request fails.

Cache lives at logs/google_trends_cache.json — refreshed once per calendar day.
"""

import json
import logging
import os
import time

from config import LOG_DIR, today_et

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "google_trends_cache.json")
_REQ_DELAY = 1.0  # pytrends rate-limit is strict — 1s between requests
_SPIKE_THRESHOLD = 1.5  # current week / 12w avg must exceed this
_MIN_BASELINE = 10  # skip symbols with near-zero baseline interest


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
        logger.warning(f"google_trends_cache: write error: {e}")


# ── live fetch ────────────────────────────────────────────────────────────────


def _live_fetch_trends(symbols: list[str]) -> dict[str, bool]:
    """Fetch Google Trends data for each symbol. Returns {sym: spike_detected}."""
    try:
        from pytrends.request import TrendReq  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("google_trends: pytrends not installed — signals disabled")
        return dict.fromkeys(symbols, False)

    result: dict[str, bool] = {}
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
    except Exception as exc:
        logger.debug(f"google_trends: TrendReq init failed: {exc}")
        return dict.fromkeys(symbols, False)

    for sym in symbols:
        try:
            time.sleep(_REQ_DELAY)
            pytrends.build_payload([sym], cat=0, timeframe="today 3-m", geo="US")
            df = pytrends.interest_over_time()
            if df is None or df.empty or sym not in df.columns:
                result[sym] = False
                continue

            values = df[sym].tolist()
            if len(values) < 4:
                result[sym] = False
                continue

            baseline_avg = sum(values[:-1]) / len(values[:-1])
            current = values[-1]

            if baseline_avg < _MIN_BASELINE:
                result[sym] = False
                continue

            result[sym] = current >= baseline_avg * _SPIKE_THRESHOLD
        except Exception as exc:
            logger.debug(f"google_trends: fetch failed for {sym}: {exc}")
            result[sym] = False

    return result


# ── Public getter ──────────────────────────────────────────────────────────────


def get_google_trends_signals(symbols: list[str]) -> dict[str, bool]:
    """Return {symbol: spike_detected} for each symbol.

    True when current-week search volume exceeds 150% of the 12-week baseline.
    Uses same-day cache. Falls back to live fetch on miss.
    Always returns an entry for every input symbol (False when no spike or on error).
    """
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, bool] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if missing:
        fresh = _live_fetch_trends(missing)
        today_cache.update(fresh)
        _save_cache({today: today_cache})

    return {sym: today_cache.get(sym, False) for sym in symbols}
