"""FRED (Federal Reserve Economic Data) client with daily disk cache.

Fetches macro series used by regime_v2 and macro signals.
All functions degrade gracefully when FRED_API_KEY is unset or network fails.

Cached series (updated daily):
  T10Y2Y  — 10-year minus 2-year treasury yield spread (yield curve)
  ICSA    — Initial jobless claims (weekly, seasonally adjusted)
  UMCSENT — University of Michigan consumer sentiment (monthly)
  FEDFUNDS — Effective federal funds rate (daily)
"""

import json
import logging
import os
from math import isnan
from pathlib import Path

from config import LOG_DIR, today_et

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(LOG_DIR) / "fred_cache.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str | None:
    """Return FRED_API_KEY from environment, or None if not set."""
    return os.environ.get("FRED_API_KEY") or None


def _load_cache() -> dict:
    """Load full cache dict from disk. Returns {} on miss/corrupt."""
    try:
        with open(_CACHE_PATH) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("fred_cache: failed to load cache: %s", exc)
        return {}


def _save_cache(cache: dict) -> None:
    """Save cache to disk. Silently swallows OSError."""
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w") as fh:
            json.dump(cache, fh)
    except OSError as exc:
        logger.warning("fred_cache: failed to save cache: %s", exc)


def _is_series_stale(entry: dict) -> bool:
    """True if cached entry's fetched_date is not today (ET)."""
    return entry.get("fetched_date") != today_et().isoformat()


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------


def fetch_series(series_id: str, observation_start: str = "2020-01-01") -> list[tuple[str, float]]:
    """Fetch a FRED series. Returns list of (date_str, value) pairs, newest last.

    Uses disk cache — only hits FRED API when cache is stale (not today).
    Returns [] when API key unset, network fails, or series unavailable.
    Uses fredapi.Fred(api_key=key).get_series(series_id, observation_start=start).
    Converts the resulting pd.Series (DatetimeIndex → float) to list of
    (isoformat_date, float). Filters out NaN values.
    """
    key = _get_api_key()
    if not key:
        return []

    cache = _load_cache()
    entry = cache.get(series_id, {})

    if entry and not _is_series_stale(entry):
        return [tuple(pair) for pair in entry["data"]]  # type: ignore[return-value]

    try:
        from fredapi import Fred  # type: ignore[import-untyped]

        fred = Fred(api_key=key)
        raw = fred.get_series(series_id, observation_start=observation_start)
        pairs: list[tuple[str, float]] = [
            (ts.date().isoformat(), float(val)) for ts, val in raw.items() if not isnan(float(val))
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("fred_client: failed to fetch %s: %s", series_id, exc)
        return []

    today_str = today_et().isoformat()
    cache[series_id] = {"fetched_date": today_str, "data": pairs}
    _save_cache(cache)
    return pairs


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_yield_curve() -> float | None:
    """Return latest T10Y2Y value (10y minus 2y spread in percentage points).

    Positive = normal curve. Negative = inverted (recession risk).
    Returns None on any failure.
    """
    data = fetch_series("T10Y2Y")
    if not data:
        return None
    return data[-1][1]


def get_yield_curve_inverted_days() -> int:
    """Return number of consecutive recent days where T10Y2Y < 0.

    Returns 0 if curve is not currently inverted or data unavailable.
    """
    data = fetch_series("T10Y2Y")
    if not data:
        return 0
    count = 0
    for _, value in reversed(data):
        if value < 0:
            count += 1
        else:
            break
    return count


def get_claims_4w_ma() -> float | None:
    """Return 4-week moving average of initial jobless claims (ICSA).

    Returns None if fewer than 4 data points available.
    """
    data = fetch_series("ICSA")
    if len(data) < 4:
        return None
    recent = [v for _, v in data[-4:]]
    return sum(recent) / 4


def get_claims_trend() -> dict:
    """Return a dict describing the jobless claims trend.

    Returns:
        {
            "latest": float | None,       # most recent weekly reading
            "ma_4w": float | None,        # 4-week MA
            "rising_weeks": int,          # consecutive weeks MA has been rising
            "deteriorating": bool,        # True when rising_weeks >= 6
        }
    Returns all-None/False dict on any failure.
    """
    _empty: dict = {
        "latest": None,
        "ma_4w": None,
        "rising_weeks": 0,
        "deteriorating": False,
    }
    data = fetch_series("ICSA")
    if len(data) < 4:
        return _empty

    latest = data[-1][1]
    ma_4w = get_claims_4w_ma()

    # Count consecutive weeks where the 4-week MA has been rising.
    # We compute rolling 4-week MAs starting from the most recent window and
    # walk backwards to find the first non-rising comparison.
    rising_weeks = 0
    values = [v for _, v in data]
    # Need at least 5 points to compare two consecutive 4-week windows.
    for i in range(len(values) - 4, 0, -1):
        current_ma = sum(values[i : i + 4]) / 4
        previous_ma = sum(values[i - 1 : i + 3]) / 4
        if current_ma > previous_ma:
            rising_weeks += 1
        else:
            break

    return {
        "latest": latest,
        "ma_4w": ma_4w,
        "rising_weeks": rising_weeks,
        "deteriorating": rising_weeks >= 6,
    }


def get_macro_snapshot() -> dict:
    """Return combined macro snapshot for regime classification.

    Returns:
        {
            "yield_curve": float | None,          # T10Y2Y latest
            "yield_curve_inverted": bool,          # T10Y2Y < 0
            "yield_curve_inverted_days": int,      # consecutive inversion days
            "claims_deteriorating": bool,          # rising_weeks >= 6
            "claims_ma_4w": float | None,
            "fedfunds": float | None,              # latest fed funds rate
            "data_available": bool,                # False when key unset or all None
        }
    """
    yield_curve = get_yield_curve()
    inverted_days = get_yield_curve_inverted_days()
    claims = get_claims_trend()
    fedfunds_data = fetch_series("FEDFUNDS")
    fedfunds = fedfunds_data[-1][1] if fedfunds_data else None

    data_available = any(v is not None for v in (yield_curve, claims["ma_4w"], fedfunds))

    return {
        "yield_curve": yield_curve,
        "yield_curve_inverted": yield_curve is not None and yield_curve < 0,
        "yield_curve_inverted_days": inverted_days,
        "claims_deteriorating": claims["deteriorating"],
        "claims_ma_4w": claims["ma_4w"],
        "fedfunds": fedfunds,
        "data_available": data_available,
    }
