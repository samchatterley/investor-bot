"""IPO lockup expiry calendar.

Tracks 180-day lockup expiry dates for recently-IPO'd stocks. In the 5–10 days
before lockup expiry, insiders and early investors can first sell shares, creating
supply-side pressure. This is a reliable short setup in the absence of strong
fundamental catalysts.

Data source: IPO dates are stored in logs/ipo_dates.json, maintained by the
weekly prefetch job via yfinance ticker.info["ipoExpectedDate"] / earliest
price-history date. Entries older than 18 months are automatically pruned.

Result schema::

    {
        "lockup_expiry_soon":  bool,   # True when lockup expires within 5–10 days
        "days_to_lockup":      int,    # Days until lockup expiry (negative if passed)
        "ipo_date":            str,    # ISO date of IPO
    }
"""

import json
import logging
import os
from datetime import date, timedelta

import yfinance as yf

from config import LOG_DIR, STOCK_UNIVERSE, today_et

logger = logging.getLogger(__name__)

_IPO_CACHE_PATH = os.path.join(LOG_DIR, "ipo_dates.json")
_LOCKUP_DAYS = 180
_ALERT_WINDOW_EARLY = 10  # start alerting 10 days before lockup
_ALERT_WINDOW_LATE = 5  # stop alerting 5 days before (earliest visible pressure)
_IPO_MAX_AGE_MONTHS = 18  # only track stocks IPO'd within 18 months


# ── cache I/O ─────────────────────────────────────────────────────────────────


def _load_ipo_cache() -> dict:
    try:
        with open(_IPO_CACHE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_ipo_cache(cache: dict) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_IPO_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError as e:
        logger.warning(f"ipo_dates_cache: write error: {e}")


# ── IPO date detection ────────────────────────────────────────────────────────


def _detect_ipo_date(sym: str) -> str | None:
    """Estimate IPO date from earliest available yfinance price history.

    Returns ISO date string or None if unavailable / older than 18 months.
    """
    try:
        t = yf.Ticker(sym)

        # Prefer the ipoExpectedDate field if available
        info = t.info or {}
        ipo_str = info.get("ipoExpectedDate") or info.get("firstTradeDateEpochUtc")
        if ipo_str:
            try:
                if isinstance(ipo_str, (int, float)):
                    from datetime import UTC, datetime

                    ipo_date = datetime.fromtimestamp(ipo_str, tz=UTC).date()
                else:
                    ipo_date = date.fromisoformat(str(ipo_str)[:10])
                cutoff = today_et() - timedelta(days=_IPO_MAX_AGE_MONTHS * 30)
                if ipo_date >= cutoff:
                    return ipo_date.isoformat()
            except (ValueError, OSError, OverflowError):
                pass

        # Fallback: earliest date in 2-year history
        hist = t.history(period="2y", auto_adjust=False)
        if hist.empty:
            return None
        earliest = hist.index.min().date()
        cutoff = today_et() - timedelta(days=_IPO_MAX_AGE_MONTHS * 30)
        if earliest < cutoff:
            return None
        return earliest.isoformat()
    except Exception as exc:
        logger.debug(f"lockup_calendar: IPO date detection failed for {sym}: {exc}")
        return None


# ── Public refresh ────────────────────────────────────────────────────────────


def refresh_ipo_dates(symbols: list[str] | None = None) -> int:
    """Refresh the IPO date cache for recently-listed symbols.

    Only queries symbols not already in the cache. Prunes entries older than
    18 months automatically.

    Returns number of new entries added.
    """
    if symbols is None:
        symbols = list(STOCK_UNIVERSE)

    cache = _load_ipo_cache()
    today = today_et()
    cutoff = (today - timedelta(days=_IPO_MAX_AGE_MONTHS * 30)).isoformat()

    # Prune stale entries
    stale = [k for k, v in cache.items() if v < cutoff]
    for k in stale:
        del cache[k]

    added = 0
    for sym in symbols:
        if sym in cache:
            continue
        ipo_date = _detect_ipo_date(sym)
        if ipo_date is not None:
            cache[sym] = ipo_date
            added += 1

    if added or stale:
        _save_ipo_cache(cache)
    return added


# ── Public getter ──────────────────────────────────────────────────────────────


def get_lockup_expiry_flags(symbols: list[str]) -> dict[str, dict]:
    """Return lockup expiry data for symbols approaching their 180-day lockup.

    A symbol is flagged when its lockup expiry is 5–10 calendar days away.
    Symbols with no IPO date in cache, or whose lockup already passed, are omitted.

    Returns:
        dict[sym, {"lockup_expiry_soon": bool, "days_to_lockup": int, "ipo_date": str}]
    """
    cache = _load_ipo_cache()
    today = today_et()
    result: dict[str, dict] = {}

    for sym in symbols:
        ipo_str = cache.get(sym)
        if not ipo_str:
            continue
        try:
            ipo_date = date.fromisoformat(ipo_str)
        except ValueError:
            continue

        lockup_expiry = ipo_date + timedelta(days=_LOCKUP_DAYS)
        days_to_lockup = (lockup_expiry - today).days

        if _ALERT_WINDOW_LATE <= days_to_lockup <= _ALERT_WINDOW_EARLY:
            result[sym] = {
                "lockup_expiry_soon": True,
                "days_to_lockup": days_to_lockup,
                "ipo_date": ipo_str,
            }

    return result
