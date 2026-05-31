"""Short-side squeeze risk detection.

is_squeeze_risk() is a live-only gate called in _execute_shorts() before
sizing and placing a short order.  It is intentionally NOT wired into the
backtest — historical short-interest data is unavailable, and adding a
momentum proxy to backtests would bias results by eliminating fast-moving
names retroactively.

Three independent components:
  1. Reported short interest / float  — yfinance .info (FINRA bi-monthly);
     stale by up to 4 weeks.  Used as a coarse filter only.
  2. Days-to-cover                    — same source and staleness caveat.
  3. 5-day price momentum             — computed from snapshot, real-time;
     catches active squeezes regardless of reported short interest level.
"""

from __future__ import annotations

import logging

import yfinance as yf

logger = logging.getLogger(__name__)

SHORT_PCT_FLOAT_MAX: float = 0.20  # block if reported short interest > 20% of float
DAYS_TO_COVER_MAX: float = 5.0  # block if days-to-cover > 5
RET_5D_MAX: float = 15.0  # block if stock is up > 15% over 5 days (active squeeze proxy)


def fetch_squeeze_info(symbol: str) -> dict[str, float | None]:
    """Return squeeze-relevant fields from yfinance .info for *symbol*.

    Keys returned: ``short_pct_float``, ``days_to_cover``.
    Either value is ``None`` when the data field is absent or unavailable.

    This call hits the yfinance API (FINRA bi-monthly report cadence).
    On any exception the function returns ``None`` for all fields so that a
    transient API failure does not silently block all short entries.
    """
    try:
        info = yf.Ticker(symbol).info
        return {
            "short_pct_float": info.get("shortPercentOfFloat"),
            "days_to_cover": info.get("shortRatio"),
        }
    except Exception:
        logger.debug("fetch_squeeze_info(%s): yfinance fetch failed — treating as safe", symbol)
        return {"short_pct_float": None, "days_to_cover": None}


def is_squeeze_risk(
    symbol: str,
    snapshot: dict,
    *,
    short_pct_float: float | None = None,
    days_to_cover: float | None = None,
    short_pct_float_max: float = SHORT_PCT_FLOAT_MAX,
    days_to_cover_max: float = DAYS_TO_COVER_MAX,
    ret_5d_max: float = RET_5D_MAX,
) -> tuple[bool, str]:
    """Return ``(True, reason)`` if *symbol* has elevated short-squeeze risk.

    Parameters
    ----------
    symbol:
        Ticker symbol — used in log / reason strings only.
    snapshot:
        Market snapshot dict; must contain ``ret_5d_pct`` for the momentum
        check (defaults to 0.0 if absent).
    short_pct_float:
        Short interest as a fraction of float (e.g. 0.25 = 25 %).
        Pass ``None`` to skip this check (data unavailable).
    days_to_cover:
        Days-to-cover ratio.  Pass ``None`` to skip.
    short_pct_float_max:
        Block threshold for ``short_pct_float``.
    days_to_cover_max:
        Block threshold for ``days_to_cover``.
    ret_5d_max:
        Block threshold for 5-day return in percent.

    Returns
    -------
    tuple[bool, str]
        ``(is_risky, reason)``.  ``reason`` is ``""`` when not risky.
    """
    if short_pct_float is not None and short_pct_float > short_pct_float_max:
        return True, (f"short_pct_float={short_pct_float:.1%} > {short_pct_float_max:.0%}")

    if days_to_cover is not None and days_to_cover > days_to_cover_max:
        return True, f"days_to_cover={days_to_cover:.1f} > {days_to_cover_max:.0f}"

    ret_5d = snapshot.get("ret_5d_pct", 0.0)
    if ret_5d > ret_5d_max:
        return True, f"ret_5d_pct={ret_5d:.1f}% > {ret_5d_max:.0f}% (active squeeze)"

    return False, ""
