"""Short interest data fetcher — live signal enrichment only.

Fetches yfinance.info["shortRatio"] (days-to-cover) for each symbol.
A high days-to-cover ratio (≥ 5) indicates heavy short positioning and
elevated squeeze / mean-reversion risk, which reinforces a bearish thesis
when combined with technical breakdown signals.

NOT BACKTESTABLE: yfinance.info provides the current snapshot value only;
no time-series history is available.  This signal is therefore live-only
and is excluded from backtest/engine.py.

Threshold: shortRatio >= _MIN_SHORT_RATIO_DAYS (default 5 days).
"""

import logging
import time

import yfinance as yf

from config import ETF_SYMBOLS

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_MIN_SHORT_RATIO_DAYS = 5.0  # days-to-cover threshold for "high short interest"
_REQ_DELAY = 0.05


def get_short_interest(
    symbols: list[str],
    min_short_ratio: float = _MIN_SHORT_RATIO_DAYS,
) -> dict[str, dict]:
    """Return short interest data for each symbol with shortRatio >= min_short_ratio.

    Result schema per symbol::

        {
            "short_ratio":              float,  # days-to-cover from yfinance.info
            "high_short_interest":      bool,   # always True when present
        }

    Symbols with no qualifying data (or below threshold) are omitted.
    All network errors are caught and logged; callers always receive a plain dict.
    """
    result: dict[str, dict] = {}

    for sym in symbols:
        if sym in ETF_SYMBOLS:
            continue
        try:
            time.sleep(_REQ_DELAY)
            info = yf.Ticker(sym).info
        except Exception as exc:
            logger.debug(f"short_interest fetch failed for {sym}: {exc}")
            continue

        short_ratio = info.get("shortRatio")
        if short_ratio is None:
            continue

        try:
            ratio = float(short_ratio)
        except (TypeError, ValueError):
            continue

        if ratio < min_short_ratio:
            continue

        result[sym] = {
            "short_ratio": ratio,
            "high_short_interest": True,
        }

    return result
