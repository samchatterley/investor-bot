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
"""

import logging
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf

from config import ETF_SYMBOLS

logger = logging.getLogger(__name__)
# yfinance logs at ERROR when no earnings exist (normal for ETFs) — suppress that noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_MIN_SURPRISE_PCT = 5.0  # EPS beat threshold (%)
_PEAD_WINDOW_DAYS = 30  # only consider surprises within this many days
_REQ_DELAY = 0.05  # modest delay — yfinance is not rate-limited by key


def get_earnings_surprise(
    symbols: list[str],
    lookback_days: int = _PEAD_WINDOW_DAYS,
    min_surprise: float = _MIN_SURPRISE_PCT,
) -> dict[str, dict]:
    """Return PEAD candidate data for each symbol that beat recent estimates.

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
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    result: dict[str, dict] = {}

    for sym in symbols:
        if sym in ETF_SYMBOLS:
            continue
        try:
            time.sleep(_REQ_DELAY)
            df: pd.DataFrame | None = yf.Ticker(sym).earnings_dates
        except Exception as exc:
            logger.debug(f"earnings_dates fetch failed for {sym}: {exc}")
            continue

        if df is None or df.empty:
            continue

        # Find the most recent completed earnings event within the lookback window.
        # Completed = has a non-null Reported EPS value.
        df = df.dropna(subset=["Reported EPS", "Surprise(%)"])
        if df.empty:
            continue

        # earnings_dates index is tz-aware; cutoff is UTC-aware — both comparable.
        df = df[df.index >= pd.Timestamp(cutoff)]
        if df.empty:
            continue

        # Most recent row first (DataFrame is sorted newest-first by yfinance).
        row = df.iloc[0]
        surprise_pct = float(row["Surprise(%)"])
        if surprise_pct < min_surprise:
            continue

        earnings_ts = row.name  # tz-aware Timestamp
        earnings_date_str = earnings_ts.date().isoformat()
        days_ago = (datetime.now(UTC) - earnings_ts.to_pydatetime()).days

        result[sym] = {
            "earnings_surprise_pct": round(surprise_pct, 2),
            "earnings_date": earnings_date_str,
            "earnings_days_ago": days_ago,
            "pead_candidate": True,
        }
        logger.debug(
            f"PEAD {sym}: +{surprise_pct:.1f}% surprise on {earnings_date_str} ({days_ago}d ago)"
        )

    logger.info(f"PEAD: {len(result)}/{len(symbols)} symbols with qualifying surprise")
    return result
