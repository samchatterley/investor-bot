"""
Correlation-based concentration filter.

`correlated_with_held` returns True when a buy candidate's 20-day daily
returns exceed a Pearson correlation threshold with any currently-held
symbol.  The check fails open: if price data is unavailable the trade is
allowed through so a network hiccup never silently blocks buying.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 20
CORRELATION_THRESHOLD = 0.7
_MIN_OVERLAP = 10  # minimum shared trading days to trust the result


def _fetch_closes(symbols: list[str], days: int) -> dict[str, list[float]]:
    """Return {symbol: [close, ...]} for each symbol with enough history."""
    if not symbols:
        return {}
    end = datetime.now()
    start = end - timedelta(days=days + 14)  # buffer for weekends/holidays
    try:
        raw = yf.download(
            tickers=symbols,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            threads=False,
            progress=False,
        )
    except Exception as e:
        logger.warning(f"Correlation price fetch failed: {e}")
        return {}

    if raw is None or raw.empty:
        return {}

    result: dict[str, list[float]] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"]
        for sym in symbols:
            if sym in close_df.columns:
                series = close_df[sym].dropna()
                if len(series) >= _MIN_OVERLAP:
                    result[sym] = series.tolist()
    elif len(symbols) == 1:
        series = raw["Close"].dropna()
        if len(series) >= _MIN_OVERLAP:
            result[symbols[0]] = series.tolist()

    return result


def _pearson(prices_a: list[float], prices_b: list[float]) -> float:
    """Pearson correlation of daily returns on the overlapping tail."""
    n = min(len(prices_a), len(prices_b))
    if n < _MIN_OVERLAP + 1:
        return 0.0

    a = prices_a[-n:]
    b = prices_b[-n:]

    ret_a = [a[i] / a[i - 1] - 1 for i in range(1, n)]
    ret_b = [b[i] / b[i - 1] - 1 for i in range(1, n)]
    m = len(ret_a)

    mean_a = sum(ret_a) / m
    mean_b = sum(ret_b) / m

    cov = sum((ret_a[i] - mean_a) * (ret_b[i] - mean_b) for i in range(m))
    var_a = sum((x - mean_a) ** 2 for x in ret_a)
    var_b = sum((x - mean_b) ** 2 for x in ret_b)

    denom = (var_a * var_b) ** 0.5
    if denom < 1e-10:
        return 0.0
    return cov / denom


def correlated_with_held(
    candidate: str,
    held_symbols: set[str],
    threshold: float = CORRELATION_THRESHOLD,
    _fetch_fn=None,
) -> bool:
    """Return True if candidate is too correlated with any held position.

    Fails open: returns False (allow trade) when price data is unavailable.
    `_fetch_fn` is injectable for testing — defaults to `_fetch_closes`.
    """
    if not held_symbols:
        return False

    fetch = _fetch_fn if _fetch_fn is not None else _fetch_closes
    symbols = [candidate] + sorted(held_symbols)
    closes = fetch(symbols, LOOKBACK_DAYS)

    candidate_closes = closes.get(candidate)
    if not candidate_closes:
        logger.debug(f"Correlation check: no data for {candidate} — allowing trade")
        return False

    for held in held_symbols:
        held_closes = closes.get(held)
        if not held_closes:
            continue
        corr = _pearson(candidate_closes, held_closes)
        if corr > threshold:
            logger.info(
                f"Correlation filter: {candidate} ↔ {held} r={corr:.2f} > {threshold:.2f} — skipping"
            )
            return True

    return False
