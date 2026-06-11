"""Rolling 20-day stock-vs-sector-ETF correlation utility.

Used by the correlation_regime_gate position-size scalar:
  - corr > 0.75: size ×0.85  (high sector-beta — systemic drag)
  - corr < 0.35: size ×1.10  (decorrelated idiosyncratic mover)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CORR_WINDOW = 20  # trading days
_FETCH_DAYS = 60  # calendar days to fetch when no pre-loaded data


def compute_stock_sector_corr(
    symbol: str,
    etf_ticker: str,
    price_data: dict[str, pd.DataFrame] | None = None,
    window: int = _CORR_WINDOW,
) -> float | None:
    """Compute rolling ``window``-day Pearson correlation between a stock
    and its sector ETF using daily close returns.

    Returns the latest correlation value in [-1, 1], or None when there is
    insufficient overlapping price history or any data fetch fails.
    """
    try:
        stock_df = _get_df(symbol, price_data)
        etf_df = _get_df(etf_ticker, price_data)
        if stock_df is None or etf_df is None:
            return None

        stock_ret = stock_df["Close"].dropna().pct_change().dropna()
        etf_ret = etf_df["Close"].dropna().pct_change().dropna()

        combined = pd.concat([stock_ret.rename("s"), etf_ret.rename("e")], axis=1).dropna()
        if len(combined) < window + 2:
            return None

        corr_series = combined["s"].rolling(window).corr(combined["e"])
        latest = corr_series.iloc[-1]
        if pd.isna(latest):
            return None
        return round(float(latest), 4)
    except Exception as exc:
        logger.debug(f"sector_correlation: {symbol}/{etf_ticker}: {exc}")
        return None


def _get_df(ticker: str, price_data: dict[str, pd.DataFrame] | None) -> pd.DataFrame | None:
    """Return price DataFrame from pre-loaded cache or a fresh yfinance download."""
    if price_data is not None and ticker in price_data:
        return price_data[ticker]
    try:
        end = datetime.now()
        start = end - timedelta(days=_FETCH_DAYS)
        df = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        if df.empty:
            return None
        return df
    except Exception:
        return None
