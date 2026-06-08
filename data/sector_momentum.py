"""Sector momentum ranking via 11 SPDR sector ETFs.

Fetches 20-day returns for the 11 Select Sector SPDR ETFs weekly (cached for 24h),
ranks them 1–11 (1 = strongest momentum), and provides gates for long/short entries:

  - Long allowed: sector rank ≤ 4 (top 4 by momentum)
  - Short allowed: sector rank ≥ 9 (bottom 3 by momentum)

The GICS sector labels used here match those returned by ``data.sector_data.get_sector``.
"""

import json
import logging
import os
import time
from datetime import date

import yfinance as yf

from config import LOG_DIR

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "sector_momentum_cache.json")
_CACHE_TTL_SECONDS = 86_400  # 24 hours — refresh once per trading day

# SPDR ETF ticker → GICS sector label (must match sector_data.get_sector output)
_SPDR_MAP: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLU": "Utilities",
}

_LONG_TOP_N = 4  # allow longs in top N sectors by momentum
_SHORT_BOTTOM_N = 3  # allow shorts in bottom N sectors by momentum


def _load_cache() -> dict | None:
    try:
        with open(_CACHE_PATH) as f:
            cache = json.load(f)
        age = time.time() - cache.get("ts", 0)
        if age < _CACHE_TTL_SECONDS:
            return cache.get("ranks")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def _save_cache(ranks: dict) -> None:
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump({"ts": time.time(), "date": date.today().isoformat(), "ranks": ranks}, f)
    except Exception as exc:
        logger.warning(f"sector_momentum: cache save failed: {exc}")


def get_sector_momentum_ranks() -> dict[str, int]:
    """Return a dict mapping GICS sector label → rank (1=best momentum, 11=worst).

    Fetches 20-day returns for the 11 SPDR ETFs.  Results are cached for 24 hours.
    Returns an empty dict on failure (callers must treat missing sector as neutral/allowed).
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    try:
        tickers = list(_SPDR_MAP.keys())
        df = yf.download(tickers, period="30d", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return {}

        import pandas as pd

        if isinstance(df.columns, pd.MultiIndex):
            closes = df["Close"]
        else:
            closes = df

        ret_20d: dict[str, float] = {}
        for ticker in tickers:
            if ticker not in closes.columns:
                continue
            col = closes[ticker].dropna()
            if len(col) < 5:
                continue
            ret = float((col.iloc[-1] / col.iloc[0] - 1) * 100)
            ret_20d[ticker] = ret

        if not ret_20d:
            return {}

        sorted_tickers = sorted(ret_20d, key=ret_20d.__getitem__, reverse=True)
        ranks: dict[str, int] = {}
        for rank, ticker in enumerate(sorted_tickers, start=1):
            sector = _SPDR_MAP[ticker]
            ranks[sector] = rank
            logger.debug(f"  {ticker} ({sector}): ret={ret_20d[ticker]:.1f}% rank={rank}")

        logger.info(
            "Sector momentum ranks: "
            + ", ".join(f"{s}={r}" for s, r in sorted(ranks.items(), key=lambda x: x[1]))
        )
        _save_cache(ranks)
        return ranks

    except Exception as exc:
        logger.warning(f"get_sector_momentum_ranks failed: {exc}")
        return {}


def sector_allowed_long(sector: str, ranks: dict[str, int]) -> bool:
    """Return True when *sector* is in the top 4 momentum sectors (or ranks are unavailable)."""
    if not ranks:
        return True  # fail-open: don't block longs when ETF data is unavailable
    rank = ranks.get(sector)
    if rank is None:
        return True  # unknown sector (ETFs, ADRs, etc.) → allow
    return rank <= _LONG_TOP_N


def sector_allowed_short(sector: str, ranks: dict[str, int]) -> bool:
    """Return True when *sector* is in the bottom 3 momentum sectors (or ranks are unavailable)."""
    if not ranks:
        return True
    rank = ranks.get(sector)
    if rank is None:
        return True
    return rank >= (11 - _SHORT_BOTTOM_N + 1)
