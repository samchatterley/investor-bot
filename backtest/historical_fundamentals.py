"""Historical fundamental data pre-fetcher for backtesting.

Enables point-in-time simulation of two fundamental signals that cannot be
computed from OHLCV data alone:

  pead          — EPS surprise ≥ 5% within the last 30 days
                  Source: yfinance earnings_dates (free, ~8 quarters history)

  insider_buying — ≥2 corporate insiders made open-market Form 4 purchases
                   within the last 10 days
                   Source: SEC EDGAR submissions API (free, ~2 years history)

Both data sets are pre-fetched once at backtest startup and stored in plain
Python dicts.  During simulation, ``pead_active_on_date`` and
``insider_state_on_date`` do O(n) walks over the sorted event lists to
compute what was known at each simulation date — no lookahead.

Usage (inside backtest/engine.py)::

    from backtest.historical_fundamentals import (
        prefetch_earnings_history,
        prefetch_insider_history,
        pead_active_on_date,
        insider_state_on_date,
    )

    earnings_hist = prefetch_earnings_history(symbols)
    insider_hist  = prefetch_insider_history(symbols)

    # Per simulation date:
    pead   = pead_active_on_date(sym, sim_date, earnings_hist)
    insider = insider_state_on_date(sym, sim_date, insider_hist)
"""

import logging
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from data.insider_feed import _get_cik_map, _parse_form4, _recent_form4_filings

logger = logging.getLogger(__name__)

_PEAD_MIN_SURPRISE = 5.0  # EPS beat threshold (%) — mirrors data/earnings_surprise.py
_YF_DELAY = 0.05  # yfinance is not key-gated; modest courtesy delay


# ── Earnings / PEAD ──────────────────────────────────────────────────────────


def prefetch_earnings_history(symbols: list[str]) -> dict[str, list[dict]]:
    """Fetch all available historical earnings surprise events for each symbol.

    Returns ``{sym: [{"date": date, "surprise_pct": float}, ...]}`` sorted
    oldest-first so callers can stop the walk early.

    Coverage: yfinance typically provides 6–12 quarters of history.  That is
    enough to cover the standard 2025–2026 backtest window.
    """
    result: dict[str, list[dict]] = {}
    for i, sym in enumerate(symbols):
        try:
            time.sleep(_YF_DELAY)
            df: pd.DataFrame | None = yf.Ticker(sym).earnings_dates
        except Exception as exc:
            logger.debug(f"earnings_dates fetch failed for {sym}: {exc}")
            continue

        if df is None or df.empty:
            continue

        df = df.dropna(subset=["Reported EPS", "Surprise(%)"])
        if df.empty:
            continue

        events = []
        for ts, row in df.iterrows():
            try:
                events.append(
                    {
                        "date": ts.date() if hasattr(ts, "date") else ts,
                        "surprise_pct": float(row["Surprise(%)"]),
                    }
                )
            except (TypeError, ValueError):
                continue

        if events:
            result[sym] = sorted(events, key=lambda e: e["date"])

        if (i + 1) % 10 == 0:
            logger.info(f"Earnings history: {i + 1}/{len(symbols)} symbols fetched")

    logger.info(f"Earnings history complete: {len(result)}/{len(symbols)} symbols have data")
    return result


def pead_active_on_date(
    sym: str,
    sim_date: date,
    earnings_history: dict[str, list[dict]],
    lookback_days: int = 30,
    min_surprise: float = _PEAD_MIN_SURPRISE,
) -> bool:
    """Return True if sym had a qualifying EPS beat in the window
    ``[sim_date - lookback_days, sim_date)``.

    Point-in-time safe: only considers events strictly before sim_date.
    """
    events = earnings_history.get(sym, [])
    if not events:
        return False
    cutoff = sim_date - timedelta(days=lookback_days)
    for event in events:
        ed = event["date"]
        if ed < cutoff:
            continue
        if ed >= sim_date:
            break  # sorted oldest-first; no further events can be in-window
        if event["surprise_pct"] >= min_surprise:
            return True
    return False


# ── Insider buying ────────────────────────────────────────────────────────────


def prefetch_insider_history(
    symbols: list[str],
    lookback_days: int = 730,
) -> dict[str, list[dict]]:
    """Fetch all available historical Form 4 open-market purchases for each symbol.

    Uses the same SEC EDGAR submissions API as ``data/insider_feed.py``.
    ``lookback_days=730`` (~2 years) captures the full depth of most CIK
    submissions JSON responses; only code='P' / AD='A' transactions are kept.

    Returns ``{sym: [{"filing_date": date, "reporter": str,
                       "shares": float, "price": float}, ...]}`` sorted oldest-first.
    """
    cik_map = _get_cik_map()
    result: dict[str, list[dict]] = {}

    for i, sym in enumerate(symbols):
        cik = cik_map.get(sym.upper())
        if not cik:
            logger.debug(f"No CIK found for {sym} — skipping")
            continue

        filings = _recent_form4_filings(cik, lookback_days)
        txns: list[dict] = []
        for filing in filings:
            try:
                fd = date.fromisoformat(filing["filing_date"])
            except ValueError:
                continue
            for t in _parse_form4(cik, filing["accession"], filing["doc"]):
                txns.append(
                    {
                        "filing_date": fd,
                        "reporter": t["reporter"],
                        "shares": t["shares"],
                        "price": t["price"],
                    }
                )

        if txns:
            result[sym] = sorted(txns, key=lambda t: t["filing_date"])
            logger.debug(f"Insider history {sym}: {len(txns)} open-market purchases")

        if (i + 1) % 10 == 0:
            logger.info(f"Insider history: {i + 1}/{len(symbols)} symbols fetched")

    logger.info(f"Insider history complete: {len(result)}/{len(symbols)} symbols have data")
    return result


def insider_state_on_date(
    sym: str,
    sim_date: date,
    insider_history: dict[str, list[dict]],
    lookback_days: int = 10,
    large_buy_usd: float = 100_000.0,
) -> dict:
    """Return insider signal state for sym as of sim_date.

    Point-in-time safe: only considers filings strictly before sim_date.

    Returns::

        {
            "insider_cluster":          bool,   # ≥2 distinct insiders bought
            "insider_large_buy":        bool,   # single buy > large_buy_usd
            "insider_unique_insiders":  int,
        }
    """
    txns = insider_history.get(sym, [])
    if not txns:
        return {"insider_cluster": False, "insider_large_buy": False, "insider_unique_insiders": 0}

    cutoff = sim_date - timedelta(days=lookback_days)
    window = [t for t in txns if cutoff <= t["filing_date"] < sim_date]

    if not window:
        return {"insider_cluster": False, "insider_large_buy": False, "insider_unique_insiders": 0}

    unique_insiders = len({t["reporter"] for t in window})
    max_notional = max(t["shares"] * t["price"] for t in window)

    return {
        "insider_cluster": unique_insiders >= 2,
        "insider_large_buy": max_notional >= large_buy_usd,
        "insider_unique_insiders": unique_insiders,
    }
