"""Cross-asset macro signals from ETF and index price data.

Provides a daily MacroSnapshot covering:
  credit_spread_roc     — HYG/LQD price ratio 10-day ROC (%); negative = credit tightening
  credit_stress         — True when credit_spread_roc ≤ -2.0%
  tlt_spy_spread_5d     — TLT 5-day return minus SPY 5-day return (%)
  duration_flight       — True when tlt_spy_spread_5d > 3.0% (flight-to-safety active)
  copper_gold_trend_20d — CPER/GLD ratio 20-day ROC (%); positive = economic expansion
  copper_gold_positive  — True when copper_gold_trend_20d > 0
  usd_trend_20d         — UUP 20-day return (%); positive = USD strengthening
  usd_strong            — True when usd_trend_20d > 1.0%
  hyg_ief_roc_10d       — HYG/IEF ratio 10-day ROC (% credit risk appetite)
  data_available        — False when all signal fields are None (data unavailable)

Cache: logs/macro_data_cache.json, refreshed once per calendar day.
All public functions degrade gracefully — return None/False on any data failure.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import LOG_DIR, today_et

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_CACHE_PATH = os.path.join(LOG_DIR, "caching", "macro_data_cache.json")
_TICKERS: list[str] = ["HYG", "LQD", "IEF", "TLT", "CPER", "GLD", "UUP", "SPY"]
_LOOKBACK_DAYS = 75  # calendar days → ~50 trading days (enough for 20d trend + buffer)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MacroSnapshot:
    credit_spread_roc: float | None
    credit_stress: bool
    tlt_spy_spread_5d: float | None
    duration_flight: bool
    copper_gold_trend_20d: float | None
    copper_gold_positive: bool
    usd_trend_20d: float | None
    usd_strong: bool
    hyg_ief_roc_10d: float | None
    data_available: bool


def _zero_snapshot() -> MacroSnapshot:
    return MacroSnapshot(
        credit_spread_roc=None,
        credit_stress=False,
        tlt_spy_spread_5d=None,
        duration_flight=False,
        copper_gold_trend_20d=None,
        copper_gold_positive=False,
        usd_trend_20d=None,
        usd_strong=False,
        hyg_ief_roc_10d=None,
        data_available=False,
    )


# ── Cache I/O ─────────────────────────────────────────────────────────────────


def _load_cache() -> MacroSnapshot | None:
    """Return today's cached MacroSnapshot or None on miss/stale/corrupt."""
    try:
        with open(_CACHE_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if data.get("_date") != today_et().isoformat():
            return None
        payload = {k: v for k, v in data.items() if k != "_date"}
        return MacroSnapshot(**payload)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("macro_data: cache read error: %s", exc)
        return None


def _save_cache(snapshot: MacroSnapshot) -> None:
    """Persist snapshot to disk for today."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        payload = asdict(snapshot)
        payload["_date"] = today_et().isoformat()
        with open(_CACHE_PATH, "w") as f:
            json.dump(payload, f)
    except OSError as exc:
        logger.warning("macro_data: cache write error: %s", exc)


# ── Price download ────────────────────────────────────────────────────────────


def _download_macro_prices() -> dict[str, pd.Series]:
    """Download close prices for all macro ETFs. Returns {ticker: close_series}."""
    end = datetime.now(_ET)
    start = (end - timedelta(days=_LOOKBACK_DAYS + 15)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            tickers=_TICKERS,
            start=start,
            end=end_str,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.warning("macro_data: yfinance download failed: %s", exc)
        return {}

    if raw is None or raw.empty:
        return {}

    result: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            close = raw["Close"]
        except KeyError:
            return {}
        for ticker in _TICKERS:
            if ticker in close.columns:
                series = close[ticker].dropna()
                if len(series) >= 22:
                    result[ticker] = series
    elif len(_TICKERS) == 1:
        series = raw["Close"].dropna()
        if len(series) >= 22:
            result[_TICKERS[0]] = series

    logger.debug("macro_data: downloaded %d/%d tickers", len(result), len(_TICKERS))
    return result


# ── Signal computation ────────────────────────────────────────────────────────


def _period_roc(series: pd.Series, n: int) -> float | None:
    """Compute n-period price ROC in %. Returns None when insufficient data."""
    if series is None or len(series) <= n:
        return None
    try:
        prev = float(series.iloc[-(n + 1)])
        if prev == 0:
            return None
        return round(float((series.iloc[-1] / prev - 1) * 100), 3)
    except (IndexError, TypeError, ZeroDivisionError):
        return None


def _ratio_roc(a: pd.Series | None, b: pd.Series | None, n: int) -> float | None:
    """Compute n-period ROC of ratio a/b on their common index."""
    if a is None or b is None:
        return None
    try:
        combined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        if len(combined) <= n:
            return None
        ratio = (
            (combined["a"] / combined["b"])
            .replace([float("inf"), float("-inf")], float("nan"))
            .dropna()
        )
        if len(ratio) <= n:
            return None
        prev = float(ratio.iloc[-(n + 1)])
        if prev == 0:
            return None
        return round(float((ratio.iloc[-1] / prev - 1) * 100), 3)
    except Exception:
        return None


def _compute_snapshot(prices: dict[str, pd.Series]) -> MacroSnapshot:
    """Compute MacroSnapshot from a {ticker: close_series} dict."""
    hyg = prices.get("HYG")
    lqd = prices.get("LQD")
    ief = prices.get("IEF")
    tlt = prices.get("TLT")
    cper = prices.get("CPER")
    gld = prices.get("GLD")
    uup = prices.get("UUP")
    spy = prices.get("SPY")

    credit_spread_roc = _ratio_roc(hyg, lqd, 10)
    credit_stress = credit_spread_roc is not None and credit_spread_roc <= -2.0

    tlt_spy_spread_5d: float | None = None
    tlt_5d = _period_roc(tlt, 5)
    spy_5d = _period_roc(spy, 5)
    if tlt_5d is not None and spy_5d is not None:
        tlt_spy_spread_5d = round(tlt_5d - spy_5d, 3)
    duration_flight = tlt_spy_spread_5d is not None and tlt_spy_spread_5d > 3.0

    copper_gold_trend_20d = _ratio_roc(cper, gld, 20)
    copper_gold_positive = copper_gold_trend_20d is not None and copper_gold_trend_20d > 0

    usd_trend_20d = _period_roc(uup, 20)
    usd_strong = usd_trend_20d is not None and usd_trend_20d > 1.0

    hyg_ief_roc_10d = _ratio_roc(hyg, ief, 10)

    data_available = any(
        v is not None
        for v in [credit_spread_roc, tlt_spy_spread_5d, copper_gold_trend_20d, usd_trend_20d]
    )

    return MacroSnapshot(
        credit_spread_roc=credit_spread_roc,
        credit_stress=credit_stress,
        tlt_spy_spread_5d=tlt_spy_spread_5d,
        duration_flight=duration_flight,
        copper_gold_trend_20d=copper_gold_trend_20d,
        copper_gold_positive=copper_gold_positive,
        usd_trend_20d=usd_trend_20d,
        usd_strong=usd_strong,
        hyg_ief_roc_10d=hyg_ief_roc_10d,
        data_available=data_available,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def get_macro_snapshot(force_refresh: bool = False) -> MacroSnapshot:
    """Return today's MacroSnapshot, using daily cache.

    Falls back to _zero_snapshot() on any data failure so callers never raise.
    Set force_refresh=True to bypass cache and re-download.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            logger.debug("macro_data: cache hit")
            return cached

    prices = _download_macro_prices()
    if not prices:
        logger.warning("macro_data: no price data — returning zero snapshot")
        return _zero_snapshot()

    snapshot = _compute_snapshot(prices)
    _save_cache(snapshot)
    logger.info(
        "macro_data: snapshot — credit_stress=%s duration_flight=%s "
        "copper_gold=%s usd_strong=%s data_available=%s",
        snapshot.credit_stress,
        snapshot.duration_flight,
        snapshot.copper_gold_positive,
        snapshot.usd_strong,
        snapshot.data_available,
    )
    return snapshot


def get_combined_macro_flags() -> dict:
    """Return combined macro flags from ETF signals and FRED data.

    Merges the ETF-based MacroSnapshot with FRED macro series (yield curve,
    claims, PMI) into a single flat dict suitable for injection into stock
    snapshot dicts.  All keys are prefixed with ``macro_`` to avoid collision
    with stock-level fields.  Degrades gracefully — all flags default to
    neutral (False/0/None) when data is unavailable.
    """
    from data.fred_client import get_macro_snapshot as _fred_snapshot

    etf = get_macro_snapshot()
    fred = _fred_snapshot()

    return {
        "macro_credit_stress": etf.credit_stress,
        "macro_duration_flight": etf.duration_flight,
        "macro_copper_gold_positive": etf.copper_gold_positive,
        "macro_usd_strong": etf.usd_strong,
        "macro_yield_curve": fred.get("yield_curve"),
        "macro_yield_curve_inverted_days": fred.get("yield_curve_inverted_days", 0),
        "macro_claims_deteriorating": fred.get("claims_deteriorating", False),
        "macro_data_available": etf.data_available or fred.get("data_available", False),
    }


def get_credit_stress() -> bool:
    """True when HYG/LQD ratio has fallen >2% over the last 10 trading days."""
    return get_macro_snapshot().credit_stress


def get_duration_flight() -> bool:
    """True when TLT has outperformed SPY by >3% over the last 5 trading days."""
    return get_macro_snapshot().duration_flight


def get_copper_gold_positive() -> bool:
    """True when the CPER/GLD ratio has risen over the last 20 trading days (expansion)."""
    return get_macro_snapshot().copper_gold_positive


def get_usd_strong() -> bool:
    """True when UUP has risen >1% over the last 20 trading days (strengthening dollar)."""
    return get_macro_snapshot().usd_strong
