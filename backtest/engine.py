"""
Rule-based backtester — validates technical signal quality on historical data
without calling Claude (avoids API cost).

RULE PROXY ONLY: This engine implements deterministic rule proxies for ten
daily signals (mean_reversion, momentum, momentum_12_1, macd_crossover,
bb_squeeze, inside_day_breakout, trend_pullback, breakout_52w, rs_leader,
gap_and_go, vix_fear_reversion) and three intraday signals (vwap_reclaim,
orb_breakout, intraday_momentum). Intraday signals require Alpaca API
credentials and --use-intraday.

This engine does not use Claude's judgment, news, options flow, or macro
context. Results measure signal quality only and must not be interpreted as
deployed-strategy validation.

Usage:
    python backtest/engine.py --start 2025-01-01 --end 2025-12-31
    python backtest/engine.py --start 2025-01-01 --end 2025-12-31 --use-intraday
    python backtest/engine.py --start 2025-01-01 --end 2025-12-31 --capital 25000
"""

import argparse
import contextlib
import json
import logging
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import product
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands

from backtest.historical_fundamentals import (
    earnings_miss_active_on_date,
    insider_state_on_date,
    pead_active_on_date,
    prefetch_earnings_history,
    prefetch_insider_history,
    recent_earnings_date,
)
from config import (
    BACKTEST_DEFAULT_START,
    HOLDOUT_START_DATE,
    LOG_DIR,
    SLIPPAGE_BPS,
    SPREAD_BPS,
    STOCK_UNIVERSE,
)
from data.market_regime import compute_regime_series
from data.universe_history import get_universe_for_date
from execution.short_universe import STATIC_SHORT_UNIVERSE
from risk.risk_config import RiskConfig
from signals.evaluator import (
    DEFAULT_SHORT_SIGNAL_PARAMS,
    DEFAULT_SIGNAL_PARAMS,
    INTRADAY_SIGNALS,
    REGIME_BLOCKED,
    SHORT_ALLOWED_REGIMES,
    SHORT_SIGNAL_PRIORITY,
    SIGNAL_PRIORITY,
    evaluate_short_signals,
    evaluate_signals,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Core indicator columns that must be non-NaN for a row to be used
_CORE_COLS = ["rsi", "macd_diff", "ema9", "ema21", "bb_pct", "vol_ratio", "ret_5d"]

# Signal priority — imported from signals.evaluator (canonical source).
_SIGNAL_PRIORITY = SIGNAL_PRIORITY

# Default signal thresholds — imported from signals.evaluator (canonical source).
_DEFAULT_PARAMS = DEFAULT_SIGNAL_PARAMS

# Search space for walk-forward parameter optimisation
_DEFAULT_PARAM_GRID: dict[str, list] = {
    "rsi_threshold": [25, 30, 35, 40],
    "bb_threshold": [0.15, 0.20, 0.25, 0.30],
    "mr_vol_threshold": [1.0, 1.2, 1.5],
    "mom_vol_threshold": [1.1, 1.3, 1.5],
    "mom_ret5d_threshold": [0.5, 1.0, 1.5, 2.0],
}

# Minimum trades in the train window for a param set to be considered valid
_MIN_TRAIN_TRADES = 20

# Short-side simulation constants
_SHORT_MAX_HOLD_DAYS = 5
_SHORT_RS_RANK_GATE = 25.0  # earnings_miss: must be in weakest quartile
# failed_breakout / high_vol_reversal target extended stocks (recently strong), not already-broken ones.
# Gate at top ~35% RS rank to avoid catching dead-cat bounces in true laggards.
_REVERSAL_SHORT_RS_GATE = 65.0


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    df = df.copy()

    # ── Core indicators ───────────────────────────────────────────────────────
    df["rsi"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close)
    df["macd_diff"] = macd.macd_diff()
    df["ema9"] = EMAIndicator(close=close, window=9).ema_indicator()
    df["ema21"] = EMAIndicator(close=close, window=21).ema_indicator()
    bb = BollingerBands(close=close, window=20)
    df["bb_pct"] = bb.bollinger_pband()
    vol_ma20 = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_ma20
    df["avg_volume_20"] = vol_ma20
    df["ret_5d"] = close.pct_change(5) * 100

    # ── Extended indicators for new daily signals ─────────────────────────────
    df["ret_10d"] = close.pct_change(10) * 100
    df["ret_20d"] = close.pct_change(20) * 100

    # MACD cross: diff crosses from <= 0 to > 0 (NaN comparisons return False)
    df["macd_cross"] = (df["macd_diff"].shift(1) <= 0) & (df["macd_diff"] > 0)

    # BB squeeze: bandwidth in bottom 20% of its recent range
    bb_bw = bb.bollinger_wband()
    bw_min = bb_bw.rolling(20, min_periods=10).min()
    bw_max = bb_bw.rolling(20, min_periods=10).max()
    bw_range = bw_max - bw_min
    bw_norm = (bb_bw - bw_min) / bw_range.where(bw_range > 0, other=float("nan"))
    df["bb_squeeze"] = (bw_norm < 0.2).fillna(False)

    # Price vs EMA21 (for trend_pullback)
    df["pct_vs_ema21"] = (close / df["ema21"] - 1) * 100

    # 52-week high — rolling max with min_periods=20 so warmup rows get a value
    df["high_52w"] = close.rolling(252, min_periods=20).max()
    df["price_vs_52w_high_pct"] = (close / df["high_52w"] - 1) * 100

    # Inside day and ADX trend strength (both need High/Low)
    if "High" in df.columns and "Low" in df.columns:
        df["is_inside_day"] = (
            (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
        ).fillna(False)
        df["adx"] = (
            ADXIndicator(high=df["High"], low=df["Low"], close=close, window=14).adx().fillna(0)
        )

    # Gap and go (needs Open)
    if "Open" in df.columns:
        df["gap_pct"] = ((df["Open"] / close.shift(1)) - 1) * 100
        df["close_above_open"] = (close > df["Open"]).fillna(False)

    # 12-1 medium-term momentum (Jegadeesh-Titman): 12-month return minus 1-month return.
    # Positive values indicate sustained trend that hasn't overextended short-term.
    df["ret_12m"] = close.pct_change(252) * 100
    df["ret_1m"] = close.pct_change(21) * 100
    df["mom_12_1"] = df["ret_12m"] - df["ret_1m"]

    # Historical volatility percentile: where today's 20-day annualized HV sits in its
    # 252-day range (0 = all-time annual low, 1 = all-time annual high).
    # hv_rank < 0.20 → bottom quintile → IV compression → expansion likely.
    daily_returns = close.pct_change()
    df["hv_20d"] = daily_returns.rolling(20).std() * math.sqrt(252) * 100
    df["hv_rank"] = df["hv_20d"].rolling(252, min_periods=30).rank(pct=True)

    # RSI divergence: price lower than 5 days ago but RSI higher — structural bullish divergence.
    # Gates (adx < 25, rsi < 45, vol) are applied in evaluate_signals(); this column captures
    # the pure price/RSI structural pattern.
    df["rsi_divergence"] = ((close < close.shift(5)) & (df["rsi"] > df["rsi"].shift(5))).fillna(
        False
    )

    # failed_breakout: stock hit a new 20-day closing high yesterday, failed back below it today.
    # high_20d_lag2 = the 20d rolling max as of 2 days ago — this is the resistance level
    # that was in place BEFORE yesterday's attempted breakout.
    high_20d_lag2 = close.rolling(20, min_periods=10).max().shift(2)
    df["failed_breakout_flag"] = (
        (close.shift(1) > high_20d_lag2) & (close <= high_20d_lag2)
    ).fillna(False)

    # close_pct_of_range: where today's close sits within the day's High–Low range.
    # 0 = closed at the low (bearish rejection), 1 = closed at the high (bullish).
    # Used by high_vol_reversal to detect distribution candles.
    if "High" in df.columns and "Low" in df.columns:
        daily_range = df["High"] - df["Low"]
        df["close_pct_of_range"] = (
            (close - df["Low"]) / daily_range.where(daily_range > 0)
        ).fillna(0.5)

    # Drop rows where any core indicator is NaN (warmup period)
    return df.dropna(subset=_CORE_COLS)


def _row_to_snapshot(
    row: pd.Series,
    intraday: dict | None = None,
    spy_ret_5d: float | None = None,
    spy_ret_10d: float | None = None,
    fundamentals: dict | None = None,
) -> dict:
    """Convert an engine indicator row to the canonical snapshot format used by evaluate_signals."""
    snap: dict = {
        "rsi_14": float(row.get("rsi", 50)),
        "bb_pct": float(row.get("bb_pct", 0.5)),
        "vol_ratio": float(row.get("vol_ratio", 1.0)),
        "macd_diff": float(row.get("macd_diff", 0)),
        "macd_crossed_up": bool(row.get("macd_cross", False)),
        "ema9_above_ema21": bool(row["ema9"] > row["ema21"])
        if "ema9" in row.index and "ema21" in row.index
        else False,
        "adx": float(row.get("adx", 30)),
        "ret_5d_pct": float(row.get("ret_5d", 0)),
        "ret_10d_pct": float(row.get("ret_10d", 0)),
        "price_vs_ema21_pct": float(row.get("pct_vs_ema21", 0)),
        "price_vs_52w_high_pct": float(row.get("price_vs_52w_high_pct", -999)),
        "hv_rank": float(row.get("hv_rank", 1.0)),
        "bb_squeeze": bool(row.get("bb_squeeze", False)),
        "is_inside_day": bool(row.get("is_inside_day", False)),
        "gap_pct": float(row.get("gap_pct", 0)),
        "close_above_open": bool(row.get("close_above_open", False)),
        "rsi_divergence": bool(row.get("rsi_divergence", False)),
        "failed_breakout_flag": bool(row.get("failed_breakout_flag", False)),
        "close_pct_of_range": float(row.get("close_pct_of_range", 0.5)),
        "high_short_interest": bool(row.get("high_short_interest", False)),
        "spy_ret_5d": spy_ret_5d,
        "spy_ret_10d": spy_ret_10d,
    }
    _m121 = row.get("mom_12_1")
    if _m121 is not None:
        snap["mom_12_1_pct"] = float(_m121)
    if fundamentals:
        snap["insider_cluster"] = bool(fundamentals.get("insider_cluster", False))
        snap["pead_candidate"] = bool(fundamentals.get("pead_active", False))
    if intraday:
        snap.update(
            {
                "intraday_change_pct": intraday.get("intraday_change_pct"),
                "price_above_vwap": intraday.get("price_above_vwap"),
                "pct_vs_vwap": float(intraday.get("pct_vs_vwap", 0)),
                "orb_breakout_up": bool(intraday.get("orb_breakout_up", False)),
                "intraday_rsi": intraday.get("intraday_rsi"),
            }
        )
    return snap


def _entry_signal(
    row: pd.Series,
    params: dict | None = None,
    intraday: dict | None = None,
    spy_ret_5d: float | None = None,
    spy_ret_10d: float | None = None,
    regime: str | None = None,
    vix_spike: bool = False,
    fundamentals: dict | None = None,
    disabled_signals: frozenset[str] | None = None,
) -> str | None:
    """Return the highest-priority matching signal, or None.

    Delegates signal logic to signals.evaluator.evaluate_signals() — the single
    canonical implementation shared with the live scanner.
    """
    blocked = REGIME_BLOCKED.get(regime or "", frozenset())
    if disabled_signals:
        blocked = blocked | disabled_signals

    snap = _row_to_snapshot(
        row,
        intraday=intraday,
        spy_ret_5d=spy_ret_5d,
        spy_ret_10d=spy_ret_10d,
        fundamentals=fundamentals,
    )
    signals = evaluate_signals(
        snap,
        blocked=blocked,
        params=params,
        vix_spike=vix_spike,
        spy_ret_5d=spy_ret_5d,
        spy_ret_10d=spy_ret_10d,
    )
    return signals[0] if signals else None


def _short_entry_signal(
    row: pd.Series,
    rs_rank_pct: float | None,
    spy_ret_20d: float | None,
    regime: str | None = None,
    fundamentals: dict | None = None,
    short_params: dict | None = None,
    rs_rank_pct_10d_ago: float | None = None,
    vix_term_inverted: bool = True,
) -> list[str] | None:
    """Return matched bearish signals if this row qualifies as a short entry, else None.

    Regime gate: only enters in SHORT_ALLOWED_REGIMES.
    VIX term gate: skips entry when VIX9D/VIX ≤ 1.05 (normal contango = less short edge).
      Default True (allow) when VIX9D data unavailable.

    Three RS-rank paths:
    - Deterioration path (rs_rank_pct_10d_ago > _REVERSAL_SHORT_RS_GATE AND rs_rank_pct < 65):
      leader-to-laggard rotation — was top-35% of universe 10d ago, now fallen.
      Catches the early stage of distribution before technical signals mature.
    - Reversal path (rs_rank_pct >= _REVERSAL_SHORT_RS_GATE): recently-strong stocks showing
      exhaustion. Checks failed_breakout and high_vol_reversal.
    - Fundamental path (rs_rank_pct < _SHORT_RS_RANK_GATE): bottom-quartile laggards with an
      earnings miss catalyst.

    Stocks in the middle band (25–65%) without a prior leader history produce no signal.
    """
    if regime is not None and regime not in SHORT_ALLOWED_REGIMES:
        return None

    if not vix_term_inverted:
        return None

    snap = _row_to_snapshot(row)
    snap["rs_rank_pct"] = rs_rank_pct if rs_rank_pct is not None else 50.0
    if rs_rank_pct_10d_ago is not None:
        snap["rs_rank_pct_10d_ago"] = rs_rank_pct_10d_ago

    if fundamentals:
        snap["earnings_miss_candidate"] = bool(fundamentals.get("earnings_miss_active", False))
        if "earnings_gap_pct" in fundamentals:
            snap["earnings_gap_pct"] = fundamentals["earnings_gap_pct"]

    # Event path — earnings gap-down: fires on all RS rank tiers.  Returns early if the
    # earnings_gap_down signal specifically fires (not just any signal from the snapshot).
    if snap.get("earnings_gap_pct") is not None:
        event_sigs = evaluate_short_signals(snap, params=short_params)
        if "earnings_gap_down" in event_sigs:
            return event_sigs

    # Deterioration path — leader-to-laggard: was top-35% 10d ago, now fallen below
    if (
        rs_rank_pct_10d_ago is not None
        and rs_rank_pct_10d_ago > _REVERSAL_SHORT_RS_GATE
        and (rs_rank_pct is None or rs_rank_pct < _REVERSAL_SHORT_RS_GATE)
    ):
        det_sigs = evaluate_short_signals(
            snap,
            params=short_params,
            blocked=frozenset({"earnings_miss", "failed_breakout", "high_vol_reversal"}),
        )
        if det_sigs:
            return det_sigs

    # Reversal path — extended/recently-strong stocks caught in a trap
    if rs_rank_pct is not None and rs_rank_pct >= _REVERSAL_SHORT_RS_GATE:
        rev_sigs = evaluate_short_signals(
            snap,
            params=short_params,
            blocked=frozenset({"earnings_miss"}),
        )
        if rev_sigs:
            return rev_sigs

    # Fundamental path — bottom-quartile laggards with an earnings miss catalyst
    if rs_rank_pct is None or rs_rank_pct >= _SHORT_RS_RANK_GATE:
        return None

    signals = evaluate_short_signals(
        snap,
        params=short_params,
        blocked=frozenset({"failed_breakout", "high_vol_reversal"}),
    )
    return signals if signals else None


def _compute_intraday_day(date_str: str, timed_bars: list) -> dict | None:
    """
    Pure computation: derive intraday signal inputs from a sorted list of
    (datetime, bar) pairs for one trading day.
    Returns a metrics dict or None if the day has insufficient data.
    bar objects must expose .open, .high, .low, .close, .volume attributes.
    """
    times = [t for t, _ in timed_bars]
    bars = [b for _, b in timed_bars]

    opens = [b.open for b in bars]
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    vols = [b.volume for b in bars]

    total_vol = sum(vols)
    if total_vol <= 0 or not closes:
        return None

    typical = [(hi + lo + c) / 3 for hi, lo, c in zip(highs, lows, closes, strict=True)]
    vwap = sum(tp * v for tp, v in zip(typical, vols, strict=True)) / total_vol

    orb_cutoff = datetime.strptime(f"{date_str} 10:00", "%Y-%m-%d %H:%M").replace(tzinfo=_ET)
    orb_idxs = [i for i, t in enumerate(times) if t <= orb_cutoff]
    post_orb_idxs = [i for i, t in enumerate(times) if t > orb_cutoff]

    orb_high = max(highs[i] for i in orb_idxs) if len(orb_idxs) >= 5 else None
    orb_low = min(lows[i] for i in orb_idxs) if len(orb_idxs) >= 5 else None

    avg_bar_vol = total_vol / len(bars)
    orb_breakout_up = False
    if orb_high and post_orb_idxs:
        orb_breakout_up = any(closes[i] > orb_high and vols[i] > avg_bar_vol for i in post_orb_idxs)

    last_close = closes[-1]
    intraday_change_pct = (last_close / opens[0] - 1) * 100 if opens and opens[0] > 0 else None
    price_above_vwap = last_close > vwap
    pct_vs_vwap = (last_close / vwap - 1) * 100 if vwap > 0 else 0.0

    intraday_rsi = None
    closes_5m = closes[::5]
    if len(closes_5m) >= 14:
        try:
            close_s = pd.Series(closes_5m, dtype=float)
            intraday_rsi = float(RSIIndicator(close=close_s, window=14).rsi().iloc[-1])
        except Exception:
            pass

    return {
        "vwap": vwap,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_breakout_up": orb_breakout_up,
        "intraday_change_pct": intraday_change_pct,
        "price_above_vwap": price_above_vwap,
        "pct_vs_vwap": pct_vs_vwap,
        "intraday_rsi": intraday_rsi,
    }


def _fetch_intraday_bars(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, dict]]:
    """
    Fetch Alpaca 1-min bars and compute per-day intraday signal inputs.
    Returns {symbol: {date_str: {vwap, orb_high, orb_low, orb_breakout_up,
                                  intraday_change_pct, price_above_vwap,
                                  pct_vs_vwap, intraday_rsi}}}
    Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in environment.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        import config as _cfg
    except ImportError as exc:
        logger.error(f"Alpaca SDK unavailable — cannot fetch intraday bars: {exc}")
        return {}

    if not (_cfg.ALPACA_API_KEY and _cfg.ALPACA_SECRET_KEY):
        logger.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not set — skipping intraday fetch")
        return {}

    client = StockHistoricalDataClient(
        api_key=_cfg.ALPACA_API_KEY,
        secret_key=_cfg.ALPACA_SECRET_KEY,
    )
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=_ET)
    end_dt = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=_ET)

    result: dict[str, dict[str, dict]] = {}

    for idx, sym in enumerate(symbols):
        logger.info(f"Intraday fetch {sym} ({idx + 1}/{len(symbols)})")
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            req = StockBarsRequest(
                symbol_or_symbols=sym,
                start=start_dt,
                end=end_dt,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                feed="iex",  # type: ignore[arg-type]
            )
            bars_resp = client.get_stock_bars(req)
            bars_data = bars_resp.data.get(sym, [])  # type: ignore[union-attr]
            if not bars_data:
                continue

            bars_by_date: dict[str, list] = defaultdict(list)
            for bar in bars_data:
                bar_et = bar.timestamp.astimezone(_ET)
                bars_by_date[bar_et.strftime("%Y-%m-%d")].append((bar_et, bar))

            sym_result: dict[str, dict] = {}
            for date_str, timed_bars in bars_by_date.items():
                timed_bars.sort(key=lambda x: x[0])
                metrics = _compute_intraday_day(date_str, timed_bars)
                if metrics is not None:  # pragma: no branch
                    sym_result[date_str] = metrics

            result[sym] = sym_result

        except Exception as exc:
            logger.warning(f"Intraday fetch failed for {sym}: {exc}")

    logger.info(f"Intraday bars fetched for {len(result)}/{len(symbols)} symbols")
    return result


def _build_regime_map(
    spy_indicators: pd.DataFrame,
    vix_df_for_regime: pd.DataFrame | None,
) -> dict[str, str]:
    """Build a {date_str: regime_name} map using the shared 5-state classifier."""
    spy_close = pd.DataFrame({"Close": spy_indicators["Close"].astype(float)}).dropna()
    spy_close.index = pd.DatetimeIndex(spy_close.index).tz_localize(None)
    trading_dates = [ts.strftime("%Y-%m-%d") for ts in spy_indicators.index]
    result = compute_regime_series(spy_close, vix_df_for_regime, trading_dates)
    stress_days = sum(1 for r in result.values() if r == "STRESS_RISK_OFF")
    logger.info(f"Regime map computed — {stress_days} STRESS_RISK_OFF sessions")
    return result


def _binomial_p_value(wins: int, n: int, p0: float = 0.5) -> float:
    """One-sided p-value: P(X >= wins) under H0: win_rate = p0 (Binomial exact test).

    Uses the regularised incomplete beta function via the relation
    P(X >= k | n, p) = I_p(k, n-k+1), computed with the log-beta recurrence.
    Returns 1.0 for degenerate inputs (n=0 or wins=0).
    """
    if n <= 0 or wins <= 0:
        return 1.0
    if wins > n:
        return 0.0

    # P(X >= wins) = sum_{k=wins}^{n} C(n,k) * p0^k * (1-p0)^(n-k)
    # Compute in log-space to avoid overflow, then sum probabilities.
    log_p0 = math.log(p0)
    log_q0 = math.log(1 - p0)
    total = 0.0
    for k in range(wins, n + 1):
        lc = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        total += math.exp(lc + k * log_p0 + (n - k) * log_q0)
    return min(1.0, total)


def _holm_bonferroni(
    p_values: dict[tuple, float],
    alpha: float = 0.05,
) -> set[tuple]:
    """Holm-Bonferroni correction across a family of hypothesis tests.

    Parameters
    ----------
    p_values : dict[tuple, float]
        {cell_key: p_value} for every regime×signal cell tested.
    alpha : float
        Family-wise error rate (default 0.05).

    Returns
    -------
    set[tuple]
        Cell keys that FAIL to reject H0 after correction — i.e. cells whose
        win rate is NOT significantly above chance.  Callers use this to build
        the blocked-signal set.
    """
    if not p_values:
        return set()
    m = len(p_values)
    sorted_cells = sorted(p_values.items(), key=lambda x: x[1])
    failed: set[tuple] = set()
    for i, (_cell, p) in enumerate(sorted_cells):
        threshold = alpha / (m - i)
        if p > threshold:
            # Once a test fails, all remaining (larger p-values) also fail
            failed.update(c for c, _ in sorted_cells[i:])
            break
    return failed


def compute_regime_blocked(
    regime_stats: dict[str, dict[str, dict]],
    min_trades: int = 20,
    alpha: float = 0.05,
    null_win_rate: float = 0.5,
) -> dict[str, set[str]]:
    """Derive data-driven signal-blocking rules from regime_stats.

    For each regime×signal cell with >= min_trades, run a one-sided binomial
    test (H0: win_rate <= null_win_rate).  Apply Holm-Bonferroni correction
    across all cells.  Cells that fail to reject H0 (win rate not
    significantly above chance) are candidates for blocking.

    Parameters
    ----------
    regime_stats : dict
        Output of run_signal_analysis()["regime_stats"]:
        {signal: {regime: {wins, losses, total_return}}}.
    min_trades : int
        Minimum trades in a cell to include in the test (default 20).
    alpha : float
        Family-wise error rate (default 0.05).
    null_win_rate : float
        Null hypothesis win rate (default 0.5 = coin flip).

    Returns
    -------
    dict[str, set[str]]
        {regime: {signals_to_block}} — signals whose win rate in that regime
        is NOT statistically distinguishable from chance after correction.
        Only includes cells that had enough trades to test.
    """
    p_values: dict[tuple, float] = {}
    for sig, reg_dict in regime_stats.items():
        for reg, cell in reg_dict.items():
            n = cell["wins"] + cell["losses"]
            if n < min_trades:
                continue
            p = _binomial_p_value(cell["wins"], n, p0=null_win_rate)
            p_values[(sig, reg)] = p

    failed_cells = _holm_bonferroni(p_values, alpha=alpha)

    blocked: dict[str, set[str]] = {}
    for sig, reg in failed_cells:
        blocked.setdefault(reg, set()).add(sig)
    return blocked


def _bootstrap_cell_ci(
    outcomes: list[float],
    n_boot: int = 2000,
    block_len: int = 5,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Block-bootstrap 95% CI on win rate for a regime×signal cell.

    Preserves serial autocorrelation (momentum, mean-reversion streaks) by
    resampling non-overlapping blocks of `block_len` trades rather than i.i.d.
    Returns (ci_low, ci_high) as fractions; returns (nan, nan) if n < 10.
    """
    n = len(outcomes)
    if n < 10:
        return (float("nan"), float("nan"))
    rng = None
    try:
        import random

        rng = random.Random(42)
    except Exception:
        pass
    arr = list(outcomes)
    blocks = [arr[i : i + block_len] for i in range(0, n, block_len)]
    boot_stats: list[float] = []
    for _ in range(n_boot):
        if rng is not None:
            sample_blocks = rng.choices(blocks, k=len(blocks))
        else:
            sample_blocks = blocks
        flat = [x for blk in sample_blocks for x in blk][:n]
        boot_stats.append(sum(flat) / len(flat))
    boot_stats.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return (round(boot_stats[lo_idx], 4), round(boot_stats[hi_idx], 4))


# Signals exempt from the cross-sectional RS rank filter.
# mean_reversion deliberately buys beaten-down stocks (low RS expected).
# insider_buying and pead are fundamental/event signals, not price-momentum.
_RS_EXEMPT_SIGNALS = frozenset(
    {"mean_reversion", "range_reversion", "rsi_divergence", "insider_buying", "pead"}
)


def _compute_rs_ranks(
    indicators: dict[str, pd.DataFrame],
    spy_indicators: pd.DataFrame | None,
) -> dict[str, dict[str, float]]:
    """Cross-sectional 20d relative-strength percentile ranks.

    For each trading date ranks every symbol by (ret_20d − SPY_ret_20d).
    Returns {symbol: {date_str: percentile}} where 100 = top of universe.
    Returns {} when spy_indicators is None or lacks ret_20d.
    """
    if spy_indicators is None or "ret_20d" not in spy_indicators.columns:
        return {}

    frames: dict[str, pd.Series] = {
        sym: df["ret_20d"] for sym, df in indicators.items() if "ret_20d" in df.columns
    }
    if len(frames) < 4:
        return {}

    wide = pd.DataFrame(frames)
    spy_ret20 = spy_indicators["ret_20d"].reindex(wide.index)
    excess = wide.sub(spy_ret20, axis=0)
    rs_rank_wide = excess.rank(axis=1, pct=True, na_option="keep") * 100

    result: dict[str, dict[str, float]] = {}
    for sym in rs_rank_wide.columns:
        sym_series = rs_rank_wide[sym].dropna()
        if not sym_series.empty:
            result[sym] = {ts.strftime("%Y-%m-%d"): float(v) for ts, v in sym_series.items()}
    return result


def _compute_rs_rank_lag10(
    rs_ranks: dict[str, dict[str, float]],
    trading_dates: pd.DatetimeIndex,
) -> dict[str, dict[str, float]]:
    """Shift rs_ranks forward by 10 trading days.

    For each symbol, maps every date D to the rank that existed 10 trading
    days before D.  Used by the short simulation to check whether a stock
    was a relative-strength leader 10 days prior (rs_deterioration signal).
    """
    date_strs = [d.strftime("%Y-%m-%d") for d in trading_dates]
    result: dict[str, dict[str, float]] = {}
    for sym, rank_by_date in rs_ranks.items():
        shifted: dict[str, float] = {}
        for i in range(10, len(date_strs)):
            past = date_strs[i - 10]
            if past in rank_by_date:
                shifted[date_strs[i]] = rank_by_date[past]
        if shifted:
            result[sym] = shifted
    return result


def _liquidity_spread_bps(adv_usd: float) -> float:
    """Liquidity-scaled half-spread: wider for illiquid names.

    Formula: max(SPREAD_BPS, 50 / sqrt(ADV_USD / 1e6))
    Examples: $10M ADV → 15.8 bps; $100M ADV → 5 bps; $1B ADV → SPREAD_BPS floor.
    """
    if adv_usd <= 0:
        return float(SPREAD_BPS)
    return max(float(SPREAD_BPS), 50.0 / math.sqrt(adv_usd / 1_000_000))


def _market_impact_bps(notional: float, adv_usd: float) -> float:
    """Square-root market impact: 10 bps at 1% of ADV, capped at 50 bps.

    Based on Almgren/Chriss sqrt-of-participation-rate model.
    Participation rate = notional / adv_usd (as a fraction).
    """
    if adv_usd <= 0 or notional <= 0:
        return 0.0
    participation_pct = (notional / adv_usd) * 100
    return min(50.0, 10.0 * math.sqrt(participation_pct))


def _run_simulation(
    indicators: dict[str, pd.DataFrame],
    trading_dates: pd.DatetimeIndex,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    params: dict | None = None,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    intraday_data: dict[str, dict[str, dict]] | None = None,
    spy_indicators: pd.DataFrame | None = None,
    per_signal_cap: int = 2,
    regime_by_date: dict[str, str] | None = None,
    vix_spike_by_date: dict[str, bool] | None = None,
    earnings_history: dict[str, list[dict]] | None = None,
    insider_history: dict[str, list[dict]] | None = None,
    disabled_signals: frozenset[str] | None = None,
    risk_config: RiskConfig | None = None,
    rs_ranks: dict[str, dict[str, float]] | None = None,
    rs_top_pct: float = 0.75,
    stop_activation_delay: int = 2,
) -> dict:
    """Core trading simulation on pre-computed indicators. Called by both run_backtest
    and run_walk_forward_optimized (the latter avoids re-downloading data per param combo).

    stop_activation_delay: stop-loss checks are skipped for trading_days_held in
    [1, stop_activation_delay] (inclusive).  Default 2 skips Day 1 and Day 2 stop
    checks — Day 2 exits show the same gap-through pattern as Day 1 (WR 2-8%, avg
    -5 to -8%) while Day 3 recovers to 55-69% WR.
    Set to 0 to disable the delay and restore the original always-on behaviour."""
    rc = risk_config or RiskConfig.from_config()
    s_bps = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    _sp_bps_override = spread_bps  # None → liquidity-scaled per trade
    # Always block intraday signals from the multi-day track — they require same-day
    # Alpaca minute bars and an Open→Close exit that this simulation does not model.
    disabled_signals = (disabled_signals or frozenset()) | INTRADAY_SIGNALS
    cash = initial_capital
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[tuple[str, float]] = []

    for today in trading_dates:
        today_str = today.strftime("%Y-%m-%d")

        # Update equity
        portfolio_value = cash
        for sym, pos in positions.items():
            try:
                px = (
                    float(indicators[sym].loc[today, "Close"])
                    if today in indicators[sym].index
                    else pos["entry_price"]
                )
                portfolio_value += pos["shares"] * px
            except Exception:
                portfolio_value += pos["shares"] * pos["entry_price"]
        equity_curve.append((today_str, round(portfolio_value, 4)))

        # Check exits for open positions
        to_close = []
        for sym, pos in positions.items():
            try:
                row_today = indicators[sym].loc[today]
                px = float(row_today["Close"])
            except Exception:
                continue
            trading_days_held = sum(1 for _ in pd.bdate_range(pos["entry_date"], today)) - 1

            # Skip stop checks during the activation delay window.  Overnight gaps that
            # open below the stop on Day 1 historically reverse by Day 3 (0% WR on Day 1
            # exits vs 56-68% WR on Day 3); the delay avoids premature forced exits.
            # Semantics: delay=1 skips Day 1 (trading_days_held==1), delay=0 disables.
            if 0 < trading_days_held <= stop_activation_delay:
                pnl_pct_check = px / pos["entry_price"] - 1
                if pnl_pct_check >= rc.take_profit_pct:
                    reason = "take_profit"
                elif trading_days_held >= max_hold_days:
                    reason = "time_exit"
                else:
                    continue
                fill_base = px
            else:
                # Gap-through-stop: if today's open is already at or below the stop price,
                # fill at open (realistic — we can't catch the stop intrabar).
                stop_price = pos["entry_price"] * (1 - rc.stop_loss_pct)
                open_px = float(row_today.get("Open", px))
                if open_px <= stop_price:
                    reason = "stop_loss"
                    fill_base = open_px
                else:
                    pnl_pct = px / pos["entry_price"] - 1
                    reason = None
                    if pnl_pct <= -rc.stop_loss_pct:
                        reason = "stop_loss"
                    elif pnl_pct >= rc.take_profit_pct:
                        reason = "take_profit"
                    elif trading_days_held >= max_hold_days:
                        reason = "time_exit"
                    fill_base = px

            if reason:
                df_sym = indicators[sym]
                avg_vol_20 = float(
                    df_sym.loc[today, "avg_volume_20"] if "avg_volume_20" in df_sym.columns else 0
                )
                adv_usd = avg_vol_20 * fill_base
                sp = (
                    _sp_bps_override
                    if _sp_bps_override is not None
                    else _liquidity_spread_bps(adv_usd)
                )
                exit_notional = pos["shares"] * fill_base
                impact = _market_impact_bps(exit_notional, adv_usd)
                sell_factor = 1.0 - (s_bps + sp / 2 + impact) / 10_000
                exit_px = fill_base * sell_factor
                cash += pos["shares"] * exit_px
                trades.append(
                    {
                        "date": today_str,
                        "symbol": sym,
                        "action": "SELL",
                        "reason": reason,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_px,
                        "pnl_pct": round((exit_px / pos["entry_price"] - 1) * 100, 2),
                        "signal": pos["signal"],
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "entry_regime": pos.get("entry_regime"),
                        "days_held": trading_days_held,
                    }
                )
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        # Look for entries (signal from bar T-1, enter at bar T — no lookahead)
        slots = max_positions - len(positions)
        if slots <= 0:
            continue

        candidates = []
        for sym, df in indicators.items():
            if sym in positions or today not in df.index:
                continue
            today_loc = df.index.get_loc(today)
            if today_loc == 0:
                continue
            prev_row = df.iloc[today_loc - 1]
            prev_date_str = df.index[today_loc - 1].strftime("%Y-%m-%d")

            # Look up intraday data for T-1 (when signal fired)
            intraday = intraday_data.get(sym, {}).get(prev_date_str) if intraday_data else None

            # SPY returns for T-1 (for rs_leader)
            spy_5d = spy_10d = None
            if spy_indicators is not None:
                prev_ts = df.index[today_loc - 1]
                if prev_ts in spy_indicators.index:
                    spy_row = spy_indicators.loc[prev_ts]
                    spy_5d = spy_row.get("ret_5d")
                    spy_10d = spy_row.get("ret_10d")

            regime = regime_by_date.get(prev_date_str) if regime_by_date else None
            vix_spike = (
                bool(vix_spike_by_date.get(prev_date_str, False)) if vix_spike_by_date else False
            )

            # Point-in-time fundamentals (only computed when histories are loaded)
            fundamentals: dict | None = None
            if earnings_history is not None or insider_history is not None:
                prev_date = df.index[today_loc - 1].date()
                fund: dict = {}
                if earnings_history is not None:
                    fund["pead_active"] = pead_active_on_date(sym, prev_date, earnings_history)
                if insider_history is not None:
                    fund.update(insider_state_on_date(sym, prev_date, insider_history))
                fundamentals = fund

            signal = _entry_signal(
                prev_row,
                params,
                intraday=intraday,
                spy_ret_5d=spy_5d,
                spy_ret_10d=spy_10d,
                regime=regime,
                vix_spike=vix_spike,
                fundamentals=fundamentals,
                disabled_signals=disabled_signals,
            )
            if signal:
                if rs_ranks is not None and signal not in _RS_EXEMPT_SIGNALS:
                    rank_pct = rs_ranks.get(sym, {}).get(prev_date_str)
                    if rank_pct is not None and rank_pct < rs_top_pct * 100:
                        continue
                candidates.append((sym, signal, float(prev_row["rsi"])))

        def _sort_key(item: tuple) -> tuple:
            _, sig, rsi = item
            priority = _SIGNAL_PRIORITY.get(sig, 99)
            # mean_reversion: ascending RSI (most oversold first)
            # everything else: descending RSI distance from neutral (strongest confirmation first)
            rsi_key = rsi if sig == "mean_reversion" else -abs(rsi - 50)
            return (priority, rsi_key)

        candidates.sort(key=_sort_key)

        # Apply per-signal cap: at most `per_signal_cap` positions from any one
        # signal per day, so no single signal monopolises all available slots.
        signal_counts: dict[str, int] = defaultdict(int)
        capped: list[tuple] = []
        for item in candidates:
            if len(capped) >= slots:
                break
            _, sig, _ = item
            if signal_counts[sig] < per_signal_cap:
                capped.append(item)
                signal_counts[sig] += 1

        for sym, signal, _ in capped:
            try:
                try:
                    entry_px = float(indicators[sym].loc[today, "Open"])
                except (KeyError, TypeError):
                    entry_px = float(indicators[sym].loc[today, "Close"])
                df_sym = indicators[sym]
                avg_vol_20 = float(
                    df_sym.loc[today, "avg_volume_20"] if "avg_volume_20" in df_sym.columns else 0
                )
                notional = (cash / slots) * 0.9
                adv_usd = avg_vol_20 * entry_px
                sp = (
                    _sp_bps_override
                    if _sp_bps_override is not None
                    else _liquidity_spread_bps(adv_usd)
                )
                impact = _market_impact_bps(notional, adv_usd)
                buy_factor = 1.0 + (s_bps + sp / 2 + impact) / 10_000
                fill_px = entry_px * buy_factor
                shares = notional / fill_px
                cost = shares * fill_px
                if cost > cash or cost < 0.5:
                    continue
                cash -= cost
                positions[sym] = {
                    "entry_price": fill_px,
                    "entry_date": today,
                    "shares": shares,
                    "signal": signal,
                    "entry_regime": regime,
                }
                trades.append(
                    {
                        "date": today_str,
                        "symbol": sym,
                        "action": "BUY",
                        "price": fill_px,
                        "signal": signal,
                    }
                )
            except Exception:
                continue

    # Close remaining positions at end of window
    last_date = trading_dates[-1] if len(trading_dates) else None
    for sym, pos in positions.items():
        try:
            last_row = indicators[sym].iloc[-1]
            px = float(last_row["Close"])
            avg_vol_20 = float(last_row.get("avg_volume_20", 0))
            adv_usd = avg_vol_20 * px
            sp = (
                _sp_bps_override if _sp_bps_override is not None else _liquidity_spread_bps(adv_usd)
            )
            exit_notional = pos["shares"] * px
            impact = _market_impact_bps(exit_notional, adv_usd)
            sell_factor = 1.0 - (s_bps + sp / 2 + impact) / 10_000
            exit_px = px * sell_factor
            cash += pos["shares"] * exit_px
            pnl_pct = (exit_px / pos["entry_price"] - 1) * 100
            days_held = (
                sum(1 for _ in pd.bdate_range(pos["entry_date"], last_date)) - 1
                if last_date is not None
                else 0
            )
            trades.append(
                {
                    "date": "end",
                    "symbol": sym,
                    "action": "SELL",
                    "reason": "end_of_backtest",
                    "pnl_pct": round(pnl_pct, 2),
                    "signal": pos["signal"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "entry_regime": pos.get("entry_regime"),
                    "days_held": days_held,
                }
            )
        except Exception:
            cash += pos["shares"] * pos["entry_price"]

    # Compute metrics
    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100
    closed_trades = [t for t in trades if t["action"] == "SELL" and "pnl_pct" in t]
    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    avg_return = (
        sum(t["pnl_pct"] for t in closed_trades) / len(closed_trades) if closed_trades else 0
    )

    eq_values = [v for _, v in equity_curve]
    peak = eq_values[0] if eq_values else initial_capital
    max_dd = 0.0
    for v in eq_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100 if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    by_signal: dict[str, dict] = {}
    for t in closed_trades:
        s = t.get("signal", "unknown")
        by_signal.setdefault(s, {"wins": 0, "losses": 0, "total_return": 0.0})
        by_signal[s]["total_return"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            by_signal[s]["wins"] += 1
        else:
            by_signal[s]["losses"] += 1

    daily_rets = pd.Series(eq_values).pct_change().dropna()
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * (252**0.5)) if daily_rets.std() > 0 else 0.0
    )

    # Derive signals_tested from what data was provided
    signals_tested = [
        "mean_reversion",
        "momentum",
        "macd_crossover",
        "bb_squeeze",
        "trend_pullback",
        "breakout_52w",
        "rsi_divergence",
    ]
    if any("is_inside_day" in df.columns for df in indicators.values()):
        signals_tested.append("inside_day_breakout")
    if any("gap_pct" in df.columns for df in indicators.values()):
        signals_tested.append("gap_and_go")
    if any("mom_12_1" in df.columns for df in indicators.values()):
        signals_tested.append("momentum_12_1")
    if any("hv_rank" in df.columns for df in indicators.values()):
        signals_tested.append("iv_compression")
    if earnings_history is not None:
        signals_tested.append("pead")
    if insider_history is not None:
        signals_tested.append("insider_buying")
    if spy_indicators is not None:
        signals_tested.append("rs_leader")
    if vix_spike_by_date:
        signals_tested.append("vix_fear_reversion")
    if intraday_data:
        signals_tested.extend(["vwap_reclaim", "orb_breakout", "intraday_momentum"])

    all_backtestable = {
        "mean_reversion",
        "momentum",
        "macd_crossover",
        "bb_squeeze",
        "trend_pullback",
        "breakout_52w",
        "inside_day_breakout",
        "gap_and_go",
        "momentum_12_1",
        "iv_compression",
        "rs_leader",
        "vix_fear_reversion",
        "vwap_reclaim",
        "orb_breakout",
        "intraday_momentum",
        "insider_buying",
        "pead",
        "rsi_divergence",
        "range_reversion",
    }
    signals_not_tested = sorted(all_backtestable - set(signals_tested))

    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": len(closed_trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_return_per_trade_pct": round(avg_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "by_signal": by_signal,
        "equity_curve": equity_curve,
        "trades": trades,
        "validation_scope": "rule_proxy_only",
        "signals_tested": signals_tested,
        "signals_not_tested": signals_not_tested,
    }


def _run_short_simulation(
    indicators: dict[str, pd.DataFrame],
    trading_dates: pd.DatetimeIndex,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = _SHORT_MAX_HOLD_DAYS,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    spy_indicators: pd.DataFrame | None = None,
    regime_by_date: dict[str, str] | None = None,
    earnings_history: dict[str, list[dict]] | None = None,
    risk_config: RiskConfig | None = None,
    rs_ranks: dict[str, dict[str, float]] | None = None,
    short_params: dict | None = None,
    rs_rank_lag10: dict[str, dict[str, float]] | None = None,
    vix_term_by_date: dict[str, bool] | None = None,
) -> dict:
    """Short-side simulation — mirrors _run_simulation() but enters short positions.

    Entry criteria (T-1 bar evaluated, enter at T open):
    - rs_rank_pct < _SHORT_RS_RANK_GATE (weakest quartile)
    - At least one bearish signal fires from evaluate_short_signals()

    Cash model:
    - Entry: lock up ``notional`` from capital (cash -= notional)
    - Exit: return capital + PnL (cash += notional + shares * (entry_price - cover_px))
    - PnL% = (entry_price - cover_px) / entry_price * 100  (positive when price falls)

    Slippage: entry fill = market * sell_factor (receive less), cover fill = market * buy_factor
    (pay more), mirroring the double-sided cost model used for longs.
    """
    rc = risk_config or RiskConfig.from_config()
    s_bps = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    _sp_bps_override = spread_bps

    cash = initial_capital
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[tuple[str, float]] = []

    for today in trading_dates:
        today_str = today.strftime("%Y-%m-%d")

        # Mark-to-market equity: cash + unrealized PnL from all open short positions
        portfolio_value = cash
        for sym, pos in positions.items():
            try:
                px = (
                    float(indicators[sym].loc[today, "Close"])
                    if today in indicators[sym].index
                    else pos["entry_price"]
                )
                portfolio_value += pos["notional"] + pos["shares"] * (pos["entry_price"] - px)
            except Exception:
                portfolio_value += pos["notional"]
        equity_curve.append((today_str, round(portfolio_value, 4)))

        # Check exits for open short positions
        to_close = []
        for sym, pos in positions.items():
            try:
                row_today = indicators[sym].loc[today]
                px = float(row_today["Close"])
            except Exception:
                continue
            trading_days_held = sum(1 for _ in pd.bdate_range(pos["entry_date"], today)) - 1

            # Stop-loss for shorts: price RISES above entry * (1 + stop_loss_pct)
            stop_price = pos["entry_price"] * (1 + rc.stop_loss_pct)
            open_px = float(row_today.get("Open", px))
            reason = None
            fill_base = px

            # Gap-through-stop: open already at or above stop → fill at open
            if open_px >= stop_price and trading_days_held > 0:
                reason = "stop_loss"
                fill_base = open_px
            else:
                pnl_pct = (pos["entry_price"] - px) / pos["entry_price"]
                if pnl_pct <= -rc.stop_loss_pct:
                    reason = "stop_loss"
                elif pnl_pct >= rc.take_profit_pct:
                    reason = "take_profit"
                elif trading_days_held >= max_hold_days:
                    reason = "time_exit"

            if reason:
                df_sym = indicators[sym]
                avg_vol_20 = float(
                    df_sym.loc[today, "avg_volume_20"] if "avg_volume_20" in df_sym.columns else 0
                )
                adv_usd = avg_vol_20 * fill_base
                sp = (
                    _sp_bps_override
                    if _sp_bps_override is not None
                    else _liquidity_spread_bps(adv_usd)
                )
                cover_notional = pos["shares"] * fill_base
                impact = _market_impact_bps(cover_notional, adv_usd)
                buy_factor = 1.0 + (s_bps + sp / 2 + impact) / 10_000
                cover_px = fill_base * buy_factor

                pnl = pos["shares"] * (pos["entry_price"] - cover_px)
                cash += pos["notional"] + pnl

                pnl_pct_final = (pos["entry_price"] - cover_px) / pos["entry_price"] * 100
                trades.append(
                    {
                        "date": today_str,
                        "symbol": sym,
                        "action": "COVER",
                        "reason": reason,
                        "entry_price": pos["entry_price"],
                        "exit_price": cover_px,
                        "pnl_pct": round(pnl_pct_final, 2),
                        "signal": pos["signal"],
                        "signals": pos.get("signals", [pos["signal"]]),
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "entry_regime": pos.get("entry_regime"),
                        "days_held": trading_days_held,
                    }
                )
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        # Look for short entries (signal from bar T-1, enter at bar T — no lookahead)
        slots = max_positions - len(positions)
        if slots <= 0:
            continue

        candidates = []
        for sym, df in indicators.items():
            if sym in positions or today not in df.index:
                continue
            today_loc = df.index.get_loc(today)
            if today_loc == 0:
                continue
            prev_row = df.iloc[today_loc - 1]
            prev_date_str = df.index[today_loc - 1].strftime("%Y-%m-%d")

            # SPY 20d return for T-1 (for loser_momentum relative strength)
            spy_ret_20d = None
            if spy_indicators is not None:
                prev_ts = df.index[today_loc - 1]
                if prev_ts in spy_indicators.index:
                    spy_ret_20d = spy_indicators.loc[prev_ts].get("ret_20d")

            # RS rank gate
            rs_rank_pct = rs_ranks.get(sym, {}).get(prev_date_str) if rs_ranks else None
            rs_rank_pct_10d_ago = (
                rs_rank_lag10.get(sym, {}).get(prev_date_str) if rs_rank_lag10 else None
            )

            regime = regime_by_date.get(prev_date_str) if regime_by_date else None

            # VIX term structure gate — True when VIX9D/VIX > 1.05
            vix_term_inverted = (
                vix_term_by_date.get(prev_date_str, True) if vix_term_by_date else True
            )

            # Point-in-time earnings fundamentals (earnings_miss only — gap handled below)
            fundamentals: dict | None = None
            if earnings_history is not None:
                prev_date = df.index[today_loc - 1].date()
                fund: dict = {}
                fund["earnings_miss_active"] = earnings_miss_active_on_date(
                    sym, prev_date, earnings_history
                )
                fundamentals = fund

            signals = _short_entry_signal(
                prev_row,
                rs_rank_pct=rs_rank_pct,
                spy_ret_20d=spy_ret_20d,
                regime=regime,
                fundamentals=fundamentals,
                short_params=short_params,
                rs_rank_pct_10d_ago=rs_rank_pct_10d_ago,
                vix_term_inverted=vix_term_inverted,
            )
            if signals:
                key_signal = signals[0]
                candidates.append((sym, key_signal, signals, rs_rank_pct or 0.0))

            # Same-bar gap-open entry: detect and enter on the earnings reaction bar itself.
            # AMC/BMO earnings are public before the open, so using today_open is not lookahead.
            if earnings_history is not None:
                today_date_obj = df.index[today_loc].date()
                earn_dt = recent_earnings_date(sym, today_date_obj, earnings_history)
                if earn_dt is not None:
                    today_open = float(df.iloc[today_loc].get("Open", 0.0))
                    prev_close = float(df.iloc[today_loc - 1].get("Close", 0.0))
                    if prev_close > 0:
                        gap_pct = (today_open - prev_close) / prev_close * 100.0
                        gap_fund = {"earnings_gap_pct": gap_pct}
                        today_row = df.iloc[today_loc]
                        gap_signals = _short_entry_signal(
                            today_row,
                            rs_rank_pct=rs_rank_pct,
                            spy_ret_20d=spy_ret_20d,
                            regime=regime,
                            fundamentals=gap_fund,
                            short_params=short_params,
                            rs_rank_pct_10d_ago=rs_rank_pct_10d_ago,
                            vix_term_inverted=vix_term_inverted,
                        )
                        if gap_signals and "earnings_gap_down" in gap_signals:
                            candidates.append(
                                (sym, gap_signals[0], gap_signals, rs_rank_pct or 0.0)
                            )

        # Sort: most signals first (highest conviction), then lowest RS rank (weakest stock)
        candidates.sort(key=lambda item: (-len(item[2]), item[3]))

        for sym, key_signal, signals, _rank in candidates[:slots]:
            try:
                try:
                    entry_px = float(indicators[sym].loc[today, "Open"])
                except (KeyError, TypeError):
                    entry_px = float(indicators[sym].loc[today, "Close"])

                df_sym = indicators[sym]
                avg_vol_20 = float(
                    df_sym.loc[today, "avg_volume_20"] if "avg_volume_20" in df_sym.columns else 0
                )
                notional = (cash / slots) * 0.9
                adv_usd = avg_vol_20 * entry_px
                sp = (
                    _sp_bps_override
                    if _sp_bps_override is not None
                    else _liquidity_spread_bps(adv_usd)
                )
                impact = _market_impact_bps(notional, adv_usd)
                sell_factor = 1.0 - (s_bps + sp / 2 + impact) / 10_000
                fill_px = entry_px * sell_factor  # receive less for short sale
                shares = notional / fill_px
                if notional > cash or notional < 0.5:
                    continue
                cash -= notional

                regime = regime_by_date.get(today_str) if regime_by_date else None
                positions[sym] = {
                    "entry_price": fill_px,
                    "entry_date": today,
                    "shares": shares,
                    "notional": notional,
                    "signal": key_signal,
                    "signals": signals,
                    "entry_regime": regime,
                }
                trades.append(
                    {
                        "date": today_str,
                        "symbol": sym,
                        "action": "SHORT",
                        "price": fill_px,
                        "signal": key_signal,
                        "signals": signals,
                    }
                )
            except Exception:
                continue

    # Close remaining positions at end of window
    last_date = trading_dates[-1] if len(trading_dates) else None
    for sym, pos in positions.items():
        try:
            last_row = indicators[sym].iloc[-1]
            px = float(last_row["Close"])
            avg_vol_20 = float(last_row.get("avg_volume_20", 0))
            adv_usd = avg_vol_20 * px
            sp = (
                _sp_bps_override if _sp_bps_override is not None else _liquidity_spread_bps(adv_usd)
            )
            cover_notional = pos["shares"] * px
            impact = _market_impact_bps(cover_notional, adv_usd)
            buy_factor = 1.0 + (s_bps + sp / 2 + impact) / 10_000
            cover_px = px * buy_factor

            pnl = pos["shares"] * (pos["entry_price"] - cover_px)
            cash += pos["notional"] + pnl

            pnl_pct = (pos["entry_price"] - cover_px) / pos["entry_price"] * 100
            days_held = (
                sum(1 for _ in pd.bdate_range(pos["entry_date"], last_date)) - 1
                if last_date is not None
                else 0
            )
            trades.append(
                {
                    "date": "end",
                    "symbol": sym,
                    "action": "COVER",
                    "reason": "end_of_backtest",
                    "pnl_pct": round(pnl_pct, 2),
                    "signal": pos["signal"],
                    "signals": pos.get("signals", [pos["signal"]]),
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "entry_regime": pos.get("entry_regime"),
                    "days_held": days_held,
                }
            )
        except Exception:
            cash += pos["notional"]

    # Compute metrics
    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100
    closed_trades = [t for t in trades if t["action"] == "COVER" and "pnl_pct" in t]
    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    avg_return = (
        sum(t["pnl_pct"] for t in closed_trades) / len(closed_trades) if closed_trades else 0
    )

    eq_values = [v for _, v in equity_curve]
    peak = eq_values[0] if eq_values else initial_capital
    max_dd = 0.0
    for v in eq_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100 if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    by_signal: dict[str, dict] = {}
    for t in closed_trades:
        s = t.get("signal", "unknown")
        by_signal.setdefault(s, {"wins": 0, "losses": 0, "total_return": 0.0})
        by_signal[s]["total_return"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            by_signal[s]["wins"] += 1
        else:
            by_signal[s]["losses"] += 1

    daily_rets = pd.Series(eq_values).pct_change().dropna()
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * (252**0.5)) if daily_rets.std() > 0 else 0.0
    )

    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": len(closed_trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_return_per_trade_pct": round(avg_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "by_signal": by_signal,
        "equity_curve": equity_curve,
        "trades": trades,
        "validation_scope": "rule_proxy_only",
        "signals_tested": sorted(SHORT_SIGNAL_PRIORITY.keys()),
    }


def _run_combined_simulation(
    indicators: dict[str, pd.DataFrame],
    trading_dates: pd.DatetimeIndex,
    initial_capital: float = 100_000.0,
    max_long_positions: int = 5,
    max_short_positions: int = 2,
    max_hold_days: int = 3,
    max_short_hold_days: int = _SHORT_MAX_HOLD_DAYS,
    params: dict | None = None,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    spy_indicators: pd.DataFrame | None = None,
    per_signal_cap: int = 2,
    regime_by_date: dict[str, str] | None = None,
    vix_spike_by_date: dict[str, bool] | None = None,
    earnings_history: dict[str, list[dict]] | None = None,
    insider_history: dict[str, list[dict]] | None = None,
    risk_config: RiskConfig | None = None,
    rs_ranks: dict[str, dict[str, float]] | None = None,
    rs_top_pct: float = 0.75,
    stop_activation_delay: int = 2,
    rs_rank_lag10: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Combined long/short simulation with a single shared cash pool.

    Longs: same entry/exit logic as _run_simulation().
    Shorts: same entry/exit logic as _run_short_simulation().
    Regime gate: shorts only enter in SHORT_ALLOWED_REGIMES; longs follow
    the existing REGIME_BLOCKED rules. Each side has its own position count
    cap; sizing is proportional to remaining total slots across both sides.
    Equity curve: cash + long_unrealised_pnl + short_unrealised_pnl.
    """
    rc = risk_config or RiskConfig.from_config()
    s_bps = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    _sp_bps_override = spread_bps

    cash = initial_capital
    long_positions: dict[str, dict] = {}
    short_positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[tuple[str, float]] = []

    for today in trading_dates:
        today_str = today.strftime("%Y-%m-%d")

        # ── Mark-to-market equity ─────────────────────────────────────────────
        portfolio_value = cash
        for sym, pos in long_positions.items():
            try:
                px = (
                    float(indicators[sym].loc[today, "Close"])
                    if today in indicators[sym].index
                    else pos["entry_price"]
                )
                portfolio_value += pos["shares"] * px
            except Exception:
                portfolio_value += pos["shares"] * pos["entry_price"]
        for sym, pos in short_positions.items():
            try:
                px = (
                    float(indicators[sym].loc[today, "Close"])
                    if today in indicators[sym].index
                    else pos["entry_price"]
                )
                portfolio_value += pos["notional"] + pos["shares"] * (pos["entry_price"] - px)
            except Exception:
                portfolio_value += pos["notional"]
        equity_curve.append((today_str, round(portfolio_value, 4)))

        # ── Long exits ────────────────────────────────────────────────────────
        long_to_close = []
        for sym, pos in long_positions.items():
            try:
                row_today = indicators[sym].loc[today]
                px = float(row_today["Close"])
            except Exception:
                continue
            trading_days_held = sum(1 for _ in pd.bdate_range(pos["entry_date"], today)) - 1

            if 0 < trading_days_held <= stop_activation_delay:
                pnl_pct_check = px / pos["entry_price"] - 1
                if pnl_pct_check >= rc.take_profit_pct:
                    reason: str | None = "take_profit"
                elif trading_days_held >= max_hold_days:
                    reason = "time_exit"
                else:
                    continue
                fill_base = px
            else:
                stop_price = pos["entry_price"] * (1 - rc.stop_loss_pct)
                open_px = float(row_today.get("Open", px))
                if open_px <= stop_price:
                    reason = "stop_loss"
                    fill_base = open_px
                else:
                    pnl_pct = px / pos["entry_price"] - 1
                    reason = None
                    if pnl_pct <= -rc.stop_loss_pct:
                        reason = "stop_loss"
                    elif pnl_pct >= rc.take_profit_pct:
                        reason = "take_profit"
                    elif trading_days_held >= max_hold_days:
                        reason = "time_exit"
                    fill_base = px

            if reason:
                df_sym = indicators[sym]
                avg_vol_20 = float(
                    df_sym.loc[today, "avg_volume_20"] if "avg_volume_20" in df_sym.columns else 0
                )
                adv_usd = avg_vol_20 * fill_base
                sp = (
                    _sp_bps_override
                    if _sp_bps_override is not None
                    else _liquidity_spread_bps(adv_usd)
                )
                exit_notional = pos["shares"] * fill_base
                impact = _market_impact_bps(exit_notional, adv_usd)
                sell_factor = 1.0 - (s_bps + sp / 2 + impact) / 10_000
                exit_px = fill_base * sell_factor
                cash += pos["shares"] * exit_px
                trades.append(
                    {
                        "date": today_str,
                        "symbol": sym,
                        "action": "SELL",
                        "side": "long",
                        "reason": reason,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_px,
                        "pnl_pct": round((exit_px / pos["entry_price"] - 1) * 100, 2),
                        "signal": pos["signal"],
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "entry_regime": pos.get("entry_regime"),
                        "days_held": trading_days_held,
                    }
                )
                long_to_close.append(sym)

        for sym in long_to_close:
            del long_positions[sym]

        # ── Short exits ───────────────────────────────────────────────────────
        short_to_close = []
        for sym, pos in short_positions.items():
            try:
                row_today = indicators[sym].loc[today]
                px = float(row_today["Close"])
            except Exception:
                continue
            trading_days_held = sum(1 for _ in pd.bdate_range(pos["entry_date"], today)) - 1

            stop_price = pos["entry_price"] * (1 + rc.stop_loss_pct)
            open_px = float(row_today.get("Open", px))
            s_reason: str | None = None
            fill_base = px

            if open_px >= stop_price and trading_days_held > 0:
                s_reason = "stop_loss"
                fill_base = open_px
            else:
                s_pnl_pct = (pos["entry_price"] - px) / pos["entry_price"]
                if s_pnl_pct <= -rc.stop_loss_pct:
                    s_reason = "stop_loss"
                elif s_pnl_pct >= rc.take_profit_pct:
                    s_reason = "take_profit"
                elif trading_days_held >= max_short_hold_days:
                    s_reason = "time_exit"

            if s_reason:
                df_sym = indicators[sym]
                avg_vol_20 = float(
                    df_sym.loc[today, "avg_volume_20"] if "avg_volume_20" in df_sym.columns else 0
                )
                adv_usd = avg_vol_20 * fill_base
                sp = (
                    _sp_bps_override
                    if _sp_bps_override is not None
                    else _liquidity_spread_bps(adv_usd)
                )
                cover_notional = pos["shares"] * fill_base
                impact = _market_impact_bps(cover_notional, adv_usd)
                buy_factor = 1.0 + (s_bps + sp / 2 + impact) / 10_000
                cover_px = fill_base * buy_factor
                pnl = pos["shares"] * (pos["entry_price"] - cover_px)
                cash += pos["notional"] + pnl
                pnl_pct_final = (pos["entry_price"] - cover_px) / pos["entry_price"] * 100
                trades.append(
                    {
                        "date": today_str,
                        "symbol": sym,
                        "action": "COVER",
                        "side": "short",
                        "reason": s_reason,
                        "entry_price": pos["entry_price"],
                        "exit_price": cover_px,
                        "pnl_pct": round(pnl_pct_final, 2),
                        "signal": pos["signal"],
                        "signals": pos.get("signals", [pos["signal"]]),
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "entry_regime": pos.get("entry_regime"),
                        "days_held": trading_days_held,
                    }
                )
                short_to_close.append(sym)

        for sym in short_to_close:
            del short_positions[sym]

        # ── Entry scanning ────────────────────────────────────────────────────
        long_slots = max_long_positions - len(long_positions)
        short_slots = max_short_positions - len(short_positions)
        total_slots = long_slots + short_slots
        if total_slots <= 0:
            continue

        today_regime = regime_by_date.get(today_str) if regime_by_date else None
        shorts_allowed = today_regime in SHORT_ALLOWED_REGIMES if today_regime else False

        # Long candidates
        long_candidates: list[tuple] = []
        if long_slots > 0:
            for sym, df in indicators.items():
                if sym in long_positions or sym in short_positions or today not in df.index:
                    continue
                today_loc = df.index.get_loc(today)
                if today_loc == 0:
                    continue
                prev_row = df.iloc[today_loc - 1]
                prev_date_str = df.index[today_loc - 1].strftime("%Y-%m-%d")

                spy_5d = spy_10d = None
                if spy_indicators is not None:
                    prev_ts = df.index[today_loc - 1]
                    if prev_ts in spy_indicators.index:
                        spy_row = spy_indicators.loc[prev_ts]
                        spy_5d = spy_row.get("ret_5d")
                        spy_10d = spy_row.get("ret_10d")

                regime = regime_by_date.get(prev_date_str) if regime_by_date else None
                vix_spike = (
                    bool(vix_spike_by_date.get(prev_date_str, False))
                    if vix_spike_by_date
                    else False
                )

                fundamentals: dict | None = None
                if earnings_history is not None or insider_history is not None:
                    prev_date = df.index[today_loc - 1].date()
                    fund: dict = {}
                    if earnings_history is not None:
                        fund["pead_active"] = pead_active_on_date(sym, prev_date, earnings_history)
                    if insider_history is not None:
                        fund.update(insider_state_on_date(sym, prev_date, insider_history))
                    fundamentals = fund

                signal = _entry_signal(
                    prev_row,
                    params,
                    spy_ret_5d=spy_5d,
                    spy_ret_10d=spy_10d,
                    regime=regime,
                    vix_spike=vix_spike,
                    fundamentals=fundamentals,
                )
                if signal:
                    if rs_ranks is not None and signal not in _RS_EXEMPT_SIGNALS:
                        rank_pct = rs_ranks.get(sym, {}).get(prev_date_str)
                        if rank_pct is not None and rank_pct < rs_top_pct * 100:
                            continue
                    long_candidates.append((sym, signal, float(prev_row["rsi"])))

        def _long_sort_key(item: tuple) -> tuple:
            _, sig, rsi = item
            priority = _SIGNAL_PRIORITY.get(sig, 99)
            rsi_key = rsi if sig == "mean_reversion" else -abs(rsi - 50)
            return (priority, rsi_key)

        long_candidates.sort(key=_long_sort_key)

        lsig_counts: dict[str, int] = defaultdict(int)
        long_capped: list[tuple] = []
        for item in long_candidates:
            if len(long_capped) >= long_slots:
                break
            _, sig, _ = item
            if lsig_counts[sig] < per_signal_cap:
                long_capped.append(item)
                lsig_counts[sig] += 1

        # Short candidates (only in bearish regimes)
        short_candidates: list[tuple] = []
        claimed_long = {sym for sym, _, _ in long_capped}
        if short_slots > 0 and shorts_allowed:
            for sym, df in indicators.items():
                if (
                    sym in long_positions
                    or sym in short_positions
                    or sym in claimed_long
                    or today not in df.index
                ):
                    continue
                today_loc = df.index.get_loc(today)
                if today_loc == 0:
                    continue
                prev_row = df.iloc[today_loc - 1]
                prev_date_str = df.index[today_loc - 1].strftime("%Y-%m-%d")

                spy_ret_20d = None
                if spy_indicators is not None:
                    prev_ts = df.index[today_loc - 1]
                    if prev_ts in spy_indicators.index:
                        spy_ret_20d = spy_indicators.loc[prev_ts].get("ret_20d")

                rs_rank_pct = rs_ranks.get(sym, {}).get(prev_date_str) if rs_ranks else None
                rs_rank_pct_10d_ago = (
                    rs_rank_lag10.get(sym, {}).get(prev_date_str) if rs_rank_lag10 else None
                )

                fund_s: dict | None = None
                if earnings_history is not None:
                    prev_date = df.index[today_loc - 1].date()
                    fund_s = {
                        "earnings_miss_active": earnings_miss_active_on_date(
                            sym, prev_date, earnings_history
                        )
                    }

                signals = _short_entry_signal(
                    prev_row,
                    rs_rank_pct=rs_rank_pct,
                    spy_ret_20d=spy_ret_20d,
                    regime=today_regime,
                    fundamentals=fund_s,
                    rs_rank_pct_10d_ago=rs_rank_pct_10d_ago,
                    vix_term_inverted=True,
                )
                if signals:
                    short_candidates.append((sym, signals[0], signals, rs_rank_pct or 0.0))

        short_candidates.sort(key=lambda item: (-len(item[2]), item[3]))

        # ── Long entries ──────────────────────────────────────────────────────
        for sym, signal, _ in long_capped:
            try:
                try:
                    entry_px = float(indicators[sym].loc[today, "Open"])
                except (KeyError, TypeError):
                    entry_px = float(indicators[sym].loc[today, "Close"])
                df_sym = indicators[sym]
                avg_vol_20 = float(
                    df_sym.loc[today, "avg_volume_20"] if "avg_volume_20" in df_sym.columns else 0
                )
                remaining = (max_long_positions - len(long_positions)) + (
                    max_short_positions - len(short_positions)
                )
                notional = (cash / max(remaining, 1)) * 0.9
                adv_usd = avg_vol_20 * entry_px
                sp = (
                    _sp_bps_override
                    if _sp_bps_override is not None
                    else _liquidity_spread_bps(adv_usd)
                )
                impact = _market_impact_bps(notional, adv_usd)
                buy_factor = 1.0 + (s_bps + sp / 2 + impact) / 10_000
                fill_px = entry_px * buy_factor
                shares = notional / fill_px
                cost = shares * fill_px
                if cost > cash or cost < 0.5:
                    continue
                cash -= cost
                regime = regime_by_date.get(today_str) if regime_by_date else None
                long_positions[sym] = {
                    "entry_price": fill_px,
                    "entry_date": today,
                    "shares": shares,
                    "signal": signal,
                    "entry_regime": regime,
                }
                trades.append(
                    {
                        "date": today_str,
                        "symbol": sym,
                        "action": "BUY",
                        "side": "long",
                        "price": fill_px,
                        "signal": signal,
                    }
                )
            except Exception:
                continue

        # ── Short entries ─────────────────────────────────────────────────────
        for sym, key_signal, signals, _rank in short_candidates[:short_slots]:
            try:
                try:
                    entry_px = float(indicators[sym].loc[today, "Open"])
                except (KeyError, TypeError):
                    entry_px = float(indicators[sym].loc[today, "Close"])
                df_sym = indicators[sym]
                avg_vol_20 = float(
                    df_sym.loc[today, "avg_volume_20"] if "avg_volume_20" in df_sym.columns else 0
                )
                remaining = (max_long_positions - len(long_positions)) + (
                    max_short_positions - len(short_positions)
                )
                notional = (cash / max(remaining, 1)) * 0.9
                adv_usd = avg_vol_20 * entry_px
                sp = (
                    _sp_bps_override
                    if _sp_bps_override is not None
                    else _liquidity_spread_bps(adv_usd)
                )
                impact = _market_impact_bps(notional, adv_usd)
                sell_factor = 1.0 - (s_bps + sp / 2 + impact) / 10_000
                fill_px = entry_px * sell_factor
                shares = notional / fill_px
                if notional > cash or notional < 0.5:
                    continue
                cash -= notional
                regime = regime_by_date.get(today_str) if regime_by_date else None
                short_positions[sym] = {
                    "entry_price": fill_px,
                    "entry_date": today,
                    "shares": shares,
                    "notional": notional,
                    "signal": key_signal,
                    "signals": signals,
                    "entry_regime": regime,
                }
                trades.append(
                    {
                        "date": today_str,
                        "symbol": sym,
                        "action": "SHORT",
                        "side": "short",
                        "price": fill_px,
                        "signal": key_signal,
                        "signals": signals,
                    }
                )
            except Exception:
                continue

    # ── Close remaining positions at end of window ────────────────────────────
    last_date = trading_dates[-1] if len(trading_dates) else None
    for sym, pos in long_positions.items():
        try:
            last_row = indicators[sym].iloc[-1]
            px = float(last_row["Close"])
            avg_vol_20 = float(last_row.get("avg_volume_20", 0))
            adv_usd = avg_vol_20 * px
            sp = (
                _sp_bps_override if _sp_bps_override is not None else _liquidity_spread_bps(adv_usd)
            )
            exit_notional = pos["shares"] * px
            impact = _market_impact_bps(exit_notional, adv_usd)
            sell_factor = 1.0 - (s_bps + sp / 2 + impact) / 10_000
            exit_px = px * sell_factor
            cash += pos["shares"] * exit_px
            pnl_pct = (exit_px / pos["entry_price"] - 1) * 100
            days_held = (
                sum(1 for _ in pd.bdate_range(pos["entry_date"], last_date)) - 1
                if last_date is not None
                else 0
            )
            trades.append(
                {
                    "date": "end",
                    "symbol": sym,
                    "action": "SELL",
                    "side": "long",
                    "reason": "end_of_backtest",
                    "pnl_pct": round(pnl_pct, 2),
                    "signal": pos["signal"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "entry_regime": pos.get("entry_regime"),
                    "days_held": days_held,
                }
            )
        except Exception:
            cash += pos["shares"] * pos["entry_price"]

    for sym, pos in short_positions.items():
        try:
            last_row = indicators[sym].iloc[-1]
            px = float(last_row["Close"])
            avg_vol_20 = float(last_row.get("avg_volume_20", 0))
            adv_usd = avg_vol_20 * px
            sp = (
                _sp_bps_override if _sp_bps_override is not None else _liquidity_spread_bps(adv_usd)
            )
            cover_notional = pos["shares"] * px
            impact = _market_impact_bps(cover_notional, adv_usd)
            buy_factor = 1.0 + (s_bps + sp / 2 + impact) / 10_000
            cover_px = px * buy_factor
            pnl = pos["shares"] * (pos["entry_price"] - cover_px)
            cash += pos["notional"] + pnl
            pnl_pct = (pos["entry_price"] - cover_px) / pos["entry_price"] * 100
            days_held = (
                sum(1 for _ in pd.bdate_range(pos["entry_date"], last_date)) - 1
                if last_date is not None
                else 0
            )
            trades.append(
                {
                    "date": "end",
                    "symbol": sym,
                    "action": "COVER",
                    "side": "short",
                    "reason": "end_of_backtest",
                    "pnl_pct": round(pnl_pct, 2),
                    "signal": pos["signal"],
                    "signals": pos.get("signals", [pos["signal"]]),
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "entry_regime": pos.get("entry_regime"),
                    "days_held": days_held,
                }
            )
        except Exception:
            cash += pos["notional"]

    # ── Compute metrics ───────────────────────────────────────────────────────
    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100

    long_closed = [
        t for t in trades if t["action"] == "SELL" and "pnl_pct" in t and t.get("side") == "long"
    ]
    short_closed = [
        t for t in trades if t["action"] == "COVER" and "pnl_pct" in t and t.get("side") == "short"
    ]
    all_closed = long_closed + short_closed

    wins = [t for t in all_closed if t["pnl_pct"] > 0]
    long_wins = [t for t in long_closed if t["pnl_pct"] > 0]
    short_wins = [t for t in short_closed if t["pnl_pct"] > 0]

    win_rate = len(wins) / len(all_closed) * 100 if all_closed else 0.0
    long_win_rate = len(long_wins) / len(long_closed) * 100 if long_closed else 0.0
    short_win_rate = len(short_wins) / len(short_closed) * 100 if short_closed else 0.0
    avg_return = sum(t["pnl_pct"] for t in all_closed) / len(all_closed) if all_closed else 0.0

    eq_values = [v for _, v in equity_curve]
    peak = eq_values[0] if eq_values else initial_capital
    max_dd = 0.0
    for v in eq_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100 if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    by_signal: dict[str, dict] = {}
    for t in all_closed:
        s = t.get("signal", "unknown")
        by_signal.setdefault(s, {"wins": 0, "losses": 0, "total_return": 0.0})
        by_signal[s]["total_return"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            by_signal[s]["wins"] += 1
        else:
            by_signal[s]["losses"] += 1

    long_total_ret = sum(t["pnl_pct"] for t in long_closed)
    short_total_ret = sum(t["pnl_pct"] for t in short_closed)
    by_side: dict[str, dict] = {
        "long": {
            "trades": len(long_closed),
            "win_rate_pct": round(long_win_rate, 1),
            "avg_return_pct": round(long_total_ret / len(long_closed), 2) if long_closed else 0.0,
            "total_return_contribution": round(long_total_ret, 2),
        },
        "short": {
            "trades": len(short_closed),
            "win_rate_pct": round(short_win_rate, 1),
            "avg_return_pct": (
                round(short_total_ret / len(short_closed), 2) if short_closed else 0.0
            ),
            "total_return_contribution": round(short_total_ret, 2),
        },
    }

    daily_rets = pd.Series(eq_values).pct_change().dropna()
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * (252**0.5)) if daily_rets.std() > 0 else 0.0
    )

    regime_counts: dict[str, int] = {}
    for td in trading_dates:
        reg = (regime_by_date or {}).get(td.strftime("%Y-%m-%d"), "unknown")
        regime_counts[reg] = regime_counts.get(reg, 0) + 1

    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": len(all_closed),
        "long_trades": len(long_closed),
        "short_trades": len(short_closed),
        "win_rate_pct": round(win_rate, 1),
        "long_win_rate_pct": round(long_win_rate, 1),
        "short_win_rate_pct": round(short_win_rate, 1),
        "avg_return_per_trade_pct": round(avg_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "by_signal": by_signal,
        "by_side": by_side,
        "equity_curve": equity_curve,
        "trades": trades,
        "regime_distribution": regime_counts,
        "validation_scope": "rule_proxy_only",
    }


def _build_indicators(
    raw: pd.DataFrame,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Extract per-symbol OHLCV from a multi-symbol yfinance download and compute indicators."""
    close_all = raw["Close"]
    open_all = raw["Open"]
    volume_all = raw["Volume"]
    high_all = raw.get("High")
    low_all = raw.get("Low")

    indicators = {}
    for sym in symbols:
        try:
            cols = {
                "Close": close_all[sym],
                "Open": open_all[sym],
                "Volume": volume_all[sym],
            }
            if high_all is not None:
                cols["High"] = high_all[sym]
            if low_all is not None:
                cols["Low"] = low_all[sym]
            df = pd.DataFrame(cols).dropna()
            indicators[sym] = _compute_indicators(df)
        except Exception:
            pass
    return indicators


def _run_intraday_simulation(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    stop_loss_pct: float = 1.0,
    target_pct: float = 2.0,
    cache_dir: str | None = None,
    bars: dict | None = None,
) -> dict:
    """Intraday-track simulation — thin wrapper around run_intraday_backtest().

    Entry/exit: same-day Open→Close replay of Alpaca minute bars.
    Signals: INTRADAY_SIGNALS ∪ INTRADAY_SHORT_SIGNALS.
    Returns the same result schema as _run_simulation() so callers can compare
    both tracks side-by-side.
    """
    _assert_pre_holdout(end_date)
    logger.info(
        f"Intraday backtest: {start_date} → {end_date} | {len(symbols)} symbols"
        f" | ${initial_capital:.0f} capital | stop={stop_loss_pct}% target={target_pct}%"
    )
    from backtest.intraday_engine import run_intraday_backtest

    return run_intraday_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
        initial_capital=initial_capital,
        max_positions=max_positions,
        cache_dir=cache_dir,
        bars=bars,
    )


def run_backtest(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    params: dict | None = None,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    use_intraday: bool = False,
    per_signal_cap: int = 2,
    use_fundamentals: bool = False,
    use_earnings_only: bool = False,
    disabled_signals: frozenset[str] | None = None,
) -> dict:

    _assert_pre_holdout(end_date)

    logger.info(
        f"Backtest: {start_date} → {end_date} | {len(symbols)} symbols | ${initial_capital:.0f} capital"
        + (" | intraday=ON" if use_intraday else "")
        + (" | earnings=ON" if use_earnings_only else "")
        + (" | fundamentals=ON" if use_fundamentals else "")
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    indicators = _build_indicators(raw, symbols)

    # SPY indicators for rs_leader signal
    spy_indicators = indicators.get("SPY")
    if spy_indicators is None:  # pragma: no branch
        try:
            spy_raw = yf.download(
                "SPY", start=fetch_start, end=end_date, auto_adjust=True, progress=False
            )
            if not spy_raw.empty:  # pragma: no branch
                spy_close = spy_raw["Close"]
                spy_vol = spy_raw["Volume"]
                if isinstance(spy_raw.columns, pd.MultiIndex):  # pragma: no branch
                    spy_close = spy_raw["Close"]["SPY"]
                    spy_vol = spy_raw["Volume"]["SPY"]
                spy_df = pd.DataFrame({"Close": spy_close, "Volume": spy_vol}).dropna()
                spy_indicators = _compute_indicators(spy_df)
        except Exception as exc:
            logger.warning(f"SPY fetch failed — rs_leader signal disabled: {exc}")

    # VIX data for fear-reversion signal and regime detection
    vix_spike_by_date: dict[str, bool] = {}
    _vix_df_for_regime: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_for_regime = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_for_regime.index = pd.DatetimeIndex(_vix_df_for_regime.index).tz_localize(None)
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
            logger.info(f"VIX fetched — {sum(vix_spike_by_date.values())} fear-spike days")
    except Exception as exc:
        logger.warning(f"VIX fetch failed — vix_fear_reversion and regime filter disabled: {exc}")

    # Regime map for trend-signal gating (shared module with hysteresis)
    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        _spy_df_for_regime = pd.DataFrame({"Close": spy_indicators["Close"].astype(float)}).dropna()
        _spy_df_for_regime.index = pd.DatetimeIndex(_spy_df_for_regime.index).tz_localize(None)
        _trading_dates = [ts.strftime("%Y-%m-%d") for ts in spy_indicators.index]
        regime_by_date = compute_regime_series(
            _spy_df_for_regime, _vix_df_for_regime, _trading_dates
        )
        stress_days = sum(1 for r in regime_by_date.values() if r == "STRESS_RISK_OFF")
        logger.info(f"Regime map computed — {stress_days} STRESS_RISK_OFF sessions")

    # Alpaca intraday bars for vwap_reclaim / orb_breakout / intraday_momentum
    intraday_data: dict | None = None
    if use_intraday:
        intraday_data = _fetch_intraday_bars(symbols, start_date, end_date)
        if not intraday_data:  # pragma: no branch
            logger.warning("Intraday fetch returned no data — intraday signals disabled")
            intraday_data = None

    # Historical fundamentals for pead / insider_buying point-in-time simulation
    earnings_history: dict[str, list[dict]] | None = None
    insider_history: dict[str, list[dict]] | None = None
    if use_fundamentals or use_earnings_only:
        logger.info("Pre-fetching earnings history…")
        earnings_history = prefetch_earnings_history(symbols)
    if use_fundamentals and not use_earnings_only:
        logger.info("Pre-fetching insider history…")
        insider_history = prefetch_insider_history(symbols)
    if use_fundamentals:
        logger.info(
            f"Fundamentals ready: {len(earnings_history or {})} earnings, "
            f"{len(insider_history or {})} insider histories"
        )

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    rs_ranks = _compute_rs_ranks(indicators, spy_indicators)
    results = _run_simulation(
        indicators,
        trading_dates,
        initial_capital,
        max_positions,
        max_hold_days,
        params,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        intraday_data=intraday_data,
        spy_indicators=spy_indicators,
        per_signal_cap=per_signal_cap,
        regime_by_date=regime_by_date or None,
        vix_spike_by_date=vix_spike_by_date or None,
        earnings_history=earnings_history,
        insider_history=insider_history,
        rs_ranks=rs_ranks,
        disabled_signals=disabled_signals,
    )
    results["start"] = start_date
    results["end"] = end_date

    _print_results(results)
    _save_results(results)
    return results


def run_walk_forward_optimized(
    symbols: list[str],
    start_date: str,
    end_date: str,
    train_days: int = 120,
    test_days: int = 60,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    param_grid: dict | None = None,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    use_fundamentals: bool = False,
    use_earnings_only: bool = False,
    disabled_signals: frozenset[str] | None = None,
    per_signal_cap: int = 2,
) -> dict:
    """
    Walk-forward optimised backtest — genuine out-of-sample validation.

    For each fold:
      1. Grid-search all param combinations on the train window (Sharpe objective).
      2. Apply the best params to the immediately following test window.
      3. Record only the test-window (OOS) results.

    Because the test window never influences param selection, the OOS metrics are
    not contaminated by look-ahead. yfinance data is downloaded once and indicators
    are computed once; the grid search reuses both across all combos in a fold.

    Returns a dict with:
      - folds: per-fold OOS results + the params that were selected
      - summary: mean OOS return/win-rate/Sharpe and consistency (% profitable folds)
    """
    _assert_pre_holdout(end_date)

    grid = param_grid or _DEFAULT_PARAM_GRID
    keys = list(grid.keys())
    all_combos = [dict(zip(keys, vals, strict=True)) for vals in product(*[grid[k] for k in keys])]

    logger.info(
        f"Walk-forward: {start_date} → {end_date} | train={train_days}d test={test_days}d "
        f"| {len(all_combos)} param combos | {len(symbols)} symbols"
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched for walk-forward")
        return {}

    indicators = _build_indicators(raw, symbols)

    # SPY for rs_leader
    spy_indicators = indicators.get("SPY")
    if spy_indicators is None:  # pragma: no branch
        try:
            spy_raw = yf.download(
                "SPY", start=fetch_start, end=end_date, auto_adjust=True, progress=False
            )
            if not spy_raw.empty:  # pragma: no branch
                spy_close = spy_raw["Close"]
                spy_vol = spy_raw["Volume"]
                if isinstance(spy_raw.columns, pd.MultiIndex):  # pragma: no branch
                    spy_close = spy_raw["Close"]["SPY"]
                    spy_vol = spy_raw["Volume"]["SPY"]
                spy_df = pd.DataFrame({"Close": spy_close, "Volume": spy_vol}).dropna()
                spy_indicators = _compute_indicators(spy_df)
        except Exception as exc:
            logger.warning(f"SPY fetch failed for walk-forward: {exc}")

    wf_earnings_history: dict[str, list[dict]] | None = None
    wf_insider_history: dict[str, list[dict]] | None = None
    if use_fundamentals or use_earnings_only:
        logger.info("Walk-forward: pre-fetching earnings history…")
        wf_earnings_history = prefetch_earnings_history(symbols)
    if use_fundamentals and not use_earnings_only:
        logger.info("Walk-forward: pre-fetching insider history…")
        wf_insider_history = prefetch_insider_history(symbols)
    if use_fundamentals and not use_earnings_only and wf_earnings_history is not None:
        logger.info(
            f"Walk-forward fundamentals ready: {len(wf_earnings_history)} earnings, "
            f"{len(wf_insider_history or {})} insider histories"
        )

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    if len(trading_dates) < train_days + test_days:
        logger.warning(
            f"Date range too short: {len(trading_dates)} bdays available, "
            f"{train_days + test_days} required for one fold"
        )
        return {}

    folds_meta = []
    i = 0
    while i + train_days + test_days <= len(trading_dates):
        folds_meta.append(
            {
                "train_start": trading_dates[i].strftime("%Y-%m-%d"),
                "train_end": trading_dates[i + train_days - 1].strftime("%Y-%m-%d"),
                "test_start": trading_dates[i + train_days].strftime("%Y-%m-%d"),
                "test_end": trading_dates[i + train_days + test_days - 1].strftime("%Y-%m-%d"),
                "train_slice": slice(i, i + train_days),
                "test_slice": slice(i + train_days, i + train_days + test_days),
            }
        )
        i += test_days

    logger.info(f"Walk-forward: {len(folds_meta)} folds")

    fold_results = []
    for fold in folds_meta:
        train_dates = trading_dates[fold["train_slice"]]
        test_dates = trading_dates[fold["test_slice"]]

        # Point-in-time universe: exclude symbols that weren't yet public on
        # the train window's start date (prevents lookahead from future IPOs).
        fold_start_date = train_dates[0].date() if len(train_dates) else date.today()
        pit_syms = get_universe_for_date(fold_start_date, list(indicators.keys()))
        pit_indicators = {s: indicators[s] for s in pit_syms if s in indicators}
        if len(pit_indicators) < len(indicators):
            excluded = set(indicators.keys()) - set(pit_indicators.keys())
            logger.debug(f"Fold {fold['train_start']}: excluded {excluded} (not yet public)")

        best_params = all_combos[0]
        best_score = -float("inf")
        best_train_trades = 0

        for combo in all_combos:
            r = _run_simulation(
                pit_indicators,
                train_dates,
                initial_capital,
                max_positions,
                max_hold_days,
                combo,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
                spy_indicators=spy_indicators,
                earnings_history=wf_earnings_history,
                insider_history=wf_insider_history,
                disabled_signals=disabled_signals,
                per_signal_cap=per_signal_cap,
            )
            if r["total_trades"] < _MIN_TRAIN_TRADES:
                continue
            if r["sharpe_ratio"] > best_score:  # pragma: no branch
                best_score = r["sharpe_ratio"]
                best_params = combo
                best_train_trades = r["total_trades"]

        oos = _run_simulation(
            pit_indicators,
            test_dates,
            initial_capital,
            max_positions,
            max_hold_days,
            best_params,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            spy_indicators=spy_indicators,
            earnings_history=wf_earnings_history,
            insider_history=wf_insider_history,
            disabled_signals=disabled_signals,
            per_signal_cap=per_signal_cap,
        )

        baseline_rets = []
        for _sym, df in pit_indicators.items():
            try:
                start_px = float(df.loc[df.index >= test_dates[0]].iloc[0]["Close"])
                end_px = float(df.loc[df.index <= test_dates[-1]].iloc[-1]["Close"])
                baseline_rets.append((end_px / start_px - 1) * 100)
            except Exception:
                pass
        fold_baseline = round(sum(baseline_rets) / len(baseline_rets), 2) if baseline_rets else 0.0

        train_sharpe_val = round(best_score, 2) if best_score != -float("inf") else 0.0
        fold_results.append(
            {
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "best_params": best_params,
                "train_sharpe": train_sharpe_val,
                "train_total_trades": best_train_trades,
                "oos_total_return_pct": oos["total_return_pct"],
                "oos_win_rate_pct": oos["win_rate_pct"],
                "oos_total_trades": oos["total_trades"],
                "oos_sharpe": oos["sharpe_ratio"],
                "oos_degradation": round(train_sharpe_val - oos["sharpe_ratio"], 2),
                "random_baseline_return_pct": fold_baseline,
            }
        )

        logger.info(
            f"  Fold {fold['test_start']}–{fold['test_end']}: "
            f"params={best_params} | OOS return={oos['total_return_pct']:+.1f}% "
            f"vs baseline={fold_baseline:+.1f}% | WR={oos['win_rate_pct']:.0f}% trades={oos['total_trades']}"
        )

    if not fold_results:
        return {"folds": [], "summary": {}}

    n = len(fold_results)
    profitable = sum(1 for f in fold_results if f["oos_total_return_pct"] > 0)
    beat_baseline = sum(
        1 for f in fold_results if f["oos_total_return_pct"] > f["random_baseline_return_pct"]
    )

    sig_counts: dict = {}
    for f in fold_results:
        key = tuple(sorted(f["best_params"].items()))
        sig_counts[key] = sig_counts.get(key, 0) + 1
    modal_count = max(sig_counts.values()) if sig_counts else 0

    summary = {
        "n_folds": n,
        "mean_oos_return_pct": round(sum(f["oos_total_return_pct"] for f in fold_results) / n, 2),
        "mean_oos_win_rate_pct": round(sum(f["oos_win_rate_pct"] for f in fold_results) / n, 1),
        "mean_oos_sharpe": round(sum(f["oos_sharpe"] for f in fold_results) / n, 2),
        "profitable_folds": profitable,
        "beat_baseline_folds": beat_baseline,
        "consistency_pct": round(profitable / n * 100, 1),
        "beat_baseline_pct": round(beat_baseline / n * 100, 1),
        "param_stability_pct": round(modal_count / n * 100, 1),
        "mean_oos_degradation": round(sum(f["oos_degradation"] for f in fold_results) / n, 2),
        "random_baseline_return_pct": round(
            sum(f["random_baseline_return_pct"] for f in fold_results) / n, 2
        ),
    }

    logger.info(
        f"Walk-forward summary: {n} folds | mean OOS return {summary['mean_oos_return_pct']:+.2f}% "
        f"vs baseline {summary['random_baseline_return_pct']:+.2f}% "
        f"| beat baseline {summary['beat_baseline_pct']:.0f}% of folds "
        f"| consistency {summary['consistency_pct']:.0f}% | param stability {summary['param_stability_pct']:.0f}%"
    )

    return {"folds": fold_results, "summary": summary}


def run_ablation(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    params: dict | None = None,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    per_signal_cap: int = 2,
    use_fundamentals: bool = False,
    use_earnings_only: bool = False,
    disabled_signals: frozenset[str] | None = None,
) -> dict:
    """Measure each signal's marginal contribution to portfolio Sharpe.

    Fetches data once, then runs N+1 simulations: one baseline (all signals
    enabled) and one per signal (that signal disabled).  The ΔSharpe column
    answers "what happens to portfolio Sharpe when this signal is removed?"

      ΔSharpe < 0  → removing it hurts  → KEEP
      ΔSharpe > 0  → removing it helps  → REVIEW (signal is a drag)

    Returns::

        {
            "baseline":  full _run_simulation result dict,
            "ablations": [
                {
                    "signal":          str,
                    "baseline_trades": int,
                    "sharpe_delta":    float,
                    "return_delta":    float,
                    "verdict":         "KEEP" | "REVIEW",
                },
                ...
            ],
        }
    """
    _assert_pre_holdout(end_date)

    logger.info(
        f"Ablation: {start_date} → {end_date} | {len(symbols)} symbols | "
        f"{len(_SIGNAL_PRIORITY)} signals"
        + (" | earnings=ON" if use_earnings_only else "")
        + (" | fundamentals=ON" if use_fundamentals else "")
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    indicators = _build_indicators(raw, symbols)

    spy_indicators = indicators.get("SPY")
    if spy_indicators is None:  # pragma: no branch
        try:
            spy_raw = yf.download(
                "SPY", start=fetch_start, end=end_date, auto_adjust=True, progress=False
            )
            if not spy_raw.empty:  # pragma: no branch
                spy_close = spy_raw["Close"]
                spy_vol = spy_raw["Volume"]
                if isinstance(spy_raw.columns, pd.MultiIndex):  # pragma: no branch
                    spy_close = spy_raw["Close"]["SPY"]
                    spy_vol = spy_raw["Volume"]["SPY"]
                spy_df = pd.DataFrame({"Close": spy_close, "Volume": spy_vol}).dropna()
                spy_indicators = _compute_indicators(spy_df)
        except Exception as exc:
            logger.warning(f"SPY fetch failed: {exc}")

    vix_spike_by_date: dict[str, bool] = {}
    _vix_df_ablation: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_ablation = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_ablation.index = pd.DatetimeIndex(_vix_df_ablation.index).tz_localize(None)
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, _vix_df_ablation)

    earnings_history: dict[str, list[dict]] | None = None
    insider_history: dict[str, list[dict]] | None = None
    if use_fundamentals or use_earnings_only:
        logger.info("Ablation: pre-fetching earnings history…")
        earnings_history = prefetch_earnings_history(symbols)
    if use_fundamentals and not use_earnings_only:
        logger.info("Ablation: pre-fetching insider history…")
        insider_history = prefetch_insider_history(symbols)

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    sim_kwargs: dict = {
        "initial_capital": initial_capital,
        "max_positions": max_positions,
        "max_hold_days": max_hold_days,
        "params": params,
        "slippage_bps": slippage_bps,
        "spread_bps": spread_bps,
        "spy_indicators": spy_indicators,
        "per_signal_cap": per_signal_cap,
        "regime_by_date": regime_by_date or None,
        "vix_spike_by_date": vix_spike_by_date or None,
        "earnings_history": earnings_history,
        "insider_history": insider_history,
        "rs_ranks": _compute_rs_ranks(indicators, spy_indicators),
        "disabled_signals": disabled_signals,
    }

    logger.info("Ablation: running baseline…")
    baseline = _run_simulation(indicators, trading_dates, **sim_kwargs)

    ablations = []
    for signal_name in sorted(_SIGNAL_PRIORITY, key=lambda s: _SIGNAL_PRIORITY[s]):
        logger.info(f"Ablation: disabling {signal_name}…")
        result = _run_simulation(
            indicators,
            trading_dates,
            **{
                **sim_kwargs,
                "disabled_signals": frozenset({signal_name}) | (disabled_signals or frozenset()),
            },
        )
        baseline_sig = baseline["by_signal"].get(signal_name, {})
        baseline_trades = baseline_sig.get("wins", 0) + baseline_sig.get("losses", 0)
        sharpe_delta = round(result["sharpe_ratio"] - baseline["sharpe_ratio"], 3)
        return_delta = round(result["total_return_pct"] - baseline["total_return_pct"], 2)
        ablations.append(
            {
                "signal": signal_name,
                "baseline_trades": baseline_trades,
                "sharpe_delta": sharpe_delta,
                "return_delta": return_delta,
                "verdict": "KEEP" if sharpe_delta < 0 else "REVIEW",
            }
        )

    out = {"baseline": baseline, "ablations": ablations}
    _print_ablation_results(out, start_date, end_date)
    return out


def run_backward_elimination(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    params: dict | None = None,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    per_signal_cap: int = 2,
    use_fundamentals: bool = False,
    use_earnings_only: bool = False,
    disabled_signals: frozenset[str] | None = None,
) -> dict:
    """Greedy backward elimination: iteratively remove the signal that most
    improves Sharpe, until no remaining signal is a net drag.

    Unlike single-pass ablation, each step re-evaluates all remaining signals
    against the *current* disabled set — capturing slot-competition interactions
    that independent ablation misses.

    Stops when removing any remaining signal produces ΔSharpe ≤ 0.

    Returns::

        {
            "steps": [
                {
                    "step":           int,
                    "signal_removed": str,
                    "sharpe_delta":   float,
                    "sharpe_after":   float,
                    "return_after":   float,
                    "trades_removed": int,
                },
                ...
            ],
            "original_baseline": full _run_simulation result,
            "final_result":      full _run_simulation result after all removals,
            "signals_kept":      list[str],
            "signals_removed":   list[str],
        }
    """
    _assert_pre_holdout(end_date)

    use_any_fundamentals = use_fundamentals or use_earnings_only
    logger.info(
        f"Backward elimination: {start_date} → {end_date} | {len(symbols)} symbols"
        + (" | earnings=ON" if use_earnings_only else "")
        + (" | fundamentals=ON" if use_fundamentals else "")
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    indicators = _build_indicators(raw, symbols)

    spy_indicators = indicators.get("SPY")
    if spy_indicators is None:  # pragma: no branch
        try:
            spy_raw = yf.download(
                "SPY", start=fetch_start, end=end_date, auto_adjust=True, progress=False
            )
            if not spy_raw.empty:  # pragma: no branch
                spy_close = spy_raw["Close"]
                spy_vol = spy_raw["Volume"]
                if isinstance(spy_raw.columns, pd.MultiIndex):  # pragma: no branch
                    spy_close = spy_raw["Close"]["SPY"]
                    spy_vol = spy_raw["Volume"]["SPY"]
                spy_df = pd.DataFrame({"Close": spy_close, "Volume": spy_vol}).dropna()
                spy_indicators = _compute_indicators(spy_df)
        except Exception as exc:
            logger.warning(f"SPY fetch failed: {exc}")

    vix_spike_by_date: dict[str, bool] = {}
    _vix_df_belim: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_belim = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_belim.index = pd.DatetimeIndex(_vix_df_belim.index).tz_localize(None)
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, _vix_df_belim)

    earnings_history: dict[str, list[dict]] | None = None
    insider_history: dict[str, list[dict]] | None = None
    if use_any_fundamentals:
        logger.info("Backward elimination: pre-fetching earnings history…")
        earnings_history = prefetch_earnings_history(symbols)
    if use_fundamentals and not use_earnings_only:
        logger.info("Backward elimination: pre-fetching insider history…")
        insider_history = prefetch_insider_history(symbols)

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    sim_kwargs: dict = {
        "initial_capital": initial_capital,
        "max_positions": max_positions,
        "max_hold_days": max_hold_days,
        "params": params,
        "slippage_bps": slippage_bps,
        "spread_bps": spread_bps,
        "spy_indicators": spy_indicators,
        "per_signal_cap": per_signal_cap,
        "regime_by_date": regime_by_date or None,
        "vix_spike_by_date": vix_spike_by_date or None,
        "earnings_history": earnings_history,
        "insider_history": insider_history,
        "rs_ranks": _compute_rs_ranks(indicators, spy_indicators),
        "disabled_signals": disabled_signals,
    }

    logger.info("Backward elimination: running baseline…")
    original_baseline = _run_simulation(indicators, trading_dates, **sim_kwargs)

    disabled: set[str] = set()
    current_result = original_baseline
    steps: list[dict] = []

    while True:
        remaining = [s for s in _SIGNAL_PRIORITY if s not in disabled]
        if not remaining:
            break

        best_signal: str | None = None
        best_delta = 0.0
        best_result: dict | None = None
        best_trades = 0

        for signal_name in remaining:
            trial = _run_simulation(
                indicators,
                trading_dates,
                **{
                    **sim_kwargs,
                    "disabled_signals": frozenset(disabled | {signal_name})
                    | (disabled_signals or frozenset()),
                },
            )
            delta = trial["sharpe_ratio"] - current_result["sharpe_ratio"]
            sig_data = current_result["by_signal"].get(signal_name, {})
            trades = sig_data.get("wins", 0) + sig_data.get("losses", 0)
            if delta > best_delta or (delta == best_delta and trades > best_trades):
                best_signal = signal_name
                best_delta = delta
                best_result = trial
                best_trades = trades

        if best_delta <= 0 or best_result is None or best_signal is None:
            break

        disabled.add(best_signal)
        steps.append(
            {
                "step": len(steps) + 1,
                "signal_removed": best_signal,
                "sharpe_delta": round(best_delta, 3),
                "sharpe_after": round(best_result["sharpe_ratio"], 3),
                "return_after": round(best_result["total_return_pct"], 2),
                "trades_removed": best_trades,
            }
        )
        logger.info(
            f"  Step {len(steps)}: removed {best_signal} "
            f"(ΔSharpe={best_delta:+.3f}, now {best_result['sharpe_ratio']:.3f})"
        )
        current_result = best_result

    signals_kept = sorted(
        [s for s in _SIGNAL_PRIORITY if s not in disabled],
        key=lambda s: _SIGNAL_PRIORITY[s],
    )
    out = {
        "steps": steps,
        "original_baseline": original_baseline,
        "final_result": current_result,
        "signals_kept": signals_kept,
        "signals_removed": [s["signal_removed"] for s in steps],
    }
    _print_backward_elimination_results(out, start_date, end_date)
    return out


def _print_backward_elimination_results(r: dict, start_date: str, end_date: str) -> None:
    ob = r["original_baseline"]
    fr = r["final_result"]
    print("\n" + "=" * 65)
    print(f"  BACKWARD ELIMINATION  {start_date} → {end_date}")
    print("=" * 65)
    print(
        f"  Start:  Sharpe {ob['sharpe_ratio']:.3f} | "
        f"Return {ob['total_return_pct']:+.1f}% | {ob['total_trades']} trades"
    )
    print(
        f"  Final:  Sharpe {fr['sharpe_ratio']:.3f} | "
        f"Return {fr['total_return_pct']:+.1f}% | {fr['total_trades']} trades"
    )
    print()
    if r["steps"]:
        print(f"  {'Step':<5} {'Signal removed':<25} {'ΔSharpe':>8}  {'Sharpe':>8}  {'Trades':>6}")
        print("  " + "-" * 58)
        for s in r["steps"]:
            print(
                f"  {s['step']:<5} {s['signal_removed']:<25} "
                f"{s['sharpe_delta']:>+8.3f}  {s['sharpe_after']:>8.3f}  {s['trades_removed']:>6}"
            )
    else:
        print("  No signals identified as net drags — baseline is optimal.")
    print()
    print(f"  Signals kept:    {', '.join(r['signals_kept'])}")
    print(f"  Signals removed: {', '.join(r['signals_removed']) or 'none'}")
    print("=" * 65 + "\n")


def _print_ablation_results(r: dict, start_date: str, end_date: str) -> None:
    b = r["baseline"]
    print("\n" + "=" * 65)
    print(f"  ABLATION STUDY  {start_date} → {end_date}")
    print("=" * 65)
    print(
        f"  Baseline: Sharpe {b['sharpe_ratio']:.2f} | "
        f"Return {b['total_return_pct']:+.1f}% | {b['total_trades']} trades"
    )
    print()
    print(f"  {'Signal':<25} {'Trades':>6}  {'ΔSharpe':>8}  {'ΔReturn':>8}  Verdict")
    print("  " + "-" * 58)
    for a in sorted(r["ablations"], key=lambda x: x["sharpe_delta"]):
        verdict = "KEEP" if a["verdict"] == "KEEP" else "REVIEW"
        print(
            f"  {a['signal']:<25} {a['baseline_trades']:>6}  "
            f"{a['sharpe_delta']:>+8.3f}  {a['return_delta']:>+7.1f}%  {verdict}"
        )
    print("=" * 65 + "\n")


_REGIMES_ORDER = [
    "BULL_TREND",
    "NEUTRAL_CHOP",
    "DEFENSIVE_DOWNTREND",
    "HIGH_VOL_DOWNTREND",
    "STRESS_RISK_OFF",
]


def run_signal_analysis(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    params: dict | None = None,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    per_signal_cap: int = 2,
    use_fundamentals: bool = False,
    use_earnings_only: bool = False,
    disabled_signals: frozenset[str] | None = None,
) -> dict:
    """Run a single simulation with enriched trade metadata, then produce:

    1. Regime-stratified signal breakdown — win rate and avg return per signal
       per market regime (BULL_TREND / NEUTRAL_CHOP / DEFENSIVE_DOWNTREND / HIGH_VOL_DOWNTREND / STRESS_RISK_OFF).
    2. Hold-period decay — win rate and avg return per signal broken down by
       days held (1 … max_hold_days).

    end_of_backtest exits are excluded from both tables (truncated holds).

    Returns::

        {
            "baseline":     full _run_simulation result,
            "regime_stats": {signal: {regime: {wins, losses, total_return}}},
            "decay_stats":  {signal: {days_held: {wins, losses, total_return}}},
        }
    """
    _assert_pre_holdout(end_date)

    logger.info(
        f"Signal analysis: {start_date} → {end_date} | {len(symbols)} symbols"
        + (" | earnings=ON" if use_earnings_only else "")
        + (" | fundamentals=ON" if use_fundamentals else "")
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    indicators = _build_indicators(raw, symbols)

    spy_indicators = indicators.get("SPY")
    if spy_indicators is None:  # pragma: no branch
        try:
            spy_raw = yf.download(
                "SPY", start=fetch_start, end=end_date, auto_adjust=True, progress=False
            )
            if not spy_raw.empty:  # pragma: no branch
                spy_close = spy_raw["Close"]
                spy_vol = spy_raw["Volume"]
                if isinstance(spy_raw.columns, pd.MultiIndex):  # pragma: no branch
                    spy_close = spy_raw["Close"]["SPY"]
                    spy_vol = spy_raw["Volume"]["SPY"]
                spy_df = pd.DataFrame({"Close": spy_close, "Volume": spy_vol}).dropna()
                spy_indicators = _compute_indicators(spy_df)
        except Exception as exc:
            logger.warning(f"SPY fetch failed: {exc}")

    vix_spike_by_date: dict[str, bool] = {}
    _vix_df_signal: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_signal = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_signal.index = pd.DatetimeIndex(_vix_df_signal.index).tz_localize(None)
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, _vix_df_signal)

    earnings_history: dict[str, list[dict]] | None = None
    insider_history: dict[str, list[dict]] | None = None
    if use_fundamentals or use_earnings_only:
        logger.info("Signal analysis: pre-fetching earnings history…")
        earnings_history = prefetch_earnings_history(symbols)
    if use_fundamentals and not use_earnings_only:
        logger.info("Signal analysis: pre-fetching insider history…")
        insider_history = prefetch_insider_history(symbols)

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    results = _run_simulation(
        indicators,
        trading_dates,
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_hold_days=max_hold_days,
        params=params,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        spy_indicators=spy_indicators,
        per_signal_cap=per_signal_cap,
        regime_by_date=regime_by_date or None,
        vix_spike_by_date=vix_spike_by_date or None,
        earnings_history=earnings_history,
        insider_history=insider_history,
        rs_ranks=_compute_rs_ranks(indicators, spy_indicators),
        disabled_signals=disabled_signals,
    )

    closed = [
        t
        for t in results["trades"]
        if t["action"] == "SELL"
        and "pnl_pct" in t
        and t.get("reason") != "end_of_backtest"
        and t.get("entry_regime") is not None
        and t.get("days_held") is not None
    ]

    # Sort closed trades by date so block-bootstrap preserves temporal order
    closed_sorted = sorted(closed, key=lambda t: t.get("date", ""))

    # Accumulate per-cell outcomes (1.0=win, 0.0=loss) in date order for bootstrap
    regime_outcomes: dict[str, dict[str, list[float]]] = {}
    regime_stats: dict[str, dict[str, dict]] = {}
    for t in closed_sorted:
        sig = t.get("signal", "unknown")
        reg = t["entry_regime"]
        regime_stats.setdefault(sig, {}).setdefault(
            reg, {"wins": 0, "losses": 0, "total_return": 0.0}
        )
        regime_outcomes.setdefault(sig, {}).setdefault(reg, [])
        regime_stats[sig][reg]["total_return"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            regime_stats[sig][reg]["wins"] += 1
            regime_outcomes[sig][reg].append(1.0)
        else:
            regime_stats[sig][reg]["losses"] += 1
            regime_outcomes[sig][reg].append(0.0)

    # Attach block-bootstrap 95% CIs to each regime cell
    for sig, reg_dict in regime_stats.items():
        for reg, cell in reg_dict.items():
            outcomes = regime_outcomes.get(sig, {}).get(reg, [])
            ci_lo, ci_hi = _bootstrap_cell_ci(outcomes)
            cell["win_rate_ci_low"] = ci_lo
            cell["win_rate_ci_high"] = ci_hi

    decay_stats: dict[str, dict[int, dict]] = {}
    for t in closed_sorted:
        sig = t.get("signal", "unknown")
        dh = int(t["days_held"])
        decay_stats.setdefault(sig, {}).setdefault(
            dh, {"wins": 0, "losses": 0, "total_return": 0.0}
        )
        decay_stats[sig][dh]["total_return"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            decay_stats[sig][dh]["wins"] += 1
        else:
            decay_stats[sig][dh]["losses"] += 1

    regime_blocked = compute_regime_blocked(regime_stats)

    out = {
        "baseline": results,
        "regime_stats": regime_stats,
        "decay_stats": decay_stats,
        "regime_blocked": {reg: sorted(sigs) for reg, sigs in regime_blocked.items()},
    }
    _print_regime_table(out, start_date, end_date)
    _print_hold_period_table(out, start_date, end_date)
    _print_cost_sensitivity(results, start_date, end_date)
    _print_regime_blocked(regime_blocked, start_date, end_date)
    return out


def run_short_signal_analysis(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = _SHORT_MAX_HOLD_DAYS,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    use_earnings_history: bool = True,
) -> dict:
    """Backtest the three bearish signals (earnings_miss, loser_momentum, ema_breakdown)
    on historical data.

    Signal coverage:
    - earnings_miss   — EPS miss ≤ −5% within 30 days (backtestable via yfinance earnings_dates)
    - loser_momentum  — 20d relative strength vs SPY ≤ −5% (backtestable via OHLCV)
    - ema_breakdown   — price ≤ −2% below EMA21 AND EMA9 slope down (backtestable via OHLCV)

    NOT backtestable (live-only signals, yfinance lacks time-series data):
    - high_short_interest  — requires point-in-time short interest history

    Returns::

        {
            "total_trades":           int,
            "win_rate_pct":           float,
            "avg_return_per_trade_pct": float,
            "total_return_pct":       float,
            "sharpe_ratio":           float,
            "max_drawdown_pct":       float,
            "by_signal":              {signal: {wins, losses, total_return}},
            "equity_curve":           [(date_str, value), ...],
            "trades":                 [...],
            "validation_scope":       "rule_proxy_only",
            "signals_tested":         [...],
            "start":                  str,
            "end":                    str,
        }
    """
    _assert_pre_holdout(end_date)

    logger.info(
        f"Short signal analysis: {start_date} → {end_date} | {len(symbols)} symbols"
        + (" | earnings=ON" if use_earnings_history else "")
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched for short signal analysis")
        return {}

    indicators = _build_indicators(raw, symbols)

    # SPY for rel_strength_20d (loser_momentum signal)
    spy_indicators = indicators.get("SPY")
    if spy_indicators is None:
        try:
            spy_raw = yf.download(
                "SPY", start=fetch_start, end=end_date, auto_adjust=True, progress=False
            )
            if not spy_raw.empty:
                spy_close = spy_raw["Close"]
                spy_vol = spy_raw["Volume"]
                if isinstance(spy_raw.columns, pd.MultiIndex):
                    spy_close = spy_raw["Close"]["SPY"]
                    spy_vol = spy_raw["Volume"]["SPY"]
                spy_df = pd.DataFrame({"Close": spy_close, "Volume": spy_vol}).dropna()
                spy_indicators = _compute_indicators(spy_df)
        except Exception as exc:
            logger.warning(f"SPY fetch failed — loser_momentum signal disabled: {exc}")

    # Regime map (not used for entry gating but stored on trades for analysis)
    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        _spy_df_regime = pd.DataFrame({"Close": spy_indicators["Close"].astype(float)}).dropna()
        _spy_df_regime.index = pd.DatetimeIndex(_spy_df_regime.index).tz_localize(None)
        _trading_dates_str = [ts.strftime("%Y-%m-%d") for ts in spy_indicators.index]
        regime_by_date = compute_regime_series(_spy_df_regime, None, _trading_dates_str)

    # Historical earnings for earnings_miss signal
    earnings_history: dict[str, list[dict]] | None = None
    if use_earnings_history:
        logger.info("Short signal analysis: pre-fetching earnings history…")
        earnings_history = prefetch_earnings_history(symbols)

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    rs_ranks = _compute_rs_ranks(indicators, spy_indicators)
    rs_rank_lag10 = _compute_rs_rank_lag10(rs_ranks, trading_dates)

    results = _run_short_simulation(
        indicators,
        trading_dates,
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_hold_days=max_hold_days,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        spy_indicators=spy_indicators,
        regime_by_date=regime_by_date or None,
        earnings_history=earnings_history,
        rs_ranks=rs_ranks,
        rs_rank_lag10=rs_rank_lag10,
    )
    results["start"] = start_date
    results["end"] = end_date

    _print_short_signal_results(results, start_date, end_date)
    return results


def _print_short_signal_results(r: dict, start_date: str, end_date: str) -> None:
    print("\n" + "=" * 60)
    print("  SHORT SIGNAL ANALYSIS")
    print(f"  {start_date} → {end_date}")
    print("  NOTE: Rule proxy only — no Claude judgment, news, or macro context.")
    print("=" * 60)
    print(f"  Total return:      {r['total_return_pct']:+.2f}%")
    print(f"  Total trades:      {r['total_trades']}")
    print(f"  Win rate:          {r['win_rate_pct']:.1f}%")
    print(f"  Avg trade return:  {r['avg_return_per_trade_pct']:+.2f}%")
    print(f"  Max drawdown:      {r['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe ratio:      {r['sharpe_ratio']:.2f}")
    print()
    print("  By signal:")
    for sig, data in r["by_signal"].items():
        total = data["wins"] + data["losses"]
        wr = data["wins"] / total * 100 if total else 0
        avg = data["total_return"] / total if total else 0
        print(f"    {sig:<25} {total:>3} trades  WR {wr:.0f}%  avg {avg:+.2f}%")
    print("=" * 60 + "\n")


def run_combined_analysis(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    max_long_positions: int = 5,
    max_short_positions: int = 2,
    max_hold_days: int = 3,
    max_short_hold_days: int = _SHORT_MAX_HOLD_DAYS,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    per_signal_cap: int = 2,
    use_earnings_history: bool = True,
) -> dict:
    """Combined long/short backtest with regime-gated short entries and shared capital pool.

    Shorts only enter in DEFENSIVE_DOWNTREND, HIGH_VOL_DOWNTREND, or STRESS_RISK_OFF
    regimes. Longs follow the existing REGIME_BLOCKED rules.

    NOTE: survivorship bias is present — universe is today's tradable listings.
    """
    _assert_pre_holdout(end_date)

    logger.info(
        f"Combined analysis: {start_date} → {end_date} | {len(symbols)} symbols"
        f" | longs={max_long_positions} shorts={max_short_positions}"
        + (" | earnings=ON" if use_earnings_history else "")
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched for combined analysis")
        return {}

    indicators = _build_indicators(raw, symbols)

    spy_indicators = indicators.get("SPY")
    if spy_indicators is None:  # pragma: no branch
        try:
            spy_raw = yf.download(
                "SPY", start=fetch_start, end=end_date, auto_adjust=True, progress=False
            )
            if not spy_raw.empty:  # pragma: no branch
                spy_close = spy_raw["Close"]
                spy_vol = spy_raw["Volume"]
                if isinstance(spy_raw.columns, pd.MultiIndex):  # pragma: no branch
                    spy_close = spy_raw["Close"]["SPY"]
                    spy_vol = spy_raw["Volume"]["SPY"]
                spy_df = pd.DataFrame({"Close": spy_close, "Volume": spy_vol}).dropna()
                spy_indicators = _compute_indicators(spy_df)
        except Exception as exc:
            logger.warning(f"SPY fetch failed: {exc}")

    vix_spike_by_date: dict[str, bool] = {}
    _vix_df_comb: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_comb = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_comb.index = pd.DatetimeIndex(_vix_df_comb.index).tz_localize(None)
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
            logger.info(f"VIX fetched — {sum(vix_spike_by_date.values())} fear-spike days")
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, _vix_df_comb)

    earnings_history: dict[str, list[dict]] | None = None
    if use_earnings_history:
        logger.info("Combined analysis: pre-fetching earnings history…")
        earnings_history = prefetch_earnings_history(symbols)

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    rs_ranks = _compute_rs_ranks(indicators, spy_indicators)
    rs_rank_lag10 = _compute_rs_rank_lag10(rs_ranks, trading_dates)

    results = _run_combined_simulation(
        indicators,
        trading_dates,
        initial_capital=initial_capital,
        max_long_positions=max_long_positions,
        max_short_positions=max_short_positions,
        max_hold_days=max_hold_days,
        max_short_hold_days=max_short_hold_days,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        spy_indicators=spy_indicators,
        per_signal_cap=per_signal_cap,
        regime_by_date=regime_by_date or None,
        vix_spike_by_date=vix_spike_by_date or None,
        earnings_history=earnings_history,
        rs_ranks=rs_ranks,
        rs_rank_lag10=rs_rank_lag10,
    )
    results["start"] = start_date
    results["end"] = end_date

    _print_combined_results(results, start_date, end_date)
    return results


def _print_combined_results(r: dict, start_date: str, end_date: str) -> None:
    long_s = r["by_side"]["long"]
    short_s = r["by_side"]["short"]
    print("\n" + "=" * 65)
    print("  COMBINED LONG/SHORT ANALYSIS")
    print(f"  {start_date} → {end_date}")
    print("  NOTE: Rule proxy only — no Claude judgment, news, or macro context.")
    print("  Shorts: earnings_miss + ema_breakdown in DEFENSIVE/HIGH_VOL/STRESS only.")
    print("  BIAS: Universe from current tradable listings — survivorship risk.")
    print("=" * 65)
    print(f"  Total return:      {r['total_return_pct']:+.2f}%")
    print(
        f"  Total trades:      {r['total_trades']}"
        f"  (long={r['long_trades']}  short={r['short_trades']})"
    )
    print(
        f"  Win rate:          {r['win_rate_pct']:.1f}%"
        f"  (long={r['long_win_rate_pct']:.0f}%  short={r['short_win_rate_pct']:.0f}%)"
    )
    print(f"  Avg trade return:  {r['avg_return_per_trade_pct']:+.2f}%")
    print(f"  Max drawdown:      {r['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe ratio:      {r['sharpe_ratio']:.2f}")
    print()
    print("  By side:")
    print(
        f"    Long:   {long_s['trades']:>4} trades"
        f"  WR {long_s['win_rate_pct']:.0f}%"
        f"  avg {long_s['avg_return_pct']:+.2f}%"
    )
    print(
        f"    Short:  {short_s['trades']:>4} trades"
        f"  WR {short_s['win_rate_pct']:.0f}%"
        f"  avg {short_s['avg_return_pct']:+.2f}%"
    )
    print()
    print("  Long signals:")
    for sig in sorted(SIGNAL_PRIORITY, key=lambda s: SIGNAL_PRIORITY[s]):
        if sig not in r["by_signal"]:
            continue
        data = r["by_signal"][sig]
        total = data["wins"] + data["losses"]
        wr = data["wins"] / total * 100 if total else 0
        avg = data["total_return"] / total if total else 0
        print(f"    {sig:<25} {total:>3} trades  WR {wr:.0f}%  avg {avg:+.2f}%")
    print("  Short signals:")
    for sig in sorted(SHORT_SIGNAL_PRIORITY, key=lambda s: SHORT_SIGNAL_PRIORITY[s]):
        if sig not in r["by_signal"]:
            continue
        data = r["by_signal"][sig]
        total = data["wins"] + data["losses"]
        wr = data["wins"] / total * 100 if total else 0
        avg = data["total_return"] / total if total else 0
        print(f"    {sig:<25} {total:>3} trades  WR {wr:.0f}%  avg {avg:+.2f}%")
    dist = r.get("regime_distribution", {})
    total_days = sum(dist.values())
    if total_days and dist:
        print()
        print("  Regime distribution:")
        for reg in _REGIMES_ORDER:
            if reg not in dist:
                continue
            pct = dist[reg] / total_days * 100
            short_flag = "  [shorts active]" if reg in SHORT_ALLOWED_REGIMES else ""
            print(f"    {reg:<25} {pct:>4.0f}%{short_flag}")
    print("=" * 65 + "\n")


_INTRADAY_SIGNALS = {"vwap_reclaim", "orb_breakout", "intraday_momentum"}
_LOW_CONFIDENCE_N = 30


def _print_regime_table(r: dict, start_date: str, end_date: str) -> None:
    stats = r["regime_stats"]
    low_n_found = False
    intraday_found = False
    print("\n" + "=" * 68)
    print(f"  REGIME-STRATIFIED SIGNAL BREAKDOWN  {start_date} → {end_date}")
    print("=" * 68)
    print("  NOTE: Rule proxy only — survivorship risk, no delisted symbols.")
    print("  COUNTS: Signal-analysis trade counts include all signal occurrences;")
    print("  portfolio trade counts (run_backtest) apply additional RS-rank, per-signal-cap,")
    print("  and cash constraints — expect lower portfolio counts for the same run.")
    for sig in sorted(_SIGNAL_PRIORITY, key=lambda s: _SIGNAL_PRIORITY[s]):
        if sig not in stats:
            continue
        total = sum(v["wins"] + v["losses"] for v in stats[sig].values())
        intraday_marker = " [intraday†]" if sig in _INTRADAY_SIGNALS else ""
        if sig in _INTRADAY_SIGNALS:
            intraday_found = True
        print(f"\n  {sig}  ({total} trades){intraday_marker}")
        for reg in _REGIMES_ORDER:
            if reg not in stats[sig]:
                continue
            d = stats[sig][reg]
            n = d["wins"] + d["losses"]
            wr = d["wins"] / n * 100 if n else 0
            avg = d["total_return"] / n if n else 0
            drag_flag = "  ← drag" if avg < -0.1 else ""
            low_n_flag = " *" if n < _LOW_CONFIDENCE_N else ""
            if n < _LOW_CONFIDENCE_N:
                low_n_found = True
            ci_lo = d.get("win_rate_ci_low")
            ci_hi = d.get("win_rate_ci_high")
            if ci_lo is not None and not math.isnan(ci_lo):
                assert ci_hi is not None
                ci_str = f"  CI [{ci_lo * 100:.0f}–{ci_hi * 100:.0f}%]"
            else:
                ci_str = ""
            print(
                f"    {reg:<15}  WR {wr:>3.0f}%{ci_str}  avg {avg:>+5.1f}%  {n:>3} trades{drag_flag}{low_n_flag}"
            )
    print()
    if low_n_found:
        print(
            f"  * n < {_LOW_CONFIDENCE_N}: low confidence — suggestive only, not statistically reliable."
        )
    if intraday_found:
        print("  † intraday signals: backtest summarises full day; does NOT validate")
        print("    live same-session execution (VWAP/ORB/intraday_momentum).")
    print("=" * 68 + "\n")


def _print_hold_period_table(r: dict, start_date: str, end_date: str) -> None:
    stats = r["decay_stats"]
    max_hold = max((dh for sig_d in stats.values() for dh in sig_d), default=3)
    days = list(range(1, max_hold + 1))
    print("\n" + "=" * 68)
    print(f"  HOLD-PERIOD DECAY  {start_date} → {end_date}")
    print("=" * 68)
    for sig in sorted(_SIGNAL_PRIORITY, key=lambda s: _SIGNAL_PRIORITY[s]):
        if sig not in stats:
            continue
        total = sum(v["wins"] + v["losses"] for v in stats[sig].values())
        print(f"\n  {sig}  ({total} trades)")
        for d in days:
            if d not in stats[sig]:
                continue
            dh = stats[sig][d]
            n = dh["wins"] + dh["losses"]
            wr = dh["wins"] / n * 100 if n else 0
            avg = dh["total_return"] / n if n else 0
            flag = "  ← decays" if avg < -0.1 else ""
            print(f"    Day {d}:  WR {wr:>3.0f}%  avg {avg:>+5.1f}%  {n:>3} trades{flag}")
    print("\n" + "=" * 68 + "\n")


def _print_cost_sensitivity(results: dict, start_date: str, end_date: str) -> None:
    """Show net-return sensitivity at 1×/2×/3× the modelled round-trip cost.

    Costs are already baked into pnl_pct at trade time.  To model higher costs,
    we subtract additional round-trip increments of (SLIPPAGE_BPS + SPREAD_BPS) * 2.
    Signals with avg return < 2× base costs are flagged as cost-sensitive.
    """
    by_signal = results.get("by_signal", {})
    if not by_signal:
        return
    base_one_way_bps = SLIPPAGE_BPS + SPREAD_BPS
    rt_increment_pct = base_one_way_bps * 2 / 10_000
    print("\n" + "=" * 68)
    print(f"  COST SENSITIVITY  {start_date} → {end_date}")
    print("=" * 68)
    print(
        f"  Costs already applied at 1× ({SLIPPAGE_BPS} slippage + {SPREAD_BPS} spread bps one-way = "
        f"{base_one_way_bps * 2} bps round-trip)."
    )
    print("  Columns show net avg return if costs were 1×/2×/3× modelled level.")
    print(f"  ⚠ = edge < 2× base round-trip cost ({base_one_way_bps * 2} bps) — fragile.\n")
    print(f"  {'Signal':<25}  {'Trades':>6}  {'1× (now)':>9}  {'2× costs':>9}  {'3× costs':>9}")
    print("  " + "-" * 62)
    for sig in sorted(_SIGNAL_PRIORITY, key=lambda s: _SIGNAL_PRIORITY[s]):
        if sig not in by_signal:
            continue
        d = by_signal[sig]
        n = d["wins"] + d["losses"]
        if n == 0:
            continue
        avg = d["total_return"] / n
        at_2x = avg - rt_increment_pct
        at_3x = avg - 2 * rt_increment_pct
        fragile = "  ⚠" if avg < 2 * rt_increment_pct else ""
        print(f"  {sig:<25}  {n:>6}  {avg:>+8.2f}%  {at_2x:>+8.2f}%  {at_3x:>+8.2f}%{fragile}")
    print("=" * 68 + "\n")


def _print_regime_blocked(
    regime_blocked: dict[str, set[str]], start_date: str, end_date: str
) -> None:
    """Print data-driven signal-blocking recommendations after Holm-Bonferroni correction."""
    print("\n" + "=" * 68)
    print(f"  DATA-DRIVEN REGIME BLOCKS  {start_date} → {end_date}")
    print("=" * 68)
    print("  Signals whose win rate is indistinguishable from chance (p > 0.05,")
    print("  Holm-Bonferroni corrected, n >= 20 trades).  Treat as hypotheses.")
    if not regime_blocked:
        print("\n  No cells failed the win-rate test — no automatic blocks recommended.")
    else:
        for reg in sorted(regime_blocked):
            sigs = sorted(regime_blocked[reg])
            print(f"\n  {reg}:")
            for sig in sigs:
                print(f"    - {sig}")
    print("\n" + "=" * 68 + "\n")


_HOLDOUT_LOG = os.path.join(LOG_DIR, "holdout_log.jsonl")


def _assert_pre_holdout(end_date_str: str) -> None:
    """Warn loudly if end_date encroaches on the holdout period.

    Any tuning run that reaches into HOLDOUT_START_DATE contaminates the one
    honest out-of-sample evaluation we have. This does not raise — it warns —
    so that exploratory reruns don't hard-fail, but the message is impossible
    to miss.
    """
    try:
        end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        return
    if end >= HOLDOUT_START_DATE:
        logger.warning(
            "=" * 70 + f"\n  HOLDOUT CONTAMINATION WARNING\n"
            f"  end_date={end_date_str} reaches into or past holdout start "
            f"({HOLDOUT_START_DATE}). Any parameters tuned on this run have\n"
            f"  seen holdout data and must not be reported as independent OOS.\n" + "=" * 70
        )


def _print_param_sensitivity(r: dict, start_date: str, end_date: str) -> None:
    b = r["baseline"]
    print("\n" + "=" * 72)
    print(f"  PARAMETER SENSITIVITY  {start_date} → {end_date}")
    print("=" * 72)
    print(
        f"  Baseline: Sharpe {b['sharpe_ratio']:.3f} | "
        f"Return {b['total_return_pct']:+.1f}% | {b['total_trades']} trades"
    )
    for param, sweep in sorted(r["by_param"].items()):
        print(f"\n  {param}")
        print(f"  {'Value':>10}  {'Return%':>9}  {'Sharpe':>8}  {'ΔSharpe':>8}  {'Trades':>7}")
        print("  " + "-" * 52)
        for val, res in sorted(sweep.items()):
            delta = res["sharpe_ratio"] - b["sharpe_ratio"]
            marker = " ←" if val == DEFAULT_SIGNAL_PARAMS.get(param) else ""
            print(
                f"  {val:>10.4g}  {res['total_return_pct']:>9.1f}  "
                f"{res['sharpe_ratio']:>8.3f}  {delta:>+8.3f}  "
                f"{res['total_trades']:>7}{marker}"
            )
    print("=" * 72 + "\n")


def run_param_sensitivity(
    symbols: list[str],
    start_date: str,
    end_date: str,
    param_ranges: dict[str, list[float]],
    base_params: dict | None = None,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    slippage_bps: int | None = None,
    spread_bps: int | None = None,
    per_signal_cap: int = 2,
    use_fundamentals: bool = False,
    use_earnings_only: bool = False,
    disabled_signals: frozenset[str] | None = None,
) -> dict:
    """Sweep each parameter in param_ranges over its values, holding all other
    params at base_params (defaults to DEFAULT_SIGNAL_PARAMS).

    Each parameter is varied independently (one-at-a-time sensitivity), not as
    a full grid search.  Use run_walk_forward_optimized for full grid search.

    Parameters
    ----------
    param_ranges : dict[str, list[float]]
        Maps each parameter name to the list of values to test.
        Example: {"rsi_threshold": [28, 32, 35, 38, 42]}
    base_params : dict | None
        Starting parameter set; merged over DEFAULT_SIGNAL_PARAMS.

    Returns
    -------
    dict with keys:
        "baseline"  — simulation result at base_params
        "by_param"  — {param: {value: simulation_result}}
    """
    _assert_pre_holdout(end_date)

    logger.info(
        f"Param sensitivity: {start_date} → {end_date} | {len(symbols)} symbols | "
        f"{sum(len(v) for v in param_ranges.values())} total sweeps"
        + (" | earnings=ON" if use_earnings_only else "")
        + (" | fundamentals=ON" if use_fundamentals else "")
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    indicators = _build_indicators(raw, symbols)

    spy_indicators = indicators.get("SPY")
    if spy_indicators is None:  # pragma: no branch
        try:
            spy_raw = yf.download(
                "SPY", start=fetch_start, end=end_date, auto_adjust=True, progress=False
            )
            if not spy_raw.empty:  # pragma: no branch
                spy_close = spy_raw["Close"]
                spy_vol = spy_raw["Volume"]
                if isinstance(spy_raw.columns, pd.MultiIndex):  # pragma: no branch
                    spy_close = spy_raw["Close"]["SPY"]
                    spy_vol = spy_raw["Volume"]["SPY"]
                spy_df = pd.DataFrame({"Close": spy_close, "Volume": spy_vol}).dropna()
                spy_indicators = _compute_indicators(spy_df)
        except Exception as exc:
            logger.warning(f"SPY fetch failed: {exc}")

    vix_spike_by_date: dict[str, bool] = {}
    _vix_df_ps: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_ps = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_ps.index = pd.DatetimeIndex(_vix_df_ps.index).tz_localize(None)
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, _vix_df_ps)

    earnings_history: dict[str, list[dict]] | None = None
    insider_history: dict[str, list[dict]] | None = None
    if use_fundamentals or use_earnings_only:
        earnings_history = prefetch_earnings_history(symbols)
    if use_fundamentals and not use_earnings_only:
        insider_history = prefetch_insider_history(symbols)

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    effective_base = {**DEFAULT_SIGNAL_PARAMS, **(base_params or {})}

    sim_kwargs: dict = {
        "initial_capital": initial_capital,
        "max_positions": max_positions,
        "max_hold_days": max_hold_days,
        "params": effective_base,
        "slippage_bps": slippage_bps,
        "spread_bps": spread_bps,
        "spy_indicators": spy_indicators,
        "per_signal_cap": per_signal_cap,
        "regime_by_date": regime_by_date or None,
        "vix_spike_by_date": vix_spike_by_date or None,
        "earnings_history": earnings_history,
        "insider_history": insider_history,
        "rs_ranks": _compute_rs_ranks(indicators, spy_indicators),
        "disabled_signals": disabled_signals,
    }

    logger.info("Param sensitivity: running baseline…")
    baseline = _run_simulation(indicators, trading_dates, **sim_kwargs)

    by_param: dict[str, dict[float, dict]] = {}
    for param, values in param_ranges.items():
        logger.info(f"Param sensitivity: sweeping {param} over {values}…")
        by_param[param] = {}
        for val in values:
            trial_params = {**effective_base, param: val}
            result = _run_simulation(
                indicators,
                trading_dates,
                **{**sim_kwargs, "params": trial_params},
            )
            by_param[param][val] = result

    out = {"baseline": baseline, "by_param": by_param}
    _print_param_sensitivity(out, start_date, end_date)
    return out


def _print_short_param_sensitivity(r: dict, start_date: str, end_date: str) -> None:
    b = r["baseline"]
    print("\n" + "=" * 72)
    print(f"  SHORT PARAMETER SENSITIVITY  {start_date} → {end_date}")
    print("=" * 72)
    print(
        f"  Baseline: Sharpe {b['sharpe_ratio']:.3f} | "
        f"Return {b['total_return_pct']:+.1f}% | "
        f"WR {b['win_rate_pct']:.0f}% | {b['total_trades']} trades"
    )
    for param, sweep in sorted(r["by_param"].items()):
        default_val = DEFAULT_SHORT_SIGNAL_PARAMS.get(param)
        print(f"\n  {param}  (default: {default_val})")
        print(
            f"  {'Value':>10}  {'WR%':>6}  {'Avg/trade':>10}  "
            f"{'Sharpe':>8}  {'ΔSharpe':>8}  {'Trades':>7}"
        )
        print("  " + "-" * 60)
        for val, res in sorted(sweep.items()):
            delta = res["sharpe_ratio"] - b["sharpe_ratio"]
            marker = " ←" if val == default_val else ""
            print(
                f"  {val:>10.4g}  {res['win_rate_pct']:>5.0f}%  "
                f"{res['avg_return_per_trade_pct']:>+10.2f}%  "
                f"{res['sharpe_ratio']:>8.3f}  {delta:>+8.3f}  "
                f"{res['total_trades']:>7}{marker}"
            )
    print("=" * 72 + "\n")


def run_short_param_sensitivity(
    symbols: list[str],
    start_date: str,
    end_date: str,
    param_ranges: dict[str, list[float]] | None = None,
    initial_capital: float = 25_000.0,
    max_positions: int = 2,
    max_hold_days: int = _SHORT_MAX_HOLD_DAYS,
    use_earnings_only: bool = False,
) -> dict:
    """One-at-a-time sensitivity sweep for short signal parameters.

    Varies each entry in DEFAULT_SHORT_SIGNAL_PARAMS independently over the
    supplied ranges while holding all others at their defaults.  Each trial
    runs _run_short_simulation so results are directly comparable to the
    --short-signals baseline.

    Parameters
    ----------
    param_ranges : dict | None
        Maps parameter name → list of values to test.  Defaults to a
        pre-defined sweep of all three DEFAULT_SHORT_SIGNAL_PARAMS.
    """
    if param_ranges is None:
        param_ranges = {
            "fb_vol_min": [0.8, 1.0, 1.2, 1.5, 2.0],
            "fb_rsi_min": [40.0, 45.0, 50.0, 55.0, 60.0],
            "hvr_vol_min": [1.5, 2.0, 2.5, 3.0],
            "hvr_range_max": [0.2, 0.3, 0.4],
            "hvr_rsi_min": [50.0, 55.0, 60.0, 65.0],
            "hvr_ret5d_min": [1.0, 2.0, 3.0, 4.0],
        }

    _assert_pre_holdout(end_date)

    n_sweeps = sum(len(v) for v in param_ranges.values())
    logger.info(
        f"Short param sensitivity: {start_date} → {end_date} | {len(symbols)} symbols | "
        f"{n_sweeps} total sweeps" + (" | earnings=ON" if use_earnings_only else "")
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    indicators = _build_indicators(raw, symbols)

    spy_indicators = indicators.get("SPY")

    _vix_df_sps: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_sps = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_sps.index = pd.DatetimeIndex(_vix_df_sps.index).tz_localize(None)
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, _vix_df_sps)

    earnings_history: dict[str, list[dict]] | None = None
    if use_earnings_only:
        earnings_history = prefetch_earnings_history(symbols)

    rs_ranks = _compute_rs_ranks(indicators, spy_indicators)
    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    rs_rank_lag10 = _compute_rs_rank_lag10(rs_ranks, trading_dates)

    sim_kwargs: dict = {
        "initial_capital": initial_capital,
        "max_positions": max_positions,
        "max_hold_days": max_hold_days,
        "spy_indicators": spy_indicators,
        "regime_by_date": regime_by_date or None,
        "earnings_history": earnings_history,
        "rs_ranks": rs_ranks,
        "rs_rank_lag10": rs_rank_lag10,
    }

    logger.info("Short param sensitivity: running baseline…")
    baseline = _run_short_simulation(indicators, trading_dates, **sim_kwargs)

    by_param: dict[str, dict[float, dict]] = {}
    for param, values in param_ranges.items():
        logger.info(f"Short param sensitivity: sweeping {param} over {values}…")
        by_param[param] = {}
        for val in values:
            trial_params = {**DEFAULT_SHORT_SIGNAL_PARAMS, param: val}
            result = _run_short_simulation(
                indicators, trading_dates, **sim_kwargs, short_params=trial_params
            )
            by_param[param][val] = result

    out = {"baseline": baseline, "by_param": by_param}
    _print_short_param_sensitivity(out, start_date, end_date)
    return out


def run_short_walk_forward(
    symbols: list[str],
    start_date: str,
    end_date: str,
    short_params: dict | None = None,
    fold_days: int = 252,
    initial_capital: float = 25_000.0,
    max_positions: int = 2,
    max_hold_days: int = _SHORT_MAX_HOLD_DAYS,
) -> dict:
    """Walk-forward stability check for a fixed set of short signal parameters.

    No train phase — params are held constant across all folds.  Use this to
    check whether a parameter set identified via run_short_param_sensitivity
    holds up fold-by-fold, or whether the edge was concentrated in one period.

    Parameters
    ----------
    short_params : dict | None
        Fixed short signal params for every fold.  Defaults to
        DEFAULT_SHORT_SIGNAL_PARAMS.
    fold_days : int
        Trading days per fold.  Default 252 (~1 year) to get enough
        STRESS_RISK_OFF sessions per fold for meaningful stats.
    """
    _params = short_params or DEFAULT_SHORT_SIGNAL_PARAMS

    _assert_pre_holdout(end_date)

    logger.info(
        f"Short walk-forward: {start_date} → {end_date} | fold={fold_days}d | params={_params}"
    )

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    indicators = _build_indicators(raw, symbols)
    spy_indicators = indicators.get("SPY")

    _vix_df_swf: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_swf = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_swf.index = pd.DatetimeIndex(_vix_df_swf.index).tz_localize(None)
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, _vix_df_swf)

    rs_ranks = _compute_rs_ranks(indicators, spy_indicators)
    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    rs_rank_lag10 = _compute_rs_rank_lag10(rs_ranks, trading_dates)

    if len(trading_dates) < fold_days:
        logger.error(f"Not enough trading days ({len(trading_dates)}) for one {fold_days}-day fold")
        return {}

    earnings_history = prefetch_earnings_history(symbols)

    sim_kwargs: dict = {
        "initial_capital": initial_capital,
        "max_positions": max_positions,
        "max_hold_days": max_hold_days,
        "spy_indicators": spy_indicators,
        "regime_by_date": regime_by_date or None,
        "rs_ranks": rs_ranks,
        "short_params": _params,
        "rs_rank_lag10": rs_rank_lag10,
        "earnings_history": earnings_history,
    }

    fold_results = []
    i = 0
    while i + fold_days <= len(trading_dates):
        fold_dates = trading_dates[i : i + fold_days]
        fold_start = fold_dates[0].strftime("%Y-%m-%d")
        fold_end = fold_dates[-1].strftime("%Y-%m-%d")

        result = _run_short_simulation(indicators, fold_dates, **sim_kwargs)
        fold_results.append(
            {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "sharpe": result["sharpe_ratio"],
                "total_return_pct": result["total_return_pct"],
                "win_rate_pct": result["win_rate_pct"],
                "avg_trade_pct": result.get("avg_return_per_trade_pct", 0.0),
                "total_trades": result["total_trades"],
            }
        )
        logger.info(
            f"  Fold {fold_start}–{fold_end}: Sharpe={result['sharpe_ratio']:+.3f} "
            f"Return={result['total_return_pct']:+.1f}% WR={result['win_rate_pct']:.0f}% "
            f"trades={result['total_trades']}"
        )
        i += fold_days

    if not fold_results:  # pragma: no cover
        return {"folds": [], "summary": {}}

    n = len(fold_results)
    profitable = sum(1 for f in fold_results if f["total_return_pct"] > 0)
    mean_sharpe = round(sum(f["sharpe"] for f in fold_results) / n, 3)
    mean_return = round(sum(f["total_return_pct"] for f in fold_results) / n, 2)
    mean_wr = round(sum(f["win_rate_pct"] for f in fold_results) / n, 1)
    total_trades = sum(f["total_trades"] for f in fold_results)

    print(
        f"\n{'=' * 72}\n"
        f"  SHORT WALK-FORWARD  {start_date} → {end_date}  "
        f"fold={fold_days}d  params={_params}\n"
        f"{'=' * 72}"
    )
    header = (
        f"  {'Fold':>22}   {'WR%':>5}  {'Avg/tr':>8}  {'Sharpe':>7}  {'Ret%':>7}  {'Trades':>6}"
    )
    print(header)
    print("  " + "-" * 68)
    for f in fold_results:
        mark = " ✓" if f["total_return_pct"] > 0 else "  "
        print(
            f"  {f['fold_start']} – {f['fold_end']}{mark}  "
            f"{f['win_rate_pct']:>4.0f}%  "
            f"{f['avg_trade_pct']:>+7.2f}%  "
            f"{f['sharpe']:>+7.3f}  "
            f"{f['total_return_pct']:>+6.1f}%  "
            f"{f['total_trades']:>6}"
        )
    print("  " + "-" * 68)
    print(
        f"  {'MEAN':>22}    "
        f"{mean_wr:>4.0f}%           "
        f"{mean_sharpe:>+7.3f}  "
        f"{mean_return:>+6.1f}%  "
        f"{total_trades:>6}"
    )
    print(
        f"\n  Profitable folds: {profitable}/{n}  |  Mean Sharpe: {mean_sharpe:+.3f}  "
        f"|  Total trades across all folds: {total_trades}"
    )
    print(f"{'=' * 72}\n")

    summary = {
        "n_folds": n,
        "profitable_folds": profitable,
        "mean_sharpe": mean_sharpe,
        "mean_return_pct": mean_return,
        "mean_win_rate_pct": mean_wr,
        "total_trades": total_trades,
        "params": _params,
    }
    return {"folds": fold_results, "summary": summary}


def run_holdout_evaluation(
    frozen_params: dict,
    version: str,
    symbols: list[str] | None = None,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    end_date: str | None = None,
) -> dict:
    """Single evaluation on the untouched holdout period with fully frozen parameters.

    This function must be called at most once per strategy version. Every call is
    recorded to holdout_log.jsonl so repeat invocations are visible. Reading that
    log and finding more than one entry per version means the holdout is spent.

    Parameters
    ----------
    frozen_params : dict
        Parameter set selected during walk-forward (e.g. rsi_threshold, bb_threshold).
        Must not be tuned on or after HOLDOUT_START_DATE.
    version : str
        Monotonically increasing strategy version string (e.g. "v1.29"). Written
        to the log so provenance is unambiguous.
    """
    _syms = symbols or STOCK_UNIVERSE
    holdout_start_str = HOLDOUT_START_DATE.strftime("%Y-%m-%d")
    holdout_end_str = end_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(_ET).isoformat(),
        "version": version,
        "holdout_start": holdout_start_str,
        "holdout_end": holdout_end_str,
        "frozen_params": frozen_params,
    }
    with open(_HOLDOUT_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    with open(_HOLDOUT_LOG) as _f:
        prior_calls = sum(1 for line in _f if f'"version": "{version}"' in line)
    if prior_calls > 1:
        logger.warning(
            f"HOLDOUT INTEGRITY: version={version} has been evaluated {prior_calls} times. "
            "Multiple evaluations on the same version invalidate the holdout result."
        )

    logger.info(
        f"Holdout evaluation: version={version} | {holdout_start_str} → {holdout_end_str} "
        f"| params={frozen_params}"
    )

    fetch_start = (datetime.strptime(holdout_start_str, "%Y-%m-%d") - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(
        _syms, start=fetch_start, end=holdout_end_str, auto_adjust=True, progress=False
    )
    if raw.empty:
        logger.error("No data fetched for holdout evaluation")
        return {}

    indicators = _build_indicators(raw, _syms)
    trading_dates = pd.bdate_range(start=holdout_start_str, end=holdout_end_str)

    spy_indicators = indicators.get("SPY")
    vix_spike_by_date: dict[str, bool] = {}
    _vix_df_holdout: pd.DataFrame | None = None
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=holdout_end_str, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:  # pragma: no branch
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):  # pragma: no branch
                vix_close = vix_close.iloc[:, 0]
            _vix_df_holdout = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            _vix_df_holdout.index = pd.DatetimeIndex(_vix_df_holdout.index).tz_localize(None)
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, _vix_df_holdout)

    results = _run_simulation(
        indicators,
        trading_dates,
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_hold_days=max_hold_days,
        params=frozen_params,
        spy_indicators=spy_indicators,
        regime_by_date=regime_by_date or None,
        vix_spike_by_date=vix_spike_by_date or None,
        rs_ranks=_compute_rs_ranks(indicators, spy_indicators),
    )
    results["start"] = holdout_start_str
    results["end"] = holdout_end_str
    results["holdout"] = True
    results["version"] = version

    print("\n" + "=" * 65)
    print(f"  HOLDOUT EVALUATION  version={version}")
    print(f"  {holdout_start_str} → {holdout_end_str}  (NEVER USED FOR TUNING)")
    print("=" * 65)
    print(f"  Total return:     {results['total_return_pct']:+.1f}%")
    print(f"  Total trades:     {results['total_trades']}")
    print(f"  Win rate:         {results['win_rate_pct']:.0f}%")
    print(f"  Max drawdown:     {results['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe ratio:     {results['sharpe_ratio']:.3f}")
    print("=" * 65 + "\n")

    return results


def _save_results(r: dict):
    """Persist latest backtest results for the dashboard to read."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, "backtest_results.json")
        saveable = {k: v for k, v in r.items() if k != "equity_curve"}
        saveable["equity_curve"] = [[str(d), v] for d, v in r.get("equity_curve", [])]
        with open(path, "w") as f:
            json.dump(saveable, f, indent=2)
        logger.info(f"Backtest results saved to {path}")
    except Exception as e:
        logger.error(f"Could not save backtest results: {e}")


def _print_results(r: dict):
    tested = r.get("signals_tested", [])
    holdout_contaminated = False
    with contextlib.suppress(KeyError, ValueError):
        holdout_contaminated = datetime.strptime(r["end"], "%Y-%m-%d").date() >= HOLDOUT_START_DATE
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS  {r['start']} → {r['end']}")
    print("=" * 60)
    print("  NOTE: Rule proxy only — does not reflect deployed strategy")
    print("        (Claude, news, options, macro context excluded).")
    print("  BIAS: Universe from current tradable listings — survivorship")
    print("        risk present; delistings and failures not represented.")
    if holdout_contaminated:
        print(f"  OOS:  Run overlaps holdout (post {HOLDOUT_START_DATE}) —")
        print("        results are NOT independent out-of-sample evidence.")
    print(f"  Signals tested:    {', '.join(tested)}")
    print(f"  Initial capital:   ${r['initial_capital']:.2f}")
    print(f"  Final value:       ${r['final_value']:.2f}")
    print(f"  Total return:      {r['total_return_pct']:+.1f}%")
    print(f"  Total trades:      {r['total_trades']}")
    print(f"  Win rate:          {r['win_rate_pct']:.0f}%")
    print(f"  Avg return/trade:  {r['avg_return_per_trade_pct']:+.2f}%")
    print(f"  Max drawdown:      {r['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe ratio:      {r['sharpe_ratio']:.2f}")
    print()
    print("  By signal:")
    for sig, data in r["by_signal"].items():
        total = data["wins"] + data["losses"]
        wr = data["wins"] / total * 100 if total else 0
        avg = data["total_return"] / total if total else 0
        print(f"    {sig:<25} {total:>3} trades  WR {wr:.0f}%  avg {avg:+.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    _today = pd.Timestamp.today().normalize()
    _last_bday = (_today - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    parser.add_argument("--start", default=BACKTEST_DEFAULT_START)
    parser.add_argument("--end", default=_last_bday)
    parser.add_argument("--capital", type=float, default=25000.0)
    parser.add_argument(
        "--use-intraday",
        action="store_true",
        help="Fetch Alpaca minute bars to test vwap_reclaim/orb_breakout/intraday_momentum",
    )
    parser.add_argument(
        "--per-signal-cap",
        type=int,
        default=2,
        help="Max positions opened from any single signal per day (default 2)",
    )
    parser.add_argument(
        "--use-fundamentals",
        action="store_true",
        help="Pre-fetch SEC EDGAR Form 4 + yfinance EPS history to test pead/insider_buying",
    )
    parser.add_argument(
        "--use-earnings-only",
        action="store_true",
        help="Pre-fetch yfinance EPS history only (skips slow EDGAR insider fetch, enables pead)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study: disable each signal in turn to measure marginal Sharpe contribution",
    )
    parser.add_argument(
        "--backward-elimination",
        action="store_true",
        help="Greedy backward elimination: iteratively remove the worst signal until none are drags",
    )
    parser.add_argument(
        "--signal-analysis",
        action="store_true",
        help="Regime-stratified breakdown + hold-period decay for each signal",
    )
    parser.add_argument(
        "--short-signals",
        action="store_true",
        help="Backtest bearish signals (earnings_miss, loser_momentum, ema_breakdown)",
    )
    parser.add_argument(
        "--short-param-sensitivity",
        action="store_true",
        help="One-at-a-time parameter sweep for short signal thresholds",
    )
    parser.add_argument(
        "--short-walk-forward",
        action="store_true",
        help="Walk-forward stability check for short signals with fixed params",
    )
    parser.add_argument(
        "--short-fb-vol-min",
        type=float,
        default=None,
        help="Override fb_vol_min for --short-walk-forward (default: use DEFAULT_SHORT_SIGNAL_PARAMS)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Combined long/short backtest — shared capital, regime-gated short entries",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Walk-forward optimised backtest (genuine OOS validation)",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Run once-only holdout evaluation on frozen params (requires --version)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Strategy version string for holdout log (e.g. v1.29); required with --holdout",
    )
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument(
        "--disabled-signals",
        type=str,
        default="",
        help="Comma-separated signals to disable globally (e.g. rs_leader,momentum_12_1)",
    )
    args = parser.parse_args()
    _disabled = (
        frozenset(s.strip() for s in args.disabled_signals.split(",") if s.strip())
        if args.disabled_signals
        else None
    )
    if args.holdout:
        if not args.version:
            parser.error("--holdout requires --version (e.g. --version v1.29)")
        run_holdout_evaluation(
            frozen_params={},
            version=args.version,
            initial_capital=args.capital,
        )
    elif args.signal_analysis:
        run_signal_analysis(
            STOCK_UNIVERSE,
            args.start,
            args.end,
            initial_capital=args.capital,
            per_signal_cap=args.per_signal_cap,
            use_fundamentals=args.use_fundamentals,
            use_earnings_only=args.use_earnings_only,
            disabled_signals=_disabled,
        )
    elif args.short_signals:
        run_short_signal_analysis(
            STATIC_SHORT_UNIVERSE,
            args.start,
            args.end,
            initial_capital=args.capital,
            use_earnings_history=True,
        )
    elif args.short_param_sensitivity:
        run_short_param_sensitivity(
            STATIC_SHORT_UNIVERSE,
            args.start,
            args.end,
            initial_capital=args.capital,
            use_earnings_only=args.use_earnings_only,
        )
    elif args.short_walk_forward:
        _swf_params = {**DEFAULT_SHORT_SIGNAL_PARAMS}
        if args.short_fb_vol_min is not None:
            _swf_params["fb_vol_min"] = args.short_fb_vol_min
        run_short_walk_forward(
            STATIC_SHORT_UNIVERSE,
            args.start,
            args.end,
            short_params=_swf_params,
            initial_capital=args.capital,
        )
    elif args.combined:
        run_combined_analysis(
            STOCK_UNIVERSE,
            args.start,
            args.end,
            initial_capital=args.capital,
            per_signal_cap=args.per_signal_cap,
            use_earnings_history=True,
        )
    elif args.walk_forward:
        result = run_walk_forward_optimized(
            STOCK_UNIVERSE,
            args.start,
            args.end,
            train_days=args.train_days,
            test_days=args.test_days,
            initial_capital=args.capital,
            per_signal_cap=args.per_signal_cap,
            use_fundamentals=args.use_fundamentals,
            use_earnings_only=args.use_earnings_only,
            disabled_signals=_disabled,
        )
        s = result["summary"]
        print("\n" + "=" * 65)
        print("  WALK-FORWARD SUMMARY")
        print("=" * 65)
        print("  NOTE: Universe built from current tradable listings.")
        print("        Survivorship bias not controlled. Working hypotheses only.")
        print()
        print(f"  Folds:              {s['n_folds']}")
        print(f"  Mean OOS return:    {s['mean_oos_return_pct']:+.2f}%")
        print(f"  Mean OOS Sharpe:    {s['mean_oos_sharpe']:.3f}")
        print(f"  Equal-weight base:  {s['random_baseline_return_pct']:+.2f}%")
        print(
            f"  Profitable folds:   {s['profitable_folds']}/{s['n_folds']}  ({s['consistency_pct']:.0f}%)"
        )
        print(
            f"  Beat baseline:      {s['beat_baseline_folds']}/{s['n_folds']}  ({s['beat_baseline_pct']:.0f}%)"
        )
        print(f"  Param stability:    {s['param_stability_pct']:.0f}%")
        print(f"  Mean train→OOS deg: {s['mean_oos_degradation']:+.3f} Sharpe")
        print()
        print(
            f"  {'Period':<25}  {'OOS Ret':>8}  {'Baseline':>9}  {'Beat?':>5}  {'Sharpe':>7}  {'Trades':>6}"
        )
        print("  " + "-" * 65)
        for f in result["folds"]:
            beat = "yes" if f["oos_total_return_pct"] > f["random_baseline_return_pct"] else "no"
            print(
                f"  {f['test_start']} → {f['test_end']}  "
                f"{f['oos_total_return_pct']:>+7.1f}%  "
                f"{f['random_baseline_return_pct']:>+8.1f}%  "
                f"{beat:>5}  "
                f"{f['oos_sharpe']:>7.3f}  "
                f"{f['oos_total_trades']:>6}"
            )
        print("=" * 65 + "\n")
    elif args.backward_elimination:
        run_backward_elimination(
            STOCK_UNIVERSE,
            args.start,
            args.end,
            initial_capital=args.capital,
            per_signal_cap=args.per_signal_cap,
            use_fundamentals=args.use_fundamentals,
            use_earnings_only=args.use_earnings_only,
            disabled_signals=_disabled,
        )
    elif args.ablation:
        run_ablation(
            STOCK_UNIVERSE,
            args.start,
            args.end,
            initial_capital=args.capital,
            per_signal_cap=args.per_signal_cap,
            use_fundamentals=args.use_fundamentals,
            use_earnings_only=args.use_earnings_only,
            disabled_signals=_disabled,
        )
    else:
        run_backtest(
            STOCK_UNIVERSE,
            args.start,
            args.end,
            initial_capital=args.capital,
            use_intraday=args.use_intraday,
            per_signal_cap=args.per_signal_cap,
            use_fundamentals=args.use_fundamentals,
            use_earnings_only=args.use_earnings_only,
            disabled_signals=_disabled,
        )
