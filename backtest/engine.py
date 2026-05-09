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
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import product
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands

from backtest.historical_fundamentals import (
    insider_state_on_date,
    pead_active_on_date,
    prefetch_earnings_history,
    prefetch_insider_history,
)
from config import LOG_DIR, SLIPPAGE_BPS, SPREAD_BPS, STOCK_UNIVERSE, STOP_LOSS_PCT, TAKE_PROFIT_PCT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Core indicator columns that must be non-NaN for a row to be used
_CORE_COLS = ["rsi", "macd_diff", "ema9", "ema21", "bb_pct", "vol_ratio", "ret_5d"]

# Signal priority for candidate slot allocation (lower number = higher priority).
# Ordered by observed edge. Intraday signals trail daily.
_SIGNAL_PRIORITY: dict[str, int] = {
    "vix_fear_reversion": 0,
    "insider_buying": 1,
    "pead": 2,
    "rs_leader": 3,
    "breakout_52w": 4,
    "momentum_12_1": 5,
    "gap_and_go": 6,
    "inside_day_breakout": 7,
    "bb_squeeze": 8,
    "iv_compression": 9,
    "momentum": 10,
    "macd_crossover": 11,
    "trend_pullback": 12,
    "mean_reversion": 13,
    "orb_breakout": 14,
    "vwap_reclaim": 15,
    "intraday_momentum": 16,
}

# Signals blocked per market regime.
# mean_reversion and vix_fear_reversion are always permitted (counter-cyclical).
_REGIME_BLOCKED: dict[str, frozenset[str]] = {
    "BEAR_DAY": frozenset(
        {
            "rs_leader",
            "breakout_52w",
            "momentum_12_1",
            "momentum",
            "macd_crossover",
            "bb_squeeze",
            "trend_pullback",
            "inside_day_breakout",
            "gap_and_go",
            "orb_breakout",
            "intraday_momentum",
        }
    ),
    "HIGH_VOL": frozenset(
        {
            "rs_leader",
            "breakout_52w",
            "momentum",
            "gap_and_go",
            "orb_breakout",
        }
    ),
    "CHOPPY": frozenset(
        {
            "rs_leader",
            "breakout_52w",
            "momentum_12_1",
            "momentum",
            "gap_and_go",
        }
    ),
}

# Default signal thresholds — match the prefilter rules in stock_scanner.py
_DEFAULT_PARAMS: dict[str, float] = {
    "rsi_threshold": 35.0,
    "bb_threshold": 0.25,
    "mr_vol_threshold": 1.2,
    "mom_vol_threshold": 1.3,
    "mom_ret5d_threshold": 1.0,
    "mom12_1_threshold": 10.0,
}

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
    df["vol_ratio"] = volume / volume.rolling(20).mean()
    df["ret_5d"] = close.pct_change(5) * 100

    # ── Extended indicators for new daily signals ─────────────────────────────
    df["ret_10d"] = close.pct_change(10) * 100

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
    import math

    daily_returns = close.pct_change()
    df["hv_20d"] = daily_returns.rolling(20).std() * math.sqrt(252) * 100
    df["hv_rank"] = df["hv_20d"].rolling(252, min_periods=30).rank(pct=True)

    # Drop rows where any core indicator is NaN (warmup period)
    return df.dropna(subset=_CORE_COLS)


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
    """
    Returns the first matching signal in priority order, or None.

    Priority (within a single symbol):
      vix_fear_reversion > mean_reversion > macd_crossover > bb_squeeze
      > inside_day_breakout > breakout_52w > gap_and_go > rs_leader
      > trend_pullback > momentum > orb_breakout > vwap_reclaim > intraday_momentum

    Regime blocking: BEAR_DAY/HIGH_VOL/CHOPPY suppress trending signals.
    ADX gate (>= 20): required for all momentum-type signals.
    mean_reversion and vix_fear_reversion are never regime-blocked.
    Intraday signals only fire when intraday data is supplied.
    """
    p = _DEFAULT_PARAMS if params is None else {**_DEFAULT_PARAMS, **params}
    blocked = _REGIME_BLOCKED.get(regime, frozenset())
    if disabled_signals:
        blocked = blocked | disabled_signals
    # Default adx=30 (assume trending) when High/Low weren't available
    adx = float(row.get("adx", 30))

    # ── Counter-cyclical — fires during fear spikes, never regime-blocked ──────
    if vix_spike and row["vol_ratio"] > 1.0:
        return "vix_fear_reversion"

    # ── Fundamental conviction — bypass regime filter but respect disabled_signals ──
    if fundamentals:
        if fundamentals.get("insider_cluster") and "insider_buying" not in blocked:
            return "insider_buying"
        if fundamentals.get("pead_active") and row.get("ret_5d", 0) > 0 and "pead" not in blocked:
            return "pead"

    # ── Daily signals ─────────────────────────────────────────────────────────
    if (
        row["rsi"] < p["rsi_threshold"]
        and row["bb_pct"] < p["bb_threshold"]
        and row["vol_ratio"] > p["mr_vol_threshold"]
        and "mean_reversion" not in blocked
    ):
        return "mean_reversion"

    if (
        row.get("macd_cross", False)
        and row["vol_ratio"] > 1.2
        and adx >= 20
        and "macd_crossover" not in blocked
    ):
        return "macd_crossover"

    if (
        row.get("bb_squeeze", False)
        and row["vol_ratio"] > 1.2
        and (row["ema9"] > row["ema21"] or row["macd_diff"] > 0)
        and adx >= 20
        and "bb_squeeze" not in blocked
    ):
        return "bb_squeeze"

    if (
        row.get("is_inside_day", False)
        and row["vol_ratio"] > 1.1
        and (row["ema9"] > row["ema21"] or row["macd_diff"] > 0)
        and adx >= 20
        and "inside_day_breakout" not in blocked
    ):
        return "inside_day_breakout"

    if (
        row.get("price_vs_52w_high_pct", -999) >= -3.0
        and row["vol_ratio"] > 1.2
        and row["ema9"] > row["ema21"]
        and adx >= 20
        and "breakout_52w" not in blocked
    ):
        return "breakout_52w"

    if (
        row.get("mom_12_1", -999) > p["mom12_1_threshold"]
        and row["ema9"] > row["ema21"]
        and adx >= 20
        and "momentum_12_1" not in blocked
    ):
        return "momentum_12_1"

    if (
        row.get("gap_pct", 0) > 2.0
        and row.get("close_above_open", False)
        and row["vol_ratio"] > 1.5
        and adx >= 20
        and "gap_and_go" not in blocked
    ):
        return "gap_and_go"

    if spy_ret_5d is not None and spy_ret_10d is not None and "rs_leader" not in blocked:
        rel_5d = row["ret_5d"] - spy_ret_5d
        rel_10d = row.get("ret_10d", 0.0) - spy_ret_10d
        if rel_5d > 2.0 and rel_10d > 3.0 and row["ema9"] > row["ema21"] and adx >= 20:
            return "rs_leader"

    pct_vs_ema21 = row.get("pct_vs_ema21", 0.0)
    if (
        row["ema9"] > row["ema21"]
        and -3.0 <= pct_vs_ema21 <= -0.5
        and 40 <= row["rsi"] <= 58
        and row["vol_ratio"] > 1.0
        and adx >= 20
        and "trend_pullback" not in blocked
    ):
        return "trend_pullback"

    if (
        row.get("hv_rank", 1.0) < 0.20
        and (row["ema9"] > row["ema21"] or row["macd_diff"] > 0)
        and row["vol_ratio"] > 1.1
        and "iv_compression" not in blocked
    ):
        return "iv_compression"

    if (
        row["ema9"] > row["ema21"]
        and row["macd_diff"] > 0
        and row["ret_5d"] > p["mom_ret5d_threshold"]
        and row["vol_ratio"] > p["mom_vol_threshold"]
        and adx >= 20
        and "momentum" not in blocked
    ):
        return "momentum"

    # ── Intraday signals (only when Alpaca data provided) ─────────────────────
    if intraday:
        if intraday.get("orb_breakout_up") is True and "orb_breakout" not in blocked:
            return "orb_breakout"

        id_chg = intraday.get("intraday_change_pct")
        above_vwap = intraday.get("price_above_vwap")
        pct_vwap = intraday.get("pct_vs_vwap", 0)
        if (
            above_vwap is True
            and id_chg is not None
            and id_chg > 1.0
            and pct_vwap <= 3.0
            and "vwap_reclaim" not in blocked
        ):
            return "vwap_reclaim"

        id_rsi = intraday.get("intraday_rsi")
        ema_up = row["ema9"] > row["ema21"]
        if (
            id_chg is not None
            and id_chg > 2.0
            and above_vwap is True
            and (id_rsi is None or id_rsi < 75)
            and (ema_up or row["ret_5d"] > 3.0)
            and "intraday_momentum" not in blocked
        ):
            return "intraday_momentum"

    return None


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
                feed="iex",
            )
            bars_resp = client.get_stock_bars(req)
            bars_data = bars_resp.data.get(sym, [])
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
                if metrics is not None:
                    sym_result[date_str] = metrics

            result[sym] = sym_result

        except Exception as exc:
            logger.warning(f"Intraday fetch failed for {sym}: {exc}")

    logger.info(f"Intraday bars fetched for {len(result)}/{len(symbols)} symbols")
    return result


def _compute_regimes(
    spy_indicators: pd.DataFrame,
    vix_spike_by_date: dict[str, bool],
) -> dict[str, str]:
    """Map each date to a market regime string using SPY returns and VIX spike flag."""
    spy_close = spy_indicators["Close"]
    spy_ret_1d = (spy_close / spy_close.shift(1) - 1) * 100
    regimes: dict[str, str] = {}
    for ts in spy_indicators.index:
        date_str = ts.strftime("%Y-%m-%d")
        spy_1d = float(spy_ret_1d.get(ts, 0) or 0)
        spy_5d = float(spy_indicators.loc[ts].get("ret_5d", 0) or 0)
        vix_spike = vix_spike_by_date.get(date_str, False)
        if spy_1d <= -1.5:
            regimes[date_str] = "BEAR_DAY"
        elif vix_spike and spy_5d < -3:
            regimes[date_str] = "HIGH_VOL"
        elif spy_5d > 2 and spy_1d > 0:
            regimes[date_str] = "BULL_TRENDING"
        else:
            regimes[date_str] = "CHOPPY"
    return regimes


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
) -> dict:
    """Core trading simulation on pre-computed indicators. Called by both run_backtest
    and run_walk_forward_optimized (the latter avoids re-downloading data per param combo)."""
    s_bps = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    sp_bps = SPREAD_BPS if spread_bps is None else spread_bps
    buy_factor = 1.0 + (s_bps + sp_bps / 2) / 10_000
    sell_factor = 1.0 - (s_bps + sp_bps / 2) / 10_000
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
                px = float(indicators[sym].loc[today, "Close"])
            except Exception:
                continue
            pnl_pct = px / pos["entry_price"] - 1
            trading_days_held = sum(1 for _ in pd.bdate_range(pos["entry_date"], today)) - 1

            reason = None
            if pnl_pct <= -STOP_LOSS_PCT:
                reason = "stop_loss"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                reason = "take_profit"
            elif trading_days_held >= max_hold_days:
                reason = "time_exit"

            if reason:
                exit_px = px * sell_factor
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
                fill_px = entry_px * buy_factor
                notional = (cash / slots) * 0.9
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
    for sym, pos in positions.items():
        try:
            px = float(indicators[sym].iloc[-1]["Close"])
            exit_px = px * sell_factor
            cash += pos["shares"] * exit_px
            pnl_pct = (exit_px / pos["entry_price"] - 1) * 100
            trades.append(
                {
                    "date": "end",
                    "symbol": sym,
                    "action": "SELL",
                    "reason": "end_of_backtest",
                    "pnl_pct": round(pnl_pct, 2),
                    "signal": pos["signal"],
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
) -> dict:

    logger.info(
        f"Backtest: {start_date} → {end_date} | {len(symbols)} symbols | ${initial_capital:.0f} capital"
        + (" | intraday=ON" if use_intraday else "")
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
            logger.warning(f"SPY fetch failed — rs_leader signal disabled: {exc}")

    # VIX data for fear-reversion signal and regime detection
    vix_spike_by_date: dict[str, bool] = {}
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):
                vix_close = vix_close.iloc[:, 0]
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
            logger.info(f"VIX fetched — {sum(vix_spike_by_date.values())} fear-spike days")
    except Exception as exc:
        logger.warning(f"VIX fetch failed — vix_fear_reversion and regime filter disabled: {exc}")

    # Regime map for trend-signal gating
    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _compute_regimes(spy_indicators, vix_spike_by_date)
        bear_days = sum(1 for r in regime_by_date.values() if r == "BEAR_DAY")
        logger.info(f"Regime map computed — {bear_days} BEAR_DAY sessions")

    # Alpaca intraday bars for vwap_reclaim / orb_breakout / intraday_momentum
    intraday_data: dict | None = None
    if use_intraday:
        intraday_data = _fetch_intraday_bars(symbols, start_date, end_date)
        if not intraday_data:
            logger.warning("Intraday fetch returned no data — intraday signals disabled")
            intraday_data = None

    # Historical fundamentals for pead / insider_buying point-in-time simulation
    earnings_history: dict[str, list[dict]] | None = None
    insider_history: dict[str, list[dict]] | None = None
    if use_fundamentals:
        logger.info("Pre-fetching fundamental data (earnings + insider)…")
        earnings_history = prefetch_earnings_history(symbols)
        insider_history = prefetch_insider_history(symbols)
        logger.info(
            f"Fundamentals ready: {len(earnings_history)} earnings, "
            f"{len(insider_history)} insider histories"
        )

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
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
            logger.warning(f"SPY fetch failed for walk-forward: {exc}")

    wf_earnings_history: dict[str, list[dict]] | None = None
    wf_insider_history: dict[str, list[dict]] | None = None
    if use_fundamentals:
        logger.info("Walk-forward: pre-fetching fundamental data…")
        wf_earnings_history = prefetch_earnings_history(symbols)
        wf_insider_history = prefetch_insider_history(symbols)
        logger.info(
            f"Walk-forward fundamentals ready: {len(wf_earnings_history)} earnings, "
            f"{len(wf_insider_history)} insider histories"
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

        best_params = all_combos[0]
        best_score = -float("inf")
        best_train_trades = 0

        for combo in all_combos:
            r = _run_simulation(
                indicators,
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
            )
            if r["total_trades"] < _MIN_TRAIN_TRADES:
                continue
            if r["sharpe_ratio"] > best_score:
                best_score = r["sharpe_ratio"]
                best_params = combo
                best_train_trades = r["total_trades"]

        oos = _run_simulation(
            indicators,
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
        )

        baseline_rets = []
        for _sym, df in indicators.items():
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
        "consistency_pct": round(profitable / n * 100, 1),
        "param_stability_pct": round(modal_count / n * 100, 1),
        "mean_oos_degradation": round(sum(f["oos_degradation"] for f in fold_results) / n, 2),
        "random_baseline_return_pct": round(
            sum(f["random_baseline_return_pct"] for f in fold_results) / n, 2
        ),
    }

    logger.info(
        f"Walk-forward summary: {n} folds | mean OOS return {summary['mean_oos_return_pct']:+.2f}% "
        f"vs baseline {summary['random_baseline_return_pct']:+.2f}% "
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
            logger.warning(f"SPY fetch failed: {exc}")

    vix_spike_by_date: dict[str, bool] = {}
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):
                vix_close = vix_close.iloc[:, 0]
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _compute_regimes(spy_indicators, vix_spike_by_date)

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
    }

    logger.info("Ablation: running baseline…")
    baseline = _run_simulation(indicators, trading_dates, **sim_kwargs)

    ablations = []
    for signal_name in sorted(_SIGNAL_PRIORITY, key=lambda s: _SIGNAL_PRIORITY[s]):
        logger.info(f"Ablation: disabling {signal_name}…")
        result = _run_simulation(
            indicators,
            trading_dates,
            **sim_kwargs,
            disabled_signals=frozenset({signal_name}),
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
            logger.warning(f"SPY fetch failed: {exc}")

    vix_spike_by_date: dict[str, bool] = {}
    try:
        vix_raw = yf.download(
            "^VIX", start=fetch_start, end=end_date, auto_adjust=False, progress=False
        )
        if not vix_raw.empty:
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):
                vix_close = vix_close.iloc[:, 0]
            vix_ma20 = vix_close.rolling(20).mean()
            vix_spike_s = vix_close > vix_ma20 * 1.3
            vix_spike_by_date = {
                ts.strftime("%Y-%m-%d"): bool(v) for ts, v in vix_spike_s.items() if not pd.isna(v)
            }
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _compute_regimes(spy_indicators, vix_spike_by_date)

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
                **sim_kwargs,
                disabled_signals=frozenset(disabled | {signal_name}),
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
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS  {r['start']} → {r['end']}")
    print("=" * 60)
    print("  NOTE: Rule proxy only — does not reflect deployed strategy")
    print("        (Claude, news, options, macro context excluded).")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _yesterday = datetime.today() - timedelta(days=1)
    _last_bday = pd.bdate_range(end=_yesterday, periods=1)[0].strftime("%Y-%m-%d")
    parser.add_argument("--start", default="2025-01-01")
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
    args = parser.parse_args()
    if args.backward_elimination:
        run_backward_elimination(
            STOCK_UNIVERSE,
            args.start,
            args.end,
            initial_capital=args.capital,
            per_signal_cap=args.per_signal_cap,
            use_fundamentals=args.use_fundamentals,
            use_earnings_only=args.use_earnings_only,
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
        )
