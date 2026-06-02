"""Dynamic exit level computation for open positions.

Three improvements over fixed stop/target config values:

1. Partial exit at 1:2 R:R — sell half when gain >= 2× stop distance, not a flat %.
2. ATR-adjusted full target — 2× ATR provides a volatility-scaled floor; config value
   is the floor so low-vol names still exit at the configured target.
3. Time-decay stop tightening — on the penultimate hold day, tighten the effective stop
   to break-even (0%) so a flat/losing trade exits rather than drifting to max-hold.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_atr_pct(symbol: str, period: int = 14) -> float | None:
    """Return 14-period ATR as a percentage of the current closing price.

    Uses yfinance daily bars (30-day window → enough for 14-period ATR).
    Returns None on any failure so callers fall back to config defaults.
    """
    try:
        import pandas as pd
        import yfinance as yf

        df = yf.download(symbol, period="30d", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < period + 1:
            return None
        # Flatten MultiIndex columns that yfinance sometimes emits for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(tr.rolling(period).mean().iloc[-1])
        current_price = float(close.iloc[-1])
        if current_price <= 0 or atr != atr:  # NaN check
            return None
        return round(atr / current_price * 100, 2)
    except Exception as e:
        logger.debug(f"compute_atr_pct({symbol}): {e}")
        return None


def compute_exit_levels(
    stop_loss_pct: float,
    take_profit_pct: float,
    atr_pct: float | None,
    days_held: int,
    max_hold_days: int,
) -> dict:
    """Compute dynamic exit thresholds for a single position.

    Parameters
    ----------
    stop_loss_pct : float
        Hard stop as a fraction (e.g. 0.07 for 7%).
    take_profit_pct : float
        Full-exit target as a fraction (e.g. 0.20 for 20%).
    atr_pct : float | None
        14-day ATR as % of price (e.g. 3.5). None → pure R:R / config values.
    days_held : int
        Trading days since entry (1 = same day).
    max_hold_days : int
        Signal-specific max hold limit.

    Returns
    -------
    dict with keys:
        partial_pct : float   — unrealized_plpc threshold to trigger half-sell (%)
        full_target_pct : float — full-exit profit target (%)
        stop_pct : float      — hard stop (negative %, e.g. -7.0)
        timedecay_stop_pct : float — tightened stop on penultimate day (0.0 = break-even)
        apply_timedecay : bool — True when timedecay stop should be used instead of hard stop
    """
    stop_pct_abs = stop_loss_pct * 100  # e.g. 7.0
    take_profit_pct_abs = take_profit_pct * 100  # e.g. 20.0

    # 1:2 R:R minimum; ATR gives an upside floor for volatile names
    partial_pct = stop_pct_abs * 2.0
    if atr_pct is not None:
        partial_pct = max(partial_pct, atr_pct)

    # 2× ATR as the full-target floor; config value is the hard floor
    full_target_pct = take_profit_pct_abs
    if atr_pct is not None:
        full_target_pct = max(full_target_pct, atr_pct * 2.0)

    stop_pct = -stop_pct_abs

    # Penultimate day: tighten stop to break-even to protect from flat/losing trades
    apply_timedecay = days_held >= max(1, max_hold_days - 1)
    timedecay_stop_pct = 0.0  # break-even

    return {
        "partial_pct": round(partial_pct, 2),
        "full_target_pct": round(full_target_pct, 2),
        "stop_pct": round(stop_pct, 2),
        "timedecay_stop_pct": timedecay_stop_pct,
        "apply_timedecay": apply_timedecay,
    }


def rs_decay_triggered(
    current_rs_rank_pct: float,
    entry_rs_rank_pct: float,
    decay_threshold: float = 25.0,
) -> bool:
    """Return True when RS rank has decayed enough to exit.

    Fires when current_rs_rank_pct has dropped >= decay_threshold percentile points
    below entry_rs_rank_pct.

    Only meaningful for momentum signals (rs_leader, momentum, momentum_12_1).
    For other signals, callers should not call this function.

    current_rs_rank_pct: 0–100, where 100 = strongest stock in universe
    entry_rs_rank_pct: RS rank at entry (stored in position record)
    decay_threshold: percentile point drop that triggers exit (default 25)
    """
    return (entry_rs_rank_pct - current_rs_rank_pct) >= decay_threshold


def adverse_volume_triggered(
    vol_ratio_today: float,
    day_return_today: float,
    vol_ratio_yesterday: float,
    day_return_yesterday: float,
    vol_threshold: float = 2.5,
    return_threshold: float = -1.5,
) -> bool:
    """Return True when two consecutive adverse-volume days signal institutional distribution.

    Fires when BOTH:
      - Today: vol_ratio >= vol_threshold AND day_return <= return_threshold
      - Yesterday: vol_ratio >= vol_threshold AND day_return <= return_threshold

    vol_ratio: today's volume / 20-day average volume
    day_return: % return for the day (e.g. -2.1 for -2.1%)
    """
    today_adverse = vol_ratio_today >= vol_threshold and day_return_today <= return_threshold
    yesterday_adverse = (
        vol_ratio_yesterday >= vol_threshold and day_return_yesterday <= return_threshold
    )
    return today_adverse and yesterday_adverse


def profit_acceleration_triggered(
    unrealised_pct: float,
    days_held: int,
    signal: str,
    fast_exit_signals: frozenset[str] | None = None,
) -> str:
    """Return exit action for mean-reversion signals with rapid early gains.

    Only applies to fast_exit_signals (default: mean_reversion, range_reversion).
    Returns one of: "full_exit", "partial_exit", "hold"

    Rules:
      - unrealised_pct >= 8.0 AND days_held <= 2 → "full_exit"
      - unrealised_pct >= 5.0 AND days_held <= 1 → "partial_exit"
      - otherwise → "hold"

    Returns "hold" immediately for signals not in fast_exit_signals.
    """
    if fast_exit_signals is None:
        fast_exit_signals = frozenset({"mean_reversion", "range_reversion"})
    if signal not in fast_exit_signals:
        return "hold"
    if unrealised_pct >= 8.0 and days_held <= 2:
        return "full_exit"
    if unrealised_pct >= 5.0 and days_held <= 1:
        return "partial_exit"
    return "hold"
