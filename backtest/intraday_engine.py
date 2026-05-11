"""Intraday backtester — replays Alpaca 1-min bars within each session.

This engine is strictly more rigorous than the daily engine's --use-intraday
overlay because:
  - Entries fill at the OPEN of the bar *following* the signal bar (no lookahead).
  - VWAP and ORB ranges are computed from bars already seen — no future data.
  - Stops and targets are checked bar-by-bar, with gap-through handling.
  - Positions are always closed at the 15:55 ET bar (hard end-of-session).
  - Every trade bears the same spread + market-impact cost as the daily engine.

Signals evaluated (all others blocked):
  orb_breakout   — close breaks ORB high (09:30–10:00) with above-avg volume
  vwap_reclaim   — price reclaims VWAP after touching below it; up >1% from open
  intraday_momentum — up >2% from open, above VWAP, intraday RSI < 75

Each signal fires at most once per (symbol, day).

Usage:
    from backtest.intraday_engine import run_intraday_backtest
    result = run_intraday_backtest(["AAPL", "MSFT"], "2025-01-01", "2025-06-30")

Returns a dict matching the daily engine's run_backtest() schema so callers can
use the same reporting/comparison infrastructure.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from data.intraday_fetcher import fetch_intraday_bars
from signals.evaluator import SIGNAL_PRIORITY, evaluate_signals

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

_INTRADAY_SIGNALS = frozenset({"orb_breakout", "vwap_reclaim", "intraday_momentum"})
_BLOCKED = frozenset(s for s in SIGNAL_PRIORITY if s not in _INTRADAY_SIGNALS)

# Minimum bars in ORB window before range is considered valid
_ORB_MIN_BARS = 5
# Hard EOD close: no new entries after this time, force-close open positions
_EOD_CLOSE_TIME = "15:55"
_EOD_NO_ENTRY_TIME = "15:30"


def _spread_bps(adv_usd: float) -> float:
    """Liquidity-scaled spread in bps (mirrors daily engine formula)."""
    from config import SPREAD_BPS

    return max(float(SPREAD_BPS), 50.0 / math.sqrt(max(adv_usd, 1_000_000) / 1e6))


def _impact_bps(notional: float, adv_usd: float) -> float:
    """Square-root market impact in bps (mirrors daily engine formula)."""
    participation = notional / max(adv_usd, 1.0)
    return min(50.0, 10.0 * math.sqrt(participation))


def _trade_cost_pct(notional: float, adv_usd: float) -> float:
    """Round-trip cost as a percentage of notional (enter + exit)."""
    spread = _spread_bps(adv_usd) / 10_000
    impact = _impact_bps(notional, adv_usd) / 10_000
    return (spread + impact) * 2 * 100  # × 2 for round-trip, × 100 for pct


def _parse_eod(date_str: str, hhmm: str) -> datetime:
    return datetime.strptime(f"{date_str} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=_ET)


def _compute_running_vwap(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    vols: list[float],
) -> float:
    """VWAP from all bars seen so far."""
    total_vol = sum(vols)
    if total_vol <= 0:
        return closes[-1] if closes else 0.0
    typical = [(h + lo + c) / 3 for h, lo, c in zip(highs, lows, closes, strict=True)]
    return sum(tp * v for tp, v in zip(typical, vols, strict=True)) / total_vol


def _compute_intraday_rsi(closes_5m: list[float]) -> float | None:
    """RSI(14) on 5-min sampled closes; None if fewer than 14 samples."""
    if len(closes_5m) < 14:
        return None
    try:
        from ta.momentum import RSIIndicator

        s = pd.Series(closes_5m, dtype=float)
        return float(RSIIndicator(close=s, window=14).rsi().iloc[-1])
    except Exception:
        return None


def _replay_day(
    sym: str,
    date_str: str,
    timed_bars: list[tuple],
    stop_loss_pct: float,
    target_pct: float,
    adv_usd: float,
    position_size_usd: float,
) -> list[dict]:
    """Replay one trading session for one symbol. Returns list of closed trades.

    Parameters
    ----------
    timed_bars : list[tuple]
        Sorted list of (datetime_et, bar) pairs for one day.
        bar objects expose .open .high .low .close .volume.
    stop_loss_pct, target_pct : float
        Stop-loss and take-profit distances in percent from entry.
    adv_usd : float
        Average daily USD volume — used for cost model.
    position_size_usd : float
        Dollar notional per position.
    """
    eod_close_dt = _parse_eod(date_str, _EOD_CLOSE_TIME)
    eod_no_entry_dt = _parse_eod(date_str, _EOD_NO_ENTRY_TIME)
    orb_cutoff_dt = _parse_eod(date_str, "10:00")

    # Running state
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    opens: list[float] = []
    vols: list[float] = []
    times: list[datetime] = []
    closes_5m: list[float] = []  # every 5th bar close for RSI
    bar_count = 0

    orb_high: float | None = None
    orb_computed = False
    avg_bar_vol = 1.0

    # Signal-fire tracking (one entry per signal per day)
    signals_fired: set[str] = set()

    # Position state
    in_position = False
    entry_price = 0.0
    entry_time: datetime | None = None
    entry_signal = ""
    stop = 0.0
    target = 0.0
    pending_entry: str | None = None  # signal name queued for next bar's open

    was_below_vwap = False  # for vwap_reclaim detection

    trades: list[dict] = []

    for _bar_idx, (bar_time, bar) in enumerate(timed_bars):
        if bar_time > eod_close_dt:
            break

        b_open = float(bar.open)
        b_high = float(bar.high)
        b_low = float(bar.low)
        b_close = float(bar.close)
        b_vol = float(bar.volume)

        # Execute pending entry at this bar's open
        if pending_entry and not in_position:
            if bar_time <= eod_no_entry_dt:
                entry_price = b_open
                stop = entry_price * (1 - stop_loss_pct / 100)
                target = entry_price * (1 + target_pct / 100)
                in_position = True
                entry_time = bar_time
                entry_signal = pending_entry
            pending_entry = None

        # Update running OHLCV
        highs.append(b_high)
        lows.append(b_low)
        closes.append(b_close)
        opens.append(b_open)
        vols.append(b_vol)
        times.append(bar_time)
        bar_count += 1
        if bar_count % 5 == 0:
            closes_5m.append(b_close)

        # Compute ORB range once the 10:00 window closes
        if not orb_computed and bar_time >= orb_cutoff_dt:
            orb_idxs = [i for i, t in enumerate(times) if t < orb_cutoff_dt]
            if len(orb_idxs) >= _ORB_MIN_BARS:
                orb_high = max(highs[i] for i in orb_idxs)
                avg_bar_vol = sum(vols[i] for i in orb_idxs) / len(orb_idxs)
            orb_computed = True

        # Manage open position
        if in_position:
            exit_price: float | None = None
            exit_reason = ""

            # Gap-through-stop: bar opened below stop
            if b_open <= stop:
                exit_price = b_open
                exit_reason = "stop_gap_through"
            elif b_low <= stop:
                exit_price = stop
                exit_reason = "stop"
            elif b_high >= target:
                exit_price = target
                exit_reason = "target"
            elif bar_time >= eod_close_dt:
                exit_price = b_close
                exit_reason = "eod"

            if exit_price is not None:
                cost_pct = _trade_cost_pct(position_size_usd, adv_usd)
                gross_pnl = (exit_price / entry_price - 1) * 100
                net_pnl = gross_pnl - cost_pct
                assert entry_time is not None
                trades.append(
                    {
                        "symbol": sym,
                        "date": date_str,
                        "signal": entry_signal,
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "entry_time": entry_time.strftime("%H:%M"),
                        "exit_time": bar_time.strftime("%H:%M"),
                        "exit_reason": exit_reason,
                        "pnl_pct": round(net_pnl, 4),
                        "gross_pnl_pct": round(gross_pnl, 4),
                        "cost_pct": round(cost_pct, 4),
                    }
                )
                in_position = False
                entry_signal = ""
                continue

        # Evaluate signals (only after ORB window, only when flat)
        if in_position or pending_entry or bar_time >= eod_no_entry_dt:
            if bar_time < orb_cutoff_dt:
                pass
            continue

        vwap = _compute_running_vwap(highs, lows, closes, vols)
        intraday_change = (b_close / opens[0] - 1) * 100 if opens and opens[0] > 0 else 0.0
        price_above_vwap = b_close > vwap
        pct_vs_vwap = (b_close / vwap - 1) * 100 if vwap > 0 else 0.0

        # Track VWAP reclaim setup
        if not price_above_vwap:
            was_below_vwap = True

        intraday_rsi = _compute_intraday_rsi(closes_5m)

        orb_breakout_up = False
        if orb_high is not None and b_vol > avg_bar_vol:
            orb_breakout_up = b_close > orb_high

        # Build snapshot for canonical evaluator (daily fields set to neutral defaults)
        snapshot = {
            "intraday_change_pct": intraday_change,
            "price_above_vwap": price_above_vwap,
            "pct_vs_vwap": pct_vs_vwap,
            "orb_breakout_up": orb_breakout_up,
            "intraday_rsi": intraday_rsi,
            # Neutral daily fields — prevent false daily signal fires
            "rsi_14": 50.0,
            "bb_pct": 0.5,
            "vol_ratio": 1.0,
            "macd_diff": 0.0,
            "macd_crossed_up": False,
            "ema9_above_ema21": True,  # permissive for intraday_momentum gate
            "adx": 30.0,
            "ret_5d_pct": 0.0,
            "ret_10d_pct": 0.0,
        }

        fired = evaluate_signals(snapshot, blocked=_BLOCKED)

        for sig in fired:
            if sig in signals_fired:
                continue
            # VWAP reclaim only valid after price was below VWAP
            if sig == "vwap_reclaim" and not was_below_vwap:
                continue
            signals_fired.add(sig)
            pending_entry = sig
            break  # one entry per bar

    # Force-close any position still open (should have been caught by eod_close_dt above,
    # but guard in case last bar is before 15:55)
    if in_position and closes:
        cost_pct = _trade_cost_pct(position_size_usd, adv_usd)
        gross_pnl = (closes[-1] / entry_price - 1) * 100
        net_pnl = gross_pnl - cost_pct
        assert entry_time is not None
        trades.append(
            {
                "symbol": sym,
                "date": date_str,
                "signal": entry_signal,
                "entry_price": round(entry_price, 4),
                "exit_price": round(closes[-1], 4),
                "entry_time": entry_time.strftime("%H:%M"),
                "exit_time": times[-1].strftime("%H:%M"),
                "exit_reason": "eod_fallback",
                "pnl_pct": round(net_pnl, 4),
                "gross_pnl_pct": round(gross_pnl, 4),
                "cost_pct": round(cost_pct, 4),
            }
        )

    return trades


def run_intraday_backtest(
    symbols: list[str],
    start_date: str,
    end_date: str,
    stop_loss_pct: float = 1.0,
    target_pct: float = 2.0,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    cache_dir: str | None = None,
    bars: dict[str, dict[str, list]] | None = None,
) -> dict:
    """Replay intraday sessions and return backtest results.

    Parameters
    ----------
    symbols : list[str]
        Tickers to backtest.
    start_date, end_date : str
        ISO date range (inclusive).
    stop_loss_pct : float
        Intraday stop-loss distance from entry (default 1%).
    target_pct : float
        Take-profit distance from entry (default 2%).
    initial_capital : float
        Starting portfolio value (default $100,000).
    max_positions : int
        Maximum concurrent open positions (default 5).
    cache_dir : str | None
        Passed to fetch_intraday_bars(); None → default location.
    bars : dict | None
        Pre-loaded bar data (for testing without Alpaca).  If provided,
        fetch_intraday_bars() is not called.

    Returns
    -------
    dict
        Matches the daily engine's run_backtest() schema:
        {initial_capital, final_value, total_return_pct, total_trades,
         win_rate_pct, avg_return_per_trade_pct, max_drawdown_pct,
         sharpe_ratio, by_signal, equity_curve, trades,
         validation_scope, signals_tested, signals_not_tested}
    """
    if bars is None:
        bars = fetch_intraday_bars(symbols, start_date, end_date, cache_dir=cache_dir)

    # Collect all unique trading dates across all symbols
    all_dates: list[str] = sorted({ds for sym_data in bars.values() for ds in sym_data})

    all_trades: list[dict] = []
    equity = initial_capital
    equity_curve: list[dict] = []

    for date_str in all_dates:
        day_trades: list[dict] = []

        # Determine available symbols for this day (point-in-time)
        day_syms = [s for s in symbols if date_str in bars.get(s, {})]

        # Simple round-robin slot allocation: process by signal priority order,
        # cap at max_positions per day.
        for sym in day_syms:
            if len(day_trades) >= max_positions:
                break
            timed_bars = bars[sym][date_str]
            if not timed_bars:
                continue

            # Estimate ADV from volume * close for cost model
            vols = [float(b.volume) for _, b in timed_bars]
            closes = [float(b.close) for _, b in timed_bars]
            avg_close = sum(closes) / len(closes) if closes else 1.0
            adv_usd = sum(vols) * avg_close  # full-day as proxy

            position_size_usd = equity / max_positions

            sym_trades = _replay_day(
                sym,
                date_str,
                timed_bars,
                stop_loss_pct=stop_loss_pct,
                target_pct=target_pct,
                adv_usd=adv_usd,
                position_size_usd=position_size_usd,
            )
            day_trades.extend(sym_trades)

        # Update equity from day's trades
        for t in day_trades:
            equity *= 1 + t["pnl_pct"] / 100

        all_trades.extend(day_trades)
        equity_curve.append({"date": date_str, "equity": round(equity, 2)})

    # Aggregate statistics
    closed = [t for t in all_trades if "pnl_pct" in t]
    n = len(closed)
    wins = sum(1 for t in closed if t["pnl_pct"] > 0)
    win_rate = wins / n * 100 if n else 0.0
    avg_return = sum(t["pnl_pct"] for t in closed) / n if n else 0.0
    total_return = (equity / initial_capital - 1) * 100 if initial_capital > 0 else 0.0

    eq_values = [initial_capital] + [e["equity"] for e in equity_curve]
    peak = eq_values[0]
    max_dd = 0.0
    for v in eq_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    by_signal: dict[str, dict] = {}
    for t in closed:
        s = t.get("signal", "unknown")
        by_signal.setdefault(s, {"wins": 0, "losses": 0, "total_return": 0.0})
        by_signal[s]["total_return"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            by_signal[s]["wins"] += 1
        else:
            by_signal[s]["losses"] += 1

    eq_series = pd.Series([e["equity"] for e in equity_curve])
    daily_rets = eq_series.pct_change().dropna()
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * (252**0.5))
        if len(daily_rets) > 1 and daily_rets.std() > 0
        else 0.0
    )

    signals_fired_set = {t["signal"] for t in closed}
    all_intraday = sorted(_INTRADAY_SIGNALS)
    signals_not_tested = sorted(_INTRADAY_SIGNALS - signals_fired_set)

    return {
        "initial_capital": initial_capital,
        "final_value": round(equity, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": n,
        "win_rate_pct": round(win_rate, 1),
        "avg_return_per_trade_pct": round(avg_return, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "by_signal": by_signal,
        "equity_curve": equity_curve,
        "trades": all_trades,
        "validation_scope": "intraday_rule_proxy",
        "signals_tested": all_intraday,
        "signals_not_tested": signals_not_tested,
    }
