"""
Rule-based backtester — validates technical signal quality on historical data
without calling Claude (avoids API cost).

Usage:
    python backtest/engine.py --start 2025-01-01 --end 2025-12-31
    python backtest/engine.py --start 2025-01-01 --end 2025-12-31 --capital 25000
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from itertools import product

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

from config import LOG_DIR, STOCK_UNIVERSE, STOP_LOSS_PCT, TAKE_PROFIT_PCT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Default signal thresholds — match the prefilter rules in stock_scanner.py
_DEFAULT_PARAMS: dict[str, float] = {
    "rsi_threshold": 35.0,
    "bb_threshold": 0.25,
    "mr_vol_threshold": 1.2,
    "mom_vol_threshold": 1.3,
    "mom_ret5d_threshold": 1.0,
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
_MIN_TRAIN_TRADES = 5


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    df = df.copy()
    df["rsi"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close)
    df["macd_diff"] = macd.macd_diff()
    df["ema9"] = EMAIndicator(close=close, window=9).ema_indicator()
    df["ema21"] = EMAIndicator(close=close, window=21).ema_indicator()
    bb = BollingerBands(close=close, window=20)
    df["bb_pct"] = bb.bollinger_pband()
    df["vol_ratio"] = volume / volume.rolling(20).mean()
    df["ret_5d"] = close.pct_change(5) * 100
    return df.dropna()


def _entry_signal(row: pd.Series, params: dict | None = None) -> str | None:
    """
    Simplified entry rules (proxy for Claude's analysis):
    - Mean reversion: RSI < rsi_threshold AND bb_pct < bb_threshold AND vol_ratio > mr_vol_threshold
    - Momentum: ema9 > ema21 AND macd_diff > 0 AND ret_5d > mom_ret5d_threshold AND vol_ratio > mom_vol_threshold
    """
    p = _DEFAULT_PARAMS if params is None else {**_DEFAULT_PARAMS, **params}
    if row["rsi"] < p["rsi_threshold"] and row["bb_pct"] < p["bb_threshold"] and row["vol_ratio"] > p["mr_vol_threshold"]:
        return "mean_reversion"
    if (row["ema9"] > row["ema21"] and row["macd_diff"] > 0
            and row["ret_5d"] > p["mom_ret5d_threshold"] and row["vol_ratio"] > p["mom_vol_threshold"]):
        return "momentum"
    return None


def _run_simulation(
    indicators: dict[str, pd.DataFrame],
    trading_dates: pd.DatetimeIndex,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    params: dict | None = None,
) -> dict:
    """Core trading simulation on pre-computed indicators. Called by both run_backtest
    and run_walk_forward_optimized (the latter avoids re-downloading data per param combo)."""
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
                px = float(indicators[sym].loc[today, "Close"]) if today in indicators[sym].index else pos["entry_price"]
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
            pnl_pct = (px / pos["entry_price"] - 1)
            trading_days_held = sum(1 for _ in pd.bdate_range(pos["entry_date"], today)) - 1

            reason = None
            if pnl_pct <= -STOP_LOSS_PCT:
                reason = "stop_loss"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                reason = "take_profit"
            elif trading_days_held >= max_hold_days:
                reason = "time_exit"

            if reason:
                cash += pos["shares"] * px
                trades.append({
                    "date": today_str, "symbol": sym, "action": "SELL",
                    "reason": reason, "entry_price": pos["entry_price"],
                    "exit_price": px, "pnl_pct": round(pnl_pct * 100, 2),
                    "signal": pos["signal"],
                })
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
            signal = _entry_signal(prev_row, params)
            if signal:
                candidates.append((sym, signal, float(prev_row["rsi"])))

        candidates.sort(key=lambda x: x[2])
        for sym, signal, _ in candidates[:slots]:
            try:
                px = float(indicators[sym].loc[today, "Close"])
                notional = (cash / slots) * 0.9
                shares = notional / px
                cost = shares * px
                if cost > cash or cost < 0.5:
                    continue
                cash -= cost
                positions[sym] = {
                    "entry_price": px, "entry_date": today,
                    "shares": shares, "signal": signal,
                }
                trades.append({
                    "date": today_str, "symbol": sym, "action": "BUY",
                    "price": px, "signal": signal,
                })
            except Exception:
                continue

    # Close remaining positions at end of window
    for sym, pos in positions.items():
        try:
            px = float(indicators[sym].iloc[-1]["Close"])
            cash += pos["shares"] * px
            pnl_pct = (px / pos["entry_price"] - 1) * 100
            trades.append({
                "date": "end", "symbol": sym, "action": "SELL",
                "reason": "end_of_backtest", "pnl_pct": round(pnl_pct, 2),
                "signal": pos["signal"],
            })
        except Exception:
            cash += pos["shares"] * pos["entry_price"]

    # Compute metrics
    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100
    closed_trades = [t for t in trades if t["action"] == "SELL" and "pnl_pct" in t]
    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    avg_return = sum(t["pnl_pct"] for t in closed_trades) / len(closed_trades) if closed_trades else 0

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
    sharpe = float(daily_rets.mean() / daily_rets.std() * (252 ** 0.5)) if daily_rets.std() > 0 else 0.0

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
    }


def run_backtest(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    params: dict | None = None,
) -> dict:

    logger.info(f"Backtest: {start_date} → {end_date} | {len(symbols)} symbols | ${initial_capital:.0f} capital")

    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    close_all = raw["Close"]
    volume_all = raw["Volume"]

    indicators = {}
    for sym in symbols:
        try:
            df = pd.DataFrame({"Close": close_all[sym], "Volume": volume_all[sym]}).dropna()
            df = _compute_indicators(df)
            indicators[sym] = df
        except Exception:
            pass

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    results = _run_simulation(indicators, trading_dates, initial_capital, max_positions, max_hold_days, params)
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

    # Download once — indicators are computed once and reused across all folds and combos
    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched for walk-forward")
        return {}

    close_all = raw["Close"]
    volume_all = raw["Volume"]

    indicators = {}
    for sym in symbols:
        try:
            df = pd.DataFrame({"Close": close_all[sym], "Volume": volume_all[sym]}).dropna()
            df = _compute_indicators(df)
            indicators[sym] = df
        except Exception:
            pass

    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    if len(trading_dates) < train_days + test_days:
        logger.warning(
            f"Date range too short: {len(trading_dates)} bdays available, "
            f"{train_days + test_days} required for one fold"
        )
        return {}

    # Generate non-overlapping test windows (train windows can overlap)
    folds_meta = []
    i = 0
    while i + train_days + test_days <= len(trading_dates):
        folds_meta.append({
            "train_start": trading_dates[i].strftime("%Y-%m-%d"),
            "train_end": trading_dates[i + train_days - 1].strftime("%Y-%m-%d"),
            "test_start": trading_dates[i + train_days].strftime("%Y-%m-%d"),
            "test_end": trading_dates[i + train_days + test_days - 1].strftime("%Y-%m-%d"),
            "train_slice": slice(i, i + train_days),
            "test_slice": slice(i + train_days, i + train_days + test_days),
        })
        i += test_days

    logger.info(f"Walk-forward: {len(folds_meta)} folds")

    fold_results = []
    for fold in folds_meta:
        train_dates = trading_dates[fold["train_slice"]]
        test_dates = trading_dates[fold["test_slice"]]

        # Grid search: find params with best Sharpe on the train window
        best_params = _DEFAULT_PARAMS.copy()
        best_score = -float("inf")

        for combo in all_combos:
            r = _run_simulation(indicators, train_dates, initial_capital, max_positions, max_hold_days, combo)
            if r["total_trades"] < _MIN_TRAIN_TRADES:
                continue
            if r["sharpe_ratio"] > best_score:
                best_score = r["sharpe_ratio"]
                best_params = combo

        # OOS test — test window data was never seen during param selection
        oos = _run_simulation(indicators, test_dates, initial_capital, max_positions, max_hold_days, best_params)

        fold_results.append({
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "best_params": best_params,
            "train_sharpe": round(best_score, 2) if best_score != -float("inf") else 0.0,
            "oos_total_return_pct": oos["total_return_pct"],
            "oos_win_rate_pct": oos["win_rate_pct"],
            "oos_total_trades": oos["total_trades"],
            "oos_sharpe": oos["sharpe_ratio"],
        })

        logger.info(
            f"  Fold {fold['test_start']}–{fold['test_end']}: "
            f"params={best_params} | OOS return={oos['total_return_pct']:+.1f}% "
            f"WR={oos['win_rate_pct']:.0f}% trades={oos['total_trades']}"
        )

    if not fold_results:
        return {"folds": [], "summary": {}}

    n = len(fold_results)
    profitable = sum(1 for f in fold_results if f["oos_total_return_pct"] > 0)
    summary = {
        "n_folds": n,
        "mean_oos_return_pct": round(sum(f["oos_total_return_pct"] for f in fold_results) / n, 2),
        "mean_oos_win_rate_pct": round(sum(f["oos_win_rate_pct"] for f in fold_results) / n, 1),
        "mean_oos_sharpe": round(sum(f["oos_sharpe"] for f in fold_results) / n, 2),
        "profitable_folds": profitable,
        "consistency_pct": round(profitable / n * 100, 1),
    }

    logger.info(
        f"Walk-forward summary: {n} folds | mean OOS return {summary['mean_oos_return_pct']:+.2f}% "
        f"| consistency {summary['consistency_pct']:.0f}%"
    )

    return {"folds": fold_results, "summary": summary}


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
    print("\n" + "=" * 50)
    print(f"  BACKTEST RESULTS  {r['start']} → {r['end']}")
    print("=" * 50)
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
        print(f"    {sig:<20} {total:>3} trades  WR {wr:.0f}%  avg {avg:+.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--capital", type=float, default=25000.0)
    args = parser.parse_args()
    run_backtest(STOCK_UNIVERSE, args.start, args.end, initial_capital=args.capital)
