"""
Rule-based backtester — validates technical signal quality on historical data
without calling Claude (avoids API cost).

Usage:
    python backtester.py --start 2024-01-01 --end 2024-12-31
    python backtester.py --start 2025-01-01 --end 2025-12-31 --capital 25000
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from config import STOCK_UNIVERSE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, LOG_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


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


def _entry_signal(row: pd.Series) -> Optional[str]:
    """
    Simplified entry rules (proxy for Claude's analysis):
    - Mean reversion: RSI < 35 AND bb_pct < 0.25 AND vol_ratio > 1.2
    - Momentum: ema9 > ema21 AND macd_diff > 0 AND ret_5d > 1 AND vol_ratio > 1.3
    """
    if row["rsi"] < 35 and row["bb_pct"] < 0.25 and row["vol_ratio"] > 1.2:
        return "mean_reversion"
    if (row["ema9"] > row["ema21"] and row["macd_diff"] > 0
            and row["ret_5d"] > 1.0 and row["vol_ratio"] > 1.3):
        return "momentum"
    return None


def run_backtest(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
) -> dict:

    logger.info(f"Backtest: {start_date} → {end_date} | {len(symbols)} symbols | ${initial_capital:.0f} capital")

    # Fetch historical data (extra buffer for indicator warmup)
    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    raw = yf.download(symbols, start=fetch_start, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("No data fetched")
        return {}

    close_all = raw["Close"]
    volume_all = raw["Volume"]

    # Compute indicators per symbol
    indicators = {}
    for sym in symbols:
        try:
            df = pd.DataFrame({"Close": close_all[sym], "Volume": volume_all[sym]}).dropna()
            df = _compute_indicators(df)
            indicators[sym] = df
        except Exception:
            pass

    # Simulate trading
    cash = initial_capital
    positions: dict[str, dict] = {}   # {symbol: {entry_price, entry_date, signal}}
    trades: list[dict] = []
    equity_curve: list[tuple[str, float]] = []

    trading_dates = pd.bdate_range(start=start_date, end=end_date)

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
            trading_days_held = sum(1 for d in pd.bdate_range(pos["entry_date"], today)) - 1

            reason = None
            if pnl_pct <= -STOP_LOSS_PCT:
                reason = "stop_loss"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                reason = "take_profit"
            elif trading_days_held >= max_hold_days:
                reason = "time_exit"

            if reason:
                proceeds = pos["shares"] * px
                cash += proceeds
                trades.append({
                    "date": today_str,
                    "symbol": sym,
                    "action": "SELL",
                    "reason": reason,
                    "entry_price": pos["entry_price"],
                    "exit_price": px,
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "signal": pos["signal"],
                })
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        # Look for entries
        slots = max_positions - len(positions)
        if slots <= 0:
            continue

        candidates = []
        for sym, df in indicators.items():
            if sym in positions or today not in df.index:
                continue
            row = df.loc[today]
            signal = _entry_signal(row)
            if signal:
                candidates.append((sym, signal, float(row["rsi"])))

        # Sort by RSI ascending (most oversold first for mean reversion)
        candidates.sort(key=lambda x: x[2])
        for sym, signal, _ in candidates[:slots]:
            try:
                px = float(indicators[sym].loc[today, "Close"])
                notional = (cash / (slots or 1)) * 0.9   # deploy 90% of cash per slot
                shares = notional / px
                cost = shares * px
                if cost > cash or cost < 0.5:
                    continue
                cash -= cost
                positions[sym] = {
                    "entry_price": px,
                    "entry_date": today,
                    "shares": shares,
                    "signal": signal,
                }
                trades.append({
                    "date": today_str,
                    "symbol": sym,
                    "action": "BUY",
                    "price": px,
                    "signal": signal,
                })
            except Exception:
                continue

    # Close remaining positions at end
    final_date = trading_dates[-1]
    for sym, pos in positions.items():
        try:
            px = float(indicators[sym].iloc[-1]["Close"])
            cash += pos["shares"] * px
            pnl_pct = (px / pos["entry_price"] - 1) * 100
            trades.append({"date": "end", "symbol": sym, "action": "SELL", "reason": "end_of_backtest", "pnl_pct": round(pnl_pct, 2), "signal": pos["signal"]})
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
    peak = eq_values[0]
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

    # Sharpe ratio (annualised, daily returns, risk-free ≈ 0)
    eq_values = [v for _, v in equity_curve]
    daily_rets = pd.Series(eq_values).pct_change().dropna()
    sharpe = float(daily_rets.mean() / daily_rets.std() * (252 ** 0.5)) if daily_rets.std() > 0 else 0.0

    results = {
        "start": start_date,
        "end": end_date,
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

    _print_results(results)
    _save_results(results)
    return results


def _save_results(r: dict):
    """Persist latest backtest results for the dashboard to read."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, "backtest_results.json")
        # equity_curve contains datetime objects — convert to strings
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
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=25.0)
    args = parser.parse_args()
    run_backtest(STOCK_UNIVERSE, args.start, args.end, initial_capital=args.capital)
