"""Isolation event-study of retirement candidates + the lottery-day gate (fast substitute for the
full engine ablation, which is impractically slow at universe scale).

Fable flagged golden_cross and macd_crossover as likely wrong-horizon / subsumed. Rather than the
N+1 engine ablation (hours at 907 names), this measures each signal's *standalone* forward edge the
same way the new-signal studies did — the isolation question is simply "does it have edge alone?".
Also tests the MAX / lottery-day gate (Bali-Cakici-Whitelaw): does a >=+10% single-day pop predict
UNDER-performance over the next few days (which would justify blocking momentum entries after one)?

  golden_cross   SMA50 crosses above SMA200 today + vol >= 0.8x 20d avg           (hold 5)
  macd_crossover MACD(12,26) crosses above signal(9) today + vol > 20d avg        (hold 4)
  lottery_max    single-day return >= +10% (the pop day)                           (hold 3; expect <0)

Full ~9y OHLCV, winsorised excess vs SPY, cost sweep, vs universe baseline. No look-ahead
(triggers use <=t data; entry t+1). Retire a signal if its standalone edge is <=0 / not > baseline.

Usage: python scripts/existing_signal_isolation.py [--limit N] [--costs 0,4,7,10,14]
       [--winsor 25] [--start 2015-01-01]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf  # noqa: E402

from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _sweep(label: str, rows: list[tuple[float, int]], costs_bps: list[float]) -> None:
    if not rows:
        print(f"  {label:28} (no trades)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:28} n={n:6}  gross={gross:+.3f}%  break-even={gross * 100.0:+.1f}bps rt")
    for c in costs_bps:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def _ema(df, span):
    return df.ewm(span=span, adjust=False).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description="Retirement-candidate isolation + lottery gate")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--costs", default="0,4,7,10,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    w = args.winsor

    universe = [to_yf_symbol(s) for s in list(_UNIVERSE)[: args.limit]]
    print(f"isolation: {len(universe)} names, winsor +/-{w}%, from {args.start}")
    syms = sorted(set(universe) | {"SPY"})
    raw = yf.download(syms, start=args.start, auto_adjust=True, progress=False)
    close = raw["Close"].dropna(how="all")
    idx = close.index
    names = [c for c in close.columns if c != "SPY"]
    spy = close["SPY"]
    vol = raw["Volume"][names].reindex(idx)
    close = close[names]
    print(f"Loaded {len(names)} names over {len(idx)} sessions ({idx[0].date()}+)")

    ret = close.pct_change()
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    volavg = vol.rolling(20).mean()
    gc = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1)) & (vol >= 0.8 * volavg)
    macd = _ema(close, 12) - _ema(close, 26)
    macd_sig = _ema(macd, 9)
    mx = (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1)) & (vol > volavg)
    lott = ret >= 0.10

    def _fwd(h: int):
        f = close.shift(-1 - h) / close.shift(-1) - 1.0
        sf = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(idx)
        return (f.sub(sf, axis=0) * 100.0).clip(-w, w)

    fwd = {3: _fwd(3), 4: _fwd(4), 5: _fwd(5)}
    triggers = {
        "golden_cross (hold5)": (gc, 5),
        "macd_crossover (hold4)": (mx, 4),
        "lottery_max +10% (hold3)": (lott, 3),
    }
    out: dict[str, list[tuple[float, int]]] = {k: [] for k in triggers}
    base: dict[int, list[tuple[float, int]]] = {3: [], 4: [], 5: []}
    for i in range(210, len(idx) - 6):
        yr = idx[i].year
        for h in (3, 4, 5):
            frow = fwd[h].iloc[i]
            for sym in names:
                ex = frow.get(sym)
                if ex is not None and not (isinstance(ex, float) and math.isnan(ex)):
                    base[h].append((float(ex), yr))
        for label, (trig, h) in triggers.items():
            trow, frow = trig.iloc[i], fwd[h].iloc[i]
            for sym in names:
                if bool(trow.get(sym, False)):
                    ex = frow.get(sym)
                    if ex is not None and not (isinstance(ex, float) and math.isnan(ex)):
                        out[label].append((float(ex), yr))

    print("\n=== Retirement isolation + lottery gate — winsorised excess vs SPY, COST SWEEP ===")
    print(
        "  ('gross' = pre-cost; break-even = round-trip cost the edge covers; * = net>0 & |t|>=2)"
    )
    _sweep("universe baseline (hold3)", base[3], costs)
    _sweep("universe baseline (hold5)", base[5], costs)
    for label in triggers:
        _sweep(label, out[label], costs)
    print(
        "\n  RETIRE golden_cross / macd_crossover if standalone edge <=0 or not clearly > baseline. "
        "LOTTERY gate is justified if lottery_max is negative (a +10% pop precedes under-performance)."
    )


if __name__ == "__main__":
    main()
