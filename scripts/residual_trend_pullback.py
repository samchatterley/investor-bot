"""Workshop v2 #3 — residual_trend_pullback: buy the residual-momentum leader on a mild dip.

Blitz-Huij-Martens residual momentum (raw momentum's premium with half the crash risk): top-decile
trailing 12-1 idiosyncratic return continues. The mild-pullback entry (this week's market-excess 5d
return in [-3%, 0%]) buys that continuation at a local discount — and is DISJOINT from N1 by
construction (N1 fires at me5 <= -7%; this fires at me5 in [-3, 0]), so it can add momentum-family
P&L the book currently gets ~nothing from.

Signal (weekly rebalance, entry t+1, hold 5, market-excess = beta-1 residual, winsorised vs SPY):
  resid_mom = (close_{t-5}/close_{t-60} - 1) - (spy_{t-5}/spy_{t-60} - 1)   -> top decile = leaders
  me5       = (close_t/close_{t-5} - 1) - (spy_t/spy_{t-5} - 1)
Arms: leaders(all) / leaders & pullback me5 in [-3,0] (SIGNAL) / leaders & still-rising me5>0 (control).

Ships only if the pullback arm beats the unconditional-leader arm net at 7bps with |t|>=2 and
consistent +yrs; else it's just slow momentum, too weak for a 5d hold (the falsification).

Usage: python scripts/residual_trend_pullback.py [--hold 5] [--costs 0,7,14] [--winsor 25]
       [--start 2015-01-01] [--dip -3,0]
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


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float]) -> None:
    if not rows:
        print(f"  {label:30} (no events)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:30} n={n:6}  gross={gross:+.3f}%")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="residual_trend_pullback backtest (v2 #3)")
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--dip", default="-3,0", help="pullback me5 band low,high (%)")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    lo, hi = (float(x) for x in args.dip.split(","))
    w, h = args.winsor, args.hold

    universe = [to_yf_symbol(s) for s in _UNIVERSE]
    syms = sorted(set(universe) | {"SPY"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"].dropna(
        how="all"
    )
    spy = px["SPY"]
    cols = [c for c in px.columns if c != "SPY"]
    close = px[cols]
    print(f"prices: {len(cols)} names, {len(close)} sessions ({close.index[0].date()}+)")

    mom = (close.shift(5) / close.shift(60) - 1.0) * 100.0
    spymom = (spy.shift(5) / spy.shift(60) - 1.0) * 100.0
    resid_mom = mom.sub(spymom, axis=0)
    me5 = ((close / close.shift(5) - 1.0) * 100.0).sub((spy / spy.shift(5) - 1.0) * 100.0, axis=0)
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-w, w)

    arms: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i in range(65, len(close) - h - 2, 5):
        rm = resid_mom.iloc[i].dropna()
        if len(rm) < 100:
            continue
        yr = close.index[i].year
        top = rm.quantile(0.9)
        mrow, fx = me5.iloc[i], fwd_excess.iloc[i]
        for sym, r in rm.items():
            if r < top:
                continue
            ex = fx.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            arms["leaders (all)"].append((float(ex), yr))
            mv = mrow.get(sym)
            if mv is None or (isinstance(mv, float) and math.isnan(mv)):
                continue
            if lo <= mv <= hi:
                arms["leaders + pullback [-3,0] (SIGNAL)"].append((float(ex), yr))
            elif mv > hi:
                arms["leaders + still-rising (control)"].append((float(ex), yr))

    print(f"\n=== residual_trend_pullback — hold {h}d, t+1, winsorised excess vs SPY ===")
    for label in (
        "leaders (all)",
        "leaders + pullback [-3,0] (SIGNAL)",
        "leaders + still-rising (control)",
    ):
        _sweep(label, arms[label], costs)
    print(
        "\n  SHIPS only if the pullback arm beats 'leaders (all)' net @7bps with |t|>=2 and "
        "consistent +yrs; if ~equal, it's just slow momentum (too weak for a 5d hold) — kill."
    )


if __name__ == "__main__":
    main()
