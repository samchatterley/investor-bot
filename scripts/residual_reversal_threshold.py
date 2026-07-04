"""N1 wiring validation — does the SHIPPABLE condition (market-excess 5d return <= fixed threshold)
reproduce the decile edge?

The decile study (residual_reversal_backtest.py) confirmed idiosyncratic 5d losers revert. But a
cross-sectional decile needs heavy dual-path rank plumbing to ship. This tests the far simpler live
condition that needs NO new data plumbing — both ret_5d and spy_ret_5d already flow into
evaluate_signals — using market-excess 5d return (beta=1 residual) against a fixed threshold:

    me5 = (close_t/close_{t-5} - 1) - (spy_t/spy_{t-5} - 1)      # market-excess 5d return
    SIGNAL fires when me5 <= threshold

Full ~9y OHLCV, weekly rebalance, winsorised excess vs SPY, cost sweep. If a threshold around the
decile boundary (~-6 to -9%) stays net>0 & significant at ~4-7bps, we wire that exact condition.

Usage: python scripts/residual_reversal_threshold.py [--limit N] [--hold 3]
       [--thresholds -5,-7,-9,-12] [--costs 0,4,7,10,14] [--winsor 25] [--start 2015-01-01]
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
        print(f"  {label:26} (no trades)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:26} n={n:6}  gross={gross:+.3f}%  break-even={gross * 100.0:+.1f}bps rt")
    for c in costs_bps:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="N1 fixed-threshold market-excess validation")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--thresholds", default="-5,-7,-9,-12", help="me5 <= t (percent)")
    ap.add_argument("--costs", default="0,4,7,10,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--rebalance", type=int, default=5)
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    w, h = args.winsor, args.hold

    universe = [to_yf_symbol(s) for s in list(_UNIVERSE)[: args.limit]]
    print(f"N1 threshold validation: {len(universe)} names, hold {h}d, thresholds {thresholds}")
    syms = sorted(set(universe) | {"SPY"})
    close = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"].dropna(
        how="all"
    )
    spy = close["SPY"]
    cols = [c for c in close.columns if c != "SPY"]
    close = close[cols]
    print(f"Loaded {len(cols)} names over {len(close)} sessions ({close.index[0].date()}+)")

    spy5 = spy / spy.shift(5) - 1.0
    me5 = (close / close.shift(5) - 1.0).sub(spy5, axis=0) * 100.0  # market-excess 5d return (%)
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-w, w)

    groups: dict[float, list[tuple[float, int]]] = {t: [] for t in thresholds}
    allrows: list[tuple[float, int]] = []
    for i in range(60, len(close) - h - 2, args.rebalance):
        yr = close.index[i].year
        mrow, frow = me5.iloc[i], fwd_excess.iloc[i]
        for sym in cols:
            ex, mv = frow.get(sym), mrow.get(sym)
            if ex is None or mv is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            if isinstance(mv, float) and math.isnan(mv):
                continue
            allrows.append((float(ex), yr))
            for t in thresholds:
                if mv <= t:
                    groups[t].append((float(ex), yr))

    print(f"\n=== N1 market-excess threshold — hold {h}d, winsorised excess vs SPY, COST SWEEP ===")
    print(
        "  ('gross' = pre-cost; break-even = round-trip cost the edge covers; * = net>0 & |t|>=2)"
    )
    _sweep("universe (all)", allrows, costs)
    for t in thresholds:
        _sweep(f"me5 <= {t:+.0f}% (SIGNAL)", groups[t], costs)
    print(
        "\n  Wire the threshold that stays net>0 & |t|>=2 at ~4-7bps with the best +yrs and a usable "
        "fire rate (n/12y)."
    )


if __name__ == "__main__":
    main()
