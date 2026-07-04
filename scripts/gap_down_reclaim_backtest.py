"""N7 gap_down_reclaim backtest — seller exhaustion in a quality name.

Fable new signal #7, the long mirror of the one working short (post_earnings_gapdown_failed_bounce):
a non-earnings gap-down in a quality name that *reclaims* the pre-gap close within 2 sessions was
supply-driven, not informational — the reclaim marks exhaustion, and modest continuation follows.

Event study over ~9y OHLCV.

Entry:
  gap_down_g = open_g / close_{g-1} - 1 <= -3%          (a real gap, not drift)
  quality: close_{g-1} > SMA200_{g-1}                    (was in an uptrend pre-gap; not a falling knife)
  reclaim: close_g > close_{g-1}  (enter close_g)  OR  close_{g+1} > close_{g-1}  (enter close_{g+1})
           — the first session that reclaims the pre-gap close; if neither reclaims, no trade
Exit: hold `--hold` sessions from the reclaim close. Winsorised excess vs SPY, cost sweep vs the
universe baseline. No look-ahead (entry is the reclaim close; forward uses only later closes).

CAVEAT: no earnings filter over 9y (yfinance ~2-3y only), so some gaps are earnings gaps — the
SMA200 quality gate removes the worst, but treat as a slight upper bound. Overlaps candle_exhaustion
(both are exhaustion longs); keep at most one if both clear.

Usage: python scripts/gap_down_reclaim_backtest.py [--limit N] [--hold 3] [--gap 3.0]
       [--costs 0,4,7,10,14] [--winsor 25] [--start 2015-01-01]
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
    """rows = [(pre_cost_excess_%, year), ...]. Print N + break-even + net mean/t at each cost."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="N7 gap-down reclaim event study")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--gap", type=float, default=3.0, help="min gap-down %")
    ap.add_argument("--costs", default="0,4,7,10,14", help="round-trip cost sweep (bps)")
    ap.add_argument("--winsor", type=float, default=25.0, help="cap |forward excess| at this %")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]

    universe = [to_yf_symbol(s) for s in list(_UNIVERSE)[: args.limit]]
    print(f"gap-down reclaim: {len(universe)} names, gap<=-{args.gap}%, hold {args.hold}d")
    syms = sorted(set(universe) | {"SPY"})
    raw = yf.download(syms, start=args.start, auto_adjust=True, progress=False)
    close = raw["Close"].dropna(how="all")
    idx = close.index
    names = [c for c in close.columns if c != "SPY"]
    spy_vals = close["SPY"].to_numpy()
    op = raw["Open"][names].reindex(idx)
    close_df = close[names]
    print(f"Loaded {len(names)} names over {len(idx)} sessions ({idx[0].date()}+)")

    gap = (op / close_df.shift(1) - 1.0) <= (-args.gap / 100.0)
    sma200 = close_df.rolling(200).mean()
    quality = close_df.shift(1) > sma200.shift(1)
    years = [d.year for d in idx]
    h, w = args.hold, args.winsor
    n = len(idx)

    events: list[tuple[float, int]] = []
    baseline: list[tuple[float, int]] = []
    for sym in names:
        cvals = close_df[sym].to_numpy()
        gcol = gap[sym].to_numpy()
        qcol = quality[sym].to_numpy()
        for g in range(200, n - h - 2):
            c_prev = cvals[g - 1]
            # Universe baseline: every valid name-day's forward excess (entry g+1).
            if not math.isnan(cvals[g]) and not math.isnan(cvals[g + 1 + h]):
                ex = (cvals[g + 1 + h] / cvals[g + 1] - 1.0) - (
                    spy_vals[g + 1 + h] / spy_vals[g + 1] - 1.0
                )
                baseline.append((max(-w, min(w, ex * 100.0)), years[g]))
            if not (gcol[g] and qcol[g]) or math.isnan(c_prev):
                continue
            # First reclaim of the pre-gap close within 2 sessions → entry there.
            entry = g if cvals[g] > c_prev else (g + 1 if cvals[g + 1] > c_prev else -1)
            if entry < 0 or entry + h >= n or math.isnan(cvals[entry + h]):
                continue
            ex = (cvals[entry + h] / cvals[entry] - 1.0) - (
                spy_vals[entry + h] / spy_vals[entry] - 1.0
            )
            events.append((max(-w, min(w, ex * 100.0)), years[g]))

    print(f"\n=== N7 gap-down reclaim — hold {h}d, winsorised excess vs SPY, COST SWEEP ===")
    print(
        "  ('gross' = pre-cost; break-even = round-trip cost the edge covers; * = net>0 & |t|>=2)"
    )
    _sweep("universe baseline", baseline, costs)
    _sweep("gap-down reclaim (SIGNAL)", events, costs)
    print(
        "\n  Ships if the signal stays net>0 with |t|>=2 at ~4-9bps AND beats the universe baseline."
    )


if __name__ == "__main__":
    main()
