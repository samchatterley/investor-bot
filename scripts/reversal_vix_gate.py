"""De-contaminate the vol-regime finding — is the reversal vol-gate tradable with a CAUSAL threshold?

reversal_enhance.py found reversal (me5<=-7, hold 3) is almost entirely a high-vol phenomenon:
high SPY-vol +0.455% net/3d @7bps (robust train+test), low SPY-vol +0.040% (dead). BUT that split
used a FULL-SAMPLE median of realized vol = look-ahead. A tradable gate must use only information
available at decision time. This script re-tests the conditioning with VIX (the bot can read it live,
zero look-ahead) using causal thresholds, split train(<2021)/test(2021+):

  - fixed VIX buckets: <15 / 15-20 / 20-30 / >30  (absolute, trivially causal)
  - VIX vs its TRAILING 252d median (causal, adaptive) -> the shippable gate arm

Ships the gate only if 'VIX >= trailing median' (or a fixed level) clearly beats the low-VIX arm net
@7bps in BOTH train and test, so the doubling survives out-of-sample AND is implementable live.

Usage: python scripts/reversal_vix_gate.py [--gate -7] [--hold 3] [--costs 0,7,14] [--start 2015-01-01]
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

_SPLIT = 2021


def _report(label: str, rows: list[tuple[float, int]], cost: float) -> None:
    if not rows:
        print(f"  {label:26} (no events)")
        return
    allv = [v for v, _ in rows]
    trv = [v for v, y in rows if y < _SPLIT]
    tev = [v for v, y in rows if y >= _SPLIT]
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, y in rows:
        by_year[y].append(v)

    def st3(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        net = statistics.mean(vals) - cost / 100.0
        se = (statistics.stdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return net, (net / se if se else 0.0)

    na, ta = st3(allv)
    ntr, ttr = st3(trv)
    nte, tte = st3(tev)
    pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - cost / 100.0) > 0)
    print(
        f"  {label:26} n={len(allv):6}  net@{cost:.0f}={na:+.3f}% t={ta:+.1f}  "
        f"| train={ntr:+.3f}%(t{ttr:+.1f}) test={nte:+.3f}%(t{tte:+.1f})  +yrs={pos}/{len(by_year)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="reversal VIX-gate (causal) study")
    ap.add_argument("--gate", type=float, default=-7.0)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    cost = 7.0 if 7.0 in costs else costs[min(1, len(costs) - 1)]
    w, gate, h = args.winsor, args.gate, args.hold

    universe = [to_yf_symbol(s) for s in _UNIVERSE]
    syms = sorted(set(universe) | {"SPY", "^VIX"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"].dropna(
        how="all"
    )
    spy = px["SPY"]
    vix = px["^VIX"]
    cols = [c for c in px.columns if c not in ("SPY", "^VIX")]
    close = px[cols]
    print(f"prices: {len(cols)} names, {len(close)} sessions ({close.index[0].date()}+)")

    me5 = ((close / close.shift(5) - 1.0) * 100.0).sub((spy / spy.shift(5) - 1.0) * 100.0, axis=0)
    f = close.shift(-1 - h) / close.shift(-1) - 1.0
    sf = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (f.sub(sf, axis=0) * 100.0).clip(-w, w)
    vix_med = vix.rolling(252, min_periods=60).median()  # CAUSAL trailing median

    fixed: dict[str, list[tuple[float, int]]] = defaultdict(list)
    gatearm: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i in range(25, len(close) - h - 2, 5):
        yr = close.index[i].year
        v = vix.iloc[i]
        vm = vix_med.iloc[i]
        if v is None or math.isnan(v):
            continue
        if v < 15:
            fb = "VIX <15"
        elif v < 20:
            fb = "VIX 15-20"
        elif v < 30:
            fb = "VIX 20-30"
        else:
            fb = "VIX >30"
        above = (not math.isnan(vm)) and v >= vm
        me_row, fx = me5.iloc[i], fwd_excess.iloc[i]
        for sym in cols:
            mv = me_row.get(sym)
            if mv is None or (isinstance(mv, float) and math.isnan(mv)) or float(mv) > gate:
                continue
            ex = fx.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            exf = float(ex)
            fixed[fb].append((exf, yr))
            if not math.isnan(vm):
                gatearm["VIX >= trail-median" if above else "VIX < trail-median"].append((exf, yr))

    print(
        f"\n=== reversal VIX-gate (me5<={gate:.0f}, hold {h}, t+1, excess vs SPY) — net @{cost:.0f}bps ==="
    )
    print("\nFIXED VIX buckets (absolute, causal):")
    for b in ("VIX <15", "VIX 15-20", "VIX 20-30", "VIX >30"):
        _report(b, fixed[b], cost)
    print("\nCAUSAL trailing-252d-median gate (the shippable arm):")
    for b in ("VIX >= trail-median", "VIX < trail-median"):
        _report(b, gatearm[b], cost)
    print(
        "\n  SHIPS the gate if 'VIX >= trail-median' clearly beats 'VIX < trail-median' net @7bps in "
        "BOTH train & test (the vol-doubling is real AND causal/tradable)."
    )


if __name__ == "__main__":
    main()
