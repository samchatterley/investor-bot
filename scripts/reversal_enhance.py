"""Deepen the one durable edge — reversal (N1) enhancement study, walk-forward guarded.

The whole v2 workshop established that short-horizon mean-reversion (N1: own market-excess 5d return
me5 <= -7%, hold ~3d, ~+0.28% net/3d @7bps, 10-12/12 yrs) is the ONLY robust edge in this liquid
~906-name universe; new signal families are arbitraged, beta, or undeployable. So the highest-EV move
is to REFINE this edge, not replace it — but carefully, because curve-fitting the one thing that works
would degrade it. Every cut below is reported train(<=2020) / test(2021+) so conclusions rest on
out-of-sample CONSISTENCY, not in-sample point estimates.

Four questions, all at entry t+1, winsorised excess vs SPY, cost-swept, weekly step (reduces overlap):
  A) CONVICTION  — is net reversal monotonic in loss depth? buckets of me5 (hold 3). If deeper losers
     revert harder net of cost, that justifies conviction-sizing.
  B) HOLD        — where does net IR peak? me5<=-7 held 1/2/3/5d.
  C) LIQUIDITY   — does restricting to liquid names help? me5<=-7, split by 20d dollar-volume median.
     Prior: YES (reversal is liquidity provision; illiquid = falling-knife risk).
  D) VOL REGIME  — does reversal pay more in high-vol? me5<=-7, split by SPY 20d realized-vol median.

No parameter is tuned on the test window; the base -7 gate and buckets are fixed a priori. A refinement
only "wins" if it improves net @7bps AND holds up in BOTH train and test.

FINDINGS (net @7bps, train<2021 / test>=2021):
  D) VOL REGIME is the headline — reversal is almost entirely a HIGH-VOL phenomenon. High SPY-vol
     +0.455% (train +0.38 / test +0.51, both robust); low SPY-vol +0.040% (~0, dead in BOTH windows).
     Gating on vol ~DOUBLES the edge and kills the dead half. NB the split here used a full-sample
     median (look-ahead) -> re-confirmed causally with VIX in reversal_vix_gate.py before shipping.
  A) CONVICTION real but BOUNDED: [-15,-10) +0.28%, [-25,-15) +0.94% (train +0.99 / test +0.90, robust).
     But the extremes fail walk-forward: shallow [-10,-7] is train-NEGATIVE (-0.13%), and <-25 is a
     train-only artifact (+6.5% train -> +0.26% test, n=389 falling knives). Edge lives in me5∈[-25,-10];
     the -7 gate admits an unreliable shallow bucket and the deepest zone must NOT be oversized.
  C) LIQUIDITY confirms prior: liquid +0.375% (robust both) > illiquid +0.227% (train weak, t1.6).
  B) HOLD: 1d (+0.289%, train≈test) and 3d most robust; 5d higher but train-heavy (decaying).

Usage: python scripts/reversal_enhance.py [--gate -7] [--costs 0,7,14] [--winsor 25] [--start 2015-01-01]
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

_SPLIT_YEAR = 2021  # train < 2021, test >= 2021


def _stats(vals: list[float], costbps: float) -> tuple[int, float, float]:
    n = len(vals)
    if n == 0:
        return 0, 0.0, 0.0
    net = statistics.mean(vals) - costbps / 100.0
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    t = net / se if se else 0.0
    return n, net, t


def _report(label: str, rows: list[tuple[float, int]], cost: float) -> None:
    if not rows:
        print(f"  {label:30} (no events)")
        return
    allv = [v for v, _ in rows]
    trainv = [v for v, y in rows if y < _SPLIT_YEAR]
    testv = [v for v, y in rows if y >= _SPLIT_YEAR]
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, y in rows:
        by_year[y].append(v)
    pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - cost / 100.0) > 0)
    na, neta, ta = _stats(allv, cost)
    _, nettr, ttr = _stats(trainv, cost)
    _, nette, tte = _stats(testv, cost)
    print(
        f"  {label:30} n={na:6}  net@{cost:.0f}={neta:+.3f}% t={ta:+.1f}  "
        f"| train={nettr:+.3f}%(t{ttr:+.1f}) test={nette:+.3f}%(t{tte:+.1f})  +yrs={pos}/{len(by_year)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="reversal (N1) enhancement study")
    ap.add_argument("--gate", type=float, default=-7.0, help="N1 me5 trigger (%)")
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    cost = 7.0 if 7.0 in costs else costs[min(1, len(costs) - 1)]
    w, gate = args.winsor, args.gate
    holds = [1, 2, 3, 5]

    universe = [to_yf_symbol(s) for s in _UNIVERSE]
    syms = sorted(set(universe) | {"SPY"})
    raw = yf.download(syms, start=args.start, auto_adjust=True, progress=False)
    closeall = raw["Close"].dropna(how="all")
    volall = raw["Volume"].reindex(closeall.index)
    spy = closeall["SPY"]
    cols = [c for c in closeall.columns if c != "SPY"]
    close = closeall[cols]
    vol = volall[cols]
    print(f"prices: {len(cols)} names, {len(close)} sessions ({close.index[0].date()}+)")

    me5 = ((close / close.shift(5) - 1.0) * 100.0).sub((spy / spy.shift(5) - 1.0) * 100.0, axis=0)
    dvol = (close * vol).rolling(20).mean()  # 20d avg dollar volume (liquidity proxy)
    spy_rvol = (spy / spy.shift(1) - 1.0).rolling(20).std()
    rvol_med = spy_rvol.median()

    fwd_excess = {}
    for h in holds:
        f = close.shift(-1 - h) / close.shift(-1) - 1.0
        sf = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
        fwd_excess[h] = (f.sub(sf, axis=0) * 100.0).clip(-w, w)

    conv: dict[str, list[tuple[float, int]]] = defaultdict(list)
    hold_arm: dict[int, list[tuple[float, int]]] = defaultdict(list)
    liq: dict[str, list[tuple[float, int]]] = defaultdict(list)
    volreg: dict[str, list[tuple[float, int]]] = defaultdict(list)
    conv_buckets = [(-10.0, "[-10,-7]"), (-15.0, "[-15,-10)"), (-25.0, "[-25,-15)"), (-1e9, "<-25")]

    for i in range(25, len(close) - max(holds) - 2, 5):
        yr = close.index[i].year
        me_row = me5.iloc[i]
        dv_row = dvol.iloc[i]
        dv_med = dv_row.median()
        hi_vol = bool(spy_rvol.iloc[i] >= rvol_med) if not math.isnan(spy_rvol.iloc[i]) else None
        f3 = fwd_excess[3].iloc[i]
        for sym in cols:
            mv = me_row.get(sym)
            if mv is None or (isinstance(mv, float) and math.isnan(mv)) or float(mv) > gate:
                continue
            mvf = float(mv)
            e3 = f3.get(sym)
            if e3 is not None and not (isinstance(e3, float) and math.isnan(e3)):
                e3f = float(e3)
                for lo, name in conv_buckets:
                    if mvf > lo:
                        conv[name].append((e3f, yr))
                        break
                dvv = dv_row.get(sym)
                if dvv is not None and not (isinstance(dvv, float) and math.isnan(dvv)):
                    liq["liquid (>med $vol)" if float(dvv) >= dv_med else "illiquid (<med)"].append(
                        (e3f, yr)
                    )
                if hi_vol is not None:
                    volreg["high SPY-vol" if hi_vol else "low SPY-vol"].append((e3f, yr))
            for h in holds:
                eh = fwd_excess[h].iloc[i].get(sym)
                if eh is not None and not (isinstance(eh, float) and math.isnan(eh)):
                    hold_arm[h].append((float(eh), yr))

    print(
        f"\n=== reversal enhancement (me5<={gate:.0f}%, t+1, winsor excess vs SPY) — net @{cost:.0f}bps ==="
    )
    print("\nA) CONVICTION (hold 3) — does deeper loss revert harder net of cost?")
    for _, name in conv_buckets:
        _report(name, conv[name], cost)
    print(f"\nB) HOLD PERIOD (me5<={gate:.0f}) — where does net peak?")
    for h in holds:
        _report(f"hold {h}d", hold_arm[h], cost)
    print("\nC) LIQUIDITY (hold 3) — does restricting to liquid names help?")
    for name in ("liquid (>med $vol)", "illiquid (<med)"):
        _report(name, liq[name], cost)
    print("\nD) VOL REGIME (hold 3) — does reversal pay more in high-vol?")
    for name in ("high SPY-vol", "low SPY-vol"):
        _report(name, volreg[name], cost)
    print(
        "\n  A refinement WINS only if it lifts net @7bps AND holds in BOTH train(<2021) & test(>=2021)."
    )


if __name__ == "__main__":
    main()
