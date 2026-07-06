"""Reconcile the VIX>=20 reversal finding with the LIVE regime blocks before shipping.

reversal_vix_gate.py showed reversal (me5<=-7, hold 3) is dead below VIX 20 and strong at/above it,
measured UNCONDITIONALLY. But the live evaluator already BLOCKS residual_reversal in STRESS_RISK_OFF
and HIGH_VOL_DOWNTREND (both require a downtrend/shock: HVD = VIX>=25 AND spy_5d<=-3%; STRESS = a
-1.8%/1d shock + VIX/drawdown convergence, or sustained -5%/5d + VIX>=30). So the raw VIX buckets
MIX blocked stress-days with the allowed non-downtrend high-VIX days. This script partitions me5<=-7
reversal into the actual live buckets to answer two things before any code change:

  A) VIX<20 (calm)                — currently ALLOWED  -> is it dead? (does a VIX>=20 floor help?)
  B) VIX>=20, NOT stress/HVD      — currently ALLOWED  -> the target: is this the real edge?
  C) HIGH_VOL_DOWNTREND           — currently BLOCKED  -> is the block removing NEGATIVE reversal?
  D) STRESS_RISK_OFF              — currently BLOCKED  -> same question

Decision: if B strongly beats A in BOTH train & test, ship a VIX>=20 floor on residual_reversal.
If C/D are negative, the existing blocks are validated (keep them). If C/D are positive, flag that
the downtrend block may be leaving money on the table (a separate follow-up).

FINDINGS (net @7bps, train<2021 / test>=2021) — this OVERTURNED the simple VIX-floor plan:
  A) VIX<20 calm    -0.011% (train -0.05 / test +0.02)  DEAD both windows — removable but ~0.
  B) VIX>=20 !stress +0.250% (train -0.075 / test +0.468) train-NEGATIVE — a VIX>=20 floor alone is
     NOT robust; the "elevated-not-stress" bucket only worked post-2021.
  C) HIGH_VOL_DOWNTREND +0.273% (train -1.21 / test +0.88, n=450) train strongly NEGATIVE -> block
     validated (falling knives); test flip is small-n noise. Keep blocked.
  D) STRESS_RISK_OFF +1.711% (train +1.91 / test +1.25, 6/7 yrs) — reversal's BIGGEST, most robust
     bucket (capitulation bounce), and it is currently BLOCKED. Unblocking = large alpha BUT:
       * SURVIVORSHIP: universe = current constituents; bankrupt falling-knives excluded, worst bias
         exactly in stress -> magnitude inflated.
       * WINSOR ±25% caps the crash tail -> understates real downside.
       * It means buying beaten-down names INTO crashes = a risk-posture change, not pure alpha.
  Live now (A+B, C+D blocked) = +0.094% net (train-negative) — mediocre; the block removed the good
  bucket. CONCLUSION: no clean autonomous ship. The one robust edge (D) is risk-encumbered ->
  escalate the risk-appetite decision to the user rather than flipping the crash protection.

Usage: python scripts/reversal_regime_reconcile.py [--gate -7] [--hold 3] [--start 2015-01-01]
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
# live RegimeThresholds (data/market_regime.py)
_VIX_HIGH_VOL, _SPY_5D_HVD = 25.0, -3.0
_SPY_BEAR_1D, _SPY_5D_STRESS, _DRAWDOWN_STRESS = -1.8, -5.0, -8.0
_VIX_STRESS_ABS, _VIX_SPIKE_RATIO, _VIX_5D_SURGE = 30.0, 1.4, 30.0
_VIX_FLOOR = 20.0


def _report(label: str, rows: list[tuple[float, int]], cost: float = 7.0) -> None:
    if not rows:
        print(f"  {label:34} (no events)")
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
        f"  {label:34} n={len(allv):6}  net@7={na:+.3f}% t={ta:+.1f}  "
        f"| train={ntr:+.3f}%(t{ttr:+.1f}) test={nte:+.3f}%(t{tte:+.1f})  +yrs={pos}/{len(by_year)}"
    )


def _is_stress(r1: float, r5: float, dd: float, vix: float, vix5: float, vixma: float) -> bool:
    return (
        (r1 <= _SPY_BEAR_1D and r5 <= _SPY_5D_STRESS)
        or (r1 <= _SPY_BEAR_1D and dd <= _DRAWDOWN_STRESS and vix >= _VIX_HIGH_VOL)
        or (r1 <= _SPY_BEAR_1D and vix >= _VIX_STRESS_ABS)
        or (r1 <= _SPY_BEAR_1D and vix5 >= _VIX_5D_SURGE)
        or (r5 <= _SPY_5D_STRESS and vix >= _VIX_STRESS_ABS)
        or (r5 <= _SPY_5D_STRESS and vixma >= _VIX_SPIKE_RATIO)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="reversal x live-regime reconciliation")
    ap.add_argument("--gate", type=float, default=-7.0)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    w, gate, h = args.winsor, args.gate, args.hold

    universe = [to_yf_symbol(s) for s in _UNIVERSE]
    syms = sorted(set(universe) | {"SPY", "^VIX"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"].dropna(
        how="all"
    )
    spy, vix = px["SPY"], px["^VIX"]
    cols = [c for c in px.columns if c not in ("SPY", "^VIX")]
    close = px[cols]
    print(f"prices: {len(cols)} names, {len(close)} sessions ({close.index[0].date()}+)")

    me5 = ((close / close.shift(5) - 1.0) * 100.0).sub((spy / spy.shift(5) - 1.0) * 100.0, axis=0)
    f = close.shift(-1 - h) / close.shift(-1) - 1.0
    sf = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (f.sub(sf, axis=0) * 100.0).clip(-w, w)
    # SPY / VIX regime features
    spy_r1 = (spy / spy.shift(1) - 1.0) * 100.0
    spy_r5 = (spy / spy.shift(5) - 1.0) * 100.0
    spy_dd = (spy / spy.rolling(252, min_periods=30).max() - 1.0) * 100.0
    vix_5d = (vix / vix.shift(5) - 1.0) * 100.0
    vix_ma = vix / vix.rolling(20, min_periods=10).mean()

    arms: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i in range(25, len(close) - h - 2, 5):
        yr = close.index[i].year
        v = float(vix.iloc[i]) if not math.isnan(vix.iloc[i]) else None
        if v is None:
            continue
        r1, r5, dd = float(spy_r1.iloc[i]), float(spy_r5.iloc[i]), float(spy_dd.iloc[i])
        v5 = 0.0 if math.isnan(vix_5d.iloc[i]) else float(vix_5d.iloc[i])
        vma = 1.0 if math.isnan(vix_ma.iloc[i]) else float(vix_ma.iloc[i])
        if _is_stress(r1, r5, dd, v, v5, vma):
            bucket = "D) STRESS_RISK_OFF (BLOCKED)"
        elif v >= _VIX_HIGH_VOL and r5 <= _SPY_5D_HVD:
            bucket = "C) HIGH_VOL_DOWNTREND (BLOCKED)"
        elif v < _VIX_FLOOR:
            bucket = "A) VIX<20 calm (ALLOWED now)"
        else:
            bucket = "B) VIX>=20 not-stress (ALLOWED)"
        me_row, fx = me5.iloc[i], fwd_excess.iloc[i]
        for sym in cols:
            mv = me_row.get(sym)
            if mv is None or (isinstance(mv, float) and math.isnan(mv)) or float(mv) > gate:
                continue
            ex = fx.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            arms[bucket].append((float(ex), yr))

    print(
        f"\n=== reversal x live-regime (me5<={gate:.0f}, hold {h}, t+1, excess vs SPY) — net @7bps ==="
    )
    for b in (
        "A) VIX<20 calm (ALLOWED now)",
        "B) VIX>=20 not-stress (ALLOWED)",
        "C) HIGH_VOL_DOWNTREND (BLOCKED)",
        "D) STRESS_RISK_OFF (BLOCKED)",
    ):
        _report(b, arms[b])
    # combined live-allowed today vs live-allowed WITH a VIX>=20 floor
    allowed_now = arms["A) VIX<20 calm (ALLOWED now)"] + arms["B) VIX>=20 not-stress (ALLOWED)"]
    _report("LIVE now (A+B)", allowed_now)
    _report("LIVE + VIX>=20 floor (B only)", arms["B) VIX>=20 not-stress (ALLOWED)"])
    print(
        "\n  Ship VIX>=20 floor if B >> A in BOTH train & test. Blocks validated if C,D <= 0; "
        "if C/D > 0 the downtrend block may leave money on the table (separate follow-up)."
    )


if __name__ == "__main__":
    main()
