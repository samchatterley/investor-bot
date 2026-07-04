"""N1 residual_reversal_long backtest — do idiosyncratic 5-day losers revert over 1-3 days?

Fable's #1 new signal. Cross-sectional decile study over the full ~9y OHLCV history (no
earnings-window limit — pure price/volume, so this is the rigorous test).

Mechanism: uninformed flow (deleveraging, retail exits, index dust) pushes price off fair
value; liquidity providers demand a premium to absorb it; price snaps back. The edge is the
*idiosyncratic* loser (market move removed) whose decline was *diffuse* (noise), not a single
big down-day on heavy volume (information — which should continue, not revert).

Per rebalance date (weekly, to blunt the same-day cross-sectional dependence Fable flagged):
  raw5      = close_t / close_{t-5} - 1
  beta      = rolling 60d cov(stock_ret, spy_ret) / var(spy_ret)
  resid5    = raw5 - beta * spy5              # idiosyncratic 5-day return
  diffuse   = no day in the trailing 5 had ret < -2.5*sigma20 on volume > 2x avg
Rank resid5 cross-sectionally into deciles. D1 (most-negative residual) is the long thesis;
we also split D1 into diffuse vs information-shock, and report D10 as the control.

Forward excess return is winsorised (a handful of explosive small-price names otherwise dominate
the winner decile), then reported across a COST SWEEP so viability vs execution cost is explicit
— the whole edge is ~10-20bps, so the flat-cost assumption decides the verdict. No look-ahead:
resid5/diffuse use only data up to t; entry is t+1.

Usage: python scripts/residual_reversal_backtest.py [--limit N] [--hold 3]
       [--costs 0,4,7,10,14] [--winsor 25] [--start 2015-01-01] [--rebalance 5]
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
    breakeven_bps = gross * 100.0  # pre-cost mean expressed in bps of round-trip cost
    print(f"  {label:26} n={n:6}  gross={gross:+.3f}%  break-even={breakeven_bps:+.1f}bps rt")
    for c in costs_bps:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="N1 residual reversal decile study (read-only)")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--hold", type=int, default=3, help="forward hold in sessions")
    ap.add_argument("--costs", default="0,4,7,10,14", help="round-trip cost sweep (bps)")
    ap.add_argument("--winsor", type=float, default=25.0, help="cap |forward excess| at this %")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--rebalance", type=int, default=5, help="rebalance every N sessions")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]

    universe = [to_yf_symbol(s) for s in list(_UNIVERSE)[: args.limit]]
    print(
        f"residual reversal: {len(universe)} names, hold {args.hold}d, winsor +/-{args.winsor}%, "
        f"rebalance every {args.rebalance}d, from {args.start}"
    )
    syms = sorted(set(universe) | {"SPY"})
    raw = yf.download(syms, start=args.start, auto_adjust=True, progress=False)
    close = raw["Close"].dropna(how="all")
    vol = raw["Volume"].reindex(close.index)
    spy = close["SPY"]
    cols = [c for c in close.columns if c != "SPY"]
    close, vol = close[cols], vol[cols]
    print(f"Loaded {len(cols)} names over {len(close)} sessions ({close.index[0].date()}+)")

    ret = close.pct_change()
    spy_ret = spy.pct_change()
    spy5 = spy / spy.shift(5) - 1.0

    beta = ret.rolling(60).cov(spy_ret).div(spy_ret.rolling(60).var(), axis=0)
    raw5 = close / close.shift(5) - 1.0
    resid5 = raw5.sub(beta.mul(spy5, axis=0))

    sigma20 = ret.rolling(20).std()
    volratio = vol / vol.rolling(20).mean()
    shock = (ret < -2.5 * sigma20) & (volratio > 2.0)
    shock_5d = shock.rolling(5).max().fillna(0.0) > 0

    h = args.hold
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = fwd.sub(spy_fwd, axis=0) * 100.0
    fwd_excess = fwd_excess.clip(-args.winsor, args.winsor)  # winsorise pre-cost excess

    idxs = range(60, len(close) - h - 2, args.rebalance)
    d1: list[tuple[float, int]] = []
    d1_diffuse: list[tuple[float, int]] = []
    d1_shock: list[tuple[float, int]] = []
    d10: list[tuple[float, int]] = []
    allrows: list[tuple[float, int]] = []
    for i in idxs:
        row = resid5.iloc[i].dropna()
        if len(row) < 30:
            continue
        yr = close.index[i].year
        lo, hi = row.quantile(0.1), row.quantile(0.9)
        fex = fwd_excess.iloc[i]
        sh = shock_5d.iloc[i]
        for sym, rv in row.items():
            ex = fex.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            allrows.append((float(ex), yr))
            if rv <= lo:
                d1.append((float(ex), yr))
                (d1_shock if bool(sh.get(sym, False)) else d1_diffuse).append((float(ex), yr))
            elif rv >= hi:
                d10.append((float(ex), yr))

    print(f"\n=== N1 residual reversal — hold {h}d, winsorised excess vs SPY, COST SWEEP ===")
    print(
        "  ('gross' = pre-cost; break-even = round-trip cost the edge covers; * = net>0 & |t|>=2)"
    )
    _sweep("universe (all)", allrows, costs)
    _sweep("D1 loser decile", d1, costs)
    _sweep("  D1 diffuse (signal)", d1_diffuse, costs)
    _sweep("  D1 info-shock (control)", d1_shock, costs)
    _sweep("D10 winner decile", d10, costs)
    print(
        "\n  Ships if D1 (or D1-diffuse) stays net>0 with |t|>=2 at a realistic round-trip cost "
        "for this liquid universe (~4-9bps). The diffuse split only matters if it beats the shock "
        "control consistently across holds."
    )


if __name__ == "__main__":
    main()
