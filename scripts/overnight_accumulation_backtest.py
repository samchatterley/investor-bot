"""N5 overnight_accumulation backtest — persistent overnight bid vs intraday selling.

Fable new signal #5. Lou-Polk-Skouras "tug of war": overnight and intraday returns have
different clienteles; the overnight component is the informed/institutional one (they trade the
opens). Names whose trailing-5d *overnight* return is strongly positive while the *intraday*
return is negative are being accumulated overnight against daytime noise selling — long the
divergence.

Event study over the full ~9y OHLCV history. Fable's key caution: raw overnight strength
correlates with momentum, so the control matters. We report three arms:
  SIGNAL       overnight_5 >= +1.5%  AND  intraday_5 <= -1%   (the divergence)
  overnight-only  overnight_5 >= +1.5%   (ignore intraday — the momentum-confound control)
  universe baseline
The signal only earns its keep if it beats BOTH the universe and the overnight-only control.

overnight_daily = open_t / close_{t-1} - 1 ;  intraday_daily = close_t / open_t - 1
(_5 = rolling 5-session sum). Exit t+1 close, hold `--hold`. Winsorised excess vs SPY, cost
sweep. No look-ahead. CAVEAT: no earnings filter over 9y (yfinance ~2-3y only).

Usage: python scripts/overnight_accumulation_backtest.py [--limit N] [--hold 3]
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
    ap = argparse.ArgumentParser(description="N5 overnight accumulation event study")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--costs", default="0,4,7,10,14", help="round-trip cost sweep (bps)")
    ap.add_argument("--winsor", type=float, default=25.0, help="cap |forward excess| at this %")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]

    universe = [to_yf_symbol(s) for s in list(_UNIVERSE)[: args.limit]]
    print(f"overnight accumulation: {len(universe)} names, hold {args.hold}d, from {args.start}")
    syms = sorted(set(universe) | {"SPY"})
    raw = yf.download(syms, start=args.start, auto_adjust=True, progress=False)
    close = raw["Close"].dropna(how="all")
    idx = close.index
    names = [c for c in close.columns if c != "SPY"]
    spy = close["SPY"]
    op = raw["Open"][names].reindex(idx)
    close = close[names]
    print(f"Loaded {len(names)} names over {len(idx)} sessions ({idx[0].date()}+)")

    overnight = op / close.shift(1) - 1.0
    intraday = close / op - 1.0
    overnight_5 = overnight.rolling(5).sum()
    intraday_5 = intraday.rolling(5).sum()
    sma50 = close.rolling(50).mean()

    on_strong = (overnight_5 >= 0.015) & (close > sma50)
    signal = on_strong & (intraday_5 <= -0.01)

    h = args.hold
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(idx)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-args.winsor, args.winsor)

    sig: list[tuple[float, int]] = []
    onlyon: list[tuple[float, int]] = []
    baseline: list[tuple[float, int]] = []
    for i in range(55, len(idx) - h - 2):
        yr = idx[i].year
        s_row, o_row, fex = signal.iloc[i], on_strong.iloc[i], fwd_excess.iloc[i]
        for sym in names:
            ex = fex.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            baseline.append((float(ex), yr))
            if bool(o_row.get(sym, False)):
                onlyon.append((float(ex), yr))
            if bool(s_row.get(sym, False)):
                sig.append((float(ex), yr))

    print(f"\n=== N5 overnight accumulation — hold {h}d, winsorised excess vs SPY, COST SWEEP ===")
    print(
        "  ('gross' = pre-cost; break-even = round-trip cost the edge covers; * = net>0 & |t|>=2)"
    )
    _sweep("universe baseline", baseline, costs)
    _sweep("overnight-only (control)", onlyon, costs)
    _sweep("divergence (SIGNAL)", sig, costs)
    print(
        "\n  Ships only if 'divergence' beats BOTH the universe AND the overnight-only control at "
        "~4-9bps with |t|>=2 (else it is just a momentum proxy, not the tug-of-war edge)."
    )


if __name__ == "__main__":
    main()
