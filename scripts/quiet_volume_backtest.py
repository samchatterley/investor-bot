"""N4 quiet_volume_accumulation backtest — does a volume shock WITHOUT a price move predict drift?

Fable new signal #4, highest orthogonality: every volume condition in the current book demands
price confirmation; this is deliberately the *unconfirmed* case. Gervais-Kaniel-Mingelgrin
high-volume return premium — an extreme-volume day on which price barely moved signals informed
absorption / an attention shock whose demand shows up over the next few days.

Event study over the full ~9y OHLCV history (pure price/volume — rigorous window).

Entry (all computable from data up to and including day t):
  volume_t >= vol_mult x its 20d median
  |close_t - open_t| < 0.5 * ATR14          (the shock did NOT move price)
  close_t in the upper half of the day's range   (absorption, not distribution)
  close_t > SMA50                            (uptrend context)
Exit: enter t+1 close, hold `--hold` sessions (default 5). Forward excess vs SPY is winsorised
and reported across a COST SWEEP vs the universe baseline, so the incremental drift and its
cost-viability are both explicit (same method as the N1 study). No look-ahead: conditions use
only <=t data.

CAVEAT: no earnings filter over the 9y window (yfinance earnings history is ~2-3y only), so a
minority of events sit near earnings. The small-|close-open| condition already screens the
biggest earnings reactions; treat the number as a slight upper bound.

Usage: python scripts/quiet_volume_backtest.py [--limit N] [--hold 5] [--costs 0,4,7,10,14]
       [--winsor 25] [--start 2015-01-01] [--vol-mult 3.0]
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
    ap = argparse.ArgumentParser(description="N4 quiet volume accumulation event study")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--costs", default="0,4,7,10,14", help="round-trip cost sweep (bps)")
    ap.add_argument("--winsor", type=float, default=25.0, help="cap |forward excess| at this %")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--vol-mult", type=float, default=3.0, help="volume vs 20d median multiple")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]

    universe = [to_yf_symbol(s) for s in list(_UNIVERSE)[: args.limit]]
    print(
        f"quiet volume: {len(universe)} names, hold {args.hold}d, winsor +/-{args.winsor}%, "
        f"vol>={args.vol_mult}x median, from {args.start}"
    )
    syms = sorted(set(universe) | {"SPY"})
    raw = yf.download(syms, start=args.start, auto_adjust=True, progress=False)
    close = raw["Close"].dropna(how="all")
    idx = close.index
    names = [c for c in close.columns if c != "SPY"]
    spy = close["SPY"]
    op = raw["Open"][names].reindex(idx)
    hi = raw["High"][names].reindex(idx)
    lo = raw["Low"][names].reindex(idx)
    vol = raw["Volume"][names].reindex(idx)
    close = close[names]
    print(f"Loaded {len(names)} names over {len(idx)} sessions ({idx[0].date()}+)")

    prev_close = close.shift(1)
    tr = (hi - lo).where((hi - lo) >= (hi - prev_close).abs(), (hi - prev_close).abs())
    tr = tr.where(tr >= (lo - prev_close).abs(), (lo - prev_close).abs())
    atr = tr.rolling(14).mean()

    vol_med20 = vol.rolling(20).median()
    sma50 = close.rolling(50).mean()
    rng = (hi - lo).replace(0.0, float("nan"))

    qualifies = (
        (vol >= args.vol_mult * vol_med20)
        & ((close - op).abs() < 0.5 * atr)
        & (close > (hi + lo) / 2.0)
        & (close > sma50)
        & (rng > 0)
    )

    h = args.hold
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(idx)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-args.winsor, args.winsor)

    events: list[tuple[float, int]] = []
    baseline: list[tuple[float, int]] = []
    for i in range(55, len(idx) - h - 2):
        yr = idx[i].year
        q_row = qualifies.iloc[i]
        fex = fwd_excess.iloc[i]
        for sym in names:
            ex = fex.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            baseline.append((float(ex), yr))
            if bool(q_row.get(sym, False)):
                events.append((float(ex), yr))

    print(f"\n=== N4 quiet volume — hold {h}d, winsorised excess vs SPY, COST SWEEP ===")
    print(
        "  ('gross' = pre-cost; break-even = round-trip cost the edge covers; * = net>0 & |t|>=2)"
    )
    _sweep("universe baseline", baseline, costs)
    _sweep("quiet-volume events (SIGNAL)", events, costs)
    print(
        "\n  Ships if the signal stays net>0 with |t|>=2 at ~4-9bps AND clearly beats the universe "
        "baseline (incremental drift, not just uptrend beta)."
    )


if __name__ == "__main__":
    main()
