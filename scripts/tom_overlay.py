"""Workshop v2 #6 — turn-of-month (TOM) overlay: does our long edge concentrate at month turn?

The TOM anomaly (Ariel 1987, Lakonishok-Smidt 1988): equities earn a disproportionate share of
their monthly return in the window {last trading day of month} + {first 3 trading days of next
month}. But that is a MARKET-LEVEL seasonal — SPY gets the boost too — so for a book measured in
EXCESS vs SPY it should cancel. The only way TOM adds alpha here is a CROSS-SECTIONAL differential:
if the names our signals pick (beaten-down reversal candidates = N1) get a *bigger* TOM lift than
the market (plausible: pension/401k inflows and month-end rebalancing bid up riskier, lower-priced
names more), then entering N1 in the TOM window beats entering it mid-month.

Test (daily eval, entry t+1, hold 3, winsorised excess vs SPY, cost swept). TOM = entry date is
among the first 3 trading days of its month OR the last trading day of its month.
  N1 all / N1 in-TOM / N1 non-TOM         (N1 = own me5 <= -7%, the book's one validated edge)
  all-names in-TOM / all-names non-TOM    (control: confirms market-level TOM cancels in excess)

Ships as a timing overlay (prefer/size-up TOM entries) only if N1 in-TOM beats N1 non-TOM net @7bps
with |t|>=2 and the all-names control shows ~0 (so it's cross-sectional, not just market beta).
Expected: null in excess terms — TOM is a market effect that cancels.

VERDICT: KILL (null, as expected). N1 works robustly in BOTH windows: non-TOM +0.276% net/3d @7bps
across 10-12/12 years (the bread-and-butter), in-TOM +0.370% net BUT positive in only 3-4/12 years
— a concentrated, year-fragile point estimate (outlier month-turns, mainly the 2020 COVID snapbacks),
not a robust TOM premium. The all-names control confirms the market-level TOM seasonal dies at cost
(+0.008% net @7bps, t=1.5). NB: the t-stats are massively overlap-inflated (daily eval, 3d hold ->
autocorrelation); +yrs is the honest robustness metric here. Restricting N1 to TOM entries would
sacrifice the robust broad edge for a noisy arm — no overlay, no capital.

Usage: python scripts/tom_overlay.py [--hold 3] [--n1 -7] [--costs 0,7,14] [--winsor 25]
       [--start 2015-01-01]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float]) -> None:
    if not rows:
        print(f"  {label:26} (no events)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:26} n={n:6}  gross={gross:+.3f}%")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def _tom_flags(index: pd.DatetimeIndex) -> dict[pd.Timestamp, bool]:
    """True if a trading day is among the first 3 or the last trading day of its calendar month."""
    flags: dict[pd.Timestamp, bool] = {}
    by_month: dict[tuple[int, int], list[pd.Timestamp]] = defaultdict(list)
    for ts in index:
        by_month[(ts.year, ts.month)].append(ts)
    for days in by_month.values():
        days.sort()
        tom = set(days[:3]) | {days[-1]}
        for ts in days:
            flags[ts] = ts in tom
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description="turn-of-month overlay backtest (v2 #6)")
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--n1", type=float, default=-7.0, help="N1 me5 trigger threshold (%)")
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
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

    me5 = ((close / close.shift(5) - 1.0) * 100.0).sub((spy / spy.shift(5) - 1.0) * 100.0, axis=0)
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-w, w)
    tom = _tom_flags(close.index)  # type: ignore[arg-type]

    arms: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i in range(25, len(close) - h - 2):
        ts = close.index[i]
        yr = ts.year
        is_tom = tom.get(ts, False)
        me_row, fx = me5.iloc[i], fwd_excess.iloc[i]
        for sym in cols:
            ex = fx.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            exf = float(ex)
            arms["all-names in-TOM" if is_tom else "all-names non-TOM"].append((exf, yr))
            mv = me_row.get(sym)
            if mv is None or (isinstance(mv, float) and math.isnan(mv)):
                continue
            if float(mv) <= args.n1:
                arms["N1 (all)"].append((exf, yr))
                arms["N1 in-TOM" if is_tom else "N1 non-TOM"].append((exf, yr))

    print(
        f"\n=== tom_overlay — hold {h}d, t+1, winsorised excess vs SPY (N1 me5<={args.n1:.0f}%) ==="
    )
    for label in (
        "N1 (all)",
        "N1 in-TOM",
        "N1 non-TOM",
        "all-names in-TOM",
        "all-names non-TOM",
    ):
        _sweep(label, arms[label], costs)
    print(
        "\n  SHIPS as a timing overlay only if 'N1 in-TOM' beats 'N1 non-TOM' net @7bps with |t|>=2 "
        "AND the all-names control is ~0 (cross-sectional, not market beta). Expected: null."
    )


if __name__ == "__main__":
    main()
