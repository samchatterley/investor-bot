"""Falling-knife SHORT — the mirror of the failed reversal long.

The reversal LONG on extreme idiosyncratic losers loses in the broad universe because those drops are
information-driven and CONTINUE (small-cap falling knives). This tests the other side: SHORT the
knives. The whole question is whether the edge survives borrow cost + the borrowable-name constraint.

Universe: the dynamic Alpaca universe (where the knives live). Signal per weekly rebalance: bottom
decile of market-excess 5d return me5 = ret_5d - spy_ret_5d. Split into:
  info-shock  — a day in the trailing 5 had ret < -2.5*sigma20 on volume > 2x (real bad news → continues)
  diffuse     — no such shock (quieter drift)

Short P&L = -(fwd stock excess vs SPY) - borrow_cost(hold, borrow_annual). Reported across a BORROW
sweep (0/5/15/30 %/yr) since we lack point-in-time short interest. Entry t+1, winsorised, weekly hold.
POSITIVE net short P&L with |t|>=2 that survives realistic borrow = a real short edge.

Usage: python scripts/falling_knife_short.py [--static] [--limit N] [--hold 5]
       [--borrows 0,5,15,30] [--winsor 25] [--start 2015-01-01] [--decile 0.1]
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


def _report(label: str, rows: list[tuple[float, int]], borrows: list[float], hold: int) -> None:
    """rows = [(pre_borrow_short_pnl_%, year), ...]. Print across the borrow sweep."""
    if not rows:
        print(f"  {label:24} (no events)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:24} n={n:6}  gross-short={gross:+.3f}%  (positive = knife kept falling)")
    for b in borrows:
        bc = b * hold / 252.0  # borrow cost over the hold, in %
        net = gross - bc
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - bc) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(
            f"      borrow {b:4.0f}%/yr (cost {bc:.3f}%)  net-short={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Falling-knife short backtest")
    ap.add_argument(
        "--static", action="store_true", help="use the static 907 (default: dynamic ~4k)"
    )
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--borrows", default="0,5,15,30", help="borrow %/yr sweep")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--decile", type=float, default=0.1, help="bottom fraction = knives")
    args = ap.parse_args()
    borrows = [float(b) for b in args.borrows.split(",")]
    w, h = args.winsor, args.hold

    if args.static:
        base = list(_UNIVERSE)
    else:
        from data.universe_builder import build_universe

        base = build_universe(use_cache=True) or list(_UNIVERSE)
    universe = [to_yf_symbol(s) for s in base[: args.limit]]
    print(f"falling-knife short: {len(universe)} names, hold {h}d, from {args.start}")
    syms = sorted(set(universe) | {"SPY"})

    frames_c, frames_v = [], []
    for i in range(0, len(syms), 300):
        raw = yf.download(
            syms[i : i + 300], start=args.start, auto_adjust=True, progress=False, threads=8
        )
        frames_c.append(raw["Close"])
        frames_v.append(raw["Volume"])
        print(f"  downloaded {min(i + 300, len(syms))}/{len(syms)} …", flush=True)
    close = pd.concat(frames_c, axis=1).dropna(how="all")
    close = close.loc[:, ~close.columns.duplicated()]
    vol = pd.concat(frames_v, axis=1)
    vol = vol.loc[:, ~vol.columns.duplicated()].reindex(close.index)
    spy = close["SPY"]
    cols = [c for c in close.columns if c != "SPY"]
    close, vol = close[cols], vol[cols]
    print(f"Loaded {len(cols)} names over {len(close)} sessions ({close.index[0].date()}+)")

    ret = close.pct_change()
    spy5 = spy / spy.shift(5) - 1.0
    me5 = (close / close.shift(5) - 1.0).sub(spy5, axis=0)
    sigma20 = ret.rolling(20).std()
    volratio = vol / vol.rolling(20).mean()
    shock5 = ((ret < -2.5 * sigma20) & (volratio > 2.0)).rolling(5).max().fillna(0.0) > 0

    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    # short P&L (pre-borrow) = -(stock forward excess vs SPY)
    short_pnl = (-(fwd.sub(spy_fwd, axis=0)) * 100.0).clip(-w, w)

    allk: list[tuple[float, int]] = []
    shock: list[tuple[float, int]] = []
    diffuse: list[tuple[float, int]] = []
    for i in range(60, len(close) - h - 2, h):
        row = me5.iloc[i].dropna()
        if len(row) < 30:
            continue
        yr = close.index[i].year
        thresh = row.quantile(args.decile)
        srow, sh = short_pnl.iloc[i], shock5.iloc[i]
        for sym, mv in row.items():
            if mv > thresh:
                continue  # only the bottom-decile knives
            pnl = srow.get(sym)
            if pnl is None or (isinstance(pnl, float) and math.isnan(pnl)):
                continue
            allk.append((float(pnl), yr))
            (shock if bool(sh.get(sym, False)) else diffuse).append((float(pnl), yr))

    print(f"\n=== Falling-knife SHORT — bottom {args.decile:.0%} me5, hold {h}d, winsorised ===")
    _report("knife (all)", allk, borrows, h)
    _report("  knife info-shock", shock, borrows, h)
    _report("  knife diffuse", diffuse, borrows, h)
    print(
        "\n  Real short edge if net-short stays >0 with |t|>=2 at a realistic borrow (~15-30%/yr for "
        "small-cap knives) and consistent +yrs. Info-shock should beat diffuse if drops are news-driven."
    )


if __name__ == "__main__":
    main()
