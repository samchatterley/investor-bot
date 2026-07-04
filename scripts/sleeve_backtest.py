"""Diversified statistical sleeve — premise backtest (go/no-go before building live architecture).

Thesis: the bot's cross-sectional edges (e.g. N1 residual reversal) are real but the concentrated
2-5 position book only takes a few of the qualifying names. A diversified *sleeve* holds the whole
top-N as a small-weight equal-weight basket, capturing the full cross-section while diversifying away
idiosyncratic risk. If the thesis holds, the portfolio Sharpe / information ratio RISES with basket
size N (same per-name edge, lower variance) and stays positive net of cost.

Signal (the sleeve's cross-sectional score): market-excess 5d return me5 = ret_5d - spy_ret_5d.
Each rebalance (weekly) we long the bottom-N me5 (biggest idiosyncratic losers = N1 reversal),
equal-weight, enter t+1 (matching live), hold to the next rebalance. Reported vs SPY:

  IR   = annualised Sharpe of the weekly (basket - SPY) excess return   ← the sleeve's ALPHA
  hit  = % of weeks the basket beat SPY
Sweeping N shows whether diversification lifts the risk-adjusted return. Winsorised per-name forward
returns; a flat round-trip cost is charged each week (assumes full turnover — conservative).

Usage: python scripts/sleeve_backtest.py [--limit N] [--sizes 5,10,20,30,50] [--hold 5]
       [--cost-bps 14] [--winsor 25] [--start 2015-01-01]
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


def _report(label: str, weekly: list[tuple[float, float, int]], cost_wk: float) -> None:
    """weekly = [(basket_ret_%, spy_ret_%, year), ...]. Print IR (excess-Sharpe) + stats."""
    if len(weekly) < 10:
        print(f"  {label:16} (too few weeks)")
        return
    excess = [b - s - cost_wk for b, s, _ in weekly]  # net-of-cost weekly alpha vs SPY
    basket = [b - cost_wk for b, _, _ in weekly]
    n = len(excess)
    mean_x = statistics.mean(excess)
    sd_x = statistics.stdev(excess)
    ir = (mean_x / sd_x) * math.sqrt(52) if sd_x else 0.0  # annualised info ratio (alpha Sharpe)
    sharpe_b = (statistics.mean(basket) / statistics.stdev(basket)) * math.sqrt(52)
    hit = sum(1 for x in excess if x > 0) / n * 100
    by_year: dict[int, list[float]] = defaultdict(list)
    for x, (_, _, yr) in zip(excess, weekly, strict=True):
        by_year[yr].append(x)
    pos_yrs = sum(1 for xs in by_year.values() if statistics.mean(xs) > 0)
    total_x = sum(excess)
    print(
        f"  N={label:4}  IR(alpha)={ir:+.2f}  basket-Sharpe={sharpe_b:+.2f}  "
        f"wk-alpha={mean_x:+.3f}%  hit={hit:.0f}%  +yrs={pos_yrs}/{len(by_year)}  "
        f"cumα={total_x:+.0f}%  ({n} wks)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Diversified sleeve premise backtest")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--sizes", default="5,10,20,30,50", help="basket sizes N to sweep")
    ap.add_argument("--hold", type=int, default=5, help="rebalance/hold in sessions (weekly=5)")
    ap.add_argument("--cost-bps", type=float, default=14.0, help="round-trip cost / rebalance")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    w, h = args.winsor, args.hold

    universe = [to_yf_symbol(s) for s in list(_UNIVERSE)[: args.limit]]
    print(
        f"sleeve backtest: {len(universe)} names, sizes {sizes}, hold {h}d, cost {args.cost_bps}bps"
    )
    syms = sorted(set(universe) | {"SPY"})
    close = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"].dropna(
        how="all"
    )
    spy = close["SPY"]
    cols = [c for c in close.columns if c != "SPY"]
    close = close[cols]
    print(f"Loaded {len(cols)} names over {len(close)} sessions ({close.index[0].date()}+)")

    spy5 = spy / spy.shift(5) - 1.0
    me5 = (close / close.shift(5) - 1.0).sub(spy5, axis=0)  # market-excess 5d return
    fwd = (close.shift(-1 - h) / close.shift(-1) - 1.0).clip(-w / 100.0, w / 100.0) * 100.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index) * 100.0

    weekly: dict[int, list[tuple[float, float, int]]] = {n: [] for n in sizes}
    for i in range(60, len(close) - h - 2, h):
        row = me5.iloc[i].dropna()
        if len(row) < max(sizes) + 5:
            continue
        yr = close.index[i].year
        srow = fwd.iloc[i]
        spy_r = float(spy_fwd.iloc[i]) if not math.isnan(spy_fwd.iloc[i]) else None
        if spy_r is None:
            continue
        ranked = row.sort_values().index  # ascending me5 → most-negative first
        for n in sizes:
            picks = [s for s in ranked[: n * 2] if not math.isnan(srow.get(s, float("nan")))][:n]
            if len(picks) < n:
                continue
            basket_ret = statistics.mean(float(srow[s]) for s in picks)
            weekly[n].append((basket_ret, spy_r, yr))

    cost_wk = args.cost_bps / 100.0  # round-trip cost as % (full turnover each rebalance)
    print("\n=== Diversified reversal sleeve vs SPY (weekly, net cost) ===")
    print(
        "  IR(alpha) = annualised Sharpe of weekly (basket - SPY); rises with N if diversification works"
    )
    for n in sizes:
        _report(str(n), weekly[n], cost_wk)
    print(
        "\n  GO if IR(alpha) is clearly positive AND rises with N (diversification lifts risk-adjusted "
        "return) with good +yrs. NO-GO if IR is ~0/negative or doesn't improve with breadth."
    )


if __name__ == "__main__":
    main()
