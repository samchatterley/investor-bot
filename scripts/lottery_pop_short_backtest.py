"""Workshop v2 #1 — lottery_pop_short: short the >=+10% single-day popper.

The lottery-gate study (1.138) measured -0.44%/3d (t=-5.1, n=7,739) after +10% pops and shipped it
as a BUY-block only. This tests the other side as a tradeable SHORT with the frictions that decide
it: cost + borrow sweep, per-year tails (meme years), and — new since the FINRA feed — a short-flow
crowding gate (our own finding: heavy-SVR names get squeezed, so a popper short must AVOID crowded
names; SVR below the daily median is the borrowable/uncrowded proxy, 2018+ where the panel exists).

Entry t+1 after the pop day (EOD signal), hold 3, short P&L = -(fwd excess vs SPY), winsorised.
CAVEATS: no earnings exclusion over 9y (earnings pops drift UP per PEAD, so their inclusion makes
this CONSERVATIVE); no stop modelled (winsor +/-25% bounds the tail read); borrow swept flat.

PARAM ROBUSTNESS (crowded arm, sweep pop threshold, net @7bps+borrow): pop>=8%:+0.51%(4/9yrs) /
10%(live):+0.78%(6/9) / 12%:+0.88%(6/9, worst -1.73%) / 15%:+1.32%(4/9, worst -3.72%). Magnitude
rises monotonically with the threshold but year-consistency degrades and tail years fatten; 10%
is the consistency sweet spot (best +yrs, survives 15%/yr borrow at +0.61%). Well-chosen — not a
spike. (Shadow-only signal; still needs forward evidence before live capital.)

Usage: python scripts/lottery_pop_short_backtest.py [--pop 10] [--hold 3] [--winsor 25]
       [--costs 7] [--borrows 0.5,5,15,30] [--start 2015-01-01]
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
from scripts.short_flow_backtest import _load_panel  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _report(
    label: str, rows: list[tuple[float, int]], cost: float, borrows: list[float], h: int
) -> None:
    if not rows:
        print(f"  {label:30} (no events)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:30} n={n:5}  gross-short={gross:+.3f}%")
    for b in borrows:
        bc = cost / 100.0 + b * h / 252.0
        net = gross - bc
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - bc) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(
            f"      cost {cost:.0f}bps + borrow {b:4.1f}%/yr  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}"
        )
    worst = sorted(by_year.items(), key=lambda kv: statistics.mean(kv[1]))[:3]
    print("      worst years: " + ", ".join(f"{y}: {statistics.mean(v):+.2f}%" for y, v in worst))


def main() -> None:
    ap = argparse.ArgumentParser(description="lottery_pop_short backtest (v2 #1)")
    ap.add_argument("--pop", type=float, default=10.0)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--costs", type=float, default=7.0)
    ap.add_argument("--borrows", default="0.5,5,15,30")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    borrows = [float(b) for b in args.borrows.split(",")]
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

    ret1 = close.pct_change() * 100.0
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    short_pnl = (-(fwd.sub(spy_fwd, axis=0)) * 100.0).clip(-w, w)

    # FINRA SVR panel for the crowding gate (2018+ where cached)
    panel = _load_panel(list(close.index[close.index >= "2018-01-01"]), set(_UNIVERSE))
    panel.columns = [to_yf_symbol(c) for c in panel.columns]
    panel = panel.loc[:, ~panel.columns.duplicated()].reindex(close.index)

    all_pops: list[tuple[float, int]] = []
    gated: list[tuple[float, int]] = []  # 2018+, SVR below daily median (uncrowded)
    crowded: list[tuple[float, int]] = []  # 2018+, SVR top quartile (avoid per our own finding)
    pop_sweep = (8.0, 10.0, 12.0, 15.0)  # PARAM ROBUSTNESS: crowded arm across pop thresholds
    crowded_by_pop: dict[float, list[tuple[float, int]]] = {p: [] for p in pop_sweep}
    for i in range(1, len(close) - h - 2):
        yr = close.index[i].year
        r_row = ret1.iloc[i]
        s_row = short_pnl.iloc[i]
        svr_row = panel.iloc[i].dropna()
        med = svr_row.median() if len(svr_row) >= 100 else None
        q3 = svr_row.quantile(0.75) if len(svr_row) >= 100 else None
        for sym in cols:
            rv = r_row.get(sym)
            if rv is None or (isinstance(rv, float) and math.isnan(rv)):
                continue
            pnl = s_row.get(sym)
            if pnl is None or (isinstance(pnl, float) and math.isnan(pnl)):
                continue
            sv = svr_row.get(sym) if med is not None else None
            sv_ok = sv is not None and not (isinstance(sv, float) and math.isnan(sv))
            if rv >= args.pop:
                all_pops.append((float(pnl), yr))
                if med is not None and sv_ok:
                    if sv < med:
                        gated.append((float(pnl), yr))
                    if q3 is not None and sv >= q3:
                        crowded.append((float(pnl), yr))
            # pop-threshold sweep, crowded (top-quartile SVR) arm only
            if q3 is not None and sv_ok and sv >= q3:
                for pth in pop_sweep:
                    if rv >= pth:
                        crowded_by_pop[pth].append((float(pnl), yr))

    print(f"\n=== lottery_pop_short (pop>={args.pop:.0f}%, hold {h}d, t+1, winsorised) ===")
    _report("all pops (2015+)", all_pops, args.costs, borrows, h)
    _report("SVR-gated uncrowded (2018+)", gated, args.costs, borrows, h)
    _report("SVR crowded top-qtl (2018+)", crowded, args.costs, borrows, h)
    print("\n--- crowded-arm pop-threshold robustness (plateau => robust) ---")
    for pth in pop_sweep:
        _report(f"crowded pop>={pth:.0f}%", crowded_by_pop[pth], args.costs, borrows, h)
    print(
        "\n  SHIPS if the SVR-gated arm is net>0 with |t|>=2 at realistic borrow and no meme-year "
        "blowup dominating; crowded arm should be WORSE (squeeze fuel) per the N1xSVR finding."
    )


if __name__ == "__main__":
    main()
