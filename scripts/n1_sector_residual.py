"""Workshop v2 #2 — sector-residual upgrade to residual_reversal (N1).

Live N1 residualises against SPY only, so a -7% "idiosyncratic" drop during a sector rout is really
sector beta — and sector moves continue more than idiosyncratic ones revert. Hypothesis: residualise
against the name's OWN SECTOR (equal-weight index of universe peers, built from the same price
panel — no new data) and the reversion arm gets cleaner (higher net mean / better year consistency).

Head-to-head on identical weeks, hold 3 (live), entry t+1, winsorised excess vs SPY, cost swept:
  spy-residual   me5_spy    = ret5 - spy5            <= -7%   (the live signal)
  sector-residual me5_sect  = ret5 - own_sector_ew5  <= -7%
  both           fire on both measures (the intersection — pure idiosyncratic)
Also decile-matched arms (bottom decile of each measure) so threshold-induced N differences don't
confound the comparison. Sector map from the bot's live cache (coverage reported).

Usage: python scripts/n1_sector_residual.py [--hold 3] [--thresh -7] [--costs 0,7,14]
       [--winsor 25] [--start 2015-01-01]
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
from data.sector_data import get_sector  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float]) -> None:
    if not rows:
        print(f"  {label:28} (no events)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:28} n={n:6}  gross={gross:+.3f}%")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="N1 sector-residual head-to-head (v2 #2)")
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=-7.0)
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    w, h = args.winsor, args.hold

    universe = [to_yf_symbol(s) for s in _UNIVERSE]
    sectors = {to_yf_symbol(s): get_sector(s) for s in _UNIVERSE}
    known = sum(1 for v in sectors.values() if v and v != "Unknown")
    print(f"sector coverage: {known}/{len(sectors)} names")

    syms = sorted(set(universe) | {"SPY"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"].dropna(
        how="all"
    )
    spy = px["SPY"]
    cols = [c for c in px.columns if c != "SPY"]
    close = px[cols]
    print(f"prices: {len(cols)} names, {len(close)} sessions ({close.index[0].date()}+)")

    ret5 = (close / close.shift(5) - 1.0) * 100.0
    spy5 = (spy / spy.shift(5) - 1.0) * 100.0
    # Equal-weight sector 5d return from the panel itself
    by_sector: dict[str, list[str]] = defaultdict(list)
    for sym in cols:
        sec = sectors.get(sym) or "Unknown"
        by_sector[sec].append(sym)
    sect5 = {sec: ret5[mem].mean(axis=1) for sec, mem in by_sector.items() if len(mem) >= 5}

    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-w, w)

    arms: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i in range(25, len(close) - h - 2, 5):
        yr = close.index[i].year
        r_row, fx = ret5.iloc[i], fwd_excess.iloc[i]
        sv = float(spy5.iloc[i])
        if math.isnan(sv):
            continue
        me_spy_vals: dict[str, float] = {}
        me_sect_vals: dict[str, float] = {}
        for sym in cols:
            rv = r_row.get(sym)
            if rv is None or (isinstance(rv, float) and math.isnan(rv)):
                continue
            me_spy_vals[sym] = rv - sv
            sec = sectors.get(sym) or "Unknown"
            if sec in sect5:
                s5 = float(sect5[sec].iloc[i])
                if not math.isnan(s5):
                    me_sect_vals[sym] = rv - s5
        spy_dec = (
            sorted(me_spy_vals.values())[max(0, len(me_spy_vals) // 10 - 1)] if me_spy_vals else 0
        )
        sect_dec = (
            sorted(me_sect_vals.values())[max(0, len(me_sect_vals) // 10 - 1)]
            if me_sect_vals
            else 0
        )
        for sym in cols:
            ex = fx.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            a = me_spy_vals.get(sym)
            b = me_sect_vals.get(sym)
            if a is not None and a <= args.thresh:
                arms["spy-resid <= -7 (LIVE)"].append((float(ex), yr))
                if b is not None and b <= args.thresh:
                    arms["both <= -7 (intersection)"].append((float(ex), yr))
            if b is not None and b <= args.thresh:
                arms["sector-resid <= -7"].append((float(ex), yr))
            if a is not None and a <= spy_dec:
                arms["spy-resid bottom decile"].append((float(ex), yr))
            if b is not None and b <= sect_dec:
                arms["sector-resid bottom decile"].append((float(ex), yr))

    print(f"\n=== N1 residualisation head-to-head — hold {h}d, t+1, winsorised excess vs SPY ===")
    for label in (
        "spy-resid <= -7 (LIVE)",
        "sector-resid <= -7",
        "both <= -7 (intersection)",
        "spy-resid bottom decile",
        "sector-resid bottom decile",
    ):
        _sweep(label, arms[label], costs)
    print(
        "\n  UPGRADE ships if sector-residual (or the intersection) beats the live spy-residual on "
        "net mean AND +yrs at 7bps; otherwise keep the live construction."
    )


if __name__ == "__main__":
    main()
