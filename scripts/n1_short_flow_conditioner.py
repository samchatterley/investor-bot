"""N1 x short-flow conditioner — does FINRA short-flow split reversal winners from knives?

Hypothesis: an idiosyncratic -7% 5d drop (the live residual_reversal trigger) accompanied by HEAVY
or SPIKING short-flow is *informed* (shorts pressing real bad news → weaker reversion / knives),
while the same drop with LIGHT/normal short-flow is *uninformed* (liquidity flow → reverts harder).
N1's volume-based diffuse filter failed to make this split; short-flow is an orthogonal dataset with
the same point-in-time coverage (data/short_flow, panel cached by short_flow_backtest).

Arms, per weekly scan (t+1 entry, hold 3 = live N1):
  trigger (all)      me5 <= -7%
    light-SVR        trigger & SVR below the day's cross-sectional median
    heavy-SVR        trigger & SVR in the day's top quartile
    SVR-spike        trigger & own-name SVR z(20d) >= +1  (abnormal shorting into the drop)
    calm-SVR         trigger & z < +1                     (no abnormal shorting)

Ships (as an N1 gate) if light/calm arms beat heavy/spike arms by a margin that survives cost with
|t|>=2 and consistent years — i.e. the filter adds precision to the already-validated edge.

Usage: python scripts/n1_short_flow_conditioner.py [--start 2018-01-01] [--hold 3]
       [--thresh -7] [--costs 0,7,14] [--winsor 25]
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


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float]) -> None:
    if not rows:
        print(f"  {label:22} (no events)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:22} n={n:5}  gross={gross:+.3f}%")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="N1 x short-flow conditioner")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=-7.0)
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
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

    panel = _load_panel(list(close.index), set(_UNIVERSE))
    panel = panel.reindex(close.index)
    panel.columns = [to_yf_symbol(c) for c in panel.columns]
    panel = panel.loc[:, ~panel.columns.duplicated()]
    svr_z = (panel - panel.rolling(20).mean()) / panel.rolling(20).std()

    spy5 = spy / spy.shift(5) - 1.0
    me5 = ((close / close.shift(5) - 1.0).sub(spy5, axis=0)) * 100.0
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-w, w)

    arms: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i in range(25, len(close) - h - 2, 5):
        yr = close.index[i].year
        mrow, fx = me5.iloc[i], fwd_excess.iloc[i]
        svr_row, z_row = panel.iloc[i], svr_z.iloc[i]
        svr_valid = svr_row.dropna()
        if len(svr_valid) < 100:
            continue
        med, q3 = svr_valid.median(), svr_valid.quantile(0.75)
        for sym in cols:
            mv = mrow.get(sym)
            if mv is None or (isinstance(mv, float) and math.isnan(mv)) or mv > args.thresh:
                continue
            ex = fx.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            arms["trigger (all)"].append((float(ex), yr))
            sv, zv = svr_row.get(sym), z_row.get(sym)
            if sv is not None and not (isinstance(sv, float) and math.isnan(sv)):
                if sv < med:
                    arms["light-SVR (below med)"].append((float(ex), yr))
                if sv >= q3:
                    arms["heavy-SVR (top qtl)"].append((float(ex), yr))
            if zv is not None and not (isinstance(zv, float) and math.isnan(zv)):
                (arms["SVR-spike (z>=1)"] if zv >= 1.0 else arms["calm-SVR (z<1)"]).append(
                    (float(ex), yr)
                )

    print(f"\n=== N1 (me5<={args.thresh:.0f}%) split by short-flow — hold {h}d, t+1 ===")
    for label in (
        "trigger (all)",
        "light-SVR (below med)",
        "heavy-SVR (top qtl)",
        "calm-SVR (z<1)",
        "SVR-spike (z>=1)",
    ):
        _sweep(label, arms[label], costs)
    print(
        "\n  Gate ships if light/calm clearly beats heavy/spike net of cost (informed-shorting "
        "knife-detection). If arms are ~equal, short-flow adds nothing to N1."
    )


if __name__ == "__main__":
    main()
