"""Short-FLOW backtest — does FINRA daily short volume predict returns in the liquid 907?

The flow/positioning family (workshop v2's unexhausted space), now testable via data/short_flow.
Boehmer-Jones-Zhang: heavy shorting predicts NEGATIVE forward returns in large caps (shorts are
informed); light shorting predicts positive. Both fit the liquid universe, t+1 entry, 1-5d holds —
and large-cap borrow is general-collateral (~0.5%/yr), so the short side is NOT friction-dead here.

Per weekly rebalance, two constructions:
  level   — cross-sectional decile of short_volume_ratio (SVR): D10 heavy-short, D1 light-short
  spike   — SVR z-score vs own trailing 20d: top decile = abnormal shorting burst

Arms: LONG D1/low-z (light shorting) and SHORT D10/high-z (heavy shorting, GC borrow swept).
Panel is built once from FINRA daily files (cached compactly per day). Entry t+1, winsorised excess
vs SPY, cost-swept, +yrs. Standard: >=6/9 +yrs, |t|>=2 net.

Usage: python scripts/short_flow_backtest.py [--start 2018-01-01] [--hold 5]
       [--costs 0,7,14] [--winsor 25]
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
from data.short_flow import _fetch_day_text, _parse_day  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402

_PANEL_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    "caching",
    "finra_short_flow_panel",
)


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float], short: bool) -> None:
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
    tag = "short P&L" if short else "long excess"
    print(f"  {label:26} n={n:6}  gross={gross:+.3f}% ({tag})")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def _load_panel(dates: list, symbols: set[str]) -> pd.DataFrame:
    """SVR panel (dates x symbols), built from FINRA files with a compact per-day CSV cache."""
    os.makedirs(_PANEL_CACHE, exist_ok=True)
    rows = {}
    for i, d in enumerate(dates):
        day = d.date()
        cpath = os.path.join(_PANEL_CACHE, f"{day.isoformat()}.csv")
        if os.path.exists(cpath):
            try:
                rows[d] = pd.read_csv(cpath, index_col=0)["svr"]
                continue
            except Exception:
                pass
        text = _fetch_day_text(day)
        if text is None:
            continue
        svr = _parse_day(text, symbols)
        s = pd.Series(svr, name="svr")
        s.to_csv(cpath)
        rows[d] = s
        if (i + 1) % 100 == 0:
            print(f"  ...panel {i + 1}/{len(dates)} days", flush=True)
    return pd.DataFrame(rows).T.sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="FINRA short-flow backtest (907 universe)")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--costs", default="0,7,14", help="long-side cost sweep; shorts add GC borrow")
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

    print("building FINRA SVR panel (one-time download, cached) …", flush=True)
    panel = _load_panel(list(close.index), set(_UNIVERSE))
    panel = panel.reindex(close.index)
    # map FINRA symbols (plain) onto yf column names where they differ (e.g. BRK.B vs BRK-B)
    panel.columns = [to_yf_symbol(c) for c in panel.columns]
    panel = panel.loc[:, ~panel.columns.duplicated()]
    print(f"panel: {panel.notna().sum(axis=1).median():.0f} names/day median coverage")

    svr_z = (panel - panel.rolling(20).mean()) / panel.rolling(20).std()
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-w, w)

    arms: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i in range(25, len(close) - h - 2, h):
        yr = close.index[i].year
        lv = panel.iloc[i].dropna()
        zv = svr_z.iloc[i].dropna()
        fx = fwd_excess.iloc[i]
        if len(lv) < 100:
            continue
        l_lo, l_hi = lv.quantile(0.1), lv.quantile(0.9)
        z_hi = zv.quantile(0.9) if len(zv) >= 100 else None
        for sym, v in lv.items():
            ex = fx.get(sym)
            if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                continue
            if v <= l_lo:
                arms["LONG light-short D1"].append((float(ex), yr))
            elif v >= l_hi:
                arms["SHORT heavy-short D10"].append((-float(ex), yr))
        if z_hi is not None:
            for sym, z in zv.items():
                if z < z_hi:
                    continue
                ex = fx.get(sym)
                if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                    continue
                arms["SHORT SVR-spike z-D10"].append((-float(ex), yr))

    print(f"\n=== FINRA short-flow — hold {h}d, t+1 entry, winsorised excess vs SPY ===")
    _sweep("LONG light-short D1", arms["LONG light-short D1"], costs, short=False)
    # shorts: add GC borrow ~0.5%/yr (large caps) => +1bp over 5d; also show 5%/yr stress
    short_costs = sorted({c + 0.5 * h / 252 * 100 for c in costs} | {costs[-1] + 5 * h / 252 * 100})
    _sweep("SHORT heavy-short D10", arms["SHORT heavy-short D10"], short_costs, short=True)
    _sweep("SHORT SVR-spike z-D10", arms["SHORT SVR-spike z-D10"], short_costs, short=True)
    print(
        "\n  Informed-short-flow validates if a SHORT arm is net>0, |t|>=2, consistent +yrs at GC "
        "borrow; light-short LONG validates symmetrically. First flow/positioning test in this book."
    )


if __name__ == "__main__":
    main()
