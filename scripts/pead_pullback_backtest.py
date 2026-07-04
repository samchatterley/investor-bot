"""N3 pead_pullback backtest — re-enter the drift on the first pullback that holds the anchor.

Fable new signal #3. After a beat and initial drift, the first 2-3d pullback that *holds above the
pre-announcement close* is a re-entry with the drift still live (the anchor holds; fast money is
shaken out). Fable's control: it must beat a naive day-3 pead extension, else just do that.

Event study over qualifying EPS beats. EARNINGS-WINDOW LIMITED (~2-3y, yfinance earnings history),
so this is corroborative, not a 9y test. Winsorised excess vs SPY, cost sweep. No look-ahead:
entry is a post-announcement pullback close; forward uses only later closes.

  reaction session r = first session >= announcement date;  anchor = close_{r-1}
  SIGNAL: first session k in r+3..r+8 with 2 consecutive down closes, close > anchor, RSI14 in
          40-55 -> enter that close, hold `--hold`
  CONTROL (naive day-3): enter close_{r+3}, hold `--hold`

Usage: python scripts/pead_pullback_backtest.py [--limit N] [--hold 4] [--min-surprise 10]
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

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from backtest.historical_fundamentals import prefetch_earnings_history  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _sweep(label: str, rows: list[tuple[float, int]], costs_bps: list[float]) -> None:
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


def _rsi14(closes: pd.Series) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0.0, float("nan"))
    return 100.0 - 100.0 / (1.0 + rs)


def main() -> None:
    ap = argparse.ArgumentParser(description="N3 pead pullback event study")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--min-surprise", type=float, default=10.0)
    ap.add_argument("--costs", default="0,4,7,10,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    w, h = args.winsor, args.hold

    universe = list(_UNIVERSE)[: args.limit]
    print(f"pead pullback: {len(universe)} names, hold {h}d, min-surprise {args.min_surprise}%")
    hist = prefetch_earnings_history(universe)
    start = pd.Timestamp(args.start).date()
    events = [
        (s, e["date"])
        for s, evs in hist.items()
        for e in evs
        if e["surprise_pct"] >= args.min_surprise and e["date"] >= start
    ]
    print(f"Collected {len(events)} beat events")
    if not events:
        return
    syms = sorted({to_yf_symbol(s) for s, _ in events} | {"SPY"})
    px = yf.download(syms, start=str(min(d for _, d in events)), auto_adjust=True, progress=False)[
        "Close"
    ]
    spy = px["SPY"].dropna()

    def _excess(closes: pd.Series, entry: int) -> float | None:
        if entry < 0 or entry + h >= len(closes):
            return None
        e_date, x_date = closes.index[entry], closes.index[entry + h]
        si, xi = int(spy.index.searchsorted(e_date)), int(spy.index.searchsorted(x_date))
        if si >= len(spy) or xi >= len(spy):
            return None
        r = (float(closes.iloc[entry + h]) / float(closes.iloc[entry]) - 1.0) * 100.0
        sr = (float(spy.iloc[xi]) / float(spy.iloc[si]) - 1.0) * 100.0
        return max(-w, min(w, r - sr))

    sig: list[tuple[float, int]] = []
    ctrl: list[tuple[float, int]] = []
    for sym, earn_date in events:
        ysym = to_yf_symbol(sym)
        if ysym not in px:
            continue
        closes = px[ysym].dropna()
        rsi = _rsi14(closes)
        r = int(closes.index.searchsorted(pd.Timestamp(earn_date)))
        if r < 15 or r + 3 >= len(closes):
            continue
        yr = pd.Timestamp(earn_date).year
        anchor = float(closes.iloc[r - 1])
        cx = _excess(closes, r + 3)  # naive day-3 control
        if cx is not None:
            ctrl.append((cx, yr))
        for k in range(r + 3, min(r + 9, len(closes) - 1)):
            two_down = closes.iloc[k] < closes.iloc[k - 1] < closes.iloc[k - 2]
            rv = rsi.iloc[k]
            if (
                two_down
                and float(closes.iloc[k]) > anchor
                and not math.isnan(rv)
                and 40.0 <= rv <= 55.0
            ):
                ex = _excess(closes, k)
                if ex is not None:
                    sig.append((ex, yr))
                break

    print(f"\n=== N3 pead pullback — hold {h}d, winsorised excess vs SPY, COST SWEEP ===")
    print(
        "  ('gross' = pre-cost; break-even = round-trip cost the edge covers; * = net>0 & |t|>=2)"
    )
    _sweep("naive day-3 (control)", ctrl, costs)
    _sweep("pullback entry (SIGNAL)", sig, costs)
    print(
        "\n  Ships only if the pullback entry beats the naive day-3 control net at ~4-9bps; note the "
        "~2-3y earnings window limits confidence (corroborative, not definitive)."
    )


if __name__ == "__main__":
    main()
