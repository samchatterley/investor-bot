"""Demonstration: backtest the secondary-offering catalyst now that EDGAR history exists.

Uses the new data/edgar_event_history feed to pull every 424B* / S-3 / S-1 (dilution / supply-shock)
filing for the universe, then measures the stock's forward excess return vs SPY after each — the
secondary_offering_short thesis (dilution → underperformance), which was previously UN-backtestable
(live-only, no historical event feed). This is the proof the feed unlocks the catalyst class.

Entry t+1 after the filing date (matching live: signals are EOD → next session). Winsorised excess
vs SPY, cost sweep, multiple holds (offering effects can play out over weeks). A NEGATIVE forward
return validates shorting the offering; POSITIVE would refute it.

Usage: python scripts/offering_backtest.py [--limit N] [--holds 3,5,10]
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
from data.edgar_event_history import OFFERING_FORMS, fetch_events  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _sweep(label: str, rows: list[tuple[float, int]], costs_bps: list[float]) -> None:
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
    print(f"  {label:22} n={n:5}  gross={gross:+.3f}%  (short profits if negative)")
    for c in costs_bps:
        # short P&L ≈ -stock_excess - borrow; here we just show the raw forward excess net of a
        # round-trip cost so the sign/size is visible (borrow not modelled — upper bound).
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) < 0)
        star = " *" if abs(t) >= 2 and net < 0 else ""
        print(
            f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  short-neg-yrs={pos}/{len(by_year)}{star}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Secondary-offering catalyst backtest (EDGAR feed)")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--holds", default="3,5,10")
    ap.add_argument("--costs", default="0,4,7,10,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    holds = [int(h) for h in args.holds.split(",")]
    costs = [float(c) for c in args.costs.split(",")]
    w = args.winsor

    universe = list(_UNIVERSE)[: args.limit]
    print(
        f"offering backtest: {len(universe)} names — fetching EDGAR offering history …", flush=True
    )
    events: list[tuple[str, str]] = []  # (sym, filing_date)
    for i, sym in enumerate(universe, 1):
        for e in fetch_events(sym, OFFERING_FORMS):
            if e["date"] >= args.start:
                events.append((sym, e["date"]))
        if i % 100 == 0:
            print(f"  ...{i}/{len(universe)} names, {len(events)} offerings", flush=True)
    print(f"Collected {len(events)} offering events from {len(universe)} names")
    if not events:
        return

    syms = sorted({to_yf_symbol(s) for s, _ in events} | {"SPY"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"]
    spy = px["SPY"].dropna()

    def _fwd_excess(closes: pd.Series, fdate: str, h: int) -> tuple[float, int] | None:
        i = int(closes.index.searchsorted(pd.Timestamp(fdate)))  # first session >= filing date
        entry = i + 1  # enter next session
        if entry + h >= len(closes):
            return None
        e_d, x_d = closes.index[entry], closes.index[entry + h]
        si, xi = int(spy.index.searchsorted(e_d)), int(spy.index.searchsorted(x_d))
        if si >= len(spy) or xi >= len(spy):
            return None
        r = (float(closes.iloc[entry + h]) / float(closes.iloc[entry]) - 1.0) * 100.0
        sr = (float(spy.iloc[xi]) / float(spy.iloc[si]) - 1.0) * 100.0
        return max(-w, min(w, r - sr)), pd.Timestamp(fdate).year

    by_hold: dict[int, list[tuple[float, int]]] = {h: [] for h in holds}
    for sym, fdate in events:
        ysym = to_yf_symbol(sym)
        if ysym not in px:
            continue
        closes = px[ysym].dropna()
        if len(closes) < 30:
            continue
        for h in holds:
            m = _fwd_excess(closes, fdate, h)
            if m is not None:
                by_hold[h].append(m)

    print("\n=== Secondary-offering forward excess vs SPY (entry t+1) ===")
    print(
        "  Thesis (secondary_offering_short): NEGATIVE forward excess = dilution underperformance"
    )
    for h in holds:
        print(f"\n  hold {h}d:")
        _sweep(f"offerings {h}d", by_hold[h], costs)
    print(
        "\n  The point isn't just this one signal — it's that the EDGAR feed makes the entire "
        "catalyst class (offerings/13D/insider/8-K) backtestable for the first time."
    )


if __name__ == "__main__":
    main()
