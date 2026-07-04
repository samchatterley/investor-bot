"""pead conditioning backtest — sharpen the one signal that carries the book.

Event study over every qualifying EPS-beat (surprise_pct >= threshold) in STOCK_UNIVERSE.
Tests the Fable pead conditionings that are *feasible with current data* (yfinance earnings
dates + daily OHLCV):

  (b) day-0 reaction cap  — is post-earnings drift concentrated in the MUTED-reaction beats?
  (c) entry window        — days 1-3 (fresh drift) vs 4-7 (stale)
  (d) hold extension      — 3d (current live) vs 5d
  + surprise-magnitude buckets and the value of the live ``ret_5d > 0`` drift filter.

NOT tested (flagged, not feasible with the current feed):
  (a) double-beat  — needs revenue surprise; the feed carries EPS surprise only
  (e) accruals     — needs point-in-time accruals per event

DATA CAVEAT: yfinance ``earnings_dates`` only reaches ~6-12 quarters back, so the effective
window is ~2-3 years regardless of ``--start``. pead is among the most-replicated anomalies
(Bernard & Thomas 1989), so treat a positive result here as corroboration at scale, not a
first discovery. A deeper FMP earnings-surprise feed would extend the window (see TODO).

No look-ahead: entry is strictly AFTER the announcement session; the ret_5d filter uses only
pre-entry closes; forward returns use only post-entry closes. Forward return is entry-close to
exit-close, excess vs SPY over the same window, net a flat round-trip cost. Read-only.

Usage: python scripts/pead_conditioning_backtest.py [--limit N] [--min-surprise 10]
       [--cost-bps 14] [--start 2015-01-01]
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

# (entry_offset, day0_reaction_%, ret_5d_%, net_excess_%) for one event under one (k, h) config.
_EventRow = tuple[float, float, float]


def _event_metrics(
    closes: pd.Series, spy: pd.Series, earn_date, k: int, h: int, cost_bps: float
) -> _EventRow | None:
    """Metrics for entering ``k`` sessions after the announcement session and holding ``h``.

    Returns (day0_reaction_%, ret_5d_at_entry_%, net_excess_%) or None if out of range.
    """
    idx = int(closes.index.searchsorted(pd.Timestamp(earn_date)))  # first session >= earn_date
    entry = idx + k
    if idx < 6 or entry - 6 < 0 or entry + h >= len(closes):
        return None
    day0 = (float(closes.iloc[idx]) / float(closes.iloc[idx - 1]) - 1.0) * 100.0
    ret5d = (float(closes.iloc[entry - 1]) / float(closes.iloc[entry - 6]) - 1.0) * 100.0
    stock_r = (float(closes.iloc[entry + h]) / float(closes.iloc[entry]) - 1.0) * 100.0

    e_date, x_date = closes.index[entry], closes.index[entry + h]
    si = int(spy.index.searchsorted(e_date))
    xi = int(spy.index.searchsorted(x_date))
    if si >= len(spy) or xi >= len(spy):
        return None
    spy_r = (float(spy.iloc[xi]) / float(spy.iloc[si]) - 1.0) * 100.0
    net_excess = (stock_r - spy_r) - cost_bps / 100.0
    return day0, ret5d, net_excess


def _summary(label: str, rows: list[tuple[float, int]]) -> None:
    """rows = [(net_excess_%, year), ...]. Print N / mean / hit / IR / t / positive-years."""
    if not rows:
        print(f"  {label:34} (no trades)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    mean = statistics.mean(vals)
    sd = statistics.pstdev(vals) if n < 2 else statistics.stdev(vals)
    hit = sum(1 for v in vals if v > 0) / n * 100.0
    ir = mean / sd if sd else 0.0  # per-trade info ratio (mean / std)
    t = mean / (sd / math.sqrt(n)) if sd else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    pos_years = sum(1 for ys in by_year.values() if statistics.mean(ys) > 0)
    print(
        f"  {label:34} n={n:5}  mean={mean:+.2f}%  hit={hit:4.0f}%  IR={ir:+.3f}  "
        f"t={t:+.2f}  +yrs={pos_years}/{len(by_year)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="pead conditioning event study (read-only)")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--min-surprise", type=float, default=10.0, help="EPS beat threshold (%)")
    ap.add_argument("--cost-bps", type=float, default=14.0, help="flat round-trip cost (bps)")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    universe = list(_UNIVERSE)[: args.limit]
    print(
        f"pead conditioning: {len(universe)} names, min-surprise {args.min_surprise}%, "
        f"cost {args.cost_bps}bps rt, from {args.start}"
    )

    # 1) Gather qualifying beat events (slow: one yfinance call per name).
    hist = prefetch_earnings_history(universe)
    start = pd.Timestamp(args.start).date()
    events: list[tuple[str, object, float]] = []  # (sym, earn_date, surprise_pct)
    for sym, evs in hist.items():
        for e in evs:
            if e["surprise_pct"] >= args.min_surprise and e["date"] >= start:
                events.append((sym, e["date"], e["surprise_pct"]))
    print(f"Collected {len(events)} beat events from {len(hist)} names with earnings data")
    if not events:
        return

    # 2) Prices (one bulk download).
    syms = sorted({to_yf_symbol(s) for s, _, _ in events} | {"SPY"})
    px_start = min(e[1] for e in events)
    print(f"Downloading prices for {len(syms)} symbols from {px_start} …", flush=True)
    px = yf.download(syms, start=str(px_start), auto_adjust=True, progress=False)["Close"]
    spy = px["SPY"].dropna()

    # 3) Route each event into pre-registered buckets (no combinatorial sweep — see docstring).
    #    Baseline live proxy = enter T+1, hold 3, require ret_5d > 0.
    base: list[tuple[float, int]] = []
    react_buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    surp_buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    window: dict[str, list[tuple[float, int]]] = defaultdict(list)  # T1-3 vs T4-7 (hold 3)
    hold_cmp: dict[str, list[tuple[float, int]]] = defaultdict(list)  # hold 3 vs 5 (T+1)
    filt: dict[str, list[tuple[float, int]]] = defaultdict(list)  # ret_5d>0 vs <=0 (T+1, hold 3)

    for sym, earn_date, surprise in events:
        ysym = to_yf_symbol(sym)
        if ysym not in px:
            continue
        closes = px[ysym].dropna()
        yr = pd.Timestamp(earn_date).year

        # Baseline family (T+1, hold 3) drives day0/surprise/filter buckets.
        m1_3 = _event_metrics(closes, spy, earn_date, k=1, h=3, cost_bps=args.cost_bps)
        if m1_3 is not None:
            day0, ret5d, ex = m1_3
            drift_ok = ret5d > 0
            if drift_ok:
                base.append((ex, yr))
                bucket = (
                    "muted (<2%)"
                    if day0 < 2
                    else "moderate (2-5%)"
                    if day0 < 5
                    else "hot (5-8%)"
                    if day0 < 8
                    else "very hot (>8%)"
                )
                react_buckets[bucket].append((ex, yr))
                smag = (
                    "surprise 10-20%"
                    if surprise < 20
                    else "surprise 20-50%"
                    if surprise < 50
                    else "surprise >50%"
                )
                surp_buckets[smag].append((ex, yr))
            filt["ret_5d > 0 (filter ON)" if drift_ok else "ret_5d <= 0 (filter OFF)"].append(
                (ex, yr)
            )

        # (d) hold 3 vs 5 at T+1, ret_5d>0.
        m1_5 = _event_metrics(closes, spy, earn_date, k=1, h=5, cost_bps=args.cost_bps)
        if m1_5 is not None and m1_5[1] > 0:
            hold_cmp["T+1 hold 5d"].append((m1_5[2], yr))
        if m1_3 is not None and m1_3[1] > 0:
            hold_cmp["T+1 hold 3d"].append((m1_3[2], yr))

        # (c) entry window: pooled T+1..3 vs T+4..7 (hold 3, ret_5d>0).
        for k in range(1, 8):
            mk = _event_metrics(closes, spy, earn_date, k=k, h=3, cost_bps=args.cost_bps)
            if mk is not None and mk[1] > 0:
                window["entry T+1..3" if k <= 3 else "entry T+4..7"].append((mk[2], yr))

    # 4) Report.
    print("\n=== BASELINE (live proxy: enter T+1, hold 3d, require ret_5d>0) ===")
    _summary("baseline", base)

    print("\n=== (b) day-0 reaction bucket (baseline entries) — expect drift in MUTED ===")
    for b in ("muted (<2%)", "moderate (2-5%)", "hot (5-8%)", "very hot (>8%)"):
        _summary(b, react_buckets[b])

    print("\n=== (c) entry window (hold 3d, ret_5d>0) — expect T+1..3 > T+4..7 ===")
    for b in ("entry T+1..3", "entry T+4..7"):
        _summary(b, window[b])

    print("\n=== (d) hold extension (T+1, ret_5d>0) — expect 5d >= 3d ===")
    for b in ("T+1 hold 3d", "T+1 hold 5d"):
        _summary(b, hold_cmp[b])

    print("\n=== surprise magnitude (baseline entries) ===")
    for b in ("surprise 10-20%", "surprise 20-50%", "surprise >50%"):
        _summary(b, surp_buckets[b])

    print("\n=== ret_5d drift filter value (T+1, hold 3d) — does the filter add edge? ===")
    for b in ("ret_5d > 0 (filter ON)", "ret_5d <= 0 (filter OFF)"):
        _summary(b, filt[b])

    print(
        "\n  Read: a positive mean with +yrs at (or near) the max and |t|>2 corroborates the "
        "conditioning. Window ~2-3y limits year-consistency — see the data caveat above."
    )
    print("  NOT tested (data gaps): (a) double-beat [needs revenue], (e) accruals [needs PIT].")


if __name__ == "__main__":
    main()
