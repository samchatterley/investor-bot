"""Backtest insider_buying — the last high-prior catalyst (Cohen-Malloy-Pomorski), finally testable.

The live insider_buying signal fires when >=2 distinct insiders make open-market purchases within 10
days. It has never been backtested (no historical feed). This parses Form 4 XMLs from EDGAR (reusing
data.insider_feed._parse_form4, which extracts open-market P/A purchases) to reconstruct historical
buy-CLUSTERS, then measures forward excess vs SPY.

HEAVY: purchases are rare, so we parse every Form 4 (hundreds/name) to find the few buys — a slow
job. Bounded via --limit; --lookback-days caps history. Entry t+1 (matching live), winsorised, cost
swept, holds 5/10/20d (insider drift runs over weeks). Survivorship caveat (today's universe).

Usage: python scripts/insider_backtest.py [--limit 80] [--lookback-days 3300] [--min-insiders 2]
       [--window 10] [--holds 5,10,20] [--costs 0,7,14] [--winsor 25] [--start 2015-01-01]
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

import data.insider_feed as insf  # noqa: E402
from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float]) -> None:
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
    print(f"  {label:24} n={n:5}  gross={gross:+.3f}%  (expect POSITIVE = insiders right)")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def _buy_cluster_dates(cik: str, filings: list[dict], window: int, min_insiders: int) -> list[str]:
    """Reconstruct open-market buy events, then return dates on which the cluster threshold is met."""
    buys: list[tuple[str, str]] = []  # (transaction_date, reporter)
    for f in filings:
        for txn in insf._parse_form4(cik, f["accession"], f["doc"]):
            if txn["kind"] == "buy" and txn.get("date"):
                buys.append((txn["date"][:10], txn["reporter"]))
    buys.sort()
    events: list[str] = []
    for i, (d, _) in enumerate(buys):
        d0 = pd.Timestamp(d)
        insiders = {
            r
            for dd, r in buys[max(0, i - 40) : i + 1]
            if 0 <= (d0 - pd.Timestamp(dd)).days <= window
        }
        if len(insiders) >= min_insiders:
            events.append(d)
    return sorted(set(events))


def main() -> None:
    ap = argparse.ArgumentParser(description="insider_buying backtest (Form 4 clusters)")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--lookback-days", type=int, default=3300)
    ap.add_argument("--min-insiders", type=int, default=2)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--holds", default="5,10,20")
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    holds = [int(h) for h in args.holds.split(",")]
    costs = [float(c) for c in args.costs.split(",")]
    w = args.winsor

    universe = list(_UNIVERSE)[: args.limit]
    cik_map = insf._get_cik_map()
    print(f"insider backtest: {len(universe)} names — parsing Form 4s (slow) …", flush=True)
    events: list[tuple[str, str]] = []  # (sym, cluster_date)
    for i, sym in enumerate(universe, 1):
        cik = cik_map.get(sym)
        if not cik:
            continue
        filings = insf._recent_form4_filings(cik, args.lookback_days)
        for d in _buy_cluster_dates(cik, filings, args.window, args.min_insiders):
            if d >= args.start:
                events.append((sym, d))
        if i % 20 == 0:
            print(f"  ...{i}/{len(universe)} names, {len(events)} buy-clusters", flush=True)
    print(f"Collected {len(events)} insider buy-cluster events")
    if not events:
        return

    syms = sorted({to_yf_symbol(s) for s, _ in events} | {"SPY"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"]
    spy = px["SPY"].dropna()

    def _fwd(closes: pd.Series, d: str, h: int) -> tuple[float, int] | None:
        idx = int(closes.index.searchsorted(pd.Timestamp(d)))
        entry = idx + 1
        if entry + h >= len(closes):
            return None
        e_d, x_d = closes.index[entry], closes.index[entry + h]
        si, xi = int(spy.index.searchsorted(e_d)), int(spy.index.searchsorted(x_d))
        if si >= len(spy) or xi >= len(spy):
            return None
        r = (float(closes.iloc[entry + h]) / float(closes.iloc[entry]) - 1.0) * 100.0
        sr = (float(spy.iloc[xi]) / float(spy.iloc[si]) - 1.0) * 100.0
        return max(-w, min(w, r - sr)), pd.Timestamp(d).year

    by_hold: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for sym, d in events:
        ysym = to_yf_symbol(sym)
        if ysym not in px:
            continue
        closes = px[ysym].dropna()
        if len(closes) < 30:
            continue
        for h in holds:
            m = _fwd(closes, d, h)
            if m is not None:
                by_hold[h].append(m)

    print(
        f"\n=== insider_buying (>={args.min_insiders} insiders / {args.window}d) forward excess ==="
    )
    for h in holds:
        print(f"\n  hold {h}d:")
        _sweep(f"insider_buy {h}d", by_hold[h], costs)
    print("\n  Validates if POSITIVE, |t|>=2, consistent +yrs. Bounded sample — directional read.")


if __name__ == "__main__":
    main()
