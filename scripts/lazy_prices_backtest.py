"""Bounded Lazy Prices backtest — does the most-changed 10-K quintile underperform?

For each name, walks its 10-K history (EDGAR), computes consecutive-filing cosine similarity, then
cross-sectionally ranks the change_score WITHIN each filing year into quintiles and measures forward
excess return vs SPY. Lazy Prices (Cohen-Malloy-Nguyen) predicts Q5 (most-changed) underperforms Q1
(copy-paste), with the spread building over 1-3 months.

Bounded first pass (default 60 names) before committing to the full universe. Heavy: EDGAR doc fetches
are rate-limited (~0.15s each). Costs/borrow not modelled; longs scored as excess vs SPY. Research only
— enter T+1 after the filing date (no look-ahead).

Usage: python scripts/lazy_prices_backtest.py [--limit 60] [--years 8] [--horizons 21,63]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

import data.edgar_client as _ec  # noqa: E402
from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from data.filing_similarity import _FILING_MAX_CHARS, cosine_similarity  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _change_events(symbol: str, years: int) -> list[tuple[str, float]]:
    """[(filing_date, change_score)] from consecutive 10-K similarities, oldest-first."""
    cik = _ec._get_cik_map().get(symbol.upper())
    if not cik:
        return []
    filings = sorted(
        _ec._get_recent_filings(cik, ["10-K"], years * 366), key=lambda f: f["filing_date"]
    )
    out: list[tuple[str, float]] = []
    prev_text: str | None = None
    for f in filings:
        text = _ec._fetch_filing_text(cik, f["accession"], f["doc"], _FILING_MAX_CHARS)
        if not text:
            prev_text = None
            continue
        if prev_text is not None:
            out.append((f["filing_date"], 1.0 - cosine_similarity(text, prev_text)))
        prev_text = text
    return out


def _fwd_excess(closes: pd.Series, spy: pd.Series, filing_date: str, horizon: int) -> float | None:
    """Excess return of entering T+1 after `filing_date` and holding `horizon` sessions."""
    fd = pd.Timestamp(filing_date)
    after = closes[closes.index > fd]
    spy_after = spy[spy.index > fd]
    if len(after) <= horizon or len(spy_after) <= horizon:
        return None
    r = (float(after.iloc[horizon]) / float(after.iloc[0]) - 1) * 100
    spy_r = (float(spy_after.iloc[horizon]) / float(spy_after.iloc[0]) - 1) * 100
    return r - spy_r


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounded Lazy Prices quintile backtest")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument(
        "--horizons", default="21,63", help="comma list of forward trading-day horizons"
    )
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]

    universe = list(_UNIVERSE)[: args.limit]
    print(
        f"Lazy Prices backtest: {len(universe)} names, {args.years}y of 10-Ks, horizons={horizons}"
    )

    # 1) Gather change events (the slow EDGAR part).
    events: list[tuple[str, str, float]] = []  # (symbol, filing_date, change_score)
    for i, sym in enumerate(universe, 1):
        evs = _change_events(sym, args.years)
        for fd, cs in evs:
            events.append((sym, fd, cs))
        if i % 10 == 0:
            print(f"  ...{i}/{len(universe)} names, {len(events)} change events so far", flush=True)
    print(f"Collected {len(events)} change events from {len(universe)} names")
    if not events:
        return

    # 2) Forward returns (one bulk download).
    syms = sorted({to_yf_symbol(s) for s, _, _ in events} | {"SPY"})
    start = min(fd for _, fd, _ in events)
    print(f"Downloading prices for {len(syms)} symbols from {start} …", flush=True)
    px = yf.download(syms, start=start, auto_adjust=True, progress=False)["Close"]
    spy = px["SPY"].dropna()

    # 3) Quintile within filing-year, aggregate forward excess per quintile per horizon.
    by_year: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
    for sym, fd, cs in events:
        by_year[int(fd[:4])].append((sym, fd, cs))

    # quintile -> horizon -> list[excess]; and per-year Q5-Q1 spread for consistency
    q_excess: dict[int, dict[int, list[float]]] = {q: defaultdict(list) for q in range(1, 6)}
    spread_by_year: dict[int, dict[int, float]] = defaultdict(dict)
    for yr, evs in by_year.items():
        if len(evs) < 5:
            continue
        ranked = sorted(evs, key=lambda e: e[2])  # ascending change_score
        n = len(ranked)
        yr_q: dict[int, dict[int, list[float]]] = {q: defaultdict(list) for q in range(1, 6)}
        for idx, (sym, fd, _cs) in enumerate(ranked):
            q = min(5, idx * 5 // n + 1)
            ysym = to_yf_symbol(sym)
            if ysym not in px:
                continue
            for h in horizons:
                ex = _fwd_excess(px[ysym].dropna(), spy, fd, h)
                if ex is not None:
                    q_excess[q][h].append(ex)
                    yr_q[q][h].append(ex)
        for h in horizons:
            if yr_q[1][h] and yr_q[5][h]:
                spread_by_year[h][yr] = statistics.mean(yr_q[5][h]) - statistics.mean(yr_q[1][h])

    # 4) Report.
    print(
        "\n=== forward EXCESS vs SPY by change-score quintile (Q1=copy-paste, Q5=most-changed) ==="
    )
    for h in horizons:
        print(f"\n  horizon {h}d:")
        for q in range(1, 6):
            v = q_excess[q][h]
            if v:
                print(f"    Q{q}  n={len(v):4}  mean excess={statistics.mean(v):+.2f}%")
        q1 = q_excess[1][h]
        q5 = q_excess[5][h]
        if q1 and q5:
            spread = statistics.mean(q5) - statistics.mean(q1)
            yrs = spread_by_year[h]
            neg = sum(1 for s in yrs.values() if s < 0) / len(yrs) * 100 if yrs else 0
            print(
                f"    Q5-Q1 spread = {spread:+.2f}%  (Lazy Prices predicts NEGATIVE);  "
                f"years Q5<Q1: {neg:.0f}% of {len(yrs)}"
            )
    print("\n  Negative Q5-Q1 spread, consistent across years = Lazy Prices edge confirmed for us.")
    print("  Caveats: survivorship, costs not modelled, cosine-on-full-text proxy, bounded N.")


if __name__ == "__main__":
    main()
