"""Validate the live catalyst signals via the EDGAR feed — 13D (activist) & 8-K accounting flags.

Two of the bot's ACTIVE catalyst signals have never been backtested (no historical event feed). Both
are now testable from data/edgar_event_history with no content parsing — just filing dates and the
8-K item codes the feed already extracts:

  activist_13d_signal   (long)  → SC 13D / SC 13D/A          → expect POSITIVE forward excess
  accounting_concern_short (short) → 8-K item 4.01 (auditor change) / 4.02 (non-reliance/restatement)
                                                              → expect NEGATIVE forward excess

Forward excess vs SPY, entry t+1 (matching live), winsorised, cost-swept, holds 5/10/20d (catalyst
effects run over weeks). CAVEAT: today's-universe survivorship applies (dead names excluded), so read
the *sign and per-year consistency*, not the absolute magnitude.

Usage: python scripts/catalyst_backtest.py [--limit N] [--holds 5,10,20]
       [--costs 0,7,14] [--winsor 25] [--start 2015-01-01]
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
from data.edgar_event_history import ACCOUNTING_ITEMS, ACTIVIST_FORMS, fetch_events  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float], want_neg: bool) -> None:
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
    exp = "expect NEG" if want_neg else "expect POS"
    print(f"  {label:26} n={n:5}  gross={gross:+.3f}%  ({exp})")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        good = (net < 0) if want_neg else (net > 0)
        pos = sum(
            1 for ys in by_year.values() if ((statistics.mean(ys) - c / 100.0) < 0) == want_neg
        )
        star = " *" if abs(t) >= 2 and good else ""
        print(
            f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  as-expected-yrs={pos}/{len(by_year)}{star}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Catalyst signal validation via EDGAR feed")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--holds", default="5,10,20")
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    holds = [int(h) for h in args.holds.split(",")]
    costs = [float(c) for c in args.costs.split(",")]
    w = args.winsor

    universe = list(_UNIVERSE)[: args.limit]
    print(f"catalyst backtest: {len(universe)} names — reading EDGAR events (cached) …", flush=True)
    # (sym, date, kind) where kind in {"13d", "acct"}
    events: list[tuple[str, str, str]] = []
    for i, sym in enumerate(universe, 1):
        for e in fetch_events(sym, ACTIVIST_FORMS):
            if e["date"] >= args.start:
                events.append((sym, e["date"], "13d"))
        for e in fetch_events(sym, ("8-K",)):
            if e["date"] >= args.start and any(it in e["items"] for it in ACCOUNTING_ITEMS):
                events.append((sym, e["date"], "acct"))
        if i % 200 == 0:
            print(f"  ...{i}/{len(universe)} names, {len(events)} events", flush=True)
    kinds = defaultdict(int)
    for _, _, k in events:
        kinds[k] += 1
    print(f"Collected: 13D={kinds['13d']}, accounting-8K={kinds['acct']}")
    if not events:
        return

    syms = sorted({to_yf_symbol(s) for s, _, _ in events} | {"SPY"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"]
    spy = px["SPY"].dropna()

    def _fwd(closes: pd.Series, fdate: str, h: int) -> tuple[float, int] | None:
        i = int(closes.index.searchsorted(pd.Timestamp(fdate)))
        entry = i + 1
        if entry + h >= len(closes):
            return None
        e_d, x_d = closes.index[entry], closes.index[entry + h]
        si, xi = int(spy.index.searchsorted(e_d)), int(spy.index.searchsorted(x_d))
        if si >= len(spy) or xi >= len(spy):
            return None
        r = (float(closes.iloc[entry + h]) / float(closes.iloc[entry]) - 1.0) * 100.0
        sr = (float(spy.iloc[xi]) / float(spy.iloc[si]) - 1.0) * 100.0
        return max(-w, min(w, r - sr)), pd.Timestamp(fdate).year

    buckets: dict[tuple[str, int], list[tuple[float, int]]] = defaultdict(list)
    for sym, fdate, kind in events:
        ysym = to_yf_symbol(sym)
        if ysym not in px:
            continue
        closes = px[ysym].dropna()
        if len(closes) < 30:
            continue
        for h in holds:
            m = _fwd(closes, fdate, h)
            if m is not None:
                buckets[(kind, h)].append(m)

    print("\n=== Catalyst forward excess vs SPY (entry t+1, winsorised) ===")
    for h in holds:
        print(f"\n  hold {h}d:")
        _sweep(f"activist_13d {h}d", buckets[("13d", h)], costs, want_neg=False)
        _sweep(f"accounting_8K {h}d", buckets[("acct", h)], costs, want_neg=True)
    print(
        "\n  A signal validates if it has the expected sign, |t|>=2, and consistent as-expected-yrs. "
        "Survivorship caveat: read sign/consistency over absolute size."
    )


if __name__ == "__main__":
    main()
