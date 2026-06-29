"""Score the shadow catalyst-short log: would shorting catalyst names have paid OUTSIDE bear regimes?

Reads logs/shadow_catalyst_shorts.jsonl (written fail-safe by analysis/shadow_catalyst_shorts.capture
on every run) and, for each row matured by `--hold-days` trading days, computes the forward SHORT
return:

  raw_short      = (entry_close - exit_close) / entry_close          # short profits when price falls
  market_excess  = raw_short + spy_return                            # beta≈1 market-neutral (isolates
                                                                       the stock-specific move)
  net            = market_excess - borrow_cost(hold_days)            # after an assumed flat borrow fee

The headline question is whether `net` is positive in NON-bear regimes (where the live gate currently
forbids the trade). Borrow is assumed flat because we have no point-in-time fee feed (see the data
audit); treat results as an upper bound. Read-only; makes yfinance calls only.

Usage: python scripts/eval_shadow_catalyst_shorts.py [--hold-days 5] [--borrow-annual 3.0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf  # noqa: E402

from analysis.shadow_catalyst_shorts import SHADOW_LOG_PATH  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402

_BEAR_REGIMES = {"DEFENSIVE_DOWNTREND", "STRESS_RISK_OFF", "HIGH_VOL_DOWNTREND"}


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    try:
        with open(SHADOW_LOG_PATH) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"No shadow log at {SHADOW_LOG_PATH} yet — nothing to score.")
    return rows


def _fwd_return(closes, entry_date: str, hold_days: int) -> float | None:
    """Pct change in close from the first session >= entry_date to `hold_days` sessions later."""
    after = closes[closes.index.date >= date.fromisoformat(entry_date)]
    if len(after) <= hold_days:
        return None
    entry, exit_ = float(after.iloc[0]), float(after.iloc[hold_days])
    return (exit_ - entry) / entry * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Score shadow catalyst shorts (read-only)")
    ap.add_argument("--hold-days", type=int, default=5)
    ap.add_argument("--borrow-annual", type=float, default=3.0, help="assumed flat borrow %/yr")
    args = ap.parse_args()

    rows = _load_rows()
    cutoff = (date.today() - timedelta(days=args.hold_days + 3)).isoformat()
    matured = [r for r in rows if r.get("date", "") <= cutoff]
    print(f"{len(rows)} rows logged, {len(matured)} matured (>= {args.hold_days} sessions old)")
    if not matured:
        return

    syms = sorted({to_yf_symbol(r["symbol"]) for r in matured} | {"SPY"})
    start = min(r["date"] for r in matured)
    data = yf.download(syms, start=start, auto_adjust=True, progress=False)["Close"]
    borrow_cost = args.borrow_annual * args.hold_days / 252.0

    buckets: dict[str, list[float]] = defaultdict(list)
    per_signal: dict[str, list[float]] = defaultdict(list)
    for r in matured:
        ysym = to_yf_symbol(r["symbol"])
        if ysym not in data or "SPY" not in data:
            continue
        stock_ret = _fwd_return(data[ysym].dropna(), r["date"], args.hold_days)
        spy_ret = _fwd_return(data["SPY"].dropna(), r["date"], args.hold_days)
        if stock_ret is None or spy_ret is None:
            continue
        raw_short = -stock_ret
        net = raw_short + spy_ret - borrow_cost
        bucket = "bear" if r.get("regime") in _BEAR_REGIMES else "non-bear"
        buckets[bucket].append(net)
        for sig in r.get("catalyst_signals", []):
            per_signal[sig].append(net)

    def _summary(label: str, vals: list[float]) -> None:
        if not vals:
            return
        hit = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(
            f"  {label:28} n={len(vals):3}  net avg={sum(vals) / len(vals):+.2f}%  hit={hit:.0f}%"
        )

    print(f"\n=== net short return ({args.hold_days}d, borrow {args.borrow_annual}%/yr) ===")
    for bucket in ("non-bear", "bear"):
        _summary(bucket, buckets[bucket])
    print("\n=== by catalyst signal ===")
    for sig, vals in sorted(per_signal.items(), key=lambda kv: -len(kv[1])):
        _summary(sig, vals)


if __name__ == "__main__":
    main()
