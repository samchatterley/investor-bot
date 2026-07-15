"""Score the shadow catalyst-short log: would shorting catalyst names have paid OUTSIDE bear regimes?

Reads logs/shadow_catalyst_shorts.jsonl (written fail-safe by analysis/shadow_catalyst_shorts.capture
on every run) and, for each row matured by `--hold-days` trading days, computes the forward SHORT
return:

  net = -stock_ret + spy_ret - borrow_cost(hold_days) - slippage   # market-neutral, net of costs

(a short profits when the name falls; add SPY to isolate the stock-specific move; subtract borrow +
round-trip slippage). The headline question is whether `net` is positive in NON-bear regimes (where the
live gate currently forbids the trade). Because these catalyst names skew HARD-TO-BORROW and we have no
point-in-time fee feed, the honest read is a **borrow-rate sensitivity** (3/10/25/50%/yr), not a single
number. Writes logs/short_gate_summary.json at a conservative (25%/yr) assumption for the weekly
short-gate telemetry + un-gate trigger. Read-only; makes yfinance calls only.

Usage: python scripts/eval_shadow_catalyst_shorts.py [--hold-days 5] [--borrow-annual 3.0]
                                                     [--slippage-bps 15.0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf  # noqa: E402

from analysis.shadow_catalyst_shorts import (  # noqa: E402
    BEAR_REGIMES,
    SHADOW_LOG_PATH,
    score_short_edge,
)
from utils.symbols import to_yf_symbol  # noqa: E402


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
    ap.add_argument(
        "--slippage-bps", type=float, default=15.0, help="round-trip execution slippage (bps)"
    )
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

    non_bear: list[dict] = []
    bear: list[dict] = []
    for r in matured:
        ysym = to_yf_symbol(r["symbol"])
        if ysym not in data or "SPY" not in data:
            continue
        stock_ret = _fwd_return(data[ysym].dropna(), r["date"], args.hold_days)
        spy_ret = _fwd_return(data["SPY"].dropna(), r["date"], args.hold_days)
        if stock_ret is None or spy_ret is None:
            continue
        obs = {"stock_ret": stock_ret, "spy_ret": spy_ret, "signals": r.get("catalyst_signals", [])}
        (bear if r.get("regime") in BEAR_REGIMES else non_bear).append(obs)

    def _print_table(obs: list[dict], borrow: float) -> None:
        edges = score_short_edge(
            obs, borrow_annual_pct=borrow, hold_days=args.hold_days, slippage_bps=args.slippage_bps
        )
        ordered = [("__all__", edges["__all__"])] if "__all__" in edges else []
        ordered += sorted(
            ((k, v) for k, v in edges.items() if k != "__all__"), key=lambda kv: -kv[1][0]
        )
        for key, (n, net, hit) in ordered:
            label = "ALL" if key == "__all__" else key
            print(f"  {label:28} n={n:4}  net avg={net:+.2f}%  hit={hit:.0f}%")

    print(
        f"\n=== NON-BEAR net short return ({args.hold_days}d, borrow {args.borrow_annual}%/yr, "
        f"+{args.slippage_bps:.0f}bps slippage) ==="
    )
    _print_table(non_bear, args.borrow_annual)

    # Borrow-rate sensitivity: catalyst-short names skew HARD-TO-BORROW, so the tradeable edge depends
    # heavily on the fee we can't observe historically. Show how it erodes as borrow rises.
    print(
        f"\n=== borrow-rate sensitivity (non-bear, {args.hold_days}d, +{args.slippage_bps:.0f}bps) ==="
    )
    rates = (3.0, 10.0, 25.0, 50.0)
    print(f"  {'signal':28} " + " ".join(f"{r:.0f}%".rjust(9) for r in rates))
    for key in ("__all__", "guidance_downgrade", "eps_revision_down_short"):
        cells = []
        for rate in rates:
            e = score_short_edge(
                non_bear,
                borrow_annual_pct=rate,
                hold_days=args.hold_days,
                slippage_bps=args.slippage_bps,
            ).get(key)
            cells.append((f"{e[1]:+.2f}%" if e else "--").rjust(9))
        print(f"  {('ALL' if key == '__all__' else key):28} " + " ".join(cells))

    if bear:
        print(f"\n=== BEAR regimes (reference, borrow {args.borrow_annual}%/yr) ===")
        _print_table(bear, args.borrow_annual)

    # Persist a summary at a CONSERVATIVE (hard-to-borrow) assumption for the weekly short-gate
    # telemetry + un-gate trigger (experiment.monitoring.build_short_gate_lines reads this).
    realistic = score_short_edge(
        non_bear, borrow_annual_pct=25.0, hold_days=args.hold_days, slippage_bps=args.slippage_bps
    )
    summary_path = os.path.join(os.path.dirname(SHADOW_LOG_PATH), "short_gate_summary.json")
    try:
        with open(summary_path, "w") as fh:
            json.dump(
                {
                    "generated": date.today().isoformat(),
                    "borrow_annual_pct": 25.0,
                    "slippage_bps": args.slippage_bps,
                    "hold_days": args.hold_days,
                    "edges": realistic,
                },
                fh,
            )
        print(
            f"\nWrote short-gate summary (borrow 25%, +{args.slippage_bps:.0f}bps) -> {summary_path}"
        )
    except OSError as exc:
        print(f"(could not write summary: {exc})")


if __name__ == "__main__":
    main()
