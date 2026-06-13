"""Targeted ΔSharpe test: disable obv_divergence and obv_acceleration.

Runs the combined long/short simulation twice on the same 3-year window:
  1. Baseline  — current GLOBALLY_DISABLED
  2. Test      — GLOBALLY_DISABLED + {obv_divergence, obv_acceleration}

Prints a summary table with ΔSharpe, ΔReturn, trade-count change, and per-signal breakdown.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

import signals.evaluator as _ev  # noqa: E402
from backtest.engine import STOCK_UNIVERSE, run_combined_analysis  # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────────
START = "2020-01-01"
END = "2022-12-31"
CAPITAL = 100_000.0
MAX_LONG = 5
MAX_SHORT = 2
PER_SIGNAL_CAP = 2
SIGNALS_TO_TEST = frozenset({"obv_divergence", "obv_acceleration"})
# ──────────────────────────────────────────────────────────────────────────────


def _run(label: str, extra_disabled: frozenset[str]) -> dict:
    original = _ev.GLOBALLY_DISABLED
    _ev.GLOBALLY_DISABLED = original | extra_disabled
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  Extra disabled: {sorted(extra_disabled) or '(none)'}")
    print(f"  Period: {START} → {END} | ${CAPITAL:,.0f} capital")
    print(f"{'=' * 65}")
    try:
        result = run_combined_analysis(
            STOCK_UNIVERSE,
            START,
            END,
            initial_capital=CAPITAL,
            max_long_positions=MAX_LONG,
            max_short_positions=MAX_SHORT,
            per_signal_cap=PER_SIGNAL_CAP,
            use_earnings_history=True,
            use_quality_fundamentals=False,
        )
    finally:
        _ev.GLOBALLY_DISABLED = original
    return result


def _sig_row(stats: dict) -> tuple[int, float, float]:
    trades = stats.get("trades", 0)
    wr = stats.get("win_rate", 0.0)
    avg = stats.get("avg_return_pct", stats.get("avg_return", 0.0))
    return trades, wr, avg


def main() -> None:
    baseline = _run("BASELINE (current signal book)", frozenset())
    test = _run("TEST (obv_divergence + obv_acceleration disabled)", SIGNALS_TO_TEST)

    if not baseline or not test:
        print("\nERROR: one or both runs returned empty results")
        return

    b_sharpe = baseline.get("sharpe_ratio", baseline.get("sharpe", float("nan")))
    t_sharpe = test.get("sharpe_ratio", test.get("sharpe", float("nan")))
    b_ret = baseline.get("total_return_pct", baseline.get("total_return", float("nan")))
    t_ret = test.get("total_return_pct", test.get("total_return", float("nan")))
    b_trades = baseline.get("total_trades", baseline.get("trades", 0))
    t_trades = test.get("total_trades", test.get("trades", 0))

    print("\n" + "=" * 65)
    print("  ΔSharpe SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Baseline':>12} {'Test':>12} {'Delta':>10}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 10}")
    print(
        f"  {'Sharpe (combined)':<30} {b_sharpe:>12.3f} {t_sharpe:>12.3f} {t_sharpe - b_sharpe:>+10.3f}"
    )
    print(f"  {'Total return':<30} {b_ret:>11.1f}% {t_ret:>11.1f}% {t_ret - b_ret:>+9.1f}%")
    print(f"  {'Total trades':<30} {b_trades:>12} {t_trades:>12} {t_trades - b_trades:>+10}")

    b_by_sig = baseline.get("by_signal", baseline.get("signal_breakdown", {}))
    if b_by_sig:
        print("\n  Signal breakdown (baseline):")
        print(f"  {'Signal':<32} {'Trades':>7} {'WR':>7} {'Avg%':>8}")
        print(f"  {'-' * 60}")
        for sig in sorted(SIGNALS_TO_TEST):
            if sig in b_by_sig:
                t, wr, avg = _sig_row(b_by_sig[sig])
                print(f"  {sig:<32} {t:>7} {wr:>6.0f}% {avg:>+8.2f}%")
            else:
                print(f"  {sig:<32} {'0':>7} {'n/a':>7} {'n/a':>8}")

    delta = t_sharpe - b_sharpe
    print()
    if delta > 0.05:
        print(f"  VERDICT: DISABLE — ΔSharpe {delta:+.3f} (>{0.05:.2f} threshold)")
    elif delta > 0:
        print(f"  VERDICT: MARGINAL — ΔSharpe {delta:+.3f} (below 0.05 threshold)")
    else:
        print(f"  VERDICT: KEEP — ΔSharpe {delta:+.3f} (removing these signals hurts)")
    print()


if __name__ == "__main__":
    main()
