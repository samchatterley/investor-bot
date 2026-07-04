"""Signal ablation — leave-one-out ΔSharpe over the pre-holdout window.

Covers Fable's isolation-retirement tests (golden_cross, macd_crossover, ...) AND the
redundancy/overlap audit in ONE engine pass. run_ablation runs a baseline (all signals) plus
one simulation per signal with that signal disabled, using the engine's *calibrated* cost model:

  ΔSharpe < 0  → removing it hurts  → KEEP
  ΔSharpe > 0  → removing it helps  → REVIEW (the signal is a drag / redundant)

By default this runs OHLCV-only (no fundamentals/earnings prefetch) — fast, and it is exactly the
family the retirement + overlap questions concern (golden_cross, macd_crossover, the
bb_squeeze/iv_compression/inside_day_breakout cluster, the trend/momentum cluster). Pass
--with-fundamentals for the slower full-book run (adds the ~20-40min earnings+quality prefetch).

Pre-holdout guard: end_date must be < HOLDOUT_START_DATE (2024-01-01).

Usage: python scripts/signal_ablation.py [--limit N] [--start 2015-01-01] [--end 2023-12-31]
       [--max-positions 5] [--with-fundamentals]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from backtest.engine import run_ablation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-one-out signal ablation (pre-holdout)")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2023-12-31", help="must be < HOLDOUT_START_DATE (2024-01-01)")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--with-fundamentals", action="store_true", help="also fire pead/fcf/quality")
    args = ap.parse_args()

    universe = list(_UNIVERSE)[: args.limit]
    print(
        f"Ablation: {len(universe)} names, {args.start} → {args.end}, "
        f"max_positions={args.max_positions}, fundamentals={args.with_fundamentals}"
    )
    run_ablation(
        universe,
        args.start,
        args.end,
        max_positions=args.max_positions,
        use_fundamentals=args.with_fundamentals,
        use_earnings_only=args.with_fundamentals,
    )


if __name__ == "__main__":
    main()
