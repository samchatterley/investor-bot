"""One-at-a-time param sweep for rs_leader excess thresholds.

Isolates rs_leader (all other long signals disabled) to get clean per-threshold
win-rate and avg-return data without cross-signal noise.

Usage:
    .venv/bin/python scripts/sweep_rs_leader.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import run_param_sensitivity
from config import STOCK_UNIVERSE
from signals.evaluator import SIGNAL_PRIORITY

START = "2015-01-01"
END = "2023-12-31"
CAPITAL = 25_000

# Disable every long signal except rs_leader so results are clean
ALL_LONG_SIGNALS = frozenset(SIGNAL_PRIORITY.keys())
DISABLED = ALL_LONG_SIGNALS - {"rs_leader"}

param_ranges = {
    "rsl_excess_5d_min": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "rsl_excess_10d_min": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
}

if __name__ == "__main__":
    print(f"rs_leader param sweep — disabled signals: {sorted(DISABLED)}")
    print(f"Range: {START} → {END} | Capital: ${CAPITAL:,}\n")
    run_param_sensitivity(
        STOCK_UNIVERSE,
        START,
        END,
        param_ranges=param_ranges,
        initial_capital=CAPITAL,
        use_earnings_only=False,
        disabled_signals=DISABLED,
    )
