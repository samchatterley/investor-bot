"""Param sweep for rs_leader restricted to BULL_TREND only.

Patches REGIME_BLOCKED so rs_leader fires exclusively in BULL_TREND/BULL_TRENDING,
blocked in every other regime. All other long signals disabled for clean isolation.

Prior data at defaults: BULL_TREND n=246, WR 51%, avg -0.13% (p>0.05) — exploring
whether tighter thresholds find edge at lower trade counts.

Usage:
    .venv/bin/python scripts/sweep_rs_leader_bull.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import signals.evaluator as _ev
from backtest.engine import run_param_sensitivity
from config import STOCK_UNIVERSE
from signals.evaluator import SIGNAL_PRIORITY

START = "2015-01-01"
END = "2023-12-31"
CAPITAL = 25_000

# rs_leader fires ONLY in BULL_TREND/BULL_TRENDING; blocked in every other regime
_ev.REGIME_BLOCKED = {
    k: (v - {"rs_leader"} if k in {"BULL_TREND", "BULL_TRENDING"} else v | {"rs_leader"})
    for k, v in _ev.REGIME_BLOCKED.items()
}

ALL_LONG_SIGNALS = frozenset(SIGNAL_PRIORITY.keys())
DISABLED = ALL_LONG_SIGNALS - {"rs_leader"}

param_ranges = {
    "rsl_excess_5d_min": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "rsl_excess_10d_min": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
}

if __name__ == "__main__":
    print("rs_leader BULL_TREND sweep — blocked in all other regimes")
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
