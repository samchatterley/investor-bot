#!/usr/bin/env bash
# Full backtest suite — long + short, all analysis modes.
# Runs sequentially; output tee'd to logs/backtest_suite_<timestamp>.log
# Usage: bash scripts/run_full_backtest_suite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

START="2015-01-01"
END="2023-12-31"
CAPITAL=25000
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/backtest_suite_$(date +%Y%m%d_%H%M%S).log"

run() {
    local label="$1"; shift
    echo "" | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "  $label" | tee -a "$LOG"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    .venv/bin/python -m backtest.engine --start "$START" --end "$END" --capital "$CAPITAL" "$@" 2>&1 | tee -a "$LOG"
}

echo "Full backtest suite — universe: $(python3 -c 'from config import STOCK_UNIVERSE; print(len(STOCK_UNIVERSE))') long symbols" | tee "$LOG"
echo "Start: $START  End: $END  Capital: $CAPITAL" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

# ── LONG SIDE ──────────────────────────────────────────────────────────────────
run "1/13  Long — core simulation"
run "2/13  Long — signal analysis (regime + hold-period breakdown)" --signal-analysis --use-earnings-only
run "3/13  Long — ablation (per-signal ΔSharpe)" --ablation --use-earnings-only
run "4/13  Long — backward elimination" --backward-elimination --use-earnings-only
run "5/13  Long — signal isolation (each signal standalone)" --signal-isolation --use-earnings-only
run "6/13  Long — Monte Carlo (Sharpe permutation + bootstrap CI)" --monte-carlo --monte-carlo-n 1000 --use-earnings-only
run "7/13  Long — multi-fold walk-forward (fold sensitivity)" --multi-fold
run "8/13  Long — crisis slices (GFC / COVID / 2022)" --crisis-slices
run "9/13  Long — co-firing analysis" --co-firing --use-earnings-only
run "10/13 Long — param sensitivity" --param-sensitivity
run "11/13 Long — walk-forward optimised" --walk-forward --use-earnings-only

# ── SHORT SIDE ─────────────────────────────────────────────────────────────────
run "12/13 Short — core simulation + signal analysis" --short-signals
run "13/13 Short — ablation" --short-ablation
run "14/13 Short — backward elimination" --short-backward-elimination
run "15/13 Short — regime analysis" --short-regime-analysis
run "16/13 Short — param sensitivity" --short-param-sensitivity
run "17/13 Short — walk-forward (all signals)" --short-walk-forward
run "18/13 Short — walk-forward overbought_downtrend" --short-walk-forward-ordt
run "19/13 Short — walk-forward parabolic_exhaustion" --short-walk-forward-pe

echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  ALL DONE — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "  Full log: $LOG" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
