# ADR-006 B4 — disabled short-signal backtest findings

**Question:** should any of the 17 `SHORT_GLOBALLY_DISABLED` signals be re-enabled now that B2 routes
shorts through the AI? **Research only — no signal was re-enabled.**

**Method.** [`scripts/short_disabled_backtest.py`](../scripts/short_disabled_backtest.py) temporarily
un-disables *one* signal at a time, runs the engine's real short simulation isolated to it
(`signals_only={sig}`), and reports trades / win-rate / avg-return / Sharpe. It reuses the
`run_short_ablation` data-prep so the numbers are comparable to the rest of the short backtest suite.

**Run:** universe = full 907-symbol list · window = **2015-01-01 → 2023-12-31** (pre-holdout) ·
$25k · `max_short=2` · borrow cost **not** modelled · earnings prefetch skipped (`--no-earnings`,
yfinance's per-symbol earnings endpoint was rate-limiting; see "Not evaluated" below).

## Result — no re-enable candidates

| Signal | Trades | WR% | Avg% | Sharpe | Verdict |
|---|---:|---:|---:|---:|---|
| volume_climax_reversal_short | 22 | 59.1 | +1.49 | 0.37 | KEEP (n<30) |
| high_vol_reversal | 150 | 46.7 | −0.12 | −0.06 | KEEP |
| overbought_downtrend | 125 | 47.2 | −0.58 | −0.27 | KEEP |
| iv_compression_short (HV proxy) | 248 | 40.3 | −0.64 | −0.31 | KEEP |
| candle_exhaustion_short | 145 | 40.0 | −0.54 | −0.35 | KEEP |
| death_cross | 283 | 35.7 | −1.01 | −0.53 | KEEP |
| failed_breakout | 270 | 38.1 | −0.80 | −0.67 | KEEP |
| obv_acceleration_short | 417 | 36.2 | −2.03 | −0.67 | KEEP |
| parabolic_exhaustion | 263 | 34.2 | −1.45 | −0.68 | KEEP |
| rs_deterioration | 394 | 31.5 | −1.95 | −0.90 | KEEP |
| obv_divergence_short | 338 | 35.5 | −1.37 | −1.00 | KEEP |

**Conclusion: keep all of them disabled.** Every signal with a usable sample (n = 125–417) has
**negative** average return *and* negative Sharpe. The only positive-expectancy line,
`volume_climax_reversal_short`, has just 22 trades over 9 years — too few to act on, and it would be
the first to erode once borrow cost is charged. Borrow cost is not modelled, so the true numbers are
**worse** than shown. This independently confirms the original disable decisions (v1.80 / v1.94 / v1.99).

## Not evaluated in this run (need a dedicated pass before any verdict)

- **earnings_miss, faded_earnings_gap_up** — need `prefetch_earnings_history` (skipped here; throttled).
  A small calibration run (80 symbols, 2020–2022, earnings ON) hinted `faded_earnings_gap_up` may be
  worth a proper look (avg +3.3%, Sharpe 0.77, but only n=8); `earnings_miss` was negative there too.
  Re-run `scripts/short_disabled_backtest.py` *without* `--no-earnings` (allow it ~30–60 min) to settle.
- **altman_distress_short, gross_margin_deterioration_short** — need point-in-time quality
  fundamentals, which `_run_short_simulation` does not receive on this path → 0 trades. Evaluate via the
  `run_combined_analysis(..., use_quality_fundamentals=True)` path instead.
- **ema_breakdown, winner_reversal** — 0 trades even with OHLCV (rarely fire / superseded); no evidence
  to revisit.

## Reproduce

```bash
python scripts/short_disabled_backtest.py --start 2015-01-01 --end 2023-12-31 --no-earnings
# add the earnings signals (slow):  drop --no-earnings
# quick smoke:                       --limit 80 --start 2020-01-01 --end 2022-12-31
```

Any future re-enable must still clear the full **CLAUDE.md disable/enable checklist** plus a
borrow-cost-realistic, out-of-sample / walk-forward check — not this in-sample screen alone.
