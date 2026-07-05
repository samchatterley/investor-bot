# InvestorBot — Signal Reference

Full signal catalogue for the pre-filter layer (`execution/stock_scanner.py`). Each signal must fire before a candidate reaches Claude.

---

## Daily signals

Computed from end-of-day bar history via yfinance.

| Signal | Entry conditions | Hold limit |
|--------|-----------------|------------|
| `pead` | Post-Earnings Announcement Drift: EPS beat ≥10% within 7 days + price drifting up (ret_5d > 0); bypasses regime filter | 3 days |
| `insider_buying` | ≥2 distinct corporate insiders made open-market Form 4 purchases (SEC EDGAR) within 10 days; bypasses regime filter | 5 days |
| `activist_13d_signal` | SC 13D activist filing within 30 days + EMA aligned — management-change catalyst; bypasses quality gates | 5 days |
| `guidance_raise_signal` | Positive 8-K guidance event; fires without price confirmation (earlier entry than `pead`); blocked by gross-margin deterioration gate | 3 days |
| `fcf_yield_signal` | FCF yield > 5% + Piotroski F ≥ 5 — high-quality free-cash-flow value buy; 563 backtest trades WR 51% avg +0.16%; elevated to priority 12 in v1.98 | 5 days |
| `bb_squeeze` | Bollinger bandwidth at 20th percentile of last 20 bars + ADX ≥ 25 + RS rank ≥ 60 + directional confirmation; blocked in NEUTRAL_CHOP and DEFENSIVE_DOWNTREND | 4 days |
| `inside_day_breakout` | Prior candle's full range contains today's; breaks with directional confirmation + volume; blocked in NEUTRAL_CHOP and DEFENSIVE_DOWNTREND | 3 days |
| `trend_pullback` | EMA9 > EMA21, price 0.5–2% below EMA21, RSI 50–58 — buying the dip in a healthy trend; blocked in DEFENSIVE_DOWNTREND and NEUTRAL_CHOP | 3 days |
| `iv_compression` | Historical volatility rank in bottom 15th percentile of 52-week range + EMA/MACD confirmation + volume; blocked in NEUTRAL_CHOP and STRESS_RISK_OFF | 4 days |
| `iv_vs_rv_spread` | ATM IV / 20d realised vol < 0.70 — options market underpricing risk vs realised; EMA or MACD confirmation + volume | 4 days |
| `mean_reversion` | RSI < 35 + BB < 0.15 + vol spike; blocked in NEUTRAL_CHOP, DEFENSIVE_DOWNTREND, and STRESS_RISK_OFF | 2 days |
| `residual_reversal` | Market-excess 5d return (stock − SPY) ≤ −7% — idiosyncratic laggard reverting over 1–3d; blocked in STRESS_RISK_OFF and HIGH_VOL_DOWNTREND (2026-07 workshop, N1) | 3 days |
| `momentum` | EMA9 > EMA21 + MACD positive + positive 5d return + high volume; blocked in NEUTRAL_CHOP and DEFENSIVE_DOWNTREND | 5 days |
| `candle_exhaustion` | Hammer or bullish engulf at 20d low with vol_ratio ≥ 1.5; blocked in STRESS_RISK_OFF and HIGH_VOL_DOWNTREND | 3 days |
| `breadth_thrust` | Zweig breadth-thrust: universe breadth jumps from <40% to >60% above 50d SMA within 10 days; EMA9 > EMA21 required | 4 days |
| `gap_and_go` | Intraday gap ≥ 2% + close above open + volume + ADX ≥ 20; blocked in NEUTRAL_CHOP and DEFENSIVE_DOWNTREND | 5 days |

### Live-only signals (no historical backtest data; accumulating paper-trading evidence from v1.98)

| Signal | Entry conditions | Hold limit |
|--------|-----------------|------------|
| `options_skew_signal` | Panic put-skew spike (contrarian long) or call-skew spike (informed upside); requires options data post-filter injection | 3 days |
| `squeeze_setup_long` | High short interest + price dormant near 20d low — crowded short pre-squeeze setup | 5 days |
| `squeeze_momentum_long` | High short interest + strong 5d return + price above 20d high — active short squeeze in motion | 4 days |
| `short_interest_trend_long` | SI% falling >30% from peak + price rising — short covering into strength (informed capitulation) | 5 days |
| `analyst_upgrade_signal` | Analyst buy% rose >10pp month-over-month (min 3 analysts) | 5 days |
| `aaii_extreme_fear_long` | AAII survey bears > 50% — contrarian long backdrop | 3 days |
| `fear_greed_extreme_fear` | Composite fear/greed index < 20 (VIX 30%, AAII 25%, NH/NL 20%, SPY 15%, breadth 10%) | 3 days |
| `sector_pair_mean_reversion` | Intra-sector RS spread z-score extended — long the RS laggard as spread reverts to sector mean | 5 days |
| `google_trends_bullish` | Search-interest spike: current week ≥ 150% of 12-week baseline (minimum baseline 10); `pytrends`-powered; confirming signal only (injected post-prefilter, cannot cause candidacy) | 3 days |

---

## Intraday signals

Computed from Alpaca minute bars; available on any run during market hours.

| Signal | Entry conditions |
|--------|-----------------|
| `vwap_reclaim` | Price above VWAP + >1% gain from open + not overextended (pct vs VWAP ≤ 3%) |
| `orb_breakout` | Price broke above the first-30-minute high with above-average volume |
| `intraday_momentum` | >2% gain from open + above VWAP + intraday RSI < 75 + daily trend confirms |

Intraday signals enable the midday run (12:00 ET) to execute new buys. Intraday positions are force-covered at the 15:30 ET close run.

---

## Globally disabled signals

Signals removed after statistically evidenced negative contribution. Blocked in both live scanning and all backtest modes.

| Signal | Reason | Disabled |
|--------|--------|---------|
| `rsi_divergence` | WR 48%, avg −0.9% in NEUTRAL_CHOP (75% of its trades); Sharpe drag in every run | v1.x |
| `breakout_52w` | WR 35%, avg −1.5% in BULL_TREND (its only firing regime); consistently negative | v1.x |
| `vix_fear_reversion` | Zero trades in all backtest runs; never fires in practice | v1.x |
| `rs_leader` | Standalone Sharpe −0.93 over 9-year walk-forward; no edge at any param threshold | v1.x |
| `momentum_12_1` | WR 48%, avg −0.2% in every tested regime; ΔSharpe +0.08 from removal | v1.x |
| `range_reversion` | 2 trades in combined production backtest, WR 0%, avg −16.2%; backward elimination Step 3 (ΔSharpe +0.04); conditions too restrictive to fire reliably | v1.98 |
| `volume_climax_reversal` | 1 trade in combined production backtest, WR 0%, avg −2.8%; fires too rarely to contribute | v1.98 |
| `tax_loss_reversal` | 38 trades, WR 37%, avg −1.02%; January-only seasonal with no confirmed reversal edge | v1.98 |
| `obv_divergence` | Below-50% WR, negative avg in every window; ΔSharpe +0.12 from joint OBV removal (doc corrected — was already in `GLOBALLY_DISABLED`) | v1.x |
| `obv_acceleration` | Consistently <50% WR, negative avg; joint OBV removal ΔSharpe +0.12 (doc corrected — was already in `GLOBALLY_DISABLED`) | v1.x |
| `macd_crossover` | 2026-07 signal workshop — standalone isolation event study (907 names, 2015–2026, cost-swept): gross −0.008%/4d (t=−0.37), negative pre-cost; subsumed by `momentum` | v1.136 |
| `golden_cross` | 2026-07 signal workshop — standalone isolation: gross +0.038%/5d, break-even 3.8bps, *below* the universe baseline (5.2bps); no edge (wrong horizon) | v1.136 |
| `unusual_options_activity` | 2026-07 options kill/keep (Alpaca 2.4y history, first-ever evidence): premise **inverted** — call-volume spikes precede −0.178%/3d (t=−2.4, 0/3 yrs); retail chase, not informed buying | v1.143 |
| `put_call_contrarian` | 2026-07 options kill/keep: flat on the volume proxy (+0.037%/3d, t=0.99); no supporting evidence after 2 years live | v1.143 |

---

## Signal families

| Family | Signals |
|--------|---------|
| Mean-reversion | `mean_reversion`, `residual_reversal` |
| Volatility / IV | `bb_squeeze`, `inside_day_breakout`, `iv_compression`, `iv_vs_rv_spread` |
| Trend / momentum | `trend_pullback`, `momentum`, `gap_and_go` |
| OHLCV technical | `candle_exhaustion`, `breadth_thrust` |
| Catalyst / fundamental | `pead`, `insider_buying`, `activist_13d_signal`, `guidance_raise_signal`, `fcf_yield_signal` |
| Options (live-only) | `options_skew_signal`, `iv_vs_rv_spread` |
| Short squeeze (live-only) | `squeeze_setup_long`, `squeeze_momentum_long`, `short_interest_trend_long` |
| Sentiment / alt-data | `aaii_extreme_fear_long`, `fear_greed_extreme_fear`, `analyst_upgrade_signal`, `google_trends_bullish` |
| Cross-asset | `sector_pair_mean_reversion` |
| Intraday | `vwap_reclaim`, `orb_breakout`, `intraday_momentum` |

---

## Fundamental quality gates

Applied before signal evaluation; block specific signals when conditions are met.

| Gate | Condition | Signals blocked |
|------|-----------|-----------------|
| Altman Z distress | Z < 1.1 | All trend/breakout/momentum longs (`_DISTRESS_BLOCKED`) |
| Piotroski quality | F < 3 | `pead`, `insider_buying` |
| Forward P/E expensive | P/E > 60 | `momentum`, `macd_crossover`, `gap_and_go`, `bb_squeeze`, `inside_day_breakout`, `trend_pullback` |
| Gross margin deterioration | GM trend < −3pp | `pead`, `guidance_raise_signal`, `trend_pullback` |
| Accruals quality | Accruals ratio > 0.10 | Same set as forward P/E expensive |
| NH/NL weak breadth | NHL ratio < 0.5 | `momentum`, `gap_and_go`, `bb_squeeze`, `inside_day_breakout`, `squeeze_momentum_long` |
| Sector correlation | Sector corr > 0.75 | `momentum`, `bb_squeeze` |
| ERP gate | 1/P/E − 10y yield < 1% | Same set as forward P/E expensive |
| AAII excessive bulls | AAII bulls > 60% for ≥1 week | `momentum`, `gap_and_go` |
| Lottery / MAX | ≥ +10% single-day pop within the last 3 sessions (2026-07 workshop: −0.44%/3d, t=−5.1) | `momentum`, `gap_and_go` |

---

## Short signals

Active short signals (regime-gated: STRESS_RISK_OFF, HIGH_VOL_DOWNTREND, DEFENSIVE_DOWNTREND, CREDIT_STRESS only).

| Signal | Entry conditions |
|--------|-----------------|
| `post_earnings_gapdown_failed_bounce` | Recent ≥7% earnings/news gap-down whose low is subsequently broken (the reflexive bounce failed) + vol_ratio ≥ 1.5 — negative-PEAD continuation entered *after* the bounce, not on the gap bar. Computed live in `scan_short_universe`; the one short with a documented short-horizon edge. |
| `earnings_gap_down` | Post-earnings gap down ≥ 7% with vol_ratio ≥ 2.5 — naive gap-day negative PEAD (superseded live by the failed-bounce variant above) |
| `piotroski_distress_short` | Piotroski F ≤ 2 + price below SMA200 |
| `accruals_quality_short` | Accruals ratio > 0.15 + ret_5d > 5% (extended price) |
| `lockup_expiry_short` | IPO lockup expires in 5–10 calendar days |
| `analyst_downgrade_signal` | Consensus shift from Buy toward Hold/Sell |
| `accounting_concern_short` | An 8-K restatement, non-reliance, or auditor change (EDGAR) — a hard governance red flag. RS-rank agnostic (catalyst path) |
| `insider_selling_short` | Cluster of ≥3 distinct insiders selling open-market (Form 4 code 'S') — informed distribution. RS-rank agnostic |
| `index_deletion_short` | Name removed from a major index (news-detected) — forced index-fund selling into the effective date. RS-rank agnostic |
| `eps_revision_down_short` | ≥3 downward current-quarter EPS estimate revisions (last 30d) that outnumber raises — the estimate-revision anomaly (cuts precede negative drift). RS-rank agnostic |
| `guidance_downgrade` | Negative 8-K guidance — management lowering outlook. RS-rank agnostic |
| `secondary_offering_short` | 424B4/S-3 secondary prospectus — dilution / supply shock. RS-rank agnostic |

The catalyst shorts (`accounting_concern_short`, `insider_selling_short`, `index_deletion_short`,
`eps_revision_down_short`, `analyst_downgrade_signal`, `guidance_downgrade`, `secondary_offering_short`)
are flag-driven corporate events surfaced regardless of RS rank (ADR-006 Tier-1): the short snapshots
are now enriched with the same EDGAR + Form-4 + analyst-revision feeds the long side uses (the
analyst-revision feed was previously never wired into the pipeline at all). They are not backtestable
(no historical point-in-time event feed) and ship live on forward-evidence, gated by the AI veto (B2).
`index_deletion_short` coverage is currently
limited to names with headlines in the long-side news set.

### Borrow cost & hard-to-borrow gate

Every short is now priced against an estimated annualized stock-borrow rate (`data/borrow_cost.py`), derived from short-interest tiers since no paid cost-to-borrow feed is available:

| short % of float | Estimated borrow rate | Tier |
|------------------|----------------------|------|
| < 5% | 0.5% | general collateral |
| 5–15% | 3% | moderate |
| 15–30% | 10% | elevated |
| 30–50% | 30% | hard-to-borrow (skipped live) |
| > 50% | 80% | special / often unborrowable |

- **Backtest**: borrow cost is netted from short P&L at every cover (the combined production backtest derives rates from the short-interest data it fetches). Prior to v1.99, short backtests modelled borrow as free and overstated short returns.
- **Live**: `_execute_shorts` skips hard-to-borrow names (rate ≥ 30%) and records `borrow_rate_annual` on every short.

### Index regime hedge (opt-in)

`_execute_index_hedge` shorts an index ETF (`INDEX_HEDGE_SYMBOL`, default SPY) at `INDEX_HEDGE_WEIGHT` of the portfolio when the regime is in `INDEX_HEDGE_REGIMES` (default STRESS_RISK_OFF, HIGH_VOL_DOWNTREND), and covers when the regime exits. Index ETFs borrow cheap, are deeply liquid, and carry no single-name squeeze risk — a structurally cleaner short than crowded single names. **Disabled by default** (`INDEX_HEDGE_ENABLED`): it is a live order path and must be explicitly opted into; it honours `dry_run`/`_live_shadow`. The backtest overlay (`compute_index_hedge_pnl`) reports the hedge's P&L contribution as `result["index_hedge"]`.

### Disabled short signals

The short book's core problem is that most short signals are *confirming* indicators, not *predictive* ones: by the time they fire, the market has already shorted the name. The lagging fundamental shorts below also encode multi-month theses that cannot resolve inside our 1–5 day hold. Disabled pending a rebuild around catalyst-anchored shorts + an index regime hedge (and a borrow-cost model, which we currently lack).

| Signal | Reason | Disabled |
|--------|--------|---------|
| `death_cross` | SMA50/200 cross fires ~15–30% into a decline; WR 32% (n=25) / 40% (n=121). Lagging confirmation. | v1.99 |
| `altman_distress_short` | Distress is a quarters-long thesis; 337 trades WR 45% avg −0.17% in a 3-day hold (wrong horizon, entered after credit desks). | v1.99 |
| `gross_margin_deterioration_short` | Slow fundamental; 5 trades, WR 40%, avg −1.32%. Too few to validate, worst avg of the trio. | v1.99 |
| `earnings_miss`, `ema_breakdown`, `rs_deterioration`, `faded_earnings_gap_up`, `overbought_downtrend`, `parabolic_exhaustion`, `failed_breakout`, `high_vol_reversal`, `winner_reversal` | Negative expectancy across all isolation/backward-elimination runs (see `SHORT_GLOBALLY_DISABLED` in `signals/evaluator.py`). | v1.x–v1.80 |
| `iv_compression_short`, `candle_exhaustion_short`, `obv_divergence_short`, `obv_acceleration_short`, `volume_climax_reversal_short` | New short signals disabled pending initial backtest validation. | v1.82–v1.94 |

> **Note on borrow cost:** no short backtest currently models stock-borrow cost, so all historical short results are optimistic. A borrow-cost model is the prerequisite for the planned short-book rebuild (`post_earnings_gapdown_failed_bounce` + `index_regime_hedge`).
