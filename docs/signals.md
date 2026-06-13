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
| `momentum` | EMA9 > EMA21 + MACD positive + positive 5d return + high volume; blocked in NEUTRAL_CHOP and DEFENSIVE_DOWNTREND | 5 days |
| `macd_crossover` | MACD line crosses above signal line + volume; blocked in NEUTRAL_CHOP and DEFENSIVE_DOWNTREND | 4 days |
| `golden_cross` | SMA50 crosses above SMA200 + vol_ratio ≥ 0.8; regime-agnostic | 5 days |
| `candle_exhaustion` | Hammer or bullish engulf at 20d low with vol_ratio ≥ 1.5; blocked in STRESS_RISK_OFF and HIGH_VOL_DOWNTREND | 3 days |
| `obv_divergence` | OBV 5d slope rising while price 5d negative (accumulation divergence) + vol_ratio ≥ 1.0; blocked in STRESS_RISK_OFF | 3 days |
| `obv_acceleration` | OBV 5d slope > OBV 20d slope (accelerating into price) + EMA aligned or MACD positive + vol_ratio ≥ 1.2; blocked in STRESS_RISK_OFF | 3 days |
| `breadth_thrust` | Zweig breadth-thrust: universe breadth jumps from <40% to >60% above 50d SMA within 10 days; EMA9 > EMA21 required | 4 days |
| `gap_and_go` | Intraday gap ≥ 2% + close above open + volume + ADX ≥ 20; blocked in NEUTRAL_CHOP and DEFENSIVE_DOWNTREND | 5 days |

### Live-only signals (no historical backtest data; accumulating paper-trading evidence from v1.98)

| Signal | Entry conditions | Hold limit |
|--------|-----------------|------------|
| `options_skew_signal` | Panic put-skew spike (contrarian long) or call-skew spike (informed upside); requires options data post-filter injection | 3 days |
| `unusual_options_activity` | Large OTM call open-interest surge — informed upside conviction; requires options data | 3 days |
| `put_call_contrarian` | Put/call OI ratio > 2.5 (extreme panic hedging) + EMA or MACD confirmation — contrarian long | 3 days |
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

---

## Signal families

| Family | Signals |
|--------|---------|
| Mean-reversion | `mean_reversion` |
| Volatility / IV | `bb_squeeze`, `inside_day_breakout`, `iv_compression`, `iv_vs_rv_spread` |
| Trend / momentum | `trend_pullback`, `momentum`, `macd_crossover`, `gap_and_go` |
| OHLCV technical | `golden_cross`, `candle_exhaustion`, `obv_divergence`, `obv_acceleration`, `breadth_thrust` |
| Catalyst / fundamental | `pead`, `insider_buying`, `activist_13d_signal`, `guidance_raise_signal`, `fcf_yield_signal` |
| Options (live-only) | `options_skew_signal`, `iv_vs_rv_spread`, `unusual_options_activity`, `put_call_contrarian` |
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

---

## Short signals

Active short signals (regime-gated: STRESS_RISK_OFF, HIGH_VOL_DOWNTREND, DEFENSIVE_DOWNTREND, CREDIT_STRESS only).

| Signal | Entry conditions |
|--------|-----------------|
| `earnings_gap_down` | Post-earnings gap down ≥ 7% with vol_ratio ≥ 2.5 — negative PEAD continuation |
| `death_cross` | SMA50 crosses below SMA200 + vol_ratio ≥ 0.8 |
| `altman_distress_short` | Altman Z < 1.1 — financial distress zone |
| `piotroski_distress_short` | Piotroski F ≤ 2 + price below SMA200 |
| `gross_margin_deterioration_short` | GM trend < −3pp + price below SMA200 |
| `accruals_quality_short` | Accruals ratio > 0.15 + ret_5d > 5% (extended price) |
| `lockup_expiry_short` | IPO lockup expires in 5–10 calendar days |
| `analyst_downgrade_signal` | Consensus shift from Buy toward Hold/Sell |
