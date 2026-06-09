# Pending Work

### In Progress
- [ ] Commit Batch 1 as v1.94 and update README

### Pending
- [ ] Implement correlation_regime_gate — rolling 20d within-sector correlation; >0.75 dampen signal confidence; <0.35 boost
- [ ] Implement vol_of_vol_signal — rolling 10d std dev of daily VIX changes; >3.5 reduce all sizes 30%; <1.0 allow boosted sizing
- [ ] Implement breadth_thrust signal — Zweig % above 50d SMA from <40% to >60% in 10 days
- [ ] Implement new_high_low_ratio gate — NH/NL >2.0 for 3 days boosts momentum; <0.5 dampens longs and boosts shorts
- [ ] Implement premarket_gap_quality — yfinance prepost pre-market prices; suppress gap_and_go if gap retraces >50% by 09:35
- [ ] Implement spread_proxy_gate — (High-Low)/midpoint 20d avg >0.5% = round-trip cost too high for short-hold signals
- [ ] Implement options_skew_signal — 25-delta put/call IV ratio; >1.4 panic hedging = contrarian long; call skew spike = informed upside
- [ ] Implement iv_vs_rv_spread — replace hv_rank with ATM IV vs 20d realised vol; IV/RV <0.7 = cheap vol long; >1.8 = premium seller edge
- [ ] Implement put_call_ratio_gate — stock-level put OI / call OI; >2.5 = contrarian long; <0.4 = dampen longs; index-level feeds regime
- [ ] Implement unusual_options_activity signal — single-day OTM call OI change >300% of prior 5d avg on strikes >10% OTM
- [ ] Implement credit_spread_gate — HYG/LQD price ratio 10d ROC falling >2% = credit stress = suppress all longs
- [ ] Implement duration_flight_signal — TLT 5d return outperforming SPY by >3% = flight-to-safety = DEFENSIVE_DOWNTREND confirmation
- [ ] Implement yield_curve_regime gate — TNX minus IRX; inversion >20 days = late-cycle, dampen momentum longs at 50%; >1.5 and rising = expansion boost
- [ ] Implement yield_curve_macro — FRED T10Y2Y; sustained <0 for >60 days = recession risk; >1.5 and rising = expansion boost
- [ ] Implement pmi_regime_signal — FRED NAPM; PMI 3m trend >55 boosts cyclical signals; <45 suppresses all longs
- [ ] Implement earnings_revision_cycle — weekly % of universe with rising vs falling EPS estimates; >55% rising = revision tailwind
- [ ] Implement initial_claims_trend — FRED ICSA 4-week MA rising ≥6 consecutive weeks = labour deterioration → DEFENSIVE_DOWNTREND weight
- [ ] Implement copper_gold_ratio signal — CPER/GLD 20d trend; positive = expansion = boost cyclicals; negative = risk-off = boost defensives
- [ ] Implement dollar_strength_gate — UUP 20d trend; strong USD = dampen longs on stocks with >40% international revenue
- [ ] Implement gross_margin_trend_gate — current quarter GM vs 4-quarter avg; long gate requires stable/improving; short supplement on >3pp deterioration
- [ ] Implement piotroski_f_score gate — F≥7 required for value signals; F≤2 = standalone short candidate
- [ ] Implement altman_z_score distress filter — Z>2.6 required for all long entries; Z<1.1 = distress zone = short candidate
- [ ] Implement accruals_quality_gate — (net income - OCF) / total assets; >0.10 = suppress longs; >0.15 + extended = quality short
- [ ] Implement fcf_yield_signal — FCF / market cap >5% + rising trend = conviction boost to pead and iv_compression
- [ ] Implement forward_pe_gate — suppress momentum/breakout longs on P/E >60
- [ ] Implement earnings_yield_vs_bonds — forward earnings yield minus ^TNX/100; ERP <1% dampen longs 50%; ERP >4% boost long scoring
- [ ] Implement relative_pe_gate — stock P/E vs sector median; >2x sector = suppress momentum; <0.7x + positive momentum = value+momentum boost
- [ ] Implement short_interest_trend signal — SI% rising >20% = bearish; falling >30% from peak + price rising = squeeze long
- [ ] Implement aaii_sentiment_signal — bears >50% for 2+ weeks = contrarian long gate; bulls >60% for 3+ weeks = dampen longs
- [ ] Implement fear_greed_composite — 7-component index; <20 = extreme fear long; >80 = caution
- [ ] Implement google_trends_signal — pytrends weekly queries; search spike + positive context = long supplement
- [ ] Implement activist_13d_signal — EDGAR SC 13D filing from known activist list; long within 5 days of filing
- [ ] Implement guidance_change_signal — EDGAR 8-K FinBERT classification; positive score >0.7 = guidance raise long; negative = guidance cut short
- [ ] Implement secondary_offering_short — EDGAR 424B4/S-3 prospectus detector; new secondary priced = supply shock short
- [ ] Implement lockup_expiry_short — 180-day IPO lockup calendar; short setup 5-10 days before expiry
- [ ] Implement analyst_revision_signal — yfinance recommendations_summary; consensus shift Hold→Buy = long; Buy→Hold/Underperform = short
- [ ] Implement index_rebalance_signal — S&P 500 add/delete announcements; long additions at close; short deletions immediately
- [ ] Implement turn_of_month_gate — days -2 to +3 around month-end boost long priority
- [ ] Implement opex_week_context — week of 3rd Friday; dampen gap_and_go and momentum; boost post-OPEX directional release
- [ ] Implement halloween_regime_modifier — Nov-Apr lower confidence hurdle; May-Oct raise hurdle
- [ ] Implement quarter_end_window_dressing — last 5 trading days of each quarter; boost momentum conviction
- [ ] Implement tax_loss_reversal — track stocks down >30% YTD in Nov/Dec; flag for January long entry
- [ ] Implement pre_holiday_boost — trading day before NYSE holidays; small long-side scoring boost
- [ ] Implement sector_pair_mean_reversion — long top-quartile RS / short bottom-quartile RS within GICS sub-industry when spread >1.5x historical std dev
- [ ] Implement squeeze_setup_long — SI% >20% float + days_to_cover >5 + ret_5d <5% + price near 20d low; crowded dormant short = pre-squeeze long entry
- [ ] Implement squeeze_momentum_long — SI% >20% float + days_to_cover >5 + ret_5d >10% + price breaking above 20d high; squeeze in motion = momentum continuation long

