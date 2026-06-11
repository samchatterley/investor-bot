# Pending Work

### Pending
- [ ] Market microstructure: correlation_regime_gate — rolling 20d within-sector correlation; >0.75 dampen signal confidence; <0.35 boost
- [ ] Market microstructure: new_high_low_ratio gate — NH/NL >2.0 for 3 days boosts momentum; <0.5 dampens longs and boosts shorts
- [ ] Market microstructure: premarket_gap_quality — yfinance prepost pre-market prices; suppress gap_and_go if gap retraces >50% by 09:35
- [ ] Options signals: options_skew_signal — 25-delta put/call IV ratio; >1.4 panic hedging = contrarian long; call skew spike = informed upside
- [ ] Options signals: iv_vs_rv_spread — replace hv_rank with ATM IV vs 20d realised vol; IV/RV <0.7 = cheap vol long; >1.8 = premium seller edge
- [ ] Options signals: put_call_ratio_gate — stock-level put OI / call OI; >2.5 = contrarian long; <0.4 = dampen longs; index-level feeds regime
- [ ] Options signals: unusual_options_activity signal — single-day OTM call OI change >300% of prior 5d avg on strikes >10% OTM
- [ ] Fundamentals: gross_margin_trend_gate — current quarter GM vs 4-quarter avg; long gate requires stable/improving; short supplement on >3pp deterioration
- [ ] Fundamentals: piotroski_f_score gate — F≥7 required for value signals; F≤2 = standalone short candidate
- [ ] Fundamentals: altman_z_score distress filter — Z>2.6 required for all long entries; Z<1.1 = distress zone = short candidate
- [ ] Fundamentals: accruals_quality_gate — (net income - OCF) / total assets; >0.10 = suppress longs; >0.15 + extended = quality short
- [ ] Fundamentals: fcf_yield_signal — FCF / market cap >5% + rising trend = conviction boost to pead and iv_compression
- [ ] Fundamentals: forward_pe_gate — suppress momentum/breakout longs on P/E >60
- [ ] Fundamentals: earnings_yield_vs_bonds — forward earnings yield minus TNX/100; ERP <1% dampen longs 50%; ERP >4% boost long scoring
- [ ] Fundamentals: relative_pe_gate — stock P/E vs sector median; >2x sector = suppress momentum; <0.7x + positive momentum = value+momentum boost
- [ ] Alternative data: short_interest_trend signal — SI% rising >20% = bearish; falling >30% from peak + price rising = squeeze long
- [ ] Alternative data: aaii_sentiment_signal — bears >50% for 2+ weeks = contrarian long gate; bulls >60% for 3+ weeks = dampen longs
- [ ] Alternative data: fear_greed_composite — 7-component index; <20 = extreme fear long; >80 = caution
- [ ] Alternative data: google_trends_signal — pytrends weekly queries; search spike + positive context = long supplement
- [ ] Alternative data: activist_13d_signal — EDGAR SC 13D filing from known activist list; long within 5 days of filing
- [ ] Alternative data: guidance_change_signal — EDGAR 8-K FinBERT classification; positive score >0.7 = guidance raise long; negative = guidance cut short
- [ ] Alternative data: lockup_expiry_short — 180-day IPO lockup calendar; short setup 5-10 days before expiry
- [ ] Alternative data: analyst_revision_signal — yfinance recommendations_summary; consensus shift Hold→Buy = long; Buy→Hold/Underperform = short
- [ ] Alternative data: index_rebalance_signal — S&P 500 add/delete announcements; long additions at close; short deletions immediately
- [ ] Cross-asset: sector_pair_mean_reversion — long top-quartile RS / short bottom-quartile RS within GICS sub-industry when spread >1.5x historical std dev
- [ ] Cross-asset: squeeze_setup_long — SI% >20% float + days_to_cover >5 + ret_5d <5% + price near 20d low; crowded dormant short = pre-squeeze long
- [ ] Cross-asset: squeeze_momentum_long — SI% >20% float + days_to_cover >5 + ret_5d >10% + price breaking above 20d high; squeeze in motion
- [ ] Backtest validation: backtest and enable candle_exhaustion_short, obv_divergence_short, obv_acceleration_short, volume_climax_reversal_short, iv_compression_short

