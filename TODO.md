# Pending Work

### Pending
- [ ] FIX zero-trade signals — debug why rs_leader, insider_buying, breakout_52w, rsi_divergence, orb_breakout, vwap_reclaim, intraday_momentum fire 0 times across 509 symbols × 9 years
- [ ] FIX vix_fear_reversion — disable or redesign: standalone Sharpe -0.380, return -28.5%, WR 44%
- [ ] Analyse backtest results — bb_squeeze dominance, signal isolation rankings, backward elimination survivors
- [ ] Extend regime model v2 — add credit spread, breadth, T10Y2Y, Fear & Greed composite as classification inputs
- [ ] Add RECOVERY, LATE_CYCLE_BULL, CREDIT_STRESS regime states
- [ ] Implement signal_invalidation_exit — re-evaluate entry conditions daily; exit when fully reversed
- [ ] Improve bb_squeeze — 5-day consecutive squeeze duration, ADX floor 25, RS rank gate top 40%, $10 min price
- [ ] Improve pead — EPS surprise >10%, revenue beat required, entry within 5 days, sector context gate
- [ ] Improve momentum_12_1 — add 1-week pullback filter (bottom 40% 1w return while top 20% 12-1m)
- [ ] Implement sector_momentum_rank gate — rank 11 SPDR ETFs; only allow longs in top 4, shorts in bottom 3
- [ ] Implement garch_vol_forecast for position sizing — GARCH(1,1) per symbol; scale down when vol elevated
- [ ] Implement momentum_quality_score — composite 0-3 score (momentum rank + Piotroski + EPS revision); score 3 = 1.5x size
- [ ] Build data/fundamental_cache.py — weekly yfinance financials cache for 509 symbols
- [ ] Implement breadth_thrust signal — Zweig % above 50d SMA from <40% to >60% in 10 days
- [ ] Implement golden_death_cross signal — SMA50 cross above/below SMA200
- [ ] Implement candle_exhaustion signal — hammer/engulfing at 20d extremes with vol_ratio >1.5
- [ ] Implement options_skew_signal — 25-delta put/call IV ratio; >1.4 panic hedging, call spike = informed upside
- [ ] Implement short_interest_trend signal — SI% rising >20% = bearish; falling >30% + price rising = squeeze long
- [ ] Implement aaii_sentiment_signal — bears >50% for 2+ weeks = contrarian long gate
- [ ] Implement turn_of_month_gate — days -2 to +3 around month-end boost long priority
- [ ] Implement halloween_regime_modifier — Nov-Apr lower confidence hurdle; May-Oct raise hurdle
- [ ] Implement sector_pair_mean_reversion — long top-quartile RS / short bottom-quartile RS within sub-industry
- [ ] Implement amihud_illiquidity_gate — reduce position size 50% for top 10% illiquid symbols

