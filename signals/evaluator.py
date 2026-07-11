"""Canonical signal evaluation — single source of truth for both live scanner and backtest engine.

evaluate_signals() returns ALL signals that fire on a snapshot, sorted by priority.
Callers that need exactly one signal (backtest engine) take the first element.
Callers that want full coverage (live scanner, Claude context) use the full list.

Snapshot dict uses the market_data.py / scanner field names (rsi_14, ret_5d_pct, etc.).
The backtest engine converts its pd.Series rows via _row_to_snapshot() before calling here.
"""

from __future__ import annotations

# Priority order: lower number = higher priority within a single slot allocation.
SIGNAL_PRIORITY: dict[str, int] = {
    "vix_fear_reversion": 0,
    "insider_buying": 1,
    "activist_13d_signal": 2,  # activist SC 13D filing — catalyst-driven, bypasses quality gates
    "pead": 3,
    "guidance_raise_signal": 4,  # positive 8-K guidance event (early entry, no price-confirm required)
    # rs_leader blocked in BULL_TREND (no edge); breakout_52w < momentum_12_1 < gap_and_go
    "rs_leader": 5,
    "breakout_52w": 6,
    "momentum_12_1": 7,
    "gap_and_go": 8,
    "bb_squeeze": 9,
    "inside_day_breakout": 10,
    "trend_pullback": 11,
    # fcf_yield elevated from 29→12: 563 backtest trades WR 51% avg +0.16%; quality gate makes
    # it a fundamentals complement to the catalyst signals above.
    "fcf_yield_signal": 12,
    "iv_compression": 13,
    "iv_vs_rv_spread": 14,  # ATM IV/RV < 0.7 = vol genuinely cheap vs realised
    "range_reversion": 15,
    "rsi_divergence": 16,
    # residual_reversal (2026-07 workshop, N1): idiosyncratic 5d loser reverts over 1-3d; the
    # better-validated reversion signal, so it outranks mean_reversion when both fire.
    "residual_reversal": 16,
    # mean_reversion outranks momentum (counter-cyclical conviction beats trend-following)
    "mean_reversion": 17,
    "momentum": 18,
    "macd_crossover": 19,
    "orb_breakout": 20,
    "vwap_reclaim": 21,
    "intraday_momentum": 22,
    # ── Batch 1: OHLCV technical signals ────────────────────────────────────
    "golden_cross": 23,
    "candle_exhaustion": 24,
    "obv_divergence": 25,
    "obv_acceleration": 26,
    "volume_climax_reversal": 27,
    # ── Batch 2: universe-level signals ──────────────────────────────────────
    "breadth_thrust": 28,
    # ── Batch 3: calendar/seasonal signals ───────────────────────────────────
    "tax_loss_reversal": 29,
    # ── Batch 4: options-derived signals (live-only: no historical options data in backtest) ──
    "options_skew_signal": 30,  # panic put skew → contrarian; call skew spike → informed
    "unusual_options_activity": 31,  # OTM call OI surge — informed upside buying
    "put_call_contrarian": 32,  # extreme put-to-call ratio → contrarian long
    # ── Batch 5: short-squeeze signals (live-only: current SI data, no historical backtest) ──
    "squeeze_setup_long": 33,  # crowded dormant short at 20d low — pre-squeeze
    "squeeze_momentum_long": 34,  # squeeze in motion above 20d high
    "short_interest_trend_long": 35,  # SI% falling >30% from peak + price rising
    # ── Batch 6: analyst / institutional signals ──────────────────────────────
    "analyst_upgrade_signal": 36,  # consensus Hold→Buy shift
    # ── Batch 7: sentiment signals ────────────────────────────────────────────
    "aaii_extreme_fear_long": 37,  # AAII bears >50% → contrarian long
    "fear_greed_extreme_fear": 38,  # composite fear/greed <20 → contrarian long
    # ── Batch 8: cross-asset signals ─────────────────────────────────────────
    "sector_pair_mean_reversion": 39,  # intra-sector RS spread reversion
    # ── Batch 9: alternative data signals ────────────────────────────────────
    "google_trends_bullish": 40,  # search volume spike with positive context
}

# Signals that receive a size reduction during OPEX week (vol pinning / gamma effects).
# Applied as a 0.70× scalar in position_sizer.seasonal_scalar — not a hard block.
OPEX_WEEK_DAMPENED: frozenset[str] = frozenset({"gap_and_go", "momentum"})

# Signals that require Alpaca minute bars and must execute same-day (Open→Close).
# These must never fire in the multi-day backtest track (daily bars, overnight holds).
INTRADAY_SIGNALS: frozenset[str] = frozenset(
    {
        "orb_breakout",
        "vwap_reclaim",
        "intraday_momentum",
    }
)

# Intraday short signals — detected bar-by-bar from minute data (not via evaluate_signals).
# Classification constant only; actual detection is in backtest/intraday_engine.py.
INTRADAY_SHORT_SIGNALS: frozenset[str] = frozenset(
    {
        "orb_breakdown",  # Break below ORB low with above-avg volume
        "gap_up_failure",  # Gap-up open fails to hold; price reverses through open
        "vwap_rejection",  # Price touches VWAP from above and rejects back down
    }
)

# All long signals that belong to the multi-day track (daily bars, overnight holds).
MULTIDAY_SIGNALS: frozenset[str] = frozenset(SIGNAL_PRIORITY.keys()) - INTRADAY_SIGNALS

# Canonical default thresholds — used when no walk-forward params are supplied.
# These are the source of truth; backtest/engine.py imports them.
DEFAULT_SIGNAL_PARAMS: dict[str, float] = {
    # mean_reversion
    "rsi_threshold": 35.0,
    "bb_threshold": 0.15,  # tightened from 0.25 in v1.48
    "mr_vol_threshold": 1.2,
    # residual_reversal (N1): fire when market-excess 5d return (ret_5d - spy_ret_5d, %) <= this.
    # -7% chosen from the threshold sweep: net +0.31%/3d @7bps, t=7.6, 9/12 +yrs, break-even 37.6bps.
    "resid_rev_threshold": -7.0,
    # momentum
    "mom_vol_threshold": 1.3,
    "mom_ret5d_threshold": 1.0,
    "mom12_1_threshold": 10.0,
    "mom12_1_pullback_ret5d_max": 1.0,  # 1w return ≤ this confirms pullback entry, not chasing
    # rsi_divergence
    "rsi_div_rsi_max": 45.0,
    "rsi_div_vol_min": 1.0,
    "rsi_div_bb_max": 0.30,  # tightened from 1.0 in v1.46
    # vix_fear_reversion
    "vfr_vol_min": 1.5,  # raised from 1.0 in v1.48 — require genuine panic volume
    # rs_leader
    "rsl_excess_5d_min": 2.0,
    "rsl_excess_10d_min": 3.0,
    # breakout_52w
    "bk52_pct_min": -2.0,  # tightened from -3.0 in v1.48 — within 2% of 52w high
    "bk52_vol_min": 1.2,
    # gap_and_go
    "gap_pct_min": 2.0,
    "gap_vol_min": 2.0,  # raised from 1.5 in v1.48 — strong vol confirmation for gaps
    # bb_squeeze
    "bbs_vol_min": 1.2,
    "bbs_squeeze_days_min": 5,  # squeeze must persist ≥5 consecutive days (not a one-day dip)
    "bbs_adx_min": 25,  # raised from 20 — require more directional conviction
    "bbs_rs_rank_min": 60.0,  # only top-40% RS stocks (filters low-momentum candidates)
    # inside_day_breakout
    "idb_vol_min": 1.1,
    # trend_pullback
    "tp_ema21_lo": -2.0,  # tightened from -3.0 in v1.48 — buy closer to EMA support
    "tp_ema21_hi": -0.5,  # must be no more than this far below EMA21
    "tp_rsi_lo": 50.0,  # raised from 40.0 in v1.48 — healthy pullback only (RSI ≥50)
    "tp_rsi_hi": 58.0,  # RSI ceiling (not overbought)
    "tp_vol_min": 1.0,
    # iv_compression
    "ivc_hv_rank_max": 0.15,  # loosened from 0.10 in v1.82 — moderate compression still predictive
    "ivc_vol_min": 1.2,  # raised from 1.1 in v1.48
    # insider_buying
    "ib_comp_ratio_min": 0.02,  # standard cluster requires purchase ≥2% of annual comp
    # range_reversion
    "rr_adx_max": 20.0,  # ADX below this = range-bound
    "rr_bb_max": 0.10,  # price in extreme lower band
    "rr_rsi_max": 30.0,  # extreme oversold
    # macd_crossover
    "macd_vol_min": 1.2,
    # spread_proxy_gate — suppress execution-sensitive signals when avg daily spread > 0.5%
    "spread_proxy_max": 0.005,
    # golden_cross / death_cross — minimal vol threshold (cross event is the signal)
    "gc_vol_min": 0.8,
    "dc_vol_min": 0.8,
    # candle_exhaustion: bullish/bearish reversal candle at 20d extreme with elevated vol
    "cex_vol_min": 1.5,
    # obv_divergence: price vs OBV trend divergence with vol confirmation
    "obv_div_vol_min": 1.0,
    # obv_acceleration: short-term OBV rate faster than long-term + vol confirmation
    "obv_acc_vol_min": 1.2,
    # volume_climax_reversal: consecutive extreme-volume days at 20d extreme
    "vcr_streak_min": 3,
    # breadth_thrust: require minimum universe coverage to trust the Zweig signal
    "bt_min_symbols": 50,
    # tax_loss_reversal: 52w high drawdown threshold (stock must be >30% below 52w high)
    "tlr_52w_drawdown_max": -30.0,
    # ── Fundamental gates ────────────────────────────────────────────────────
    "az_distress_threshold": 1.1,  # Altman Z < 1.1 = distress zone → block trend longs
    "piotroski_min": 3,  # Piotroski F < 3 → block pead and insider_buying
    "fwd_pe_max": 60.0,  # forward P/E > 60 → block momentum/breakout longs
    "gm_trend_min": -0.03,  # gross margin trend < -3pp → block pead, trend_pullback
    "accruals_max": 0.10,  # accruals ratio > 0.10 → suppress longs
    "erp_min": 0.01,  # equity risk premium < 1% → block momentum longs
    # ── fcf_yield_signal ─────────────────────────────────────────────────────
    "fcf_yield_min": 0.05,  # FCF yield > 5% required
    "fcf_piotroski_min": 5,  # also requires moderate quality (F ≥ 5)
    # ── options signals ──────────────────────────────────────────────────────
    "iv_rv_spread_cheap_max": 0.70,  # IV/RV < 0.70 = vol genuinely cheap
    "put_call_contrarian_min": 2.5,  # put_call_oi_ratio > 2.5 → contrarian long
    # ── short-squeeze signals ────────────────────────────────────────────────
    "squeeze_si_min": 0.15,  # short_pct_float > 15% (0.15 as decimal)
    "squeeze_dtc_min": 5.0,  # days_to_cover > 5
    "squeeze_setup_ret5d_max": 5.0,  # dormant: ret_5d < 5%
    "squeeze_mom_ret5d_min": 10.0,  # momentum: ret_5d > 10%
    "si_trend_drop_min": 0.30,  # SI fell > 30% from peak to flag trend reversal
    # ── breadth / macro gates ────────────────────────────────────────────────
    "nhl_ratio_bear_max": 0.5,  # NHL < 0.5 → block momentum
    "sector_corr_high": 0.75,  # sector correlation > 0.75 → treat as correlated regime
    # ── analyst revision signal ──────────────────────────────────────────────
    "analyst_upgrade_pct_min": 0.10,  # buy % must rise by at least this much
    # ── pairs mean reversion ─────────────────────────────────────────────────
    "pairs_zscore_min": 1.5,  # spread z-score must exceed 1.5σ
}

# Canonical regime-blocked signal set — imported by both the backtest engine and
# live scanner so the two systems always suppress the same signals in the same regimes.
#
# These are working hypotheses derived from backtest analysis (2015–2026) and
# live paper-trading validation.  Treat every entry as a lean, not a hard rule:
# low trade counts and survivorship risk mean individual cells are suggestive, not proven.
_BEAR_DAY_BLOCKED = frozenset(
    {
        "breakout_52w",
        "momentum",
        "macd_crossover",
        "bb_squeeze",
        "trend_pullback",
        "inside_day_breakout",
        "gap_and_go",
        "orb_breakout",
        "intraday_momentum",
        "iv_compression",  # -1.3% avg in BEAR_DAY — n=24
        "mean_reversion",  # WR 47%, p>0.05 in STRESS_RISK_OFF (n=129)
        "residual_reversal",  # no reversion buying in acute stress (belt-and-suspenders; N1)
        "rsi_divergence",  # no mean-reversion buying in stress regimes
        "candle_exhaustion",  # catching falling knives in stress; need backtest to validate
        "obv_divergence",  # accumulation signals unreliable in extreme panic selling
        "obv_acceleration",  # trend confirmation meaningless in bear-day whipsaw
        "breadth_thrust",  # breadth was already declining into bear day; thrust already stale
    }
)
_HIGH_VOL_BLOCKED = frozenset(
    {
        "breakout_52w",
        "momentum",
        "gap_and_go",
        "orb_breakout",
        "candle_exhaustion",  # catching reversal candles in HV downtrend is premature
        "breadth_thrust",  # breadth thrust in high-vol downtrend can be a whipsaw
        "residual_reversal",  # no reversion buying in HV downtrend / credit stress (N1)
    }
)

# Signals suppressed when the 20-day average (High–Low)/midpoint spread > spread_proxy_max.
# Round-trip cost exceeds expected edge for these short-hold / execution-sensitive signals.
_SPREAD_PROXY_GATED: frozenset[str] = frozenset(
    {
        "gap_and_go",  # gap execution on wide spread → slippage kills the edge
        "mean_reversion",  # 2-day hold; round-trip cost is a large fraction of P&L
        "range_reversion",  # extreme-oversold mean-revert; same profile as mean_reversion
        "candle_exhaustion",  # 3-day reversal entry; tight execution required
        "orb_breakout",  # intraday — execution cost matters most at open
        "vwap_reclaim",  # intraday — execution cost matters most at open
        "intraday_momentum",  # intraday — execution cost matters most at open
    }
)
# Fallback when a snapshot has no spread_proxy_20d at all (both producers — data/market_data and the
# backtest engine — always populate it with a real value or 0.0, so this only bites hand-built /
# partial snapshot dicts). Declared direction: an ABSENT spread reads as WIDE (illiquid) so both the
# execution-cost gate (block short-hold signals) and the capitulation-bounce guardrail (require a
# liquid name) fail CLOSED on unknown liquidity — previously the gate defaulted 0.0 (fail-open) while
# the guardrail defaulted 1.0, treating a missing field as liquid in one place and illiquid in another.
_SPREAD_PROXY_ABSENT = 1.0
# DEFENSIVE_DOWNTREND: mean_reversion has edge here (WR 53%, avg +0.6%, n=112) — kept.
_DEFENSIVE_BLOCKED = frozenset(
    {
        "breakout_52w",
        "momentum",
        "gap_and_go",
        "macd_crossover",  # -2.0% avg in CHOPPY — n=33
        "inside_day_breakout",  # -0.6% avg in CHOPPY — n=101
        "range_reversion",  # WR 30%, avg -2.1% in DEFENSIVE_DOWNTREND (n=10); WR 46%, p>0.05 in NEUTRAL_CHOP (n=52)
    }
)
# NEUTRAL_CHOP: mean_reversion drags (WR 49%, avg -0.1%, n=687 — p>0.05 Holm-corrected).
# iv_compression blocked: WR 51%, avg +0.0%, n=506 — doesn't clear 0.32% round-trip cost threshold.
_NEUTRAL_CHOP_BLOCKED = frozenset({*_DEFENSIVE_BLOCKED, "mean_reversion", "iv_compression"})

# BULL_TREND: rsi_divergence is a consolidation setup, not a reversal worth buying in uptrends.
_BULL_TREND_BLOCKED = frozenset({"rsi_divergence"})

# ── Fundamental / quality gate blocked sets ───────────────────────────────────

# Signals blocked in Altman Z distress zone (Z < 1.1): all trend-following and
# breakout-oriented longs. Counter-cyclical signals (mean_reversion, range_reversion)
# and fundamental-conviction signals (insider_buying) are preserved.
_DISTRESS_BLOCKED = frozenset(
    {
        "gap_and_go",
        "bb_squeeze",
        "inside_day_breakout",
        "trend_pullback",
        "iv_compression",
        "iv_vs_rv_spread",
        "momentum",
        "macd_crossover",
        "golden_cross",
        "candle_exhaustion",
        "obv_divergence",
        "obv_acceleration",
        "volume_climax_reversal",
        "breadth_thrust",
        "tax_loss_reversal",
        "rs_leader",
        "breakout_52w",
        "momentum_12_1",
        "orb_breakout",
        "vwap_reclaim",
        "intraday_momentum",
        "fcf_yield_signal",
        "options_skew_signal",
        "unusual_options_activity",
        "put_call_contrarian",
        "short_interest_trend_long",
        "squeeze_setup_long",
        "squeeze_momentum_long",
        "analyst_upgrade_signal",
        "aaii_extreme_fear_long",
        "fear_greed_extreme_fear",
        "sector_pair_mean_reversion",
        "google_trends_bullish",
        "pead",
        "guidance_raise_signal",
        "activist_13d_signal",
    }
)

# Signals blocked on poor Piotroski quality (F < 3): value-oriented and
# quality-dependent signals. Pure technicals (momentum, breakouts) still fire.
_PIOTROSKI_GATED = frozenset({"pead", "insider_buying"})

# Signals blocked when forward P/E > 60 (stretched valuation):
# momentum and breakout strategies require reasonable valuation support.
_EXPENSIVE_BLOCKED = frozenset(
    {
        "momentum",
        "macd_crossover",
        "gap_and_go",
        "bb_squeeze",
        "inside_day_breakout",
        "trend_pullback",
        "momentum_12_1",
        "breakout_52w",
    }
)

# Signals blocked on gross margin deterioration (> 3pp decline):
# quality-dependent strategies require stable fundamentals.
_GM_GATE_BLOCKED = frozenset({"pead", "guidance_raise_signal", "trend_pullback"})

# Signals blocked on NHL ratio < nhl_ratio_bear_max (weak breadth — no rising tide):
# pure momentum and breakout strategies require broad market participation.
_WEAK_BREADTH_BLOCKED = frozenset(
    {
        "momentum",
        "macd_crossover",
        "gap_and_go",
        "bb_squeeze",
        "inside_day_breakout",
        "breakout_52w",
        "momentum_12_1",
        "short_interest_trend_long",
        "squeeze_momentum_long",
    }
)

# CREDIT_STRESS: credit tightening pre-emptively; treat same as HIGH_VOL (block pure momentum).
_CREDIT_STRESS_BLOCKED = _HIGH_VOL_BLOCKED

# LATE_CYCLE_BULL: bull price action but macro warning (inverted curve / narrow breadth).
# Quality and catalyst signals (pead, insider_buying) pass; momentum/breakout blocked.
_LATE_CYCLE_BULL_BLOCKED = _NEUTRAL_CHOP_BLOCKED

# RECOVERY: bouncing from weakness; allow mean-reversion and early trend, block pure momentum chasers.
_RECOVERY_BLOCKED = frozenset(
    {
        "breakout_52w",
        "momentum",
        "gap_and_go",
        "macd_crossover",
        "inside_day_breakout",
    }
)

# ── Globally disabled signals ─────────────────────────────────────────────────
# Signals removed from both live and backtest after statistically evidenced negative
# contribution across all analysis runs (Jan 2024 – Jun 2026).
# Evidence threshold: consistent negative Sharpe in signal isolation + ablation,
# confirmed across multiple regime conditions.
#   rsi_divergence:  WR 48%, avg -0.9% in NEUTRAL_CHOP (75% of its trades); +0.28–0.31 Sharpe drag.
#   breakout_52w:    WR 35%, avg -1.5% in BULL_TREND (its only firing regime); drag in every run.
#   vix_fear_reversion: 0 trades in all backtest runs; never fires in practice.
#   rs_leader:       Sharpe -0.93 standalone (n=3163); -0.86 Sharpe at all param thresholds
#                    in both NEUTRAL_CHOP/DEFENSIVE_DOWNTREND and BULL_TREND regime sweeps
#                    (Jun 2026). No threshold produces positive expectancy.
#   momentum_12_1:   Sharpe -0.26 standalone (n=45); blocked in all regimes except HIGH_VOL
#                    where it also produces negative expectancy. ΔSharpe +0.08 from removal.
#   range_reversion: 2 trades in combined production backtest, WR 0%, avg -16.2%.
#                    Isolation Sharpe +0.14 (n=218) does not survive into the combined pipeline;
#                    backward elimination removes it Step 3 (ΔSharpe +0.04).
#                    Conditions (ADX<20 + BB<0.10 + RSI<30) too restrictive to fire reliably.
#   volume_climax_reversal: 1 trade in combined production backtest, WR 0%, avg -2.8%.
#                    Fires too rarely to validate and contributes no edge when it does.
#   tax_loss_reversal: 38 trades, WR 37%, avg -1.02% in combined backtest (Jan 2026 test).
#                    January-only seasonal with no confirmed reversal edge; survivorship bias
#                    in the universe inflates the pool of distressed stocks that recovered.
#   obv_divergence:  161 trades WR 53% avg +0.50% on 2020-2022 window, but WR 51% avg -0.07%
#                    on full 2015-2026 run — regime-dependent and inconsistent. Critically,
#                    removing both OBV signals frees slots for pead (604→692 trades, WR 54%→56%);
#                    ΔSharpe +0.12 from joint removal (targeted elimination test, Jun 2026).
#   obv_acceleration: 171 trades WR 44% avg -0.24% (2020-2022); WR 49% avg -0.08% (2015-2026).
#                    Consistently below 50% WR; negative avg return in every window tested.
#                    Joint removal with obv_divergence: ΔSharpe +0.12, ΔReturn +7.0%.
GLOBALLY_DISABLED: frozenset[str] = frozenset(
    {
        "rsi_divergence",
        "breakout_52w",
        "vix_fear_reversion",
        "rs_leader",
        "momentum_12_1",
        "range_reversion",
        "volume_climax_reversal",
        "tax_loss_reversal",
        "obv_divergence",
        "obv_acceleration",
        # 2026-07 signal workshop — standalone isolation event study (907 names, 2015-2026,
        # winsorised excess vs SPY, cost-swept). Both have no standalone edge:
        "macd_crossover",  # gross -0.008%/4d (t=-0.37) — negative pre-cost; subsumed by momentum
        "golden_cross",  # gross +0.038%/5d, break-even 3.8bps — below the universe baseline (5.2bps)
        # 2026-07 options kill/keep (Alpaca 2.4y history, 32,555 name-weeks — first-ever evidence
        # for the v1.98 live-only options signals; iv_vs_rv_spread KEPT: +0.26%/4d, t=2.6, 3/3 yrs):
        "unusual_options_activity",  # premise INVERTED: call-vol spike -0.178%/3d, t=-2.4, 0/3 yrs
        "put_call_contrarian",  # flat on volume proxy (+0.037%, t=0.99); no supporting evidence
    }
)

# ── Short-side signal constants ───────────────────────────────────────────────

# Regimes where short entries are permitted — canonical source used by both
# backtest/engine.py and execution/stock_scanner.py.
# Shorts only make sense when the market is already under stress; entering
# short in BULL_TREND or NEUTRAL_CHOP means fighting the prevailing tide.
SHORT_ALLOWED_REGIMES: frozenset[str] = frozenset(
    {"STRESS_RISK_OFF", "HIGH_VOL_DOWNTREND", "DEFENSIVE_DOWNTREND", "CREDIT_STRESS"}
)

# Signals removed after producing negative expectancy in every isolated and
# combined backtest run (Jan 2023 – May 2026):
#   ema_breakdown:   WR 37-41%, avg -0.78 to -1.04% across all param sweeps; Sharpe -1.22.
#                    Fires after a stock has already broken down — too late; no predictive edge.
#   winner_reversal: RSI>70 + extended + ret_5d<0 is self-contradictory; barely fires.
# v1.99 — lagging fundamental shorts disabled. All three are *confirming* state descriptions,
# not forward catalysts: they fire after the market has already shorted the name, and they
# encode multi-month theses that cannot resolve inside our 1–5 day hold (holding-period
# mismatch). Combined production backtest (2015–2026), borrow cost NOT modelled (so these
# numbers are optimistic):
#   death_cross:      25 trades WR 32% avg -1.14% (combined); 121 trades WR 40% avg -0.10%
#                     (2020-2022 window). SMA50/200 cross fires ~15-30% into a decline.
#   altman_distress_short: 337 trades WR 45% avg -0.17%. Distress drift is a quarters-long
#                     thesis; we capture noise in a 3-day window and enter after credit desks.
#   gross_margin_deterioration_short: 5 trades WR 40% avg -1.32%. Too few to validate; worst
#                     avg of the three. Margin trend is a slow fundamental, wrong horizon.
# Replacement plan: borrow-cost model → post_earnings_gapdown_failed_bounce → index_regime_hedge.
SHORT_GLOBALLY_DISABLED: frozenset[str] = frozenset(
    {
        "ema_breakdown",
        "winner_reversal",
        "failed_breakout",
        "high_vol_reversal",
        "earnings_miss",
        "rs_deterioration",  # 0/11 profitable walk-forward folds, mean Sharpe -0.872 — no edge
        "faded_earnings_gap_up",  # mean Sharpe -0.201, 2/9 folds; 2020-21 fold -35% catastrophic
        "overbought_downtrend",  # backward elimination: ΔSharpe +0.060 drag — removed v1.80
        "parabolic_exhaustion",  # backward elimination: ΔSharpe +0.570, -99.5% return — removed v1.80
        "iv_compression_short",  # new signal — disabled pending initial backtest validation (v1.82)
        "candle_exhaustion_short",  # new signal — disabled pending initial backtest validation (v1.94)
        "obv_divergence_short",  # new signal — disabled pending initial backtest validation (v1.94)
        "obv_acceleration_short",  # new signal — disabled pending initial backtest validation (v1.94)
        "volume_climax_reversal_short",  # new signal — disabled pending initial backtest validation (v1.94)
        "death_cross",  # v1.99 — lagging confirming signal; WR 32%; fires after the decline
        "altman_distress_short",  # v1.99 — multi-month thesis, wrong horizon for 1–5d hold
        "gross_margin_deterioration_short",  # v1.99 — slow fundamental, n=5, worst avg
    }
)

SHORT_SIGNAL_PRIORITY: dict[str, int] = {
    "earnings_miss": 0,  # Negative PEAD — strongest bearish fundamental (disabled)
    "earnings_gap_down": 1,  # Post-earnings gap-down continuation — PEAD short (active)
    "faded_earnings_gap_up": 2,  # Earnings beat faded — distribution into gap-up (disabled — no edge)
    "rs_deterioration": 3,  # Leader-to-laggard rotation (disabled — no edge)
    "failed_breakout": 4,  # Bull trap: broke 20d high yesterday, failed back below today (disabled)
    "high_vol_reversal": 5,  # Distribution bar (disabled)
    "overbought_downtrend": 6,  # Relief rally fade — RSI cross below sma200 (disabled v1.80 — backward elim)
    "parabolic_exhaustion": 7,  # Momentum crash — up 80%+ in 60d, vol drying, RSI extended (disabled v1.80 — backward elim)
    "high_short_interest": 8,  # Crowded short + low lendable supply (live-only, not backtestable)
    "guidance_downgrade": 9,  # Negative 8-K guidance — management lowering outlook (live-only)
    "secondary_offering_short": 10,  # 424B4/S-3 secondary — supply shock dilution (live-only)
    "iv_compression_short": 11,  # HV compression in downtrend (disabled pending backtest)
    # ── Batch 1: OHLCV short signals ────────────────────────────────────────
    "death_cross": 12,  # SMA50 crosses below SMA200 (active)
    "candle_exhaustion_short": 13,  # bearish reversal candle at 20d high (disabled pending backtest)
    "obv_divergence_short": 14,  # price up / OBV down — distribution (disabled pending backtest)
    "obv_acceleration_short": 15,  # OBV selling rate accelerating (disabled pending backtest)
    "volume_climax_reversal_short": 16,  # high-vol streak at 20d high — exhaustion (disabled pending backtest)
    # ── Batch 2: fundamental distress shorts ─────────────────────────────────
    "altman_distress_short": 17,  # Altman Z < 1.1 — financial distress zone
    "piotroski_distress_short": 18,  # Piotroski F ≤ 2 — multi-factor quality failure
    "gross_margin_deterioration_short": 19,  # GM trend < -3pp — deteriorating unit economics
    "accruals_quality_short": 20,  # accruals_ratio > 0.15 + extended price — earnings quality risk
    # ── Batch 3: event-driven shorts ─────────────────────────────────────────
    "lockup_expiry_short": 21,  # IPO lockup expiry within 5-10 days
    "analyst_downgrade_signal": 22,  # consensus Buy→Hold/Sell shift
    # ── Batch: catalyst-anchored short with timing edge (v1.99) ──────────────
    # Negative-PEAD continuation entered AFTER the reflexive bounce fails, not on the
    # gap bar — the failed-bounce filter removes dead-cat-bounce losses that wreck the
    # naive earnings_gap_down. Live-only (computed in scan_short_universe from daily OHLCV);
    # accumulating paper-trading evidence — the one short with a documented short-horizon edge.
    "post_earnings_gapdown_failed_bounce": 23,
    # ── Tier-1 catalyst shorts (ADR-006) — flag-driven, live-only (not backtestable), AI-vetoed.
    # Ranked above the lower-priority technicals: a hard corporate catalyst beats a chart pattern.
    "accounting_concern_short": 24,  # restatement / non-reliance / auditor-change 8-K
    "insider_selling_short": 25,  # cluster of open-market insider sales (Form 4 code 'S')
    "index_deletion_short": 26,  # removed from a major index — forced index-fund selling
    "eps_revision_down_short": 27,  # ≥3 downward current-quarter EPS revisions (estimate-cut momentum)
}

DEFAULT_SHORT_SIGNAL_PARAMS: dict[str, float] = {
    # earnings_gap_down thresholds (PEAD short — post-earnings gap continuation)
    "egd_gap_pct_max": -7.0,  # open must be at least 7% below prior close on earnings day
    "egd_vol_min": 2.5,  # vol_ratio floor — confirms institutional selling, not noise
    # post_earnings_gapdown_failed_bounce thresholds (failed-bounce negative-PEAD short)
    "egdfb_gap_pct_max": -7.0,  # the recent earnings gap-down must be at least this large
    "egdfb_vol_min": 1.5,  # vol_ratio floor on the continuation bar
    # faded_earnings_gap_up thresholds (distribution into earnings beat)
    "fegu_gap_min": 5.0,  # gap-up must be at least 5% on earnings day
    "fegu_range_max": 0.30,  # close must be in bottom 30% of day's H-L range (sellers dominated)
    "fegu_vol_min": 1.5,  # vol_ratio floor — confirms institutional distribution, not thin tape
    # failed_breakout thresholds
    "fb_vol_min": 1.0,  # volume confirmation on the failure day (vol_ratio)
    "fb_rsi_min": 45.0,  # RSI floor — stock must have come from elevated levels
    "fb_rsi_max": 85.0,  # RSI ceiling — filter out extreme blow-off tops that may squeeze
    # high_vol_reversal thresholds
    "hvr_vol_min": 2.0,  # vol_ratio floor — must be 2× 20d average (clear distribution signal)
    "hvr_range_max": 0.3,  # close must be in bottom 30% of day's High–Low range
    "hvr_rsi_min": 55.0,  # came from overbought territory
    "hvr_ret5d_min": 2.0,  # was extended upward (5d return > 2%)
    # overbought_downtrend thresholds (relief rally fade below 200-day SMA)
    "ordt_rsi_entry": 65.0,  # RSI must have been at or above this (genuine overbought bounce)
    "ordt_rsi_exit": 60.0,  # RSI must fall below this (meaningful exhaustion, not just noise)
    "ordt_vol_min": 0.8,  # vol_ratio floor — minimal bar needed to confirm
    # parabolic_exhaustion thresholds (momentum crash — up big, volume drying, RSI extended)
    "pe_ret60d_min": 80.0,  # 60-day return must exceed this % (parabolic run)
    "pe_rsi_min": 72.0,  # RSI must be in overbought territory
    "pe_vol_ratio_max": 0.9,  # volume has dried up (below 20d average) — buyers exhausted
    # rs_deterioration thresholds (cross-sectional signal — requires universe-level data)
    "rs_det_lag_min": 65.0,  # rs_rank_pct_10d_ago must exceed this (was in top 35%)
    "rs_det_current_max": 45.0,  # rs_rank_pct today must be below this (now below median)
    "rs_det_ret5d_max": -2.0,  # ret_5d_pct must be below this (falling > 2% in 5 days)
    # iv_compression_short thresholds (HV compression in confirmed downtrend)
    "ivcs_hv_rank_max": 0.15,  # same compression threshold as long-side iv_compression
    "ivcs_vol_min": 1.0,  # vol confirmation — lower floor; downtrends can be quiet
    # Batch 1 OHLCV short signal thresholds
    "dc_vol_min": 0.8,
    "cex_vol_min": 1.5,
    "obv_div_vol_min": 1.0,
    "obv_acc_vol_min": 1.2,
    "vcr_streak_min": 3,
    # Batch 2: fundamental distress short thresholds
    "az_distress_short_threshold": 1.1,  # Altman Z < 1.1 = distress zone
    "pf_distress_max": 2,  # Piotroski F ≤ 2 = multi-factor failure
    "gm_deterioration_short_min": -0.03,  # gross margin trend < -3pp
    "accruals_short_min": 0.15,  # accruals ratio > 0.15 AND extended price
    "accruals_short_ret5d_min": 5.0,  # price must be extended >5% in 5 days
}


def evaluate_short_signals(
    snapshot: dict,
    params: dict[str, float] | None = None,
    blocked: frozenset[str] = frozenset(),
) -> list[str]:
    """Return all matching bearish signal names for snapshot, sorted by SHORT_SIGNAL_PRIORITY.

    Parameters
    ----------
    snapshot : dict
        Technical snapshot in scanner / market_data format.  Expected keys:
        failed_breakout_flag, close_pct_of_range, vol_ratio, rsi_14, rsi_prev,
        ret_5d_pct, ret_60d_pct, price_below_sma50, earnings_miss_candidate,
        earnings_gap_pct, faded_earnings_gap_up_pct.
    params : dict | None
        Override DEFAULT_SHORT_SIGNAL_PARAMS thresholds.
    blocked : frozenset[str]
        Signal names to suppress.

    Returns
    -------
    list[str]
        All bearish signals that fired, sorted ascending by SHORT_SIGNAL_PRIORITY.
    """
    blocked = blocked | SHORT_GLOBALLY_DISABLED
    p = DEFAULT_SHORT_SIGNAL_PARAMS if params is None else {**DEFAULT_SHORT_SIGNAL_PARAMS, **params}
    matched: list[str] = []

    if "earnings_miss" not in blocked and snapshot.get("earnings_miss_candidate"):
        matched.append(
            "earnings_miss"
        )  # pragma: no cover — earnings_miss in SHORT_GLOBALLY_DISABLED

    # earnings_gap_down: PEAD short — stock gapped down ≥ egd_gap_pct_max% on the first
    # session after earnings with volume ≥ egd_vol_min × 20d average.  Captures the first
    # 1–5 days of institutional selling / analyst-downgrade drift after a negative reaction.
    # RS-rank agnostic: fired via the event path in _short_entry_signal() so it bypasses the
    # reversal/fundamental path RS gates.  Backtest-only until earnings gap detection is wired
    # into scan_short_universe() for the live scanner.
    _egd_gap = snapshot.get("earnings_gap_pct")
    if (
        "earnings_gap_down" not in blocked
        and _egd_gap is not None
        and _egd_gap <= p["egd_gap_pct_max"]
        and snapshot.get("vol_ratio", 0.0) >= p["egd_vol_min"]
    ):
        matched.append("earnings_gap_down")

    # post_earnings_gapdown_failed_bounce: negative-PEAD continuation entered AFTER the reflexive
    # bounce fails — not on the gap bar. Requires a recent earnings gap-down (earnings_gap_pct ≤
    # egdfb_gap_pct_max) AND gap_failed_bounce=True (price has not reclaimed the gap-bar high and
    # has broken below the gap-bar low). The failed-bounce filter removes the dead-cat-bounce
    # losses that make the naive gap-day short unreliable. Computed live in scan_short_universe.
    _egdfb_gap = snapshot.get("earnings_gap_pct")
    if (
        "post_earnings_gapdown_failed_bounce" not in blocked
        and bool(snapshot.get("gap_failed_bounce", False))
        and _egdfb_gap is not None
        and _egdfb_gap <= p["egdfb_gap_pct_max"]
        and snapshot.get("vol_ratio", 0.0) >= p["egdfb_vol_min"]
    ):
        matched.append("post_earnings_gapdown_failed_bounce")

    # faded_earnings_gap_up: stock gapped UP on earnings (beat) but closed in the bottom
    # of the day's range on heavy volume — smart money distributing into retail excitement.
    # Detected on the earnings reaction bar; entry is the following open (T+1).
    # Complement to earnings_gap_down: opposite polarity, same earnings catalyst anchor.
    _fegu = snapshot.get("faded_earnings_gap_up_pct")
    if (
        "faded_earnings_gap_up" not in blocked
        and _fegu is not None
        and _fegu >= p["fegu_gap_min"]
        and snapshot.get("close_pct_of_range", 0.5) <= p["fegu_range_max"]
        and snapshot.get("vol_ratio", 0.0) >= p["fegu_vol_min"]
    ):
        matched.append("faded_earnings_gap_up")  # pragma: no cover — in SHORT_GLOBALLY_DISABLED

    # overbought_downtrend: stock is below its 200-day SMA (confirmed major downtrend) but has
    # staged a relief rally that pushed RSI above ordt_rsi_entry.  Signal fires when RSI
    # crosses back below ordt_rsi_exit — the bounce has exhausted, downtrend reasserting.
    # Using sma200 (not sma50) avoids triggering on normal bull-market pullbacks.
    # RS-rank agnostic: fired via the technical path in _short_entry_signal.
    if (
        "overbought_downtrend" not in blocked
        and snapshot.get("price_below_sma200", False)
        and snapshot.get("rsi_prev", 50.0) >= p["ordt_rsi_entry"]
        and snapshot.get("rsi_14", 50.0) < p["ordt_rsi_exit"]
        and snapshot.get("vol_ratio", 0.0) >= p["ordt_vol_min"]
    ):
        matched.append("overbought_downtrend")  # pragma: no cover — in SHORT_GLOBALLY_DISABLED

    # parabolic_exhaustion: stock up >pe_ret60d_min% in 60 trading days (~3 months) with
    # RSI in overbought territory and volume drying up — buyers exhausted.  Targets the
    # momentum-crash dynamic documented in Daniel & Moskowitz (2016).
    # Fired via a dedicated regime-agnostic path in _short_entry_signal so it can fire in
    # BULL_TREND and NEUTRAL_CHOP, where parabolic moves actually occur.
    if (
        "parabolic_exhaustion" not in blocked
        and snapshot.get("ret_60d_pct", 0.0) >= p["pe_ret60d_min"]
        and snapshot.get("rsi_14", 50.0) >= p["pe_rsi_min"]
        and snapshot.get("vol_ratio", 1.0) <= p["pe_vol_ratio_max"]
    ):
        matched.append("parabolic_exhaustion")  # pragma: no cover — in SHORT_GLOBALLY_DISABLED

    # rs_deterioration: cross-sectional signal — leader-to-laggard rotation.
    # Stock was in the top 35% of the universe 10 trading days ago but has now
    # dropped below the median and is down >2% over 5 days.  Captures the early
    # stage of institutional distribution before technical signals appear.
    # Requires rs_rank_pct_10d_ago in snapshot (set by backtest lag or live scan).
    _rs_lag = snapshot.get("rs_rank_pct_10d_ago")
    if (
        "rs_deterioration" not in blocked
        and _rs_lag is not None
        and _rs_lag > p["rs_det_lag_min"]
        and snapshot.get("rs_rank_pct", 50.0) < p["rs_det_current_max"]
        and snapshot.get("ret_5d_pct", 0.0) < p["rs_det_ret5d_max"]
    ):
        matched.append(
            "rs_deterioration"
        )  # pragma: no cover — rs_deterioration in SHORT_GLOBALLY_DISABLED

    # failed_breakout: stock hit a new 20-day high yesterday, closed back below it today.
    # Hypothesis: trapped longs from the breakout attempt become sellers; the failed breakout
    # level provides a natural stop for the short (stop above yesterday's high).
    if (
        "failed_breakout" not in blocked
        and snapshot.get("failed_breakout_flag", False)
        and snapshot.get("vol_ratio", 0.0) >= p["fb_vol_min"]
        and snapshot.get("rsi_14", 50.0) >= p["fb_rsi_min"]
        and snapshot.get("rsi_14", 50.0) <= p["fb_rsi_max"]
    ):
        matched.append(
            "failed_breakout"
        )  # pragma: no cover — failed_breakout in SHORT_GLOBALLY_DISABLED

    # high_vol_reversal: high-volume day where price closes in the bottom of its range
    # after an extended run. Hypothesis: institutional distribution — smart money selling
    # into retail buying on high volume; the rejection candle signals exhaustion.
    if (
        "high_vol_reversal" not in blocked
        and snapshot.get("vol_ratio", 0.0) >= p["hvr_vol_min"]
        and snapshot.get("close_pct_of_range", 0.5) <= p["hvr_range_max"]
        and snapshot.get("rsi_14", 50.0) >= p["hvr_rsi_min"]
        and snapshot.get("ret_5d_pct", 0.0) >= p["hvr_ret5d_min"]
    ):
        matched.append(
            "high_vol_reversal"
        )  # pragma: no cover — high_vol_reversal in SHORT_GLOBALLY_DISABLED

    if "high_short_interest" not in blocked and snapshot.get("high_short_interest"):
        matched.append("high_short_interest")

    # guidance_downgrade: management explicitly lowered guidance in a recent 8-K.
    # Fires on negative 8-K keyword classification (data/edgar_client.py).
    # Live-only — SEC filings are not replayed in the daily-bar backtest.
    if "guidance_downgrade" not in blocked and snapshot.get("guidance_negative", False):
        matched.append("guidance_downgrade")

    # secondary_offering_short: company filed a 424B4/S-3/S-1 secondary prospectus.
    # New share supply dilutes existing holders; tends to weigh on price 5–20 days post-filing.
    # Live-only — filing history is not replayed in the daily-bar backtest.
    if "secondary_offering_short" not in blocked and snapshot.get("secondary_offering", False):
        matched.append("secondary_offering_short")

    # iv_compression_short: stock below SMA200, EMA in downtrend, and HV compressed — coiling
    # for continuation lower.  Mirror of the long-side iv_compression setup.
    if (
        "iv_compression_short" not in blocked
        and snapshot.get("price_below_sma200", False)
        and not snapshot.get("ema9_above_ema21", True)
        and snapshot.get("hv_rank", 1.0) < p["ivcs_hv_rank_max"]
        and snapshot.get("vol_ratio", 0.0) >= p["ivcs_vol_min"]
    ):
        matched.append("iv_compression_short")  # pragma: no cover — in SHORT_GLOBALLY_DISABLED

    # ── Batch 1 OHLCV short signals ──────────────────────────────────────────

    # death_cross: SMA50 just crossed below SMA200 — major trend deterioration confirmed.
    # Fires once at the cross; subsequent trailing entries captured by other signals.
    if (
        "death_cross" not in blocked
        and snapshot.get("death_cross", False)
        and snapshot.get("vol_ratio", 0.0) >= p["dc_vol_min"]
    ):
        matched.append("death_cross")  # pragma: no cover — death_cross in SHORT_GLOBALLY_DISABLED

    # candle_exhaustion_short: bearish reversal candle (shooting star or bearish engulfing)
    # at the 20-day high with elevated volume — distribution at resistance.
    if (
        "candle_exhaustion_short" not in blocked
        and (snapshot.get("shooting_star", False) or snapshot.get("bearish_engulf", False))
        and snapshot.get("near_20d_high", False)
        and snapshot.get("vol_ratio", 0.0) >= p["cex_vol_min"]
    ):
        matched.append("candle_exhaustion_short")  # pragma: no cover — in SHORT_GLOBALLY_DISABLED

    # obv_divergence_short: price rising but OBV declining — smart money distributing
    # into retail buying pressure.  Complement to the long-side obv_divergence signal.
    if (
        "obv_divergence_short" not in blocked
        and snapshot.get("obv_divergence_bear", False)
        and snapshot.get("vol_ratio", 0.0) >= p["obv_div_vol_min"]
    ):
        matched.append("obv_divergence_short")  # pragma: no cover — in SHORT_GLOBALLY_DISABLED

    # obv_acceleration_short: short-term OBV selling rate faster than long-term baseline
    # AND EMA structure already bearish — institutional selling accelerating.
    if (
        "obv_acceleration_short" not in blocked
        and snapshot.get("obv_accelerating_down", False)
        and not snapshot.get("ema9_above_ema21", True)
        and snapshot.get("vol_ratio", 0.0) >= p["obv_acc_vol_min"]
    ):
        matched.append("obv_acceleration_short")  # pragma: no cover — in SHORT_GLOBALLY_DISABLED

    # volume_climax_reversal_short: ≥vcr_streak_min consecutive extreme-volume days at
    # the 20-day high — climactic buying exhaustion, distribution into retail enthusiasm.
    if (
        "volume_climax_reversal_short" not in blocked
        and int(snapshot.get("high_vol_streak", 0)) >= p["vcr_streak_min"]
        and snapshot.get("near_20d_high", False)
    ):
        matched.append(
            "volume_climax_reversal_short"
        )  # pragma: no cover — in SHORT_GLOBALLY_DISABLED

    # ── Batch 2: fundamental distress shorts ─────────────────────────────────

    # altman_distress_short: Altman Z < az_distress_short_threshold → financial distress.
    # Most reliable when the stock is NOT deeply oversold already (avoid catching falling knives).
    _az = snapshot.get("altman_z")
    if (
        "altman_distress_short" not in blocked
        and _az is not None
        and _az < p["az_distress_short_threshold"]
    ):
        matched.append(
            "altman_distress_short"
        )  # pragma: no cover — altman_distress_short in SHORT_GLOBALLY_DISABLED

    # piotroski_distress_short: Piotroski F ≤ pf_distress_max → multiple quality failures.
    # Low F-score alone is weak; combine with price confirmation (below SMA200) for reliability.
    _pf = snapshot.get("piotroski_f")
    if (
        "piotroski_distress_short" not in blocked
        and _pf is not None
        and int(_pf) <= p["pf_distress_max"]
        and snapshot.get("price_below_sma200", False)
    ):
        matched.append("piotroski_distress_short")

    # gross_margin_deterioration_short: gross margin trend below gm_deterioration_short_min
    # (-3pp) signals deteriorating unit economics — often precedes earnings misses.
    _gm = snapshot.get("gross_margin_trend")
    if (
        "gross_margin_deterioration_short" not in blocked
        and _gm is not None
        and _gm < p["gm_deterioration_short_min"]
        and snapshot.get("price_below_sma200", False)
    ):
        matched.append(
            "gross_margin_deterioration_short"
        )  # pragma: no cover — gross_margin_deterioration_short in SHORT_GLOBALLY_DISABLED

    # accruals_quality_short: high accruals (earnings driven by balance-sheet changes,
    # not cash) + extended price is a mean-reversion short. Sloan (1996) documented
    # the accruals anomaly; accruals > 0.15 + ret_5d > 5% is a reliable setup.
    _acc = snapshot.get("accruals_ratio")
    if (
        "accruals_quality_short" not in blocked
        and _acc is not None
        and _acc > p["accruals_short_min"]
        and snapshot.get("ret_5d_pct", 0.0) > p["accruals_short_ret5d_min"]
    ):
        matched.append("accruals_quality_short")

    # ── Batch 3: event-driven shorts ─────────────────────────────────────────

    # lockup_expiry_short: IPO lockup expiry approaching (5–10 days out).
    # Early shareholders (founders, VC) can first sell → supply pressure.
    if "lockup_expiry_short" not in blocked and snapshot.get("lockup_expiry_soon", False):
        matched.append("lockup_expiry_short")

    # analyst_downgrade_signal: consensus shift from Buy toward Hold/Sell — institutional
    # coverage degrading. Complement to analyst_upgrade_signal on the long side.
    if "analyst_downgrade_signal" not in blocked and snapshot.get("analyst_downgrade", False):
        matched.append("analyst_downgrade_signal")

    # ── Tier-1 catalyst shorts (ADR-006) — hard corporate catalysts, flag-driven. Live-only:
    # these flags are not present in the daily-bar backtest, so they never fire there. The AI veto
    # (B2) gates every one before a live order; they are active and AI-citeable.
    if "accounting_concern_short" not in blocked and snapshot.get("accounting_concern", False):
        matched.append("accounting_concern_short")

    if "insider_selling_short" not in blocked and snapshot.get("insider_sell_cluster", False):
        matched.append("insider_selling_short")

    if "index_deletion_short" not in blocked and snapshot.get("index_deletion", False):
        matched.append("index_deletion_short")

    # eps_revision_down_short: a cluster of downward current-quarter EPS estimate revisions
    # (data/analyst_revisions.py) — the estimate-revision anomaly: cuts precede negative drift.
    if "eps_revision_down_short" not in blocked and snapshot.get("eps_estimate_cut", False):
        matched.append("eps_revision_down_short")

    matched.sort(key=lambda s: SHORT_SIGNAL_PRIORITY.get(s, 99))
    return matched


REGIME_BLOCKED: dict[str, frozenset[str]] = {
    # Legacy names (kept for backward compatibility)
    "BEAR_DAY": _BEAR_DAY_BLOCKED,
    "HIGH_VOL": _HIGH_VOL_BLOCKED,
    "BULL_TRENDING": _BULL_TREND_BLOCKED,
    "CHOPPY": _NEUTRAL_CHOP_BLOCKED,
    # 5-state names
    "STRESS_RISK_OFF": _BEAR_DAY_BLOCKED,
    "HIGH_VOL_DOWNTREND": _HIGH_VOL_BLOCKED,
    "DEFENSIVE_DOWNTREND": _DEFENSIVE_BLOCKED,
    "BULL_TREND": _BULL_TREND_BLOCKED,
    "NEUTRAL_CHOP": _NEUTRAL_CHOP_BLOCKED,
    "UNKNOWN": _BEAR_DAY_BLOCKED,
    # v2 states
    "CREDIT_STRESS": _CREDIT_STRESS_BLOCKED,
    "LATE_CYCLE_BULL": _LATE_CYCLE_BULL_BLOCKED,
    "RECOVERY": _RECOVERY_BLOCKED,
}


def evaluate_signals(
    snapshot: dict,
    blocked: frozenset[str] = frozenset(),
    params: dict | None = None,
    vix_spike: bool = False,
    spy_ret_5d: float | None = None,
    spy_ret_10d: float | None = None,
    regime: str = "",
) -> list[str]:
    """Return all matching signal names for snapshot, sorted by SIGNAL_PRIORITY.

    Parameters
    ----------
    snapshot : dict
        Technical snapshot in scanner / market_data format.  Expected keys:
        rsi_14, bb_pct, vol_ratio, macd_diff, macd_crossed_up, ema9_above_ema21,
        adx (default 30 if absent), ret_5d_pct, ret_10d_pct, price_vs_ema21_pct,
        price_vs_52w_high_pct, hv_rank, bb_squeeze, is_inside_day, gap_pct,
        close_above_open, insider_cluster, pead_candidate, mom_12_1_pct (optional),
        intraday_change_pct, price_above_vwap, pct_vs_vwap, orb_breakout_up,
        intraday_rsi.
    blocked : frozenset[str]
        Signal names suppressed by regime or disabled_signals.
    params : dict | None
        Walk-forward optimised thresholds; merged over DEFAULT_SIGNAL_PARAMS.
    vix_spike : bool
        True when VIX is above the spike threshold (used for vix_fear_reversion).
    spy_ret_5d, spy_ret_10d : float | None
        SPY 5d and 10d returns for the rs_leader relative-strength gate.
        Pass None to skip rs_leader entirely.
    regime : str
        Canonical market regime name (data.market_regime). Only "STRESS_RISK_OFF" is used:
        it unblocks the capitulation-bounce leg of residual_reversal for liquid names. Empty
        string (default) disables that leg — safe for callers that do not classify regime.

    Returns
    -------
    list[str]
        All signals that fired, sorted ascending by SIGNAL_PRIORITY value.
    """
    blocked = blocked | GLOBALLY_DISABLED
    p = DEFAULT_SIGNAL_PARAMS if params is None else {**DEFAULT_SIGNAL_PARAMS, **params}
    calendar_month = int(snapshot.get("calendar_month", 0))

    # Spread proxy gate: when average daily H–L spread > threshold, execution cost exceeds
    # edge for short-hold and intraday signals — dynamically add them to blocked.
    if float(snapshot.get("spread_proxy_20d", _SPREAD_PROXY_ABSENT)) > p["spread_proxy_max"]:
        blocked = blocked | _SPREAD_PROXY_GATED

    # ── Macro gates (injected by scanner and backtest engine) ─────────────────
    # credit_stress: HYG/LQD ratio falling → treat same as HIGH_VOL_DOWNTREND
    if bool(snapshot.get("macro_credit_stress", False)):
        blocked = blocked | _HIGH_VOL_BLOCKED
    # duration_flight / claims deteriorating → defensive mode
    if bool(snapshot.get("macro_duration_flight", False)) or bool(
        snapshot.get("macro_claims_deteriorating", False)
    ):
        blocked = blocked | _DEFENSIVE_BLOCKED
    # Sustained yield-curve inversion → late-cycle; block momentum/breakout longs
    if int(snapshot.get("macro_yield_curve_inverted_days", 0)) >= 20:
        blocked = blocked | _LATE_CYCLE_BULL_BLOCKED

    # premarket_gap_quality: opening gap retraced >50% by 09:35 → weak gap momentum
    if bool(snapshot.get("premarket_gap_retrace", False)):
        blocked = blocked | frozenset({"gap_and_go"})

    # Lottery/MAX gate (2026-07 workshop): a >=+10% single-day pop within the last 3 sessions
    # precedes underperformance (-0.44%/3d, t=-5.1; Bali-Cakici-Whitelaw). Don't let the
    # momentum family chase the lottery pop into its reversal.
    if bool(snapshot.get("recent_lottery_pop", False)):
        blocked = blocked | frozenset({"momentum", "gap_and_go"})

    # ── Fundamental quality gates ─────────────────────────────────────────────

    # Altman Z distress gate: Z < threshold means likely financial distress.
    # Block all trend-following longs; preserve counter-cyclical and conviction signals.
    _altman_z = snapshot.get("altman_z")
    if _altman_z is not None and float(_altman_z) < p["az_distress_threshold"]:
        blocked = blocked | _DISTRESS_BLOCKED

    # Piotroski quality gate: F < piotroski_min = weak multi-factor quality.
    # Block pead and insider_buying (require quality support for these catalysts).
    _piotroski_f = snapshot.get("piotroski_f")
    if _piotroski_f is not None and int(_piotroski_f) < p["piotroski_min"]:
        blocked = blocked | _PIOTROSKI_GATED

    # Forward P/E gate: expensive stocks (P/E > fwd_pe_max) block momentum/breakout.
    # Stretched valuations reduce the risk/reward for trend-following strategies.
    _fwd_pe = snapshot.get("forward_pe")
    if _fwd_pe is not None and float(_fwd_pe) > p["fwd_pe_max"]:
        blocked = blocked | _EXPENSIVE_BLOCKED

    # Gross margin deterioration gate: GM trend < gm_trend_min (−3pp) blocks
    # catalyst-driven and trend-pullback longs (deteriorating unit economics undermine these).
    _gm_trend = snapshot.get("gross_margin_trend")
    if _gm_trend is not None and float(_gm_trend) < p["gm_trend_min"]:
        blocked = blocked | _GM_GATE_BLOCKED

    # Accruals quality gate: high accruals (earnings from balance-sheet, not cash)
    # suppress momentum longs — Sloan anomaly: accruals > 0.10 predicts reversals.
    _accruals = snapshot.get("accruals_ratio")
    if _accruals is not None and float(_accruals) > p["accruals_max"]:
        blocked = blocked | _EXPENSIVE_BLOCKED  # same set — block momentum/breakout

    # ── Market microstructure gates ───────────────────────────────────────────

    # NH/NL ratio breadth gate: when more stocks are making new lows than new highs,
    # the tide is against momentum and breakout strategies.
    _nhl = snapshot.get("nhl_ratio")
    if _nhl is not None and float(_nhl) < p["nhl_ratio_bear_max"]:
        blocked = blocked | _WEAK_BREADTH_BLOCKED

    # Sector correlation gate: when stocks move together (correlation > 0.75),
    # stock-specific momentum loses its edge; treat like OPEX dampening.
    _sector_corr = snapshot.get("sector_correlation_20d")
    if _sector_corr is not None and float(_sector_corr) > p["sector_corr_high"]:
        blocked = blocked | frozenset({"momentum", "breakout_52w", "bb_squeeze"})

    # Earnings yield vs bonds gate: equity risk premium < erp_min (1%) means equities
    # offer little excess return over bonds — dampen momentum longs.
    _fwd_pe_g = snapshot.get("forward_pe")
    _10y_yield = snapshot.get("macro_10y_yield")
    if _fwd_pe_g is not None and _10y_yield is not None and float(_fwd_pe_g) > 0:
        _erp = (1.0 / float(_fwd_pe_g)) - float(_10y_yield) / 100.0
        if _erp < p["erp_min"]:
            blocked = blocked | _EXPENSIVE_BLOCKED

    # AAII sentiment excessive-bulls gate: when >60% investors are bullish for ≥1 week,
    # complacency risk is elevated — dampen momentum/breakout signals.
    if bool(snapshot.get("aaii_excessive_bulls", False)):
        blocked = blocked | frozenset({"momentum", "gap_and_go", "breakout_52w"})

    def _f(key: str, default: float) -> float:
        v = snapshot.get(key)
        return float(v) if v is not None else default

    rsi = _f("rsi_14", 50)
    bb = _f("bb_pct", 0.5)
    vol = _f("vol_ratio", 1.0)
    macd_diff = _f("macd_diff", 0)
    macd_up = bool(snapshot.get("macd_crossed_up", False))
    ema_up = bool(snapshot.get("ema9_above_ema21", False))
    # Default adx=30 (assume trending) when High/Low data unavailable — same as engine
    adx = _f("adx", 30)
    ret_5d = _f("ret_5d_pct", 0)
    ret_10d = _f("ret_10d_pct", 0)
    pct_ema21 = _f("price_vs_ema21_pct", 0)
    pct_52w = _f("price_vs_52w_high_pct", -999)
    hv_rank = _f("hv_rank", 1.0)
    bb_squeeze = bool(snapshot.get("bb_squeeze", False))
    bb_squeeze_days = int(snapshot.get("bb_squeeze_days", 0))
    is_inside_day = bool(snapshot.get("is_inside_day", False))
    gap_pct = _f("gap_pct", 0)
    close_above_open = bool(snapshot.get("close_above_open", False))
    insider_cluster = bool(snapshot.get("insider_cluster", False))
    insider_strong_cluster = bool(snapshot.get("insider_strong_cluster", False))
    insider_comp_ratio = _f("insider_comp_ratio", 0.0)
    insider_large_buy = bool(snapshot.get("insider_large_buy", False))
    activist_filing = bool(snapshot.get("activist_filing", False))
    pead_candidate = bool(snapshot.get("pead_candidate", False))
    guidance_positive = bool(snapshot.get("guidance_positive", False))
    iv_cheap = bool(snapshot.get("iv_cheap", False))
    # mom_12_1_pct absent → momentum_12_1 signal not evaluated
    _m121 = snapshot.get("mom_12_1_pct")
    mom_12_1: float | None = float(_m121) if _m121 is not None else None

    intraday_chg = snapshot.get("intraday_change_pct")
    above_vwap = snapshot.get("price_above_vwap")
    pct_vwap = _f("pct_vs_vwap", 0)
    orb_up = bool(snapshot.get("orb_breakout_up", False))
    intraday_rsi = snapshot.get("intraday_rsi")

    matched: list[str] = []

    # Counter-cyclical — fires during fear spikes; never regime-blocked
    if vix_spike and vol > p["vfr_vol_min"] and "vix_fear_reversion" not in blocked:
        matched.append("vix_fear_reversion")  # pragma: no cover — in GLOBALLY_DISABLED

    # activist_13d_signal: SC 13D filing from known activist → management change catalyst.
    # Fires purely on activist_filing (subset of insider_buying trigger). Separate signal
    # allows the AI to cite the activist catalyst distinctly from cluster insider buying.
    if activist_filing and "activist_13d_signal" not in blocked:
        matched.append("activist_13d_signal")

    # Fundamental conviction — bypass regime filter
    if (
        activist_filing
        or insider_strong_cluster
        or (insider_cluster and (insider_comp_ratio >= p["ib_comp_ratio_min"] or insider_large_buy))
    ) and "insider_buying" not in blocked:
        matched.append("insider_buying")
    if (pead_candidate or guidance_positive) and ret_5d > 0 and "pead" not in blocked:
        matched.append("pead")

    # guidance_raise_signal: positive 8-K guidance event before price confirms.
    # Fires on guidance_positive without requiring ret_5d > 0 — earlier entry than pead.
    if guidance_positive and "guidance_raise_signal" not in blocked:
        matched.append("guidance_raise_signal")

    # Relative strength leader (requires SPY data; engine-only unless caller provides)
    if (
        spy_ret_5d is not None
        and spy_ret_10d is not None
        and "rs_leader" not in blocked
        and (ret_5d - spy_ret_5d) > p["rsl_excess_5d_min"]
        and (ret_10d - spy_ret_10d) > p["rsl_excess_10d_min"]
        and ema_up
        and adx >= 20
    ):
        matched.append("rs_leader")  # pragma: no cover — rs_leader in GLOBALLY_DISABLED

    # 52-week breakout
    if (
        pct_52w >= p["bk52_pct_min"]
        and vol > p["bk52_vol_min"]
        and ema_up
        and adx >= 20
        and "breakout_52w" not in blocked
    ):
        matched.append("breakout_52w")  # pragma: no cover — breakout_52w in GLOBALLY_DISABLED

    # 12-month momentum (skipped when mom_12_1_pct field absent)
    # Pullback filter: require recent 1-week return ≤ mom12_1_pullback_ret5d_max so we buy
    # on a retracement in a strong trend rather than chasing an already-extended move.
    if (
        mom_12_1 is not None
        and mom_12_1 > p["mom12_1_threshold"]
        and ret_5d <= p["mom12_1_pullback_ret5d_max"]
        and ema_up
        and adx >= 20
        and "momentum_12_1" not in blocked
    ):
        matched.append("momentum_12_1")  # pragma: no cover — momentum_12_1 in GLOBALLY_DISABLED

    # Gap and go
    if (
        gap_pct > p["gap_pct_min"]
        and close_above_open
        and vol > p["gap_vol_min"]
        and adx >= 20
        and "gap_and_go" not in blocked
    ):
        matched.append("gap_and_go")

    # Bollinger squeeze
    _bbs_price = snapshot.get("current_price", 10.0)
    if (
        bb_squeeze
        and bb_squeeze_days >= p["bbs_squeeze_days_min"]
        and vol > p["bbs_vol_min"]
        and (ema_up or macd_diff > 0)
        and adx >= p["bbs_adx_min"]
        and snapshot.get("rs_rank_pct", 0.0) >= p["bbs_rs_rank_min"]
        and _bbs_price >= 10.0
        and "bb_squeeze" not in blocked
    ):
        matched.append("bb_squeeze")

    # Inside day breakout
    if (
        is_inside_day
        and vol > p["idb_vol_min"]
        and (ema_up or macd_diff > 0)
        and adx >= 20
        and "inside_day_breakout" not in blocked
    ):
        matched.append("inside_day_breakout")

    # Trend pullback to EMA21 within uptrend
    if (
        ema_up
        and p["tp_ema21_lo"] <= pct_ema21 <= p["tp_ema21_hi"]
        and p["tp_rsi_lo"] <= rsi <= p["tp_rsi_hi"]
        and vol > p["tp_vol_min"]
        and adx >= 20
        and "trend_pullback" not in blocked
    ):
        matched.append("trend_pullback")

    # IV compression: historical vol compressed (hv_rank) OR options market pricing IV cheap vs RV
    if (
        (hv_rank < p["ivc_hv_rank_max"] or iv_cheap)
        and (ema_up or macd_diff > 0)
        and vol > p["ivc_vol_min"]
        and "iv_compression" not in blocked
    ):
        matched.append("iv_compression")

    # iv_vs_rv_spread: ATM IV / 20d realised vol ratio < iv_rv_spread_cheap_max (<0.70).
    # When the market is pricing vol below recent realised vol, long vol + trend is high EV.
    # Complement to iv_compression (which uses HV rank); this uses the absolute IV/RV ratio.
    _iv_rv = snapshot.get("iv_rv_spread")
    if (
        "iv_vs_rv_spread" not in blocked
        and _iv_rv is not None
        and float(_iv_rv) < p["iv_rv_spread_cheap_max"]
        and (ema_up or macd_diff > 0)
        and vol > p["ivc_vol_min"]
    ):
        matched.append("iv_vs_rv_spread")

    # Range reversion: extreme oversold within confirmed range-bound conditions.
    # The adx < rr_adx_max gate implicitly restricts this to non-trending regimes.
    if (
        adx < p["rr_adx_max"]
        and bb < p["rr_bb_max"]
        and rsi < p["rr_rsi_max"]
        and "range_reversion" not in blocked
    ):
        matched.append("range_reversion")  # pragma: no cover — range_reversion in GLOBALLY_DISABLED

    # RSI divergence: price lower than 5 days ago but RSI recovering — bullish structural
    # divergence in range-bound conditions.  The adx < 25 gate keeps it out of trending
    # regimes; explicit regime blocks cover BULL_TREND and STRESS_RISK_OFF.
    rsi_div = bool(snapshot.get("rsi_divergence", False))
    if (
        rsi_div
        and adx < 25
        and rsi < p["rsi_div_rsi_max"]
        and vol > p["rsi_div_vol_min"]
        and bb < p["rsi_div_bb_max"]
        and "rsi_divergence" not in blocked
    ):
        matched.append("rsi_divergence")  # pragma: no cover — rsi_divergence in GLOBALLY_DISABLED

    # Momentum
    if (
        ema_up
        and macd_diff > 0
        and ret_5d > p["mom_ret5d_threshold"]
        and vol > p["mom_vol_threshold"]
        and adx >= 20
        and "momentum" not in blocked
    ):
        matched.append("momentum")

    # Mean reversion
    if (
        rsi < p["rsi_threshold"]
        and bb < p["bb_threshold"]
        and vol > p["mr_vol_threshold"]
        and "mean_reversion" not in blocked
    ):
        matched.append("mean_reversion")

    # Residual reversal (2026-07 signal workshop, N1): an idiosyncratic 5-day loser reverts over
    # 1-3 days. Fire on market-excess 5d return (stock ret_5d - SPY ret_5d, both %) <= threshold.
    # The market-excess construction fires on stock-SPECIFIC drops — a broad selloff hits SPY too,
    # so the excess rarely triggers — which mutes the crash-tail risk of raw mean-reversion.
    # v1.144 sector conjunct: the drop must ALSO clear the threshold vs the name's own sector
    # (sector_ret_5d_pct, equal-weight universe peers) — a -7% move during a sector rout is sector
    # beta that continues, not idiosyncratic flow that reverts. Head-to-head study: the intersection
    # nets +0.374%/3d @7bps (t=7.1, 10/12 +yrs) vs +0.301% (8/12) for spy-only. Fail-open when the
    # sector field is absent (backtest engine + unknown-sector names keep the spy-only construction).
    # v1.145 capitulation bounce: residual_reversal is regime-blocked in STRESS_RISK_OFF, but the
    # reconcile study (scripts/reversal_regime_reconcile.py) shows that is reversal's BEST bucket —
    # +1.86% net/3d @7bps, robust in BOTH walk-forward halves (train +2.1 / test +1.3, 6/7 yrs): the
    # forced-liquidation overshoot in an idiosyncratic loser snaps back. Allow it there for LIQUID
    # names only (20d spread within the execution gate — the survivorship guardrail: illiquid
    # distressed names are the ones that delist; the liquid ones bounce). STRESS_RISK_OFF ONLY —
    # HIGH_VOL_DOWNTREND reversal is train-negative (kept blocked); UNKNOWN stays blocked (no info).
    _capit = (
        regime == "STRESS_RISK_OFF"
        and float(snapshot.get("spread_proxy_20d", _SPREAD_PROXY_ABSENT)) <= p["spread_proxy_max"]
    )
    _sect5 = snapshot.get("sector_ret_5d_pct")
    if (
        spy_ret_5d is not None
        and ret_5d - spy_ret_5d <= p["resid_rev_threshold"]
        and (_sect5 is None or ret_5d - float(_sect5) <= p["resid_rev_threshold"])
        and ("residual_reversal" not in blocked or _capit)
    ):
        matched.append("residual_reversal")

    # MACD crossover
    if macd_up and vol > p["macd_vol_min"] and adx >= 20 and "macd_crossover" not in blocked:
        matched.append("macd_crossover")  # pragma: no cover — macd_crossover in GLOBALLY_DISABLED

    # ── Batch 1 OHLCV signals ─────────────────────────────────────────────────

    # Golden cross: SMA50 just crossed above SMA200 — major long-term trend confirmation.
    # Low vol threshold; the cross itself is the signal, not the volume on the cross day.
    if (
        bool(snapshot.get("golden_cross", False))
        and vol >= p["gc_vol_min"]
        and "golden_cross" not in blocked
    ):
        matched.append("golden_cross")  # pragma: no cover — golden_cross in GLOBALLY_DISABLED

    # Candle exhaustion: bullish reversal candle (hammer or bullish engulfing) at the
    # 20-day low with elevated volume — panic selling into support; capitulation setup.
    if (
        (bool(snapshot.get("hammer", False)) or bool(snapshot.get("bullish_engulf", False)))
        and bool(snapshot.get("near_20d_low", False))
        and vol >= p["cex_vol_min"]
        and "candle_exhaustion" not in blocked
    ):
        matched.append("candle_exhaustion")

    # OBV divergence: price declining over 5 days but OBV rising — institutional
    # accumulation under price weakness; smart money buying into the selloff.
    if (
        bool(snapshot.get("obv_divergence_bull", False))
        and vol >= p["obv_div_vol_min"]
        and "obv_divergence" not in blocked
    ):
        matched.append("obv_divergence")  # pragma: no cover — obv_divergence in GLOBALLY_DISABLED

    # OBV acceleration: short-term OBV buying rate faster than long-term baseline
    # AND trend structure intact (EMA9 > EMA21 or MACD positive) — buying momentum
    # is picking up before price fully reflects it.
    if (
        bool(snapshot.get("obv_accelerating_up", False))
        and (ema_up or macd_diff > 0)
        and vol >= p["obv_acc_vol_min"]
        and "obv_acceleration" not in blocked
    ):
        matched.append(
            "obv_acceleration"
        )  # pragma: no cover — obv_acceleration in GLOBALLY_DISABLED

    # Volume climax reversal: ≥vcr_streak_min consecutive extreme-volume days at the
    # 20-day low — capitulation / panic selling exhaustion; mean-reversion long setup.
    if (
        int(snapshot.get("high_vol_streak", 0)) >= p["vcr_streak_min"]
        and bool(snapshot.get("near_20d_low", False))
        and "volume_climax_reversal" not in blocked
    ):
        matched.append(
            "volume_climax_reversal"
        )  # pragma: no cover — volume_climax_reversal in GLOBALLY_DISABLED

    # Breadth thrust (Zweig): universe breadth jumped from <40% to >60% above 50d SMA
    # within 10 days — rare "all-clear" thrust confirming a broad market expansion.
    # Requires individual stock EMA alignment (don't buy laggards on a broad rally).
    # Requires minimum universe coverage (bt_min_symbols) to trust the breadth reading.
    _bt_symbols = int(snapshot.get("breadth_symbols_counted", 0))
    if (
        bool(snapshot.get("breadth_thrust", False))
        and (_bt_symbols == 0 or _bt_symbols >= int(p["bt_min_symbols"]))
        and ema_up
        and "breadth_thrust" not in blocked
    ):
        matched.append("breadth_thrust")

    # ── Batch 3: calendar/seasonal signals ───────────────────────────────────

    # tax_loss_reversal: January rebound after Nov/Dec systematic tax-loss selling.
    # Stocks beaten down >30% from their 52-week high face year-end selling pressure
    # from investors harvesting losses; that pressure lifts when the new tax year resets.
    # Requires nascent EMA alignment (EMA9 > EMA21) to avoid buying into continued decline.
    # calendar_month must be present in snapshot (injected by scanner and backtest engine).
    # Guard: skip when price_vs_52w_high_pct is absent — the sentinel default of -999
    # would satisfy the drawdown threshold and fire a false signal on missing data.
    if (
        calendar_month == 1
        and snapshot.get("price_vs_52w_high_pct") is not None
        and pct_52w < p["tlr_52w_drawdown_max"]
        and ema_up
        and "tax_loss_reversal" not in blocked
    ):
        matched.append(
            "tax_loss_reversal"
        )  # pragma: no cover — tax_loss_reversal in GLOBALLY_DISABLED

    # Intraday signals (only active when intraday fields are present in snapshot)
    if orb_up and "orb_breakout" not in blocked:
        matched.append("orb_breakout")

    if (
        above_vwap is True
        and intraday_chg is not None
        and intraday_chg > 1.0
        and pct_vwap <= 3.0
        and "vwap_reclaim" not in blocked
    ):
        matched.append("vwap_reclaim")

    if (
        intraday_chg is not None
        and intraday_chg > 2.0
        and above_vwap is True
        and (intraday_rsi is None or float(intraday_rsi) < 75)
        and (ema_up or ret_5d > 3.0)
        and "intraday_momentum" not in blocked
    ):
        matched.append("intraday_momentum")

    # ── Batch 4: fundamental quality signals ─────────────────────────────────

    # fcf_yield_signal: FCF yield > fcf_yield_min (5%) with moderate quality (F ≥ 5).
    # High-FCF + quality = conviction buy; not blocked by distress gate above because
    # a truly distressed company can't generate FCF.
    _fcf = snapshot.get("fcf_yield")
    _pf_fcf = snapshot.get("piotroski_f")
    if (
        "fcf_yield_signal" not in blocked
        and _fcf is not None
        and float(_fcf) > p["fcf_yield_min"]
        and (_pf_fcf is None or int(_pf_fcf) >= p["fcf_piotroski_min"])
    ):
        matched.append("fcf_yield_signal")

    # ── Batch 5: options-derived signals ─────────────────────────────────────

    # options_skew_signal: extreme put/call IV skew (panic hedging) → contrarian long;
    # call skew spike (informed upside buying) → trend confirmation.
    if "options_skew_signal" not in blocked and (
        bool(snapshot.get("panic_put_skew", False)) or bool(snapshot.get("call_skew_spike", False))
    ):
        matched.append("options_skew_signal")

    # unusual_options_activity: large OTM call OI surge signals informed upside conviction.
    if "unusual_options_activity" not in blocked and bool(snapshot.get("unusual_call_oi", False)):
        matched.append("unusual_options_activity")  # pragma: no cover — in GLOBALLY_DISABLED

    # put_call_contrarian: extreme put-to-call OI ratio (>2.5) signals panic hedging →
    # contrarian long. Requires trend structure to avoid catching falling knives.
    _pc_ratio = snapshot.get("put_call_oi_ratio")
    if (
        "put_call_contrarian" not in blocked
        and _pc_ratio is not None
        and float(_pc_ratio) > p["put_call_contrarian_min"]
        and (ema_up or macd_diff > 0)
    ):
        matched.append("put_call_contrarian")  # pragma: no cover — in GLOBALLY_DISABLED

    # ── Batch 6: short-squeeze signals ───────────────────────────────────────

    _si_pct = snapshot.get("short_pct_float")
    _si_dtc = snapshot.get("short_ratio")

    # squeeze_setup_long: crowded dormant short at 20d low — pre-squeeze setup.
    # High short interest + dormant price (ret_5d < squeeze_setup_ret5d_max) near 20d low.
    if (
        "squeeze_setup_long" not in blocked
        and _si_pct is not None
        and float(_si_pct) > p["squeeze_si_min"]
        and _si_dtc is not None
        and float(_si_dtc) > p["squeeze_dtc_min"]
        and ret_5d < p["squeeze_setup_ret5d_max"]
        and bool(snapshot.get("near_20d_low", False))
    ):
        matched.append("squeeze_setup_long")

    # squeeze_momentum_long: squeeze in motion — shorts being forced to cover.
    # High SI + strong recent return + price above 20d high = active squeeze.
    if (
        "squeeze_momentum_long" not in blocked
        and _si_pct is not None
        and float(_si_pct) > p["squeeze_si_min"]
        and _si_dtc is not None
        and float(_si_dtc) > p["squeeze_dtc_min"]
        and ret_5d > p["squeeze_mom_ret5d_min"]
        and bool(snapshot.get("near_20d_high", False))
    ):
        matched.append("squeeze_momentum_long")

    # short_interest_trend_long: SI% falling >30% from its peak while price is rising —
    # short covering into strength = informed buying as shorts capitulate.
    # Requires si_peak field in snapshot (injected from historical cache if available).
    _si_peak = snapshot.get("si_peak")
    if (
        "short_interest_trend_long" not in blocked
        and _si_pct is not None
        and _si_peak is not None
        and _si_peak > 0
        and (float(_si_peak) - float(_si_pct)) / float(_si_peak) > p["si_trend_drop_min"]
        and ret_5d > 0
    ):
        matched.append("short_interest_trend_long")

    # ── Batch 7: analyst / institutional signals ──────────────────────────────

    # analyst_upgrade_signal: consensus hold→buy shift — institutional coverage improving.
    if "analyst_upgrade_signal" not in blocked and bool(snapshot.get("analyst_upgrade", False)):
        matched.append("analyst_upgrade_signal")

    # ── Batch 8: sentiment signals ────────────────────────────────────────────

    # aaii_extreme_fear_long: AAII survey bears > 50% = contrarian long backdrop.
    # Historically, extreme AAII bearishness has been a reliable 3-6 month contrarian indicator.
    if "aaii_extreme_fear_long" not in blocked and bool(snapshot.get("aaii_extreme_fear", False)):
        matched.append("aaii_extreme_fear_long")

    # fear_greed_extreme_fear: composite fear/greed index < 20 = extreme fear backdrop.
    # Used as a confirmation signal — fires alongside technical signals in fear regimes.
    _fg_score = snapshot.get("fear_greed_score")
    if (
        "fear_greed_extreme_fear" not in blocked
        and _fg_score is not None
        and float(_fg_score) < 20.0
    ):
        matched.append("fear_greed_extreme_fear")

    # ── Batch 9: cross-asset signals ─────────────────────────────────────────

    # sector_pair_mean_reversion: intra-sector spread is extended (z-score > pairs_zscore_min).
    # Long the top-RS stock relative to the bottom-RS when their spread is stretched.
    # Requires pairs_spread_z field (injected from data/pairs.py per symbol).
    _pairs_z = snapshot.get("pairs_spread_z")
    if (
        "sector_pair_mean_reversion" not in blocked
        and _pairs_z is not None
        and abs(float(_pairs_z)) > p["pairs_zscore_min"]
        and float(_pairs_z) > 0  # positive z = this stock is the cheap leg
        and ema_up
    ):
        matched.append("sector_pair_mean_reversion")

    # ── Batch 10: alternative data signals ───────────────────────────────────

    # google_trends_bullish: search volume spike for this ticker with trend confirmation.
    # Retail attention spike + positive technicals = participation-driven momentum.
    if "google_trends_bullish" not in blocked and bool(snapshot.get("google_trends_spike", False)):
        matched.append("google_trends_bullish")

    matched.sort(key=lambda s: SIGNAL_PRIORITY.get(s, 99))
    return matched
