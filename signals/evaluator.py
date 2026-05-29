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
    "pead": 2,
    # rs_leader blocked in BULL_TREND (no edge); breakout_52w < momentum_12_1 < gap_and_go
    "rs_leader": 3,
    "breakout_52w": 4,
    "momentum_12_1": 5,
    "gap_and_go": 6,
    "bb_squeeze": 7,
    "inside_day_breakout": 8,
    "trend_pullback": 9,
    "iv_compression": 10,
    "range_reversion": 11,
    "rsi_divergence": 12,
    # mean_reversion outranks momentum (counter-cyclical conviction beats trend-following)
    "mean_reversion": 13,
    "momentum": 14,
    "macd_crossover": 15,
    "orb_breakout": 16,
    "vwap_reclaim": 17,
    "intraday_momentum": 18,
}

# Canonical default thresholds — used when no walk-forward params are supplied.
# These are the source of truth; backtest/engine.py imports them.
DEFAULT_SIGNAL_PARAMS: dict[str, float] = {
    # mean_reversion
    "rsi_threshold": 35.0,
    "bb_threshold": 0.15,  # tightened from 0.25 in v1.48
    "mr_vol_threshold": 1.2,
    # momentum
    "mom_vol_threshold": 1.3,
    "mom_ret5d_threshold": 1.0,
    "mom12_1_threshold": 10.0,
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
    # inside_day_breakout
    "idb_vol_min": 1.1,
    # trend_pullback
    "tp_ema21_lo": -2.0,  # tightened from -3.0 in v1.48 — buy closer to EMA support
    "tp_ema21_hi": -0.5,  # must be no more than this far below EMA21
    "tp_rsi_lo": 50.0,  # raised from 40.0 in v1.48 — healthy pullback only (RSI ≥50)
    "tp_rsi_hi": 58.0,  # RSI ceiling (not overbought)
    "tp_vol_min": 1.0,
    # iv_compression
    "ivc_hv_rank_max": 0.10,  # tightened from 0.20 in v1.48 — only extreme vol compression
    "ivc_vol_min": 1.2,  # raised from 1.1 in v1.48
    # range_reversion
    "rr_adx_max": 20.0,  # ADX below this = range-bound
    "rr_bb_max": 0.10,  # price in extreme lower band
    "rr_rsi_max": 30.0,  # extreme oversold
    # macd_crossover
    "macd_vol_min": 1.2,
}

# Canonical regime-blocked signal set — imported by both the backtest engine and
# live scanner so the two systems always suppress the same signals in the same regimes.
#
# These are working hypotheses derived from backtest analysis (2015–2026) and
# live paper-trading validation.  Treat every entry as a lean, not a hard rule:
# low trade counts and survivorship risk mean individual cells are suggestive, not proven.
_BEAR_DAY_BLOCKED = frozenset(
    {
        "rs_leader",
        "breakout_52w",
        "momentum_12_1",
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
        "rsi_divergence",  # no mean-reversion buying in stress regimes
    }
)
_HIGH_VOL_BLOCKED = frozenset(
    {
        "rs_leader",
        "breakout_52w",
        "momentum",
        "gap_and_go",
        "orb_breakout",
    }
)
# DEFENSIVE_DOWNTREND: mean_reversion has edge here (WR 53%, avg +0.6%, n=112) — kept.
_DEFENSIVE_BLOCKED = frozenset(
    {
        "rs_leader",
        "breakout_52w",
        "momentum_12_1",
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

# BULL_TREND: rs_leader and momentum_12_1 have no edge (rs_leader WR 51%, avg -0.13%, n=246;
# momentum_12_1 WR 48%, avg -0.2%, n=97 — both p>0.05 Holm-corrected).
# rsi_divergence: divergence setups in uptrends are consolidations, not reversals worth buying.
_BULL_TREND_BLOCKED = frozenset({"rs_leader", "momentum_12_1", "rsi_divergence"})

# ── Short-side signal constants ───────────────────────────────────────────────

SHORT_SIGNAL_PRIORITY: dict[str, int] = {
    "earnings_miss": 0,  # Negative PEAD — strongest bearish fundamental
    "high_short_interest": 1,  # Crowded short + low lendable supply (live-only, not backtestable)
    "ema_breakdown": 2,  # Structural downtrend confirmed by price + EMA slope
}

DEFAULT_SHORT_SIGNAL_PARAMS: dict[str, float] = {
    "ema_breakdown_threshold": -2.0,  # price_vs_ema21_pct must be <= this
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
        price_vs_ema21_pct, ema9_above_ema21, earnings_miss_candidate.
    params : dict | None
        Override DEFAULT_SHORT_SIGNAL_PARAMS thresholds.
    blocked : frozenset[str]
        Signal names to suppress.

    Returns
    -------
    list[str]
        All bearish signals that fired, sorted ascending by SHORT_SIGNAL_PRIORITY.
    """
    p = DEFAULT_SHORT_SIGNAL_PARAMS if params is None else {**DEFAULT_SHORT_SIGNAL_PARAMS, **params}
    matched: list[str] = []

    if "earnings_miss" not in blocked and snapshot.get("earnings_miss_candidate"):
        matched.append("earnings_miss")

    if "high_short_interest" not in blocked and snapshot.get("high_short_interest"):
        matched.append("high_short_interest")

    if (
        "ema_breakdown" not in blocked
        and snapshot.get("price_vs_ema21_pct", 0.0) <= p["ema_breakdown_threshold"]
        and not snapshot.get("ema9_above_ema21", True)
    ):
        matched.append("ema_breakdown")

    matched.sort(key=lambda s: SHORT_SIGNAL_PRIORITY.get(s, 99))
    return matched


REGIME_BLOCKED: dict[str, frozenset[str]] = {
    # Legacy names (kept for backward compatibility)
    "BEAR_DAY": _BEAR_DAY_BLOCKED,
    "HIGH_VOL": _HIGH_VOL_BLOCKED,
    "BULL_TRENDING": _BULL_TREND_BLOCKED,
    "CHOPPY": _NEUTRAL_CHOP_BLOCKED,
    # New 5-state names
    "STRESS_RISK_OFF": _BEAR_DAY_BLOCKED,
    "HIGH_VOL_DOWNTREND": _HIGH_VOL_BLOCKED,
    "DEFENSIVE_DOWNTREND": _DEFENSIVE_BLOCKED,
    "BULL_TREND": _BULL_TREND_BLOCKED,
    "NEUTRAL_CHOP": _NEUTRAL_CHOP_BLOCKED,
    "UNKNOWN": _BEAR_DAY_BLOCKED,
}


def evaluate_signals(
    snapshot: dict,
    blocked: frozenset[str] = frozenset(),
    params: dict | None = None,
    vix_spike: bool = False,
    spy_ret_5d: float | None = None,
    spy_ret_10d: float | None = None,
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

    Returns
    -------
    list[str]
        All signals that fired, sorted ascending by SIGNAL_PRIORITY value.
    """
    p = DEFAULT_SIGNAL_PARAMS if params is None else {**DEFAULT_SIGNAL_PARAMS, **params}

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
    is_inside_day = bool(snapshot.get("is_inside_day", False))
    gap_pct = _f("gap_pct", 0)
    close_above_open = bool(snapshot.get("close_above_open", False))
    insider_cluster = bool(snapshot.get("insider_cluster", False))
    pead_candidate = bool(snapshot.get("pead_candidate", False))
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
        matched.append("vix_fear_reversion")

    # Fundamental conviction — bypass regime filter
    if insider_cluster and "insider_buying" not in blocked:
        matched.append("insider_buying")
    if pead_candidate and ret_5d > 0 and "pead" not in blocked:
        matched.append("pead")

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
        matched.append("rs_leader")

    # 52-week breakout
    if (
        pct_52w >= p["bk52_pct_min"]
        and vol > p["bk52_vol_min"]
        and ema_up
        and adx >= 20
        and "breakout_52w" not in blocked
    ):
        matched.append("breakout_52w")

    # 12-month momentum (skipped when mom_12_1_pct field absent)
    if (
        mom_12_1 is not None
        and mom_12_1 > p["mom12_1_threshold"]
        and ema_up
        and adx >= 20
        and "momentum_12_1" not in blocked
    ):
        matched.append("momentum_12_1")

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
    if (
        bb_squeeze
        and vol > p["bbs_vol_min"]
        and (ema_up or macd_diff > 0)
        and adx >= 20
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

    # IV compression
    if (
        hv_rank < p["ivc_hv_rank_max"]
        and (ema_up or macd_diff > 0)
        and vol > p["ivc_vol_min"]
        and "iv_compression" not in blocked
    ):
        matched.append("iv_compression")

    # Range reversion: extreme oversold within confirmed range-bound conditions.
    # The adx < rr_adx_max gate implicitly restricts this to non-trending regimes.
    if (
        adx < p["rr_adx_max"]
        and bb < p["rr_bb_max"]
        and rsi < p["rr_rsi_max"]
        and "range_reversion" not in blocked
    ):
        matched.append("range_reversion")

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
        matched.append("rsi_divergence")

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

    # MACD crossover
    if macd_up and vol > p["macd_vol_min"] and adx >= 20 and "macd_crossover" not in blocked:
        matched.append("macd_crossover")

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

    matched.sort(key=lambda s: SIGNAL_PRIORITY.get(s, 99))
    return matched
