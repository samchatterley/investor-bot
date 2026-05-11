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
    # rs_leader and breakout_52w outrank bb_squeeze; breakout_52w < momentum_12_1 < gap_and_go
    "rs_leader": 3,
    "breakout_52w": 4,
    "momentum_12_1": 5,
    "gap_and_go": 6,
    "bb_squeeze": 7,
    "inside_day_breakout": 8,
    "trend_pullback": 9,
    "iv_compression": 10,
    # mean_reversion outranks momentum (counter-cyclical conviction beats trend-following)
    "mean_reversion": 11,
    "momentum": 12,
    "macd_crossover": 13,
    "orb_breakout": 14,
    "vwap_reclaim": 15,
    "intraday_momentum": 16,
}

# Canonical default thresholds — used when no walk-forward params are supplied.
# These are the source of truth; backtest/engine.py imports them.
DEFAULT_SIGNAL_PARAMS: dict[str, float] = {
    "rsi_threshold": 35.0,
    "bb_threshold": 0.25,
    "mr_vol_threshold": 1.2,
    "mom_vol_threshold": 1.3,
    "mom_ret5d_threshold": 1.0,
    "mom12_1_threshold": 10.0,
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
    if vix_spike and vol > 1.0 and "vix_fear_reversion" not in blocked:
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
        and (ret_5d - spy_ret_5d) > 2.0
        and (ret_10d - spy_ret_10d) > 3.0
        and ema_up
        and adx >= 20
    ):
        matched.append("rs_leader")

    # 52-week breakout
    if pct_52w >= -3.0 and vol > 1.2 and ema_up and adx >= 20 and "breakout_52w" not in blocked:
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
        gap_pct > 2.0
        and close_above_open
        and vol > 1.5
        and adx >= 20
        and "gap_and_go" not in blocked
    ):
        matched.append("gap_and_go")

    # Bollinger squeeze
    if (
        bb_squeeze
        and vol > 1.2
        and (ema_up or macd_diff > 0)
        and adx >= 20
        and "bb_squeeze" not in blocked
    ):
        matched.append("bb_squeeze")

    # Inside day breakout
    if (
        is_inside_day
        and vol > 1.1
        and (ema_up or macd_diff > 0)
        and adx >= 20
        and "inside_day_breakout" not in blocked
    ):
        matched.append("inside_day_breakout")

    # Trend pullback to EMA21 within uptrend
    if (
        ema_up
        and -3.0 <= pct_ema21 <= -0.5
        and 40 <= rsi <= 58
        and vol > 1.0
        and adx >= 20
        and "trend_pullback" not in blocked
    ):
        matched.append("trend_pullback")

    # IV compression
    if (
        hv_rank < 0.20
        and (ema_up or macd_diff > 0)
        and vol > 1.1
        and "iv_compression" not in blocked
    ):
        matched.append("iv_compression")

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
    if macd_up and vol > 1.2 and adx >= 20 and "macd_crossover" not in blocked:
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
