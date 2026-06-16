import json
import logging
import os
from datetime import date as _date

from config import (
    KELLY_MULTIPLIER,
    LOG_DIR,
    MAX_POSITION_PCT,
    MAX_POSITION_WEIGHT,
    RISK_PER_TRADE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT,
)
from risk.risk_config import RiskConfig

logger = logging.getLogger(__name__)

_SIGNAL_STATS_PATH = os.path.join(LOG_DIR, "signal_stats.json")
_MIN_SAMPLE_SIZE = 5


def _load_signal_stats() -> dict:
    try:
        with open(_SIGNAL_STATS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _empirical_win_rate(signal: str, regime: str) -> float | None:
    """
    Return empirical win rate from signal_stats.json.
    Tries (signal, regime) bucket first, falls back to signal-only.
    Returns None when the best available bucket has fewer than _MIN_SAMPLE_SIZE trades.
    """
    stats = _load_signal_stats()
    signal_data = stats.get(signal)
    if not signal_data:
        return None

    regime_data = signal_data.get("by_regime", {}).get(regime, {})
    if regime_data.get("trades", 0) >= _MIN_SAMPLE_SIZE:
        return regime_data["wins"] / regime_data["trades"]

    if signal_data.get("trades", 0) >= _MIN_SAMPLE_SIZE:
        return signal_data["wins"] / signal_data["trades"]

    return None


def kelly_fraction(
    confidence: int,
    signal: str = "unknown",
    regime: str = "UNKNOWN",
    risk_config: RiskConfig | None = None,
) -> float:
    """
    Half-Kelly position sizing.

    Uses empirical win rate from signal_stats.json when ≥5 samples exist for the
    (signal, regime) bucket; falls back to LLM confidence score as a proxy for p.
    """
    emp = _empirical_win_rate(signal, regime)
    if emp is not None:
        p = emp
        logger.debug(f"kelly_fraction: empirical p={p:.2f} for {signal}/{regime}")
    else:
        p = confidence / 10.0
        logger.debug(
            f"kelly_fraction: no empirical data for {signal}/{regime} — using LLM confidence p={p:.2f}"
        )

    tp = risk_config.take_profit_pct if risk_config is not None else TAKE_PROFIT_PCT
    sl = risk_config.stop_loss_pct if risk_config is not None else STOP_LOSS_PCT
    q = 1.0 - p
    b = tp / max(sl, 1e-6)
    raw_kelly = (p * b - q) / b
    fraction = max(0.0, raw_kelly * KELLY_MULTIPLIER)
    return min(fraction, MAX_POSITION_PCT)


def risk_budget_size(
    equity: float,
    confidence: int,
    signal: str = "unknown",
    regime: str = "UNKNOWN",
    risk_config: RiskConfig | None = None,
) -> float:
    """
    Risk-budget position sizing, scaled by conviction.

    Base notional: if trailing stop triggers → RISK_PER_TRADE_PCT of equity lost.
    Conviction scale: multiplier derived from empirical win rate (when ≥5 samples exist)
    or from AI confidence score (linear from 0.5× at MIN_CONFIDENCE to 1.0× at 10).
    Result is capped at MAX_POSITION_WEIGHT of equity.
    """
    if equity <= 0:
        return 0.0

    tsp = risk_config.trailing_stop_pct if risk_config is not None else TRAILING_STOP_PCT
    risk_usd = equity * RISK_PER_TRADE_PCT
    stop_pct = tsp / 100.0
    base_notional = risk_usd / max(stop_pct, 1e-6)
    base_notional = min(base_notional, equity * MAX_POSITION_WEIGHT)

    emp = _empirical_win_rate(signal, regime)
    if emp is not None:
        # 60%+ win rate → full size; lower → proportionally smaller; floor at 25%
        conviction_scale = max(0.25, min(emp / 0.60, 1.0))
    else:
        # Neutral scale until empirical history accrues — LLM confidence does not drive size
        conviction_scale = 0.75

    try:
        kelly = kelly_fraction(confidence, signal, regime, risk_config=risk_config)
        logger.debug(
            f"risk_budget_size: base={base_notional:.2f} kelly_telemetry={kelly:.3f} "
            f"conviction_scale={conviction_scale:.2f} conf={confidence} {signal}/{regime}"
        )
    except Exception as e:
        logger.debug(f"kelly_fraction unavailable (telemetry only): {e}")

    return max(0.0, base_notional * conviction_scale)


def get_max_positions(portfolio_value: float) -> int:
    """
    Gradually unlock more position slots as the account grows.
    Keeps individual positions large enough to matter.
    """
    if portfolio_value >= 50000:
        return 5
    elif portfolio_value >= 20000:
        return 4
    else:
        return 3


_DRAWDOWN_REDUCE_THRESHOLD = -5.0  # % below all-time peak that triggers size reduction
_DRAWDOWN_SIZE_SCALAR = 0.5  # multiply notional by this when in drawdown


def drawdown_scalar(portfolio_history: list[dict]) -> float:
    """Return 0.5 when portfolio is >5% below its all-time peak, else 1.0.

    Guards against opening full-size positions into a sustained drawdown.
    Plausibility floor scales with account size (half the peak, floored at $10) so the
    filter works in SMALL_ACCOUNT_MODE — a hardcoded $1,000 floor previously discarded
    every record on a ~$150 account, silently disabling drawdown-adaptive sizing.
    Mirrors the filter in risk_manager.check_circuit_breaker.
    """
    if len(portfolio_history) < 2:
        return 1.0
    try:
        _raw_values = [r["account_after"]["portfolio_value"] for r in portfolio_history]
        _peak_raw = max(_raw_values) if _raw_values else 0.0
        _min_plausible = max(10.0, _peak_raw * 0.5)
        values = [v for v in _raw_values if v >= _min_plausible]
        if len(values) < 2:
            return 1.0
        peak = max(values)
        current = values[-1]
        drawdown_pct = (current / peak - 1) * 100
        if drawdown_pct <= _DRAWDOWN_REDUCE_THRESHOLD:
            logger.info(
                f"Drawdown-adaptive sizing: {drawdown_pct:.1f}% from peak "
                f"— position size reduced to {_DRAWDOWN_SIZE_SCALAR:.0%}"
            )
            return _DRAWDOWN_SIZE_SCALAR
        return 1.0
    except (KeyError, IndexError, ZeroDivisionError) as e:
        logger.warning(f"drawdown_scalar: could not compute drawdown: {e}")
        return 1.0


def small_account_size(portfolio_value: float, max_single_order: float = 55.0) -> float:
    """
    Explicit-notional sizing for small-account experiment mode (<$200 account).

    Targets 1-2 positions of $40-$55 rather than risk-budget math that produces
    unusable $5-$8 orders on a £150-scale account. Each position is sized to
    allow at least one whole share for stop protection on names up to $55.

    Returns a notional capped at max_single_order and floored at $40.
    """
    if portfolio_value <= 0:
        return 0.0
    # Aim for 2 equal positions spending ~80% of portfolio
    per_position = (portfolio_value * 0.80) / 2.0
    return max(40.0, min(per_position, max_single_order))


# Signal quality multipliers derived from backtest signal isolation (2015–2023).
# Scale position size by signal's historical Sharpe relative to the median signal.
# Updated quarterly from live trade log as empirical data accumulates.
SIGNAL_SHARPE_MULTIPLIER: dict[str, float] = {
    "iv_compression": 1.5,  # Sharpe +0.62 in isolation — strongest signal
    "pead": 1.3,  # Sharpe +0.45 — high quality, many trades
    "inside_day_breakout": 1.0,  # Sharpe +0.09
    "momentum": 1.0,  # Sharpe +0.05
    "macd_crossover": 1.0,  # Sharpe +0.03
    "bb_squeeze": 0.5,  # Sharpe +0.05 but 78% of trades, noisy — half size
    "insider_buying": 1.3,  # Fundamental conviction; treat as pead-equivalent
    "gap_and_go": 0.8,  # Sharpe -0.11 in isolation — reduce size
    "trend_pullback": 0.8,  # Sharpe -0.05
    "mean_reversion": 0.8,  # Sharpe -0.06
    # Globally disabled (R2: kept current with GLOBALLY_DISABLED) — multiplier defined only as
    # documentation. These never fire; get_signal_size_multiplier defaults to 1.0 for any
    # signal absent here, so new active signals (options/squeeze/sentiment) are unpenalised.
    "breakout_52w": 0.0,
    "rsi_divergence": 0.0,
    "rs_leader": 0.0,
    "momentum_12_1": 0.0,
    "vix_fear_reversion": 0.0,
    "range_reversion": 0.0,
    "volume_climax_reversal": 0.0,
    "tax_loss_reversal": 0.0,
    "obv_divergence": 0.0,
    "obv_acceleration": 0.0,
}


def get_signal_size_multiplier(signal: str) -> float:
    """Return the Sharpe-based size multiplier for signal.

    Returns 1.0 for unknown/new signals (conservative default — don't penalise new signals).
    Returns 0.0 for globally disabled signals.
    """
    return SIGNAL_SHARPE_MULTIPLIER.get(signal, 1.0)


def atr_position_size(
    equity: float,
    atr_pct: float,
    risk_pct: float | None = None,
) -> float:
    """ATR-based equal-risk position sizing.

    position_size = (equity * risk_pct) / (atr_pct / 100)

    Ensures each trade risks the same dollar amount regardless of volatility.
    Capped at equity * MAX_POSITION_WEIGHT.
    Floored at 0.

    atr_pct: 14-day ATR as % of current price (e.g. 3.5 means ATR is 3.5% of price)
    risk_pct: fraction of equity to risk per trade; defaults to RISK_PER_TRADE_PCT

    Returns 0.0 when atr_pct <= 0 or equity <= 0.
    """
    if equity <= 0 or atr_pct <= 0:
        return 0.0
    effective_risk_pct = risk_pct if risk_pct is not None else RISK_PER_TRADE_PCT
    position_size = (equity * effective_risk_pct) / (atr_pct / 100)
    capped = min(position_size, equity * MAX_POSITION_WEIGHT)
    return max(0.0, capped)


_MQS_BOOST = 1.5  # size multiplier when all three quality components fire


def momentum_quality_score(snapshot: dict) -> int:
    """Return a 0–3 conviction score combining momentum rank, EPS revision, and quality.

    Each component contributes 1 point:
      1. Momentum rank: ``rs_rank_pct >= 60`` (top 40% of universe by 20d RS)
      2. EPS revision:  ``pead_candidate == True``  (recent positive EPS beat ≥10%)
      3. Quality proxy: ``roe > 0 and profit_margin > 0``  (profitable on both metrics)

    Score 3 = all three fire → 1.5× size multiplier via ``mqr_size_multiplier``.
    Scores 0-2 → no penalty, no boost.
    """
    score = 0
    if (snapshot.get("rs_rank_pct") or 0.0) >= 60:
        score += 1
    if snapshot.get("pead_candidate"):
        score += 1
    roe = snapshot.get("roe")
    pm = snapshot.get("profit_margin")
    if roe is not None and pm is not None and roe > 0 and pm > 0:
        score += 1
    return score


def mqr_size_multiplier(score: int) -> float:
    """Return 1.5× when momentum quality score is 3, else 1.0."""
    return _MQS_BOOST if score >= 3 else 1.0


_AMIHUD_ILLIQUID_SCALAR = 0.5  # reduce position size 50% for top-10% illiquid symbols


def amihud_size_scalar(amihud_illiquid: bool) -> float:
    """Return 0.5 when symbol is in the top 10% least-liquid symbols, else 1.0.

    Liquidity risk: wide bid-ask spreads and thin books mean exits at stop are
    more costly on illiquid names — halving size keeps the dollar risk constant.
    """
    return _AMIHUD_ILLIQUID_SCALAR if amihud_illiquid else 1.0


def cofiring_boost(n_signals: int) -> float:
    """Return position size multiplier for co-firing signals.

    1 signal: 1.0× (no boost)
    2+ signals: 1.5× (rare convergence, high information content)

    The 1.5× is a hard cap — 3 signals don't get more than 2 do.
    """
    if n_signals >= 2:
        return 1.5
    return 1.0


_VOV_REDUCE_THRESHOLD = 3.5
_VOV_BOOST_THRESHOLD = 1.0
_VOV_REDUCE_SCALAR = 0.7
_VOV_BOOST_SCALAR = 1.2


def vol_of_vol_scalar(vov: float | None) -> float:
    """Return position-size multiplier based on 10-day VIX volatility-of-volatility.

    High VoV (>3.5) signals an unstable volatility regime — reduce all positions.
    Low VoV (<1.0) signals a quiet, mean-reverting regime — allow a small boost.
    """
    if vov is None:
        return 1.0
    if vov > _VOV_REDUCE_THRESHOLD:
        return _VOV_REDUCE_SCALAR
    if vov < _VOV_BOOST_THRESHOLD:
        return _VOV_BOOST_SCALAR
    return 1.0


_TOM_SCALAR = 1.05  # turn-of-month institutional flows boost longs
_POST_OPEX_SCALAR = 1.10  # post-OPEX directional release after gamma pinning
_QUARTER_END_SCALAR = 1.10  # window dressing: fund managers chase momentum winners
_PRE_HOLIDAY_SCALAR = 1.05  # light-tape thin-volume day before NYSE holiday
_OPEX_WEEK_SCALAR = 0.70  # OPEX week: gamma pinning suppresses gap/momentum edge
_HALLOWEEN_BULLISH_SCALAR = 1.10  # Nov–Apr: seasonally stronger market half
_HALLOWEEN_BEARISH_SCALAR = 0.90  # May–Oct: sell-in-May seasonality

_OPEX_WEEK_DAMPENED_SIGNALS: frozenset[str] = frozenset({"gap_and_go", "momentum"})
_QUARTER_END_BOOSTED_SIGNALS: frozenset[str] = frozenset(
    {"momentum", "bb_squeeze", "trend_pullback"}
)


def seasonal_scalar(signal: str, check_date: _date | None = None) -> float:
    """Return a position-size multiplier from calendar/seasonal context.

    Covers six effects: halloween seasonality, OPEX week dampening, post-OPEX boost,
    turn-of-month flows, quarter-end window dressing, and pre-holiday thin-tape boost.
    Result is clamped to [0.70, 1.25] to limit combined drift.
    """
    from risk.macro_calendar import get_seasonal_context

    ctx = get_seasonal_context(check_date)
    scalar = 1.0

    if ctx.get("halloween_bullish"):
        scalar *= _HALLOWEEN_BULLISH_SCALAR
    else:
        scalar *= _HALLOWEEN_BEARISH_SCALAR

    if ctx.get("opex_week") and signal in _OPEX_WEEK_DAMPENED_SIGNALS:
        scalar *= _OPEX_WEEK_SCALAR
    if ctx.get("post_opex"):
        scalar *= _POST_OPEX_SCALAR
    if ctx.get("turn_of_month"):
        scalar *= _TOM_SCALAR
    if ctx.get("quarter_end_dressing") and signal in _QUARTER_END_BOOSTED_SIGNALS:
        scalar *= _QUARTER_END_SCALAR
    if ctx.get("pre_holiday"):
        scalar *= _PRE_HOLIDAY_SCALAR

    return max(0.70, min(scalar, 1.25))


# Signals that represent cyclical / trend-following bets (benefit from expansion).
_CYCLICAL_SIGNALS: frozenset[str] = frozenset(
    {
        "momentum",
        "gap_and_go",
        "trend_pullback",
        "bb_squeeze",
        "inside_day_breakout",
        "golden_cross",
        "obv_acceleration",
        "rs_leader",
        "breadth_thrust",
    }
)

_MACRO_EXPANSION_SCALAR = 1.10  # steep positive yield curve → expansion boost for cyclicals
_MACRO_RECESSION_SCALAR = 0.80  # sustained yield-curve inversion → reduce all longs
_MACRO_USD_STRONG_SCALAR = 0.90  # rising USD → headwind to earnings, risk appetite
_MACRO_COPPER_CYCLICAL_SCALAR = 1.10  # copper-gold expansion ratio → boost cyclicals


_CORR_HIGH_THRESHOLD = 0.75
_CORR_LOW_THRESHOLD = 0.35
_CORR_HIGH_SCALAR = 0.85  # high sector-beta: stock tracks sector closely → reduce
_CORR_LOW_SCALAR = 1.10  # decorrelated idiosyncratic mover → boost


def correlation_scalar(corr: float | None) -> float:
    """Return position-size multiplier based on 20-day stock-vs-sector correlation.

    High correlation (>0.75): high systemic exposure — dampen size.
    Low correlation (<0.35): idiosyncratic mover — small boost.
    """
    if corr is None:
        return 1.0
    if corr > _CORR_HIGH_THRESHOLD:
        return _CORR_HIGH_SCALAR
    if corr < _CORR_LOW_THRESHOLD:
        return _CORR_LOW_SCALAR
    return 1.0


_NHL_EXPANSION_THRESHOLD = 2.0
_NHL_CONTRACTION_THRESHOLD = 0.5
_NHL_EXPANSION_SCALAR = 1.10  # NH/NL > 2 → breadth expanding → boost longs
_NHL_CONTRACTION_SCALAR = 0.80  # NH/NL < 0.5 → breadth deteriorating → dampen longs


def nhl_scalar(nhl_ratio: float | None) -> float:
    """Return position-size multiplier based on new-high / new-low breadth ratio.

    NH/NL > 2.0: expansion regime — more conviction in longs.
    NH/NL < 0.5: contraction — dampen longs.
    """
    if nhl_ratio is None:
        return 1.0
    if nhl_ratio > _NHL_EXPANSION_THRESHOLD:
        return _NHL_EXPANSION_SCALAR
    if nhl_ratio < _NHL_CONTRACTION_THRESHOLD:
        return _NHL_CONTRACTION_SCALAR
    return 1.0


def macro_scalar(snapshot: dict, signal: str) -> float:
    """Return a position-size multiplier from macro/rates context.

    Reads macro flags injected into the stock snapshot dict (macro_* keys).
    Combines ETF-based signals (credit stress, duration flight, copper-gold,
    USD) with FRED series (yield curve, PMI) into a single scalar.
    Result is clamped to [0.70, 1.25].
    """
    scalar = 1.0

    yc: float | None = snapshot.get("macro_yield_curve")
    inv_days = int(snapshot.get("macro_yield_curve_inverted_days", 0))
    copper_gold = bool(snapshot.get("macro_copper_gold_positive", False))
    usd_strong = bool(snapshot.get("macro_usd_strong", False))

    # Yield curve: sustained inversion → reduce all longs (recession risk signal)
    if yc is not None and yc < 0 and inv_days >= 60:
        scalar *= _MACRO_RECESSION_SCALAR
    # Yield curve: steep positive + cyclical → expansion boost
    elif yc is not None and yc >= 1.5 and signal in _CYCLICAL_SIGNALS:
        scalar *= _MACRO_EXPANSION_SCALAR

    # Copper-gold ratio expanding → boost cyclical signals (risk-on growth regime)
    if copper_gold and signal in _CYCLICAL_SIGNALS:
        scalar *= _MACRO_COPPER_CYCLICAL_SCALAR

    # Strong USD → dampen longs (earnings headwind, reduced global risk appetite)
    if usd_strong:
        scalar *= _MACRO_USD_STRONG_SCALAR

    return max(0.70, min(scalar, 1.25))
