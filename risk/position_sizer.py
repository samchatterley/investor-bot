import json
import logging
import os

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
