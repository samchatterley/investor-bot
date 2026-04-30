import json
import logging
import os

from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_POSITION_PCT, KELLY_MULTIPLIER, LOG_DIR

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


def kelly_fraction(confidence: int, signal: str = "unknown", regime: str = "UNKNOWN") -> float:
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
        logger.debug(f"kelly_fraction: no empirical data for {signal}/{regime} — using LLM confidence p={p:.2f}")

    q = 1.0 - p
    b = TAKE_PROFIT_PCT / max(STOP_LOSS_PCT, 1e-6)
    raw_kelly = (p * b - q) / b
    fraction = max(0.0, raw_kelly * KELLY_MULTIPLIER)
    return min(fraction, MAX_POSITION_PCT)


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
